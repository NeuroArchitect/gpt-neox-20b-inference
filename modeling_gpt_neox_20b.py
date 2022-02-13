import torch
from tokenizers import Tokenizer
from torch import nn
from typing import Union, List
import os
import glob


class SpatialGatingUnit(nn.Module):
    def __init__(self, neox_args, d_ff, d_attn=None, causal=True, mask_fn=None):
        super().__init__()
        self.causal = causal
        self.use_attn = d_attn is not None

        norm, eps = get_norm(neox_args)
        self.norm = norm(d_ff, eps=eps)
        self.proj = nn.Linear(neox_args.seq_length, neox_args.seq_length)
        if self.use_attn:
            assert mask_fn is not None
            self.attn = TinyAttention(
                neox_args=neox_args, d_attn=d_attn, d_ff=d_ff, mask_fn=mask_fn
            )

    def forward(self, x, attention_mask):
        device, n = x.device, x.shape[1]
        x = x.transpose(0, 1)  # [s, b, d] -> [b, s, d]

        res, gate = x.chunk(2, dim=-1)  # split along dim
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias
        if self.causal:
            weight, bias = weight[:n, :n], bias[:n]
            mask = torch.ones(weight.shape[:2], device=device).triu_(1).bool()
            weight = weight.masked_fill(mask, 0.0)

        gate = F.linear(gate.transpose(2, 1), weight, self.proj.bias).transpose(2, 1)

        if self.use_attn:
            gate = gate + self.attn(x, attention_mask)

        return (gate * res).transpose(0, 1)  # [b, s, d] -> [s, b, d]


class GPTNeoxAttention(nn.Module):
    """
    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
    ):
        super().__init__()

        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"

        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = neox_args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = neox_args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number
        # Per attention head and per partition values.
        self.pos_emb = neox_args.pos_emb

        # Strided linear layer.
        self.query_key_value = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=3 * neox_args.hidden_size,
            gather_output=False,
            init_method=init_method,
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        if neox_args.rotary_pct == 1:
            self.rotary_ndims = None
        else:
            assert neox_args.rotary_pct < 1
            self.rotary_ndims = int(
                self.hidden_size_per_attention_head * neox_args.rotary_pct
            )
        dim = (
            self.rotary_ndims
            if self.rotary_ndims is not None
            else self.hidden_size_per_attention_head
        )
        self.rotary_emb = RotaryEmbedding(
            dim, base=neox_args.rotary_emb_base, precision=neox_args.params_dtype
        )

        self.attention_type = neox_args.attention_config[layer_number]
        self.sparse = self.attention_type != "global"
        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.fp16,
            input_in_bf16=self.bf16,
            fusion_type=get_fusion_type(neox_args),
            mask_func=self.attention_mask_func,
            softmax_in_fp32=self.attention_softmax_in_fp32,
            scale=coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = nn.Dropout(neox_args.attention_dropout)

        # Output.
        self.dense = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
        )

    def attention(
        self, query_layer, key_layer, value_layer, layer_past, attention_mask
    ):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.use_cache:
            with torch.no_grad():
                attention_mask = attention_mask[
                    ..., : attention_scores.size(3), : attention_scores.size(3)
                ]

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def forward(self, hidden_states, attention_mask, layer_past=None):

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(
            mixed_x_layer, 3
        )

        # partial rotary
        query_rot, query_pass = (
            query_layer[..., : self.rotary_ndims],
            query_layer[..., self.rotary_ndims :],
        )
        key_rot, key_pass = (
            key_layer[..., : self.rotary_ndims],
            key_layer[..., self.rotary_ndims :],
        )

        apply_rotary_fn = (
            apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb
        )

        seq_len = key_layer.shape[0]
        offset = 0
        if exists(layer_past) and layer_past.numel() > 0:
            offset = layer_past[0].shape[0]
            seq_len += offset
        cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
        query_layer, key_layer = apply_rotary_fn(
            query_rot, key_rot, cos, sin, offset=offset
        )

        if exists(self.rotary_ndims):
            query_layer = torch.cat((query_layer, query_pass), dim=-1)
            key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # ==================================
        # Cache key and value for inference
        # ==================================

        if exists(layer_past) and layer_past.numel() > 0:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat(
                (past_value.type_as(value_layer), value_layer), dim=0
            )

        if self.use_cache:
            present = torch.stack((key_layer, value_layer))

        if not self.sparse:
            context_layer = self.attention(
                query_layer, key_layer, value_layer, layer_past, attention_mask
            )
        else:
            context_layer = self.sparse_attention(
                query_layer, key_layer, value_layer, attention_mask
            )

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if self.use_cache:
            output = [output, present]

        return output, bias


class GPTNeox20BModelConfig:
    vocab_size = "xx"
    max_position_embeddings = 2048
    hidden_size = "xx"
    num_hidden_layers = 44
    attention_config = [[["global"], 44]]
    layer_norm_epsilon: float = 1.0e-5


class GPTNeoxBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super(self).__init__()
        hidden_size = config.hidden_size
        self.input_layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTNeoxAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = nn.LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon
        )

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(
                hidden_size, eps=config.layer_norm_epsilon
            )

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(self, x, attention_mask):
        x = self.norm(x)
        x, _ = self.input_linear(x)
        x = self.activation_func(x)
        x = self.sgu(x, attention_mask)
        x, _ = self.output_linear(x)
        return x, attention_mask


class GPTNeox20BModel(nn.Module):
    def __init__(self, config: GPTNeox20BModelConfig):
        super().__init__()

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [GPTNeoxBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, position_ids, attention_mask):
        pass


def load_checkpoint(load_path):
    print(load_path)
    assert os.path.exists(load_path)

    config = GPTNeox20BModelConfig()

    model = GPTNeox20BModel(config)
    model.to(torch.device("cpu"))

    for ptfile in sorted(glob.glob(os.path.join(load_path, "*.pt"))):
        checkpoint = torch.load(ptfile)
        print(ptfile)
        for k, v in checkpoint.items():
            if v:
                print("\t", k, v.size().list())


if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("./20B_checkpoints/20B_tokenizer.json")

    encoded = tokenizer.encode("test")
    original = tokenizer.decode(encoded.ids)
    print(original)

    load_checkpoint("20B_checkpoints/global_step150000/")
