import torch
from tokenizers import Tokenizer
from torch import nn
from typing import Union, List
import os
import glob


class RotaryEmbedding(torch.nn.Module):
    """ """

    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
            if self.precision == torch.bfloat16:
                self.cos_cached = self.cos_cached.bfloat16()
                self.sin_cached = self.sin_cached.bfloat16()
        return self.cos_cached, self.sin_cached


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


def gpt2_attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores


class GPTNeoxAttention(nn.Module):
    """
    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config,
        layer_number,
    ):
        super().__init__()

        self.attention_mask_func = gpt2_attention_mask_func
        # self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        # self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        # if self.apply_query_key_layer_scaling:
        #     self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number
        # Per attention head and per partition values.
        self.pos_emb = config.pos_emb
        return

        # Strided linear layer.
        self.query_key_value = mpu.ColumnParallelLinear(
            neox_args=config,
            input_size=config.hidden_size,
            output_size=3 * config.hidden_size,
            gather_output=False,
            init_method=init_method,
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

            assert config.rotary_pct < 1
            self.rotary_ndims = int(
                self.hidden_size_per_attention_head * config.rotary_pct
            )
        dim = self.rotary_ndims
        self.rotary_emb = RotaryEmbedding(
            dim, base=config.rotary_emb_base, precision=config.params_dtype
        )

        self.attention_type = config.attention_config[layer_number]

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.fp16,
            input_in_bf16=self.bf16,
            fusion_type=get_fusion_type(config),
            mask_func=self.attention_mask_func,
            softmax_in_fp32=self.attention_softmax_in_fp32,
            scale=coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        # Output.
        self.dense = mpu.RowParallelLinear(
            neox_args=config,
            input_size=config.hidden_size,
            output_size=config.hidden_size,
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
        # with mpu.get_cuda_rng_tracker().fork():
        #   attention_probs = self.attention_dropout(attention_probs)

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
    max_position_embeddings: int = 2048
    num_hidden_layers = 44
    attention_config = [[["global"], 44]]
    layer_norm_epsilon: float = 1.0e-5
    attention_dropout: float = 0.0
    gpt_j_residual: bool = True
    hidden_dropout: float = 0.0
    hidden_size: int = 6144
    make_vocab_size_divisible_by: int = 256
    no_load_rng: bool = True
    no_weight_tying: bool = True
    norm: str = "layernorm"
    num_attention_heads: int = 64
    num_layers: int = 44
    partition_activations: bool = False
    pos_emb: str = "rotary"
    rotary_pct: float = 0.25
    seq_length: int = 2048
    split: List[int] = [995, 4, 1]
    tokenizer_type: str = "HFTokenizer"
    fp16: bool = True
    vocab_size: int = 54031


class GPTNeoxBlock(nn.Module):
    def __init__(self, config, layer_number: int):
        super().__init__()
        hidden_size = config.hidden_size
        self.input_layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTNeoxAttention(config, layer_number=layer_number)
        self.post_attention_layernorm = nn.LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon
        )

        # self.crossattention = GPTNeoxAttention(config, is_cross_attention=True)
        # self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # self.mlp = GPT2MLP(inner_dim, config)

    def forward(self, x, attention_mask):
        x = self.norm(x)
        x, _ = self.input_linear(x)
        x = self.activation_func(x)
        x = self.sgu(x, attention_mask)
        x, _ = self.output_linear(x)
        return x, attention_mask


def _pre_transformer_block(args):
    # data format change for hidden_states to avoid explicit tranposes : [b s h] --> [s b h]
    assert len(args) == 2, "Incorrect number of arguments to _pre_transformer_block"
    fn = lambda _args: (_args[0].transpose(0, 1).contiguous(), *_args[1:])
    return fn(args)


def _post_transformer_block(args):
    # from (hidden_states, attention_mask)
    # to (hidden_states.T)
    assert len(args) == 2, "Incorrect number of arguments to _post_transformer_block"
    fn = lambda _args: (_args[0].transpose(0, 1).contiguous())
    return fn(args)


class GPTNeox20BModel(nn.Module):
    def __init__(self, config: GPTNeox20BModelConfig):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.vocab_size = config.vocab_size

        # weight token embedding
        self.word_embeddings_weight = nn.Embedding(self.vocab_size, self.embed_dim)
        # wight position embedding
        self.wpe = RotaryEmbedding(self.embed_dim)

        self.drop = nn.Dropout(config.hidden_dropout)
        self.tranformer = nn.ModuleList(
            # [_pre_transformer_block]
            [
                GPTNeoxBlock(config, layer_number=i)
                for i in range(config.num_hidden_layers)
            ]
            # + [_post_transformer_block]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

    def forward(self, input_ids, position_ids, attention_mask):
        b, t = input_ids.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        token_embeddings = self.word_embeddings_weight(input_ids)
        x = self.rotary_embeddings(token_embeddings)
        x = self.drop(x)
        x = self.transformer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def vocab_size_with_padding(config, orig_vocab_size):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = config.make_vocab_size_divisible_by
    while (after % multiple) != 0:
        after += 1
    print(
        " > padded vocab (size: {}) with {} dummy tokens "
        "(new size: {})".format(orig_vocab_size, after - orig_vocab_size, after),
        flush=True,
    )
    return after


def load_checkpoint(load_path):
    print(load_path)
    assert os.path.exists(load_path)
    tokenizer = Tokenizer.from_file("./20_checkpoints_merged/20B_tokenizer.json")

    config = GPTNeox20BModelConfig()
    config.vocab_size = vocab_size_with_padding(config, tokenizer.get_vocab_size())

    model = GPTNeox20BModel(config)
    model.to(torch.device("cpu"))

    for ptfile in sorted(glob.glob(os.path.join(load_path, "*.pt"))):
        checkpoint = torch.load(ptfile)
        print(ptfile)
        for k, v in checkpoint.items():
            if v:
                print("\t", k, v.size().list())


if __name__ == "__main__":
    load_checkpoint("20_checkpoints_merged/global_step150000/")
