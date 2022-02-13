# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from megatron.model import (
    GPT2ModelPipe,
    SoftEmbedding,
)
import deepspeed
import datetime
import torch
import json
import os
import threading
import argparse
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from typing import List

from megatron import print_rank_0, mpu
from megatron.checkpointing import load_checkpoint
from megatron.utils import get_total_params
from megatron.text_generation_utils import pad_batch, forward_model, filter_logits, switch, stop_tokens_in_completion, generate_samples_from_prompt
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron

from functools import partial
import copy
from torch import functional as F
GENERATE_NUM = 0
lock = threading.Lock()


def print_latency(latency_set, title=""):
    # 10 warmup queries
    latency_set = latency_set[10:]
    count = len(latency_set)
    if count > 0:
        latency_set.sort()
        n50 = (count - 1) * 0.5 + 1
        n90 = (count - 1) * 0.9 + 1
        n95 = (count - 1) * 0.95 + 1
        n99 = (count - 1) * 0.99 + 1
        n999 = (count - 1) * 0.999 + 1

        avg = sum(latency_set) / count
        p50 = latency_set[int(n50) - 1]
        p90 = latency_set[int(n90) - 1]
        p95 = latency_set[int(n95) - 1]
        p99 = latency_set[int(n99) - 1]
        p999 = latency_set[int(n999) - 1]

        print("====== latency stats {0} ======", title)
        print("\tAvg Latency: {0:8.2f} ms".format(avg * 1000))
        print("\tP50 Latency: {0:8.2f} ms".format(p50 * 1000))
        print("\tP90 Latency: {0:8.2f} ms".format(p90 * 1000))
        print("\tP95 Latency: {0:8.2f} ms".format(p95 * 1000))
        print("\tP99 Latency: {0:8.2f} ms".format(p99 * 1000))
        print("\t999 Latency: {0:8.2f} ms".format(p999 * 1000))


class MegatronGenerate(Resource):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def send_do_generate():
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice, 0)

    def put(self):
        print("request IP: " + str(request.remote_addr))
        print(json.dumps(request.get_json()), flush=True)
        print("current time: ", datetime.datetime.now())

        if not "prompt" in request.get_json():
            return "prompt argument required", 400

        if "max_len" in request.get_json():
            return "max_len is no longer used.  Replace with tokens_to_generate", 400

        if "sentences" in request.get_json():
            return "sentences is no longer used.  Replace with prompts", 400

        prompt = request.get_json()["prompt"]

        max_tokens = 64
        if "max_tokens" in request.get_json():
            max_tokens = request.get_json()["max_tokens"]
            if not isinstance(max_tokens, int):
                return "max_tokens must be an integer greater than 0"
            if max_tokens < 0:
                return "max_tokens must be an integer greater than or equal to 0"

        logprobs = False
        if "logprobs" in request.get_json():
            logprobs = request.get_json()["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value"

        if max_tokens == 0 and not logprobs:
            return "tokens_to_generate=0 implies logprobs should be True"

        temperature = 1.0
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not (type(temperature) == int or type(temperature) == float):
                return "temperature must be a positive number less than or equal to 100.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"

        top_k = 0.0
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not (type(top_k) == int):
                return "top_k must be an integer equal to or greater than 0 and less than or equal to 1000"
            if not (0 <= top_k <= 1000):
                return "top_k must be equal to or greater than 0 and less than or equal to 1000"

        top_p = 0.0
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not (type(top_p) == float):
                return "top_p must be a positive float less than or equal to 1.0"
            if top_p > 0.0 and top_k > 0.0:
                return "cannot set both top-k and top-p samplings."
            if not (0 <= top_p <= 1.0):
                return "top_p must be less than or equal to 1.0"

        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"

        with lock:  # Need to get lock to keep multiple threads from hitting code
            MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
            response, response_seg, response_logprobs, _ = \
                generate_and_post_process(
                    self.model,
                    self.tokenizer,
                    prompts=prompt,
                    tokens_to_generate=max_tokens,
                    return_output_log_probs=logprobs,
                    top_k_sampling=top_k,
                    top_p_sampling=top_p,
                    temperature=temperature,
                    add_BOS=add_BOS,
                    use_eod_token_for_early_termination=True)

        return jsonify({"text": response,
                        "segments": response_seg,
                        "logprobs": response_logprobs})


class MegatronServer(object):
    def __init__(self, model, tokenizer):
        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/api',
                         resource_class_args=[model, tokenizer])

    def run(self, host, **options):
        self.app.run(host, threaded=True, debug=False, **options)


def stream_tokens(
    model,
    context_tokens: List[List[int]],
    eos_token_id: int = None,
    maximum_tokens: int = None,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    stop_tokens=None,
    seq_length: int = 0,
    tokenizer=None,
):

    # pad batch in order to allow conversion to tensor
    context_tokens, context_lengths = pad_batch(
        copy.deepcopy(context_tokens),
        pad_id=tokenizer.eod,
        pad_len=seq_length,
    )

    # convert to tensor and broadcast
    context_tokens = torch.cuda.LongTensor(context_tokens)
    if stop_tokens:
        stop_tokens = torch.cuda.LongTensor(stop_tokens)
        if stop_tokens.ndim == 1:
            stop_tokens = stop_tokens.unsqueeze(0)

    # Make sure context tokens + start tokens are the same across all ranks
    token_generation_start_index = torch.cuda.LongTensor(context_lengths)
    torch.distributed.broadcast(
        context_tokens,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )
    torch.distributed.broadcast(
        token_generation_start_index,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )

    # get attention mask / position ids
    # Move to GPU.
    context_tokens = context_tokens.contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=context_tokens,
        eod_token=tokenizer.eod,
        eod_mask_loss=False,
    )

    # set variables
    eos_token_id = eos_token_id or tokenizer.eod
    maximum_tokens = maximum_tokens or (
        seq_length - token_generation_start_index.max().item() - 1
    )
    batch_size = context_tokens.size(0)

    # get the context_index at which generation is to start
    # we start generation at the position where the smallest context ends
    token_index_to_generate = token_generation_start_index.min().item()
    first_token_index_to_generate = token_index_to_generate
    last_token_index_to_generate = min(
        seq_length
        - 1,  # never generate more than the model's sequence length
        token_index_to_generate + maximum_tokens - 1,
    )

    with torch.no_grad():
        # initialize generation variables
        state_is_done = torch.zeros([batch_size]).byte().cuda()
        token_generation_end_index = torch.ones(
            [batch_size]).long().cuda() * (-1)

        while token_index_to_generate <= last_token_index_to_generate:
            if recompute:  # recompute all tokens
                model_inputs = (
                    context_tokens,
                    position_ids,
                    attention_mask,
                )
                logits = forward_model(
                    model, model_inputs, model.is_pipe_parallel)

                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = logits[
                        :, token_index_to_generate - 1, :
                    ]  # [bs, seq, vocab_size] -> [bs, vocab_size]
            else:  # use kv cache
                if token_index_to_generate == first_token_index_to_generate:
                    tokens_to_use = context_tokens[:, :token_index_to_generate]
                    positions_to_use = position_ids[:,
                                                    :token_index_to_generate]
                else:
                    tokens_to_use = context_tokens[:, token_index_to_generate - 1].view(
                        batch_size, -1
                    )
                    positions_to_use = position_ids[
                        :, token_index_to_generate - 1
                    ].view(batch_size, -1)

                model_inputs = (
                    tokens_to_use,  # input_ids
                    positions_to_use,  # position_ids
                    attention_mask,  # attention_mask
                )

                logits = forward_model(
                    model, model_inputs, model.is_pipe_parallel)
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = (
                        logits[:, -1].view(batch_size, -1).contiguous()
                    )  # [bs, seq, vocab_size] -> [bs, vocab_size]

            if logits is not None:
                # sample token id of the to be generated token
                if temperature == 0.0 and top_k == 0 and top_p == 0.0:
                    generated_tokens = torch.argmax(
                        generated_token_logits, dim=-1
                    ).view(-1)
                else:
                    generated_token_logits = generated_token_logits.float()
                    if temperature > 0.0:
                        generated_token_logits /= temperature
                    generated_token_logits = filter_logits(
                        generated_token_logits, top_k=top_k, top_p=top_p
                    )
                    next_token_log_probs = F.softmax(
                        generated_token_logits, dim=-1)
                    generated_tokens = torch.multinomial(
                        next_token_log_probs, num_samples=1
                    ).view(-1)

            if model.is_pipe_parallel:
                # broadcast generated tokens to pipe parallel group
                src_rank = model.grid.stage_to_global(model.num_stages - 1)
                generated_tokens = (
                    generated_tokens
                    if logits is not None
                    else torch.zeros(batch_size, dtype=torch.long).cuda()
                )
                torch.distributed.broadcast(
                    tensor=generated_tokens,
                    src=src_rank,
                    group=mpu.get_pipe_parallel_group(),
                )

            # determine if state has started for each batch item
            state_started = (
                token_generation_start_index <= token_index_to_generate
            )  # check which batch items have been started

            # switch out padding tokens for generated tokens
            context_tokens[:, token_index_to_generate] = switch(
                context_tokens[:, token_index_to_generate].view(-1),
                generated_tokens,
                state_started,
            )

            # determine if state has finished for each batch item
            state_done = (
                generated_tokens == eos_token_id
            ).byte() & state_started.byte()  # check which batch items produce an eos_token in the current iteration
            state_just_finished = (state_done & ~state_is_done).bool()
            state_is_done = state_is_done | state_done
            stop_tokens_produced = torch.zeros_like(state_is_done)
            for batch_idx, ctx in enumerate(context_tokens):
                stop_tokens_produced[batch_idx] = stop_tokens_in_completion(
                    stop_tokens, context_tokens, batch_idx, token_index_to_generate
                )
            state_is_done = state_is_done | stop_tokens_produced

            token_generation_end_index[
                (state_started.byte() & ~state_is_done).bool()
            ] = token_index_to_generate

            token_index_to_generate += 1

            if torch.all(state_is_done):
                break
        return context_tokens, token_generation_start_index, token_generation_end_index


def generate_and_post_process(
    model,
    tokenizer,
    prompts,
    tokens_to_generate,
    return_output_log_probs,
    top_k_sampling,
    top_p_sampling,
    temperature,
    # add_BOS,
    # use_eod_token_for_early_termination,
):

    generated_text = ""

    (batch_context_tokens,
        batch_token_generation_start_index,
        batch_token_generation_end_index
     ) = stream_tokens(
        model=model,
        context_tokens=[context_tokens],
        eos_token_id=tokenizer.eod,
        maximum_tokens=max_seq_length,
        recompute=False,
        temperature=temperature,
        top_k=top_k_sampling,
        top_p=top_p_sampling,
        seq_length=tokens_to_generate,
        tokenizer=tokenizer,
    )

    if mpu.get_model_parallel_rank() == 0:
        generated_tokens = (
            batch_context_tokens[0]
            .cpu()
            .numpy()
            .tolist()[
                batch_token_generation_start_index[0]
                .item(): batch_token_generation_end_index[0]
                .item()
            ]
        )

        generated_text = tokenizer.detokenize(generated_tokens)
    return generated_text


def get_model(neox_args, use_cache=False):
    """Build the model."""

    print_rank_0("building GPT2 model ...")

    # Build model on cpu.
    model = GPT2ModelPipe(
        neox_args=neox_args,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    ### soft prompt tuning stuff ###
    if neox_args.soft_prompt_tuning is not None and neox_args.soft_prompt_tuning.get(
        "enabled", False
    ):
        soft_prompt = SoftEmbedding(
            neox_args,
            wte=getattr(model, "0").word_embeddings,
            n_tokens=neox_args.soft_prompt_tuning.get("n_tokens", 10),
            init_string=neox_args.soft_prompt_tuning.get("init_string", ""),
            init_range=neox_args.soft_prompt_tuning.get("init_range", 0.5),
        )
        model.insert_layers(
            layers=soft_prompt, idx=1
        )  # insert the soft prompt layer directly after the word embeddings

        # freeze everything but the soft prompt
        for name, param in model.named_parameters():
            if not "soft_embedding" in name:
                param.requires_grad = False

    if not neox_args.is_pipe_parallel:
        # Export PipeParallel model to nn.Sequential model to avoid the overhead of deepspeed's pipe parallel training
        model = model.to_sequential()

    if neox_args.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return model
    else:
        raise ValueError("Must be using deepspeed to run neox")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="GPT-NeoX Configuration", allow_abbrev=False
    )

    group = parser.add_argument_group(title="Inference Configuration")

    group.add_argument(
        "--local_rank",
        type=int,
    )

    group.add_argument(
        "--conf_dir",
        "-d",
        type=str,
        default=None,
        help="Directory to prefix to all configuration file paths",
    )

    group.add_argument(
        "conf_file",
        type=str,
        nargs="+",
        help="Configuration file path. Multiple files can be provided and will be merged.",
    )

    return parser.parse_args()


def genconfig(conf_dir, conf_files, overwrite_values):
    # load config files
    if conf_dir:
        conf_files = [os.path.join(conf_dir, f)
                      for f in conf_files]

    # enables us to pass in `small` instead of `small.yml`
    conf_files = [(cf if cf.endswith(".yml") else cf + ".yml")
                  for cf in conf_files]

    # load args
    neox_args = NeoXArgs.from_ymls(
        paths_to_yml_files=conf_files, overwrite_values=overwrite_values
    )
    return neox_args


def _get_batch(neox_args, tokenizer, keys, data, datatype):
    """Support function for get_batch / get_batch pipe (to avoid code repetition)"""
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch_pipe(data, neox_args):
    """A modification of get_batch() to work with the latest batch instead of an iterator."""
    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    tokens, labels, loss_mask, attention_mask, position_ids = _get_batch(
        neox_args, neox_args.tokenizer, keys, data, datatype
    )

    # unpack data
    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def tokenize_text(tokenizer,
                  prompt,
                  max_seq_length=2048  # FIXME: extract this from the model
                  ):
    context_tokens = tokenizer.tokenize(prompt)
    if len(context_tokens) == 0:
        context_tokens = [tokenizer.eod]
    context_tokens, pad_batch([context_tokens], tokenizer.eod, max_seq_length)
    generated_text = tokenizer.detokenize(context_tokens)
    import ipdb
    ipdb.set_trace()
    print(generated_text)
    return context_tokens, context_length


def main():
    """
    Generate text/sample model
    """
    overwrite_values = {
        "checkpoint_activations": False,
        "partition_activations": False,
        "no_load_optim": True,
        # disable zero optimization (won't be used in inference, and loading zero optimizer can cause errors)
        "zero_optimization": None,
    }

    if os.environ.get('DEEPERSPEED'):
        neox_args = NeoXArgs.consume_neox_args(
            overwrite_values=overwrite_values)
    else:
        args = parse_arguments()
        neox_args = genconfig(args.conf_dir, args.conf_file, overwrite_values)
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()

    if neox_args.load is None:
        raise ValueError("`load` parameter must be supplied to load a model`")

    # initialize megatron
    initialize_megatron(neox_args)
    use_cache = True
    # set up model and load checkpoint.
    model = get_model(neox_args=neox_args, use_cache=use_cache)

    if neox_args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, *_ = deepspeed.initialize(
            args=neox_args,
            model=model,
            optimizer=None,
            model_parameters=None,
            training_data=None,
            lr_scheduler=None,
            mpu=mpu if not neox_args.is_pipe_parallel else None,
            dist_init_required=False,
            config_params=neox_args.deepspeed_config,
        )

        model.total_params = get_total_params(model.module)
        print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')

        if neox_args.is_pipe_parallel:
            model.set_has_attention_mask(True)
            model.set_batch_fn(partial(get_batch_pipe, neox_args=neox_args))
    else:
        raise ValueError("Must be using deepspeed to run neox")

    neox_args.iteration = load_checkpoint(
        neox_args=neox_args,
        model=model,
        optimizer=None,
        lr_scheduler=None,
        iteration=None,
    )

    # monkey patch to pass this config
    model.is_pipe_parallel = neox_args.is_pipe_parallel

    print_rank_0(
        f"Loading checkpoint and starting from iteration {neox_args.iteration}"
    )
    print_rank_0("Finished loading model")

    torch.distributed.barrier(group=mpu.get_model_parallel_group())

    result = generate_samples_from_prompt(neox_args,
                                          model,
                                          text="Who is Charlize Teron?",
                                          eos_token_id=neox_args.tokenizer.eod_id,
                                          top_k=0.0,
                                          top_p=1.0,
                                          temperature=0.0
                                          )

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        import ipdb
        ipdb.set_trace()
        # tokenize_text(neox_args.tokenizer, prompt="Who is Charlize Teron?")

        # response, response_seg, response_logprobs, _ = \
        #     generate_and_post_process(model,
        #                               neox_args.tokenizer,
        #                               tokens_to_generate=60,
        #                               top_k_sampling=0.0,
        #                               top_p_sampling=1.0,
        #                               return_output_log_probs=False,
        #                               temperature=0.0
        #                               )
        # print(response)
        # print("run api server")
        # server = MegatronServer(model, neox_args.tokenizer)
        # server.run("localhost", port=8888)
        terminate = 1
    else:
        terminate = 0

    while True:
        terminate_runs_tensor = torch.cuda.LongTensor([terminate])
        torch.distributed.broadcast(
            terminate_runs_tensor,
            mpu.get_model_parallel_src_rank(),
            group=mpu.get_model_parallel_group(),
        )
        terminate_runs = terminate_runs_tensor[0].item()
        if terminate_runs == 1:
            break
        else:
            import time
            print("sleeping for few sconds")
            time.sleep(10)


if __name__ == "__main__":
    main()
