import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from functools import partial

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.datasets.utils import compile_helpers
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.training.tokenizer.tokenizer import _NullTokenizer
import argparse
from torch.testing._internal.distributed.fake_pg import FakeStore

from memory_profiler import MemoryTracer

_SEQUENCE_LENGTH = 4096
_VOCAB_SIZE = 32000


def initialize_distributed(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=1,
    fake=False,
    create_gloo_process_groups=True,
):
    parallel_state.destroy_model_parallel()

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Torch setup for distributed training
    torch.cuda.set_device(rank)
    if fake:
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", world_size=world_size, rank=rank, store=fake_store
        )
    else:
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=rank
        )

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=1,
        create_gloo_process_groups=create_gloo_process_groups,
    )


def get_train_data_iterator(batch_size):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=_VOCAB_SIZE-1),
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()

    train_dataloader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True)

    train_iterator = iter(train_dataloader)

    return train_iterator


def silicon_main(args, model_provider_func):
    initialize_distributed(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        expert_model_parallel_size=args.ep_size,
    )
    model_parallel_cuda_manual_seed(123)

    rank = torch.distributed.get_rank()

    gpt_model = model_provider_func()
    torch.cuda.set_device(rank)

    device = torch.device("cuda")
    gpt_model.to(device)

    def forward_step_func(data_iterator, model):

        def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            return loss, {"lm loss": loss}

        data = next(data_iterator)
        tokens = data["tokens"].to(device)
        attention_mask = data["attention_mask"].to(device)
        position_ids = data["position_ids"].to(device)
        labels = data["labels"].to(device)
        loss_mask = data["loss_mask"].to(device)

        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

        return output_tensor, partial(loss_func, loss_mask)

    optim = Adam(gpt_model.parameters())

    train_iterator = get_train_data_iterator(args.mbs)

    forward_backward_func = get_forward_backward_func()

    for iteration in range(2):
        optim.zero_grad()

        rank = torch.distributed.get_rank()
        if rank == 0:
            print(
                f"rank {rank} actual current_memory_allocated before forward: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB"
            )
            print(
                f"rank {rank} actual max_memory_allocated before forward: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB"
            )

        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=args.ga,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=args.mbs,
            decoder_seq_length=_SEQUENCE_LENGTH,
            forward_only=False,
        )
        if rank == 0:
            print(
                f"rank {rank} actual current_memory_allocated after forward-backward pass: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB"
            )
            print(
                f"rank {rank} actual max_memory_allocated after forward-backward pass: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB"
            )
            optim.step()
            print(
                f"rank {rank} actual current_memory_allocated after optimizer step: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB"
            )
            print(
                f"rank {rank} actual max_memory_allocated after optimizer step: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB"
            )


def estimated_main(args, model_provider_func):
    """Run memory estimation with the GPT model."""
    # Initialize distributed setup
    initialize_distributed(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        expert_model_parallel_size=args.ep_size,
        fake=True,
        create_gloo_process_groups=False,
    )
    model_parallel_cuda_manual_seed(123)
    train_iterator = get_train_data_iterator(args.mbs)

    # Create memory tracer
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)

    estimator = MemoryTracer(device="cuda", use_fake_tensor=False)

    with estimator:
        # Create model and move to CUDA
        with estimator.track_phase("model_creation"):
            gpt_model = model_provider_func()
            device = torch.device("cuda")
            gpt_model.to(device)
            optim = Adam(gpt_model.parameters())

        def forward_step_func(data_iterator, model):
            def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
                losses = output_tensor.float()
                loss_mask = loss_mask.view(-1).float()
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
                return loss, {"lm loss": loss}

            # Create fake input data
            data = next(data_iterator)

            # Convert data to fake tensors
            fake_tokens = estimator.create_fake_tensor_from_tensor(data["tokens"]).to(
                device
            )
            fake_attention_mask = estimator.create_fake_tensor_from_tensor(
                data["attention_mask"]
            ).to(device)
            fake_position_ids = estimator.create_fake_tensor_from_tensor(
                data["position_ids"]
            ).to(device)
            fake_labels = estimator.create_fake_tensor_from_tensor(data["labels"]).to(
                device
            )
            fake_loss_mask = estimator.create_fake_tensor_from_tensor(
                data["loss_mask"]
            ).to(device)

            output_tensor = model(
                fake_tokens, fake_position_ids, fake_attention_mask, labels=fake_labels
            )

            return output_tensor, partial(loss_func, fake_loss_mask)


        forward_backward_func = get_forward_backward_func()

        # Running the model for 5 iterations
        for iteration in range(2):
            optim.zero_grad()

            with estimator.track_phase(f"forward-backward-{iteration}"):
                losses_reduced = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=train_iterator,
                    model=gpt_model,
                    num_microbatches=args.ga,
                    seq_length=_SEQUENCE_LENGTH,
                    micro_batch_size=args.mbs,
                    decoder_seq_length=_SEQUENCE_LENGTH,
                    forward_only=False,
                )
            
            with estimator.track_phase(f"optimizer_step-{iteration}"):
                optim.step()

        estimator.print_memory_stats()
        estimator.memory_dispatch_mode.save_peak_memory_snapshot_to_file("peak_memory_snapshot.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimated", action="store_true")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--mbs", type=int, default=1)
    parser.add_argument("--ga", type=int, default=8)
    parser.add_argument("--model", type=str, default="gpt")
    return parser.parse_args()


def gpt_model_provider(args):
    """Build the model."""

    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=4096,
        num_attention_heads=16,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        sequence_parallel=args.tp_size > 1,
        expert_tensor_parallel_size=1,
        bf16=True,
    )
    if args.estimated:
        if hasattr(transformer_config, "memory_tracing"):
            transformer_config.memory_tracing = True
            transformer_config._update_for_memory_tracing()
        else:
            print("❗️No using patched megatron-lm for memory tracing. May encounter unexpected issues.")

    pre_process = parallel_state.is_pipeline_first_stage()
    post_process = parallel_state.is_pipeline_last_stage()
    layer_spec = get_gpt_layer_with_transformer_engine_spec()

    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=layer_spec,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=_SEQUENCE_LENGTH,
        pre_process=pre_process,
        post_process=post_process,
    ).bfloat16()

    return gpt_model


def moe_model_provider(args):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=4096,
        moe_ffn_hidden_size=8192,
        num_attention_heads=16,
        num_moe_experts=8,
        use_cpu_initialization=True,
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_aux_loss_coeff=0.01,
        moe_grouped_gemm=True,
        add_bias_linear=False,
        moe_expert_capacity_factor=1.0,
        moe_pad_expert_input_to_capacity=True,
        pipeline_dtype=torch.float32,
        expert_model_parallel_size=args.ep_size,
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        sequence_parallel=args.tp_size > 1,
        expert_tensor_parallel_size=1,
        bf16=True,
    )
    if args.estimated:
        if hasattr(config, "memory_tracing"):
            config.memory_tracing = True
            config._update_for_memory_tracing()
        else:
            print("❗️No using patched megatron-lm for memory tracing. May encounter unexpected issues.")

    pre_process = parallel_state.is_pipeline_first_stage()
    post_process = parallel_state.is_pipeline_last_stage()

    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=8, moe_grouped_gemm=True
    )
    gpt_model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=_SEQUENCE_LENGTH,
        pre_process=pre_process,
        post_process=post_process,
    ).bfloat16()
    return gpt_model


if __name__ == "__main__":
    args = parse_args()
    if args.model == "gpt":
        model_provider_func = partial(gpt_model_provider, args=args)
    elif args.model == "moe":
        model_provider_func = partial(moe_model_provider, args=args)
    else:
        raise ValueError(f"Invalid model: {args.model}")

    if args.estimated:
        estimated_main(args, model_provider_func)
    else:
        # Then run the actual model
        silicon_main(args, model_provider_func)
