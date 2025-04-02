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
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.datasets.utils import compile_helpers 
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.training.tokenizer.tokenizer import _NullTokenizer

from memory_profiler import MemoryTracer

_SEQUENCE_LENGTH = 64


def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    parallel_state.destroy_model_parallel()

    # Torch setup for distributed training
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)

def model_provider():
    """Build the model."""

    transformer_config = TransformerConfig(
        num_layers=4, 
        hidden_size=16, 
        num_attention_heads=16, 
        use_cpu_initialization=True, 
        pipeline_dtype=torch.float32,
    )

    gpt_model = GPTModel(
        config=transformer_config, 
        transformer_layer_spec=get_gpt_layer_local_spec(normalization="RMSNorm"), 
        vocab_size=100, 
        max_sequence_length=_SEQUENCE_LENGTH,
    )

    return gpt_model

def get_train_data_iterator():
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
        tokenizer=_NullTokenizer(vocab_size=_SEQUENCE_LENGTH),
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()

    train_dataloader = DataLoader(datasets[0], batch_size=8, shuffle=True)

    train_iterator = iter(train_dataloader)

    return train_iterator

def print_rank0(message):
    if torch.distributed.get_rank() == 0:
        print(message)

def silicon_main():
    initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    rank = torch.distributed.get_rank()

    gpt_model = model_provider()
    torch.cuda.set_device(rank)

    device = torch.device("cuda")
    gpt_model.to(device)

    def forward_step_func(data_iterator, model):

        def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            return loss, {'lm loss': loss}

        data = next(data_iterator)
        tokens = data['tokens'].to(device)
        attention_mask = data['attention_mask'].to(device)
        position_ids = data['position_ids'].to(device)
        labels = data['labels'].to(device)
        loss_mask = data['loss_mask'].to(device)

        output_tensor = model(tokens, position_ids, attention_mask,
                            labels=labels)

        return output_tensor, partial(loss_func, loss_mask)

    optim = Adam(gpt_model.parameters())

    train_iterator = get_train_data_iterator()

    forward_backward_func = get_forward_backward_func()

    for iteration in range(1):
        optim.zero_grad()

        print_rank0(f"actual current_memory_allocated before forward: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=1,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=8,
            decoder_seq_length=_SEQUENCE_LENGTH,
            forward_only=False)
        print_rank0(f"actual current_memory_allocated after forward-backward pass: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        optim.step()
        print_rank0(f"actual current_memory_allocated after optimizer step: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

def estimated_main():
    """Run memory estimation with the GPT model."""
    # Create memory tracer
    device = 'cuda'
    estimator = MemoryTracer(device=device)
    torch.set_default_device(device)
    
    train_iterator = get_train_data_iterator()
    # Initialize distributed setup
    initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    with estimator:

        rank = torch.distributed.get_rank()
        torch.cuda.set_device(rank)

        # Create model
        gpt_model = model_provider()
        
        # Register hooks for memory tracking
        hook_handles = estimator.memory_dispatch_mode.register_hooks_to_module(gpt_model)
        
        device = torch.device("cuda")
        gpt_model.to(device)

        def forward_step_func(data_iterator, model):
            def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
                losses = output_tensor.float()
                loss_mask = loss_mask.view(-1).float()
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
                return loss, {'lm loss': loss}

            # Create fake input data
            data = next(data_iterator)
            
            # Convert data to fake tensors
            fake_tokens = estimator.create_fake_tensor_from_tensor(data['tokens']).to(device)
            fake_attention_mask = estimator.create_fake_tensor_from_tensor(data['attention_mask']).to(device)
            fake_position_ids = estimator.create_fake_tensor_from_tensor(data['position_ids']).to(device)
            fake_labels = estimator.create_fake_tensor_from_tensor(data['labels']).to(device)
            fake_loss_mask = estimator.create_fake_tensor_from_tensor(data['loss_mask']).to(device)

            output_tensor = model(fake_tokens, fake_position_ids, fake_attention_mask, labels=fake_labels)

            return output_tensor, partial(loss_func, fake_loss_mask)

        optim = Adam(gpt_model.parameters())


        forward_backward_func = get_forward_backward_func()

        # Running the model for 5 iterations
        for iteration in range(1):
            optim.zero_grad()
            
            print_rank0(f"profiled current_memory_allocated before forward: {estimator.get_current_memory_allocated()} MB")

            losses_reduced = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=train_iterator,
                model=gpt_model,
                num_microbatches=1,
                seq_length=_SEQUENCE_LENGTH,
                micro_batch_size=8,
                decoder_seq_length=_SEQUENCE_LENGTH,
                forward_only=False)

            print_rank0(f"profiled current_memory_allocated after forward-backward pass: {estimator.get_current_memory_allocated()[0] / (1024 ** 2):.2f} MB")
            optim.step()
            print_rank0(f"profiled current_memory_allocated after optimizer step: {estimator.get_current_memory_allocated()[0] / (1024 ** 2):.2f} MB")

        # Remove hooks
        estimator.memory_dispatch_mode.remove_hooks(hook_handles)

if __name__ == "__main__":
    # First run the estimation
    estimated_main()
    
    # Reset memory stats before running the actual model
    torch.cuda.reset_peak_memory_stats()
    
    # Then run the actual model
    # silicon_main()
