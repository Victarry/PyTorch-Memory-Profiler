import torch

def print_rank_0(message):
    """Print message only on rank 0 if distributed is initialized."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message)
    else:
        print(message) 