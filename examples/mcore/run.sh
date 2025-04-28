#!/bin/bash

# Set logging level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL)
export MEMORY_PROFILER_LOG_LEVEL=20  # DEBUG level

export MEGATRON_PATH=/workspace/megatron-lm
export PYTHONPATH=/workspace/pytorch-memory-tracing:$MEGATRON_PATH:$PYTHONPATH

# Running for silicon
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py --model moe --tp-size 2 --pp-size 1 --ep-size 1 --ga 8 
# > silicon.log 2>&1


# Running for estimated
# export WORLD_SIZE=8
# export LOCAL_RANK=0

# python pretrain_gpt.py --tp-size 2 --pp-size 2 --ep-size 2 --ga 8 --estimated --model moe > estimated.log 2>&1
