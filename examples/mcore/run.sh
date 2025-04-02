export MEGATRON_PATH=/workspace/megatron-lm
export PYTHONPATH=/workspace/pytorch-memory-tracing:$MEGATRON_PATH:$PYTHONPATH
torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py 
# > output.log 2>&1