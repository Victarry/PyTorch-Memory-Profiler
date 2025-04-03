export MEGATRON_PATH=/workspace/megatron-lm
export PYTHONPATH=/workspace/pytorch-memory-tracing:$MEGATRON_PATH:$PYTHONPATH
torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py --tp-size 2 --pp-size 1 --ga 1 > silicon.log 2>&1
torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py --tp-size 2 --pp-size 1 --ga 1 --estimated > estimated.log 2>&1
