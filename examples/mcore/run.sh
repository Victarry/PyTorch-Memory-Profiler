export MEGATRON_PATH=/workspace/megatron-lm
export PYTHONPATH=/workspace/pytorch-memory-tracing:$MEGATRON_PATH:$PYTHONPATH
torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py --tp-size 2 --pp-size 2 --ga 8 > silicon.log 2>&1
torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py --tp-size 2 --pp-size 2 --ga 8 --estimated > estimated.log 2>&1
