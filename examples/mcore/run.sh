export MEGATRON_PATH=/workspace/megatron-lm
export PYTHONPATH=/workspace/pytorch-memory-tracing:$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
# torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py --tp-size 2 --pp-size 2 --ga 8 > silicon.log 2>&1
# torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py --tp-size 2 --pp-size 2 --ga 8 --estimated > estimated.log 2>&1


export WORLD_SIZE=8
export LOCAL_RANK=0

python pretrain_gpt.py --tp-size 2 --pp-size 2 --ga 8 --estimated --model moe

# nsys profile -o moe_profile --trace=cuda,nvtx --sample none --force-overwrite true 
# torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py --tp-size 2 --pp-size 2 --ga 8 --model moe




# rank 0 actual current_memory_allocated before forward: 4673.75 MB                                                                                                                         
# rank 0 actual max_memory_allocated before forward: 4673.75 MB
# rank 0 actual current_memory_allocated after forward-backward pass: 9876.63 MB                                                                                                            
# rank 0 actual max_memory_allocated after forward-backward pass: 27063.27 MB                                                                                                               
# rank 0 actual current_memory_allocated after optimizer step: 19224.13 MB                                                                                                                  
# rank 0 actual max_memory_allocated after optimizer step: 27063.27 MB


# profiled current_memory_allocated before forward: 4673.75 MB
# profiled max_memory_allocated before forward: 4865.75 MB
# profiled current_memory_allocated after forward-backward pass: 9860.38 MB
# profiled max_memory_allocated after forward-backward pass: 27047.01 MB
# profiled current_memory_allocated after optimizer step: 19207.88 MB
# profiled max_memory_allocated after optimizer step: 27047.01 MB
