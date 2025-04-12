export MEGATRON_PATH=/workspace/megatron-lm
export PYTHONPATH=/workspace/pytorch-memory-tracing:$MEGATRON_PATH:$PYTHONPATH
# torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py --tp-size 2 --pp-size 2 --ga 8 > silicon.log 2>&1
# torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py --tp-size 2 --pp-size 2 --ga 8 --estimated > estimated.log 2>&1


export WORLD_SIZE=8
export LOCAL_RANK=0

python pretrain_gpt.py --tp-size 2 --pp-size 2 --ga 8 --estimated --model moe

# torchrun --nproc_per_node=8 --nnodes=1 pretrain_gpt.py --tp-size 2 --pp-size 2 --ga 8 --model moe




# actual current_memory_allocated before forward: 4610.75 MB 
# actual max_memory_allocated before forward: 4610.75 MB                                                                                                                                                                                                                                        
# actual current_memory_allocated after forward-backward pass: 9237.89 MB                                                                                                                                                                 
# actual max_memory_allocated after forward-backward pass: 9448.17 MB                                                                                                                                                                                                                           
# actual current_memory_allocated after optimizer step: 18459.39 MB                                                                                                                                                                       
# actual max_memory_allocated after optimizer step: 23070.14 MB 


# profiled current_memory_allocated before forward: 4610.75 MB
# profiled max_memory_allocated before forward: 4802.75 MB
# profiled current_memory_allocated after forward-backward pass: 9221.64 MB
# profiled max_memory_allocated after forward-backward pass: 9431.16 MB
# profiled current_memory_allocated after optimizer step: 18443.14 MB
# profiled max_memory_allocated after optimizer step: 18667.14 MB
