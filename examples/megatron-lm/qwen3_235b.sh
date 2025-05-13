export MEGATRON_PATH=/workspace/megatron-lm
export PYTHONPATH=$MEGATRON_PATH:/workspace/pytorch-memory-tracing:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Preliminary parallel setting TP2EP8PP4
# number of params: 235B
# 256 GPUs
# TP4PP8VPP4EP8: 48500.95
# TP2PP8VPP4EP8: 67868.22
# TP2PP16VPP6EP8: 
# TP2PP8VPP4EP32: 53612.22

# TP1EP8ZP32

TP=${TP:-2}
EP=${EP:-32}
PP=${PP:-8}
VPP=${VPP:-4}
MBS=${MBS:-1}
GBS=${GBS:-2048}

num_layers=${num_layers:-94}

export MEMORY_PROFILER_LOG_LEVEL=20  # INFO level
export WORLD_SIZE=${WORLD_SIZE:-256}
export RANK=${RANK:-0}

if [ $VPP -gt 1 ]; then
    extra_args="--num-virtual-stages-per-pipeline-rank $VPP"
else
    extra_args=""
fi


python $MEGATRON_PATH/pretrain_gpt.py \
    --distributed-timeout-minutes 60 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --expert-model-parallel-size $EP \
    --context-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --no-create-attention-mask-in-dataloader \
    --use-mcore-models \
    --sequence-parallel \
    --use-flash-attn \
    --disable-bias-linear \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --train-samples 268554688 \
    --exit-duration-in-mins 230 \
    --transformer-impl transformer_engine \
    --data-cache-path /workspace/megatron-moe-scripts/data_cache \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3-235B-A22B \
    --data-path /lustre/share/coreai_dlalgo_mcore/Dataset/Slimpajama/Qwen-plus/Slimpajama_text_document \
    --split 99,1,0 \
    --no-mmap-bin-files \
    --num-workers 6 \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --normalization RMSNorm \
    --swiglu \
    --norm-epsilon 1e-06 \
    --num-layers 94 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 4 \
    --qk-layernorm \
    --seq-length 4096 \
    --max-position-embeddings 40960 \
    --make-vocab-size-divisible-by 1187 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --lr-decay-samples 255126953 \
    --lr-warmup-samples 162761 \
    --lr 1.2e-4 \
    --min-lr 1.2e-5 \
    --lr-decay-style cosine \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --num-experts 128 \
    --moe-ffn-hidden-size 1536 \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 8 \
    --moe-router-pre-softmax \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type alltoall \
    --moe-permute-fusion \
    --eval-iters 32 \
    --eval-interval 500 \
    --finetune \
    --auto-detect-ckpt-format \
    --save /lustre/fsw/portfolios/coreai/users/denliu/MoE-Dev/megatron-moe-scripts/output/Dennis-Qwen3-Benchmark/checkpoints \
    --no-ckpt-fully-parallel-save \
    --save-interval 500 \
    --dist-ckpt-strictness log_all \
    --init-method-std 0.02 \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-num-zeros-in-grad \
    --log-params-norm \
    --log-validation-ppl-to-tensorboard \
    --log-throughput \
    --log-interval 1 \
    --tensorboard-dir /lustre/fsw/portfolios/coreai/users/denliu/MoE-Dev/megatron-moe-scripts/output/Dennis-Qwen3-Benchmark/tensorboard \
    --wandb-project Dennis-Qwen3-Benchmark \
    --wandb-exp-name Qwen3-235B-A22B-TP2PP2EP4CP1VPP1-MBS1GBS256-bf16_baseline_repro \
    --bf16 \
    --account-for-embedding-in-pipeline-split \
    --account-for-loss-in-pipeline-split \
    $extra_args \
    --memory-tracing \
    --save-peak-memory-snapshot qwen3_235b_TP${TP}_PP${PP}_EP${EP}_VPP${VPP}.json \
    --recompute-granularity selective \
    --recompute-modules moe_act layernorm
