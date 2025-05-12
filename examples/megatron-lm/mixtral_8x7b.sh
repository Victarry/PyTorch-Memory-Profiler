export MEGATRON_PATH=/workspace/megatron-lm
export PYTHONPATH=$MEGATRON_PATH:/workspace/pytorch-memory-tracing:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Parallel setting configuration
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-1}
VPP=${VPP:-1}
MBS=${MBS:-1}
GBS=${GBS:-256}

num_layers_per_vpp=$((32 / PP / VPP))

export MEMORY_PROFILER_LOG_LEVEL=20  # INFO level

export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}

extra_args=""
if [ $VPP -ne 1 ]; then
    extra_args="--num-layers-per-virtual-pipeline-stage $num_layers_per_vpp"
fi

python $MEGATRON_PATH/pretrain_gpt.py \
    --exit-duration-in-mins 225 \
    --distributed-timeout-minutes 60 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-mcore-models \
    --sequence-parallel \
    --use-flash-attn \
    --disable-bias-linear \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --train-samples 51200 \
    --exit-duration-in-mins 230 \
    --transformer-impl transformer_engine \
    --data-cache-path /tmp/data-cache \
    --tokenizer-type NullTokenizer \
    --vocab-size 32000 \
    --data-path /workspace/data/moe_data/Slimpajama_text_document \
    --split 99,1,0 \
    --no-mmap-bin-files \
    --num-workers 6 \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-rope-fusion \
    --rotary-percent 1.0 \
    --normalization RMSNorm \
    --swiglu \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --make-vocab-size-divisible-by 128 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --lr-decay-samples 255126953 \
    --lr-warmup-samples 162761 \
    --lr 1.2e-5 \
    --min-lr 1.2e-6 \
    --lr-decay-style cosine \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --expert-model-parallel-size $EP \
    --num-experts 8 \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 2 \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-2 \
    --moe-token-dispatcher-type alltoall \
    --eval-iters 32 \
    --eval-interval 200 \
    --finetune \
    --auto-detect-ckpt-format \
    --no-ckpt-fully-parallel-save \
    --save-interval 500 \
    --init-method-std 0.008 \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-num-zeros-in-grad \
    --log-params-norm \
    --log-validation-ppl-to-tensorboard \
    --log-throughput \
    --log-interval 1 \
    --tensorboard-dir /lustre/fsw/coreai_dlalgo_mcore/release-testing/mcore-v0.12.0/mixtral/mixtral_8x7b_tp1pp4ep8vpp8_release/tensorboard \
    --wandb-project megatron-core-release-runs \
    --wandb-exp-name release-testing-mcore-v0.12.0-mixtral-mixtral_8x7b_tp1pp4ep8vpp8_release \
    --bf16 \
    $extra_args \
    --memory-tracing