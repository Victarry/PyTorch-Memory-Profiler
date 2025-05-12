export MEGATRON_PATH=/workspace/megatron-lm
export PYTHONPATH=$MEGATRON_PATH:/workspace/pytorch-memory-tracing:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# DeepSeek-V3 parallel configuration
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-1}
CP=${CP:-1}
ETP=${ETP:-1}
VPP=${VPP:-1}
MBS=${MBS:-1}
GBS=${GBS:-8192}

# Model has 61 layers
num_layers=61

export MEMORY_PROFILER_LOG_LEVEL=20  # INFO level

export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}

extra_args=""
if [ $VPP -ne 1 ]; then
    extra_args+=" --num-layers-per-virtual-pipeline-stage $((num_layers / PP / VPP))"
fi

python $MEGATRON_PATH/pretrain_gpt.py \
    --distributed-timeout-minutes 60 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --expert-model-parallel-size $EP \
    --context-parallel-size $CP \
    --expert-tensor-parallel-size $ETP \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-mcore-models \
    --sequence-parallel \
    --use-flash-attn \
    --disable-bias-linear \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --train-samples 585937500 \
    --exit-duration-in-mins 220 \
    --no-check-for-nan-in-loss-and-grad \
    --no-rope-fusion \
    --manual-gc \
    --manual-gc-interval 1 \
    --transformer-impl transformer_engine \
    --seq-length 4096 \
    --data-cache-path /tmp/data-cache \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model deepseek-ai/DeepSeek-V3 \
    --data-path /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/datasets/Slimpajama/DeepSeek-V3/dsv3_text_document \
    --split 99,1,0 \
    --no-mmap-bin-files \
    --no-create-attention-mask-in-dataloader \
    --num-workers 6 \
    --num-layers $num_layers \
    --hidden-size 7168 \
    --ffn-hidden-size 18432 \
    --num-attention-heads 128 \
    --kv-channels 128 \
    --max-position-embeddings 4096 \
    --position-embedding-type rope \
    --rotary-base 10000 \
    --make-vocab-size-divisible-by 3232 \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --multi-latent-attention \
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --qk-layernorm \
    --lr-decay-samples 584765624 \
    --lr-warmup-samples 1536000 \
    --lr-warmup-init 3.9e-7 \
    --lr 3.9e-6 \
    --min-lr 3.9e-7 \
    --lr-decay-style cosine \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --num-experts 256 \
    --moe-layer-freq "([0]*3+[1]*58)" \
    --moe-ffn-hidden-size 2048 \
    --moe-shared-expert-intermediate-size 2048 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-topk 8 \
    --moe-token-dispatcher-type flex \
    --moe-enable-deepep \
    --moe-router-pre-softmax \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-group-topk 4 \
    --moe-router-num-groups 8 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-bias-update-rate 1e-3 \
    --moe-router-dtype fp32 \
    --moe-permute-fusion \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --qk-head-dim 128 \
    --qk-pos-emb-head-dim 64 \
    --v-head-dim 128 \
    --rotary-scaling-factor 40 \
    --mscale 1.0 \
    --mscale-all-dim 1.0 \
    --eval-iters 32 \
    --eval-interval 200 \
    --finetune \
    --auto-detect-ckpt-format \
    --dist-ckpt-strictness log_all \
    --init-method-std 0.02 \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-num-zeros-in-grad \
    --log-params-norm \
    --log-validation-ppl-to-tensorboard \
    --log-throughput \
    --log-interval 1 \
    --logging-level 40 \
    --tensorboard-dir /lustre/fsw/portfolios/coreai/users/denliu/MoE-Dev/megatron-moe-scripts/output/Dennis-DeepSeek-V3-with-MTP-benchmark/tensorboard \
    --wandb-project Dennis-DeepSeek-V3-with-MTP-benchmark \
    --wandb-exp-name DeepSeek-V3-TP${TP}PP${PP}EP${EP}CP${CP}VPP${VPP}-MBS${MBS}GBS${GBS}-bf16 \
    --bf16 \
    $extra_args \
    --memory-tracing