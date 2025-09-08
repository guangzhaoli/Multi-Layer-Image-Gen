#!/bin/bash

# 多GPU训练启动脚本
# Multi-GPU training launch script for ART pipeline

set -e

# 默认配置
TRAIN_JSONL="train.jsonl"
EPOCHS=5
BATCH_SIZE=1
ACCUM_STEPS=1
LR=2e-4
WEIGHT_DECAY=0.01
FLUX_VAE_PATH="models/Flux_vae"
ARCH="vit-b/32"
SAVE_DIR="./checkpoints"
SEED=42
LOG_INTERVAL=20
NUM_WORKERS=2
MASTER_PORT="12355"

# 自动检测GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --train_jsonl)
            TRAIN_JSONL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --accum_steps)
            ACCUM_STEPS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --world_size)
            NUM_GPUS="$2"
            shift 2
            ;;
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --single_gpu)
            NUM_GPUS=1
            shift
            ;;
        --eval_first_item)
            EVAL_FIRST_ITEM="--eval_first_item"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --train_jsonl PATH      Training JSONL file path (default: train.jsonl)"
            echo "  --epochs N              Number of epochs (default: 5)"
            echo "  --batch_size N          Batch size (default: 1)"
            echo "  --accum_steps N         Gradient accumulation steps (default: 4)"
            echo "  --lr FLOAT              Learning rate (default: 2e-4)"
            echo "  --world_size N          Number of GPUs to use (default: auto-detect)"
            echo "  --arch ARCH             Model architecture (default: vit-b/32)"
            echo "  --save_dir PATH         Checkpoint save directory (default: ./checkpoints)"
            echo "  --single_gpu            Force single GPU training"
            echo "  --eval_first_item       Evaluate on first item each epoch"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 检查必要文件
if [ ! -f "$TRAIN_JSONL" ]; then
    echo "Error: Training JSONL file not found: $TRAIN_JSONL"
    exit 1
fi

if [ ! -f "train_pipeline_multi_gpu.py" ]; then
    echo "Error: Training script not found: train_pipeline_multi_gpu.py"
    exit 1
fi

# 检查GPU可用性
if [ "$NUM_GPUS" -gt 1 ]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: nvidia-smi not found. Cannot use multi-GPU training."
        exit 1
    fi
    
    AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
    if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
        echo "Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available."
        exit 1
    fi
fi

# 创建保存目录
mkdir -p "$SAVE_DIR"

echo "=========================================="
echo "ART Multi-GPU Training Configuration"
echo "=========================================="
echo "Training JSONL: $TRAIN_JSONL"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Accumulation Steps: $ACCUM_STEPS"
echo "Learning Rate: $LR"
echo "Architecture: $ARCH"
echo "Save Directory: $SAVE_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Master Port: $MASTER_PORT"
echo "=========================================="

# 构建训练命令
CMD="python train_pipeline_multi_gpu_v2.py \
    --train_jsonl $TRAIN_JSONL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accum_steps $ACCUM_STEPS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --flux_vae_path $FLUX_VAE_PATH \
    --arch $ARCH \
    --save_dir $SAVE_DIR \
    --seed $SEED \
    --log_interval $LOG_INTERVAL \
    --num_workers $NUM_WORKERS \
    --master_port $MASTER_PORT \
    --eval_first_item" 
    

# 添加分布式训练参数（如果使用多GPU）
if [ "$NUM_GPUS" -gt 1 ]; then
    CMD="$CMD --distributed --world_size $NUM_GPUS"
    echo "Starting distributed training on $NUM_GPUS GPUs..."
else
    echo "Starting single GPU training..."
fi

# 添加评估参数
if [ ! -z "$EVAL_FIRST_ITEM" ]; then
    CMD="$CMD $EVAL_FIRST_ITEM"
fi

# 激活conda环境并运行训练
echo "Starting training..."
echo "Command: $CMD"
echo "=========================================="

# 运行训练
eval $CMD

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $SAVE_DIR"
echo "=========================================="