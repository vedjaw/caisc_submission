#!/bin/bash
# Train the base model from scratch (no unlearning).
# Run this once before any unlearning experiment.
# Usage: bash scripts/train_base.sh [dataset] [model]

DATASET=${1:-cifar10}
MODEL=${2:-resnet18}

python src/run.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --forget_type class \
    --forget_class 0 \
    --data_dir ./data \
    --save_dir ./checkpoints \
    --epochs 0
