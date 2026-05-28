#!/bin/bash
# Evaluate an already-unlearned model checkpoint.
# The evaluation metrics are printed at the end of a regular unlearning run.
# To re-evaluate without unlearning, load the checkpoint with --epochs 0.
# Usage: bash scripts/eval.sh [checkpoint_path] [dataset] [forget_class]

CHECKPOINT=${1:-./checkpoints/base_cifar10_resnet18.pth}
DATASET=${2:-cifar10}
FORGET_CLASS=${3:-0}

python src/run.py \
    --dataset "$DATASET" \
    --model resnet18 \
    --forget_type class \
    --forget_class "$FORGET_CLASS" \
    --checkpoint "$CHECKPOINT" \
    --data_dir ./data \
    --save_dir ./checkpoints \
    --epochs 0
