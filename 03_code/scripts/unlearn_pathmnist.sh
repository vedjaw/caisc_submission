#!/bin/bash
# Run PURGE unlearning on PathMNIST with kl_retain objective.
# Usage: bash scripts/unlearn_pathmnist.sh [forget_class]

FORGET_CLASS=${1:-0}

python src/run.py \
    --dataset pathmnist \
    --model resnet18 \
    --forget_type class \
    --forget_class "$FORGET_CLASS" \
    --checkpoint ./checkpoints/base_pathmnist_resnet18.pth \
    --data_dir ./data \
    --save_dir ./checkpoints \
    --epochs 15 \
    --lr 5e-4 \
    --rep_weight 0.05 \
    --kd_weight 2.0 \
    --max_grad_norm 1.0 \
    --forget_objective kl_retain \
    --forget_gate 0 \
    --retain_budget_factor 5.0
