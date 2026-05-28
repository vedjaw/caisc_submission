#!/bin/bash
# Run PURGE unlearning on SVHN

python src/run.py \
    --dataset svhn --model resnet18 \
    --forget_type class --forget_class 0 \
    --checkpoint ./checkpoints/base_svhn_resnet18.pth \
    --data_dir ./data --save_dir ./checkpoints \
    --epochs 12 --lr 2e-4 \
    --rep_weight 0.03 --kd_weight 3.0 \
    --max_grad_norm 1.0 \
    --forget_objective kl_retain \
    --forget_gate 0 --retain_budget_factor 4.0 \
    --fa_target 10.0
