#!/bin/bash
for class in 0 1 2 3 4 5 6 7 8 9; do
  echo "===== Forgetting MNIST class $class ====="
  python src/run.py \
    --dataset mnist \
    --model resnet18 \
    --forget_type class \
    --forget_class $class \
    --checkpoint ./checkpoints/base_mnist_resnet18.pth \
    --data_dir ./data \
    --save_dir ./checkpoints \
    --epochs 12 \
    --lr 2e-4 \
    --rep_weight 0.03 \
    --kd_weight 3.0 \
    --max_grad_norm 1.0 \
    --forget_objective kl_retain \
    --forget_gate 0 \
    --retain_budget_factor 4.0 \
    --fa_target 10.0 \
    2>&1 | tee ./checkpoints/log_mnist_class${class}.txt
done
