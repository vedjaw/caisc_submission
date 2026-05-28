#!/usr/bin/env bash
# Quick standalone demonstration script for PURGE

echo "==========================================="
echo " PURGE Demo: Base Training + Unlearning    "
echo " Dataset: CIFAR-10                         "
echo "==========================================="

echo -e "\n[1/2] Training base ResNet18 model on CIFAR-10..."
python src/run.py \
    --dataset cifar10 \
    --model resnet18 \
    --forget_type class \
    --forget_class 0 \
    --epochs 0 \
    --data_dir ./data \
    --save_dir ./checkpoints

echo -e "\n[2/2] Running kl_retain unlearning algorithm..."
python src/run.py \
    --dataset cifar10 \
    --model resnet18 \
    --forget_type class \
    --forget_class 0 \
    --checkpoint ./checkpoints/base_cifar10_resnet18.pth \
    --epochs 15 \
    --lr 5e-4 \
    --forget_objective kl_retain \
    --rep_weight 0.05 \
    --kd_weight 2.0 \
    --retain_budget_factor 5.0 \
    --data_dir ./data \
    --save_dir ./checkpoints

echo -e "\n==========================================="
echo "                 Demo Complete                "
echo "==========================================="
