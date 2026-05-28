# Demo Instructions

## Prerequisites

- Python 3.10+ with deps installed (`pip install -r ../03_code/requirements.txt`)
- A pre-trained base model checkpoint (or internet to download CIFAR-10 on the fly)
- GPU recommended but CPU works (just takes ~5x longer)
- Make sure BN layers are in eval mode — the code handles this automatically but
  if you've modified anything, double-check

## Demo Flow

### Step 1: Base Model (Before Unlearning) — ~30 seconds

```bash
cd ../03_code

# If base model checkpoint exists:
python src/run.py \
    --dataset cifar10 --model resnet18 \
    --forget_type class --forget_class 0 \
    --checkpoint ./checkpoints/base_cifar10_resnet18.pth \
    --data_dir ./data --save_dir ./checkpoints \
    --epochs 0
```

The base model achieves ~93% Test Accuracy, ~93% accuracy on
class 0 (the forget class), confirming the model has learned the class.

### Step 2: Run PURGE Unlearning — ~2 minutes

```bash
python src/run.py \
    --dataset cifar10 --model resnet18 \
    --forget_type class --forget_class 0 \
    --checkpoint ./checkpoints/base_cifar10_resnet18.pth \
    --data_dir ./data --save_dir ./checkpoints \
    --epochs 15 --lr 5e-4 \
    --rep_weight 0.05 --kd_weight 2.0 \
    --max_grad_norm 1.0 \
    --forget_objective kl_retain \
    --forget_gate 0 \
    --retain_budget_factor 5.0
```
- Per-epoch Forget Accuracy (FA) dropping toward random chance (~10%)
- Retain Accuracy (RA) staying high
- Projection rate (shows gradient conflict resolution in action)
- The self-regulating stopping when FA target is reached

### Step 3: Final Results — ~30 seconds

- **Forget Accuracy (FA)** dropped to ~10% (random chance) → class is forgotten
- **Test Accuracy (TA)** remains close to original → utility preserved
- **Retain Accuracy (RA)** stays high → other classes unaffected
- **MIA AUROC ≈ 0.5** → the model is indistinguishable from one that never saw the class
- **Feature-level metrics** show representation-level erasure (not just output suppression)

### Step 4: Compared with Naive Gradient Ascent — ~1 minute

```bash
python src/run.py \
    --dataset cifar10 --model resnet18 \
    --forget_type class --forget_class 0 \
    --checkpoint ./checkpoints/base_cifar10_resnet18.pth \
    --epochs 10 --lr 5e-4 \
    --forget_objective ga
```

Naive GA often over-forgets (FA=0%) and damages TA/RA, showing why
PURGE's projection + kl_retain + stopping criteria matter.

## Backup

- Pre-computed results are in `../05_results/main_results.csv` and `ablations.csv`
- Figures are in `../05_results/figures/` (all 12 plots)
- Terminal logs from actual runs are in `../05_results/logs/`
