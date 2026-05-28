# Claimed Contribution

## What We Reproduced

- Implemented and tested 7 existing machine unlearning baselines during the midsem phase
  (gradient ascent, fine-tuning, random labels, knowledge distillation / Bad Teacher,
  SalUn-style saliency masking, Fisher forgetting, amnesiac unlearning). All were
  evaluated on CIFAR-10 with ResNet-18. We didn't reproduce SalUn ourselves for the
  endsem comparison — those numbers come from their published Table A2, which we
  acknowledge is not an apples-to-apples comparison (see Limitations).
- Reproduced the A-GEM gradient projection from Chaudhry et al. (ICLR 2019) and
  verified it works correctly as a gradient constraint. The math is straightforward
  (dot product check + projection formula) but getting it to work reliably inside
  a PyTorch training loop took some careful handling of `zero_grad` / `clone` sequences.
- Reproduced standard MIA evaluation (loss-based + confidence-based AUROC). We initially
  used threshold-based MIA accuracy like many papers do, but switched to AUROC after
  realising the threshold approach has class-imbalance artifacts.

## What We Modified

- **Adapted A-GEM from CL to MU**: A-GEM was designed to protect *past tasks* while
  learning new ones. We flipped this: the "past task" is the retain set, and the "new
  task" is the forgetting objective. The main code change is using a live retain-batch
  gradient as the reference instead of episodic memory samples.
- **KL-retain objective**: Most unlearning methods push forget outputs toward the uniform
  distribution. We noticed (while looking at MIA results) that uniform is actually quite
  easy to detect because no real model ever produces perfectly uniform outputs. So we
  replaced the target with the model's own natural confusion pattern — the averaged
  softmax over the retain set. This is what we call "retain-confusion target".
- **Entropy-based gating**: Standard gradient ascent doesn't know when to stop pushing.
  We added an entropy gate (skip the forget gradient when the batch is already near
  max-entropy) and a hard CE cap (for GA specifically). These prevent over-forgetting
  on easy-to-forget batches.
- **MIA formulation fix**: Changed from forget-train vs full-test (class-imbalanced)
  to forget-class-train vs forget-class-test (same class, different split). This is the
  right way to do MIA for class-level forgetting, but many papers don't do it this way.

## What Did Not Work

- **IEWPv2**: Used gradient norms as a proxy for influence scores
  and alternated between forget and retain steps. The alternating optimisation was
  fundamentally unstable — each step partially undid the previous one. This is why we
  switched to gradient projection, which handles both objectives jointly.
- **Damped gradient ascent** (smooth multiplicative damping `min(1, log(K)/CE)`) instead
  of a hard entropy cutoff — too smooth, still allowed over-forgetting. We went with the
  hard cutoff instead.
- **Accuracy-based forget gating** (skip if batch accuracy < threshold) — turns out batch
  accuracy is really noisy and class-dependent. Entropy is a much more stable signal.
- **KL-uniform with aggressive learning rates** — the uniform target is "artificial" and
  the model fights it. High LR + uniform = catastrophic retain damage.
- **Running BN in train mode** — this one was painful. Our first runs collapsed to ~21%
  accuracy and we spent a full afternoon debugging before finding that the class-
  homogeneous forget batches were corrupting BN running statistics.
- **Trying to frame PURGE as a DP mechanism** — the projection gives a constructive
  guarantee (retain loss doesn't increase per-step) but we couldn't figure out how to
  translate this into an (ε, δ) bound. Still an open question.

## What We Believe Is Our Contribution

1. **CL–MU Duality (the core insight)**: The idea that you can take a CL tool (A-GEM
   projection) and directly repurpose it for unlearning. The connection between CL and
   MU has been noted before (Kurmanji et al., 2023), but to our knowledge no prior work
   has constructed a complete unlearning algorithm around it. The per-step retain-safety guarantee comes
   directly from this duality.

2. **Retain-Confusion Target**: Pushing forget outputs toward the model's natural
   confusion distribution instead of uniform. This gets MIA AUROC ≈ 0.5 consistently,
   which we believe is a stronger privacy result than what other objectives achieve.

3. **Joint optimisation with projection**: Instead of alternating between forget and
   retain steps (which we found doesn't work well — see IEWPv2), PURGE handles both
   in a single projected update step. The projection resolves conflicts between the
   two objectives mathematically rather than hoping they balance out.

4. **Dual stopping criteria**: Retain-loss budget + FA target + intra-epoch checking.
   The intra-epoch part turned out to be critical on PathMNIST where FA can drop from
   99.8% to 0% within one epoch.

5. **BatchNorm freezing**: Documenting this failure mode and showing it affects any
   unlearning method that sends class-homogeneous batches through BN networks.

