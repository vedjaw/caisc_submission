# Prior Work Basis

## Core Mechanism

**A-GEM** — Chaudhry et al., "Efficient Lifelong Learning with A-GEM" (ICLR 2019)

This is the most important reference. A-GEM was designed for continual learning: it
projects new-task gradients so they don't conflict with old-task gradients, preventing
catastrophic forgetting. We realised this is exactly what we need for unlearning, just
flipped: project forget gradients so they don't conflict with retain gradients.

The projection formula is identical — the difference is what the "reference" gradient
represents. In A-GEM it comes from an episodic memory buffer of past tasks; in PURGE
it comes from a live forward pass on a retain batch.

We initially found this paper while reading the continual learning survey (Parisi et al.,
2019) and looking for gradient-based CL methods that could be adapted.

## Evaluation Methodology

**Shokri et al.** — "Membership Inference Attacks Against Machine Learning Models" (IEEE S&P 2017)

Standard reference for membership inference attacks. We use their general framework
(train shadow models → learn to distinguish members from non-members) but simplify
it to loss-based and confidence-based signals with AUROC scoring.

**Carlini et al.** — "Membership Inference Attacks From First Principles" (IEEE S&P 2022)

More recent and rigorous MIA formulation. Their key insight is that threshold-based
MIA accuracy is misleading when the member/non-member sets are imbalanced. This is
why we switched to AUROC — it's threshold-free and handles imbalance properly.

We initially used threshold-based accuracy (like many papers) and got confusingly
low numbers. Reading Carlini et al. made us realise the evaluation was broken, not
the unlearning.

## Baselines and Comparison Methods

**SalUn** — Fan et al., "SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency" (ICLR 2024)

Our primary comparison baseline. SalUn uses gradient saliency to mask which weights to modify
during unlearning. It gets excellent results on CIFAR-10 (TA=93.51%) but doesn't report
AUROC-based MIA. We use their Table A2 numbers for comparison, which is admittedly not
a perfectly fair setup since seed/augmentation differences can shift numbers.

**Bad Teacher** — Chundawat et al. (AAAI 2023)

Uses knowledge distillation with a deliberately incompetent teacher for the forget set. We
borrowed the general idea of KD-based anchoring but apply it differently: our KD component
anchors the *retain* set, not the forget set.

**Golatkar et al.** — "Eternal Sunshine of the Spotless Net" (CVPR 2020)

Showed that output-level suppression alone doesn't guarantee information erasure in
the hidden representations. This motivated our multi-layer representation erasure component.
We use MSE between forget activations and retain-mean activations instead of their
Fisher-based approach, which requires expensive Hessian approximations.

## BatchNorm Understanding

**Ioffe & Szegedy** — "Batch Normalization: Accelerating Deep Network Training" (ICML 2015)

The original BN paper. Our contribution isn't about BatchNorm itself — it's about
documenting that BN's running statistics get silently corrupted when you pass class-
homogeneous batches (which is exactly what happens during class-level unlearning).
This failure mode isn't discussed in any unlearning paper we've found.

## Other Influences

**Thudi et al.** — "Unrolling SGD" (IEEE EuroS&P 2022): Gradent ascent baseline and analysis
of what factors make approximate unlearning effective. Their verification error metric
(L2 between unlearned and retrained weights) informed our thinking about what it means
to "truly" unlearn.

**Bourtoule et al.** — "Machine Unlearning" (IEEE S&P 2021): SISA training framework. We
don't use data partitioning but this paper is the standard reference for the problem
definition and threat model.

**Kurmanji et al.** — "Towards Unbounded Machine Unlearning" (NeurIPS 2023): Mentioned the
connection between CL and MU in their related work section, but did not construct a
full unlearning algorithm around it. This confirmed we were on the right track with the duality idea.

**Hinton et al.** — "Distilling the Knowledge in a Neural Network" (NeurIPS 2015 Workshop):
Standard KD reference. We use temperature-scaled KD for the retain-anchoring component.
