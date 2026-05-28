"""
Generate all figures for the PURGE report.

Run from the 05_results/ directory:
    python3 generate_figures.py

This creates 12 plots in 05_results/figures/. The data is hardcoded in the
script (copied from our actual run logs) rather than loaded from CSVs because
we kept changing the CSV format during development and got tired of parsing bugs.

NOTE: If you re-run experiments with different seeds, you'll need to update
the numbers in this file manually. Yeah, it's not ideal.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PAL = {
    "blue":   "#1976D2",
    "red":    "#D32F2F",
    "green":  "#388E3C",
    "orange": "#F57C00",
    "purple": "#7B1FA2",
    "teal":   "#00796B",
    "grey":   "#607D8B",
    "pink":   "#C2185B",
    "cyan":   "#0097A7",
    "lime":   "#689F38",
}

DS_COLORS = {
    "CIFAR-10":  PAL["blue"],
    "MNIST":     PAL["green"],
    "SVHN":      PAL["orange"],
    "STL10":     PAL["purple"],
    "PathMNIST": PAL["teal"],
}

MNIST_DATA = {
    "classes": list(range(10)),
    "ta": [89.99, 88.30, 89.99, 90.77, 90.68, 91.24, 90.97, 90.25, 90.66, 90.01],
    "fa": [1.99, 5.38, 7.67, 9.28, 6.09, 7.38, 7.06, 8.14, 9.38, 5.31],
    "ra": [99.82, 99.87, 99.87, 99.88, 99.87, 99.88, 99.85, 99.90, 99.84, 99.91],
    "mia": [0.512, 0.653, 0.510, 0.510, 0.511, 0.521, 0.495, 0.522, 0.512, 0.524],
}

PATHMNIST_DATA = {
    "classes": list(range(9)),
    "labels": ["Adipose", "Background", "Debris", "Lympho.", "Mucus",
               "Smooth M.", "Normal M.", "Stroma", "Epithelium"],
    "ta": [74.35, 79.35, 89.30, 83.20, 77.49, 84.36, 80.93, 87.16, 75.96],
    "fa": [7.28, 9.85, 9.47, 10.58, 9.44, 9.45, 9.87, 10.71, 7.12],
    "ra": [98.45, 98.33, 98.93, 99.00, 98.53, 98.55, 98.81, 99.39, 98.74],
    "mia": [0.545, 0.614, 0.369, 0.425, 0.517, 0.501, 0.594, 0.521, 0.557],
}

CROSS_DS = {
    "names":    ["CIFAR-10", "MNIST", "SVHN", "STL10", "PathMNIST"],
    "ta":       [84.47, 90.19, 89.50, 84.95, 81.23],
    "fa":       [8.38, 6.77, 3.62, 8.80, 9.31],
    "ra":       [96.31, 99.87, 97.32, 99.07, 98.75],
    "mia":      [0.496, 0.517, 0.524, 0.479, 0.516],
    "cosine":   [0.800, 0.637, 0.758, 0.757, 0.674],
}


# =========================================================================
# Figure 1: TA / FA / RA comparison across methods (CIFAR-10)
# =========================================================================
def fig1_method_comparison():
    methods = ["Retrain", "SalUn", "l1-sparse", "PURGE\n(ours)", "IU", "RL", "GA", "FT"]
    ta = [92.47, 93.51, 92.29, 84.47, 89.10, 94.52, 38.18, 94.78]
    fa = [0.00, 0.72, 0.00, 8.38, 2.98, 10.67, 0.09, 68.31]
    ra = [100.0, 99.40, 97.92, 96.31, 94.78, 99.92, 38.92, 99.92]

    x = np.arange(len(methods))
    w = 0.24

    fig, ax = plt.subplots(figsize=(12, 5.5))
    b1 = ax.bar(x - w, ta, w, label="Test Acc (TA)", color=PAL["blue"], edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x, fa, w, label="Forget Acc (FA \u2193)", color=PAL["red"], edgecolor="white", linewidth=0.5)
    b3 = ax.bar(x + w, ra, w, label="Retain Acc (RA)", color=PAL["green"], edgecolor="white", linewidth=0.5)

    ax.axhline(y=10, color=PAL["grey"], linestyle="--", alpha=0.5, linewidth=1)
    ax.text(7.6, 11.5, "random chance", fontsize=8, color=PAL["grey"], alpha=0.7)

    highlight_idx = 3
    ax.axvspan(highlight_idx - 0.45, highlight_idx + 0.45, alpha=0.07, color=PAL["blue"])

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CIFAR-10 Class-Level Forgetting \u2014 Method Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.2)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 4:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=6.5)

    fig.savefig(os.path.join(OUT, "fig1_method_comparison.png"))
    plt.close(fig)
    print("  Saved fig1_method_comparison.png")


# =========================================================================
# Figure 2: MIA AUROC across all 5 datasets
# =========================================================================
def fig2_mia_comparison():
    names = CROSS_DS["names"]
    aurocs = CROSS_DS["mia"]
    colors = [DS_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, aurocs, color=colors, edgecolor="white", width=0.55, linewidth=0.5)

    ax.axhline(y=0.5, color=PAL["red"], linestyle="--", alpha=0.6, linewidth=1.5)
    ax.axhspan(0.45, 0.55, alpha=0.06, color=PAL["green"])
    ax.text(4.35, 0.555, "ideal zone", fontsize=8, color=PAL["green"], fontstyle="italic")

    ax.set_ylabel("MIA AUROC")
    ax.set_title("Membership Inference Attack: AUROC Across Datasets\n(closer to 0.5 = better privacy)", fontweight="bold")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", alpha=0.2)

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.savefig(os.path.join(OUT, "fig2_mia_auroc.png"))
    plt.close(fig)
    print("  Saved fig2_mia_auroc.png")


# =========================================================================
# Figure 3: FA vs Epoch — CIFAR-10 kl_retain progression
# =========================================================================
def fig3_fa_vs_epoch():
    epochs_c = [0, 1, 1.75]
    fa_c = [98.44, 31.3, 8.38]
    epochs_p = [0, 1, 1.14]
    fa_p = [99.83, 21.9, 7.28]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_c, fa_c, "o-", color=PAL["blue"], linewidth=2.5, markersize=9,
            label="CIFAR-10", zorder=5)
    ax.plot(epochs_p, fa_p, "s-", color=PAL["teal"], linewidth=2.5, markersize=9,
            label="PathMNIST", zorder=5)

    ax.axhline(y=10, color=PAL["red"], linestyle=":", alpha=0.6, linewidth=1.5, label="Random chance (10%)")
    ax.fill_between([0, 2.5], 0, 12, alpha=0.06, color=PAL["green"])

    ax.annotate("intra-epoch stop\n(batch 30)", xy=(1.75, 8.38), xytext=(2.15, 35),
                arrowprops=dict(arrowstyle="-|>", color=PAL["grey"], lw=1.2),
                fontsize=9, ha="center", color=PAL["blue"])
    ax.annotate("intra-epoch stop\n(batch 10)", xy=(1.14, 7.28), xytext=(1.7, 50),
                arrowprops=dict(arrowstyle="-|>", color=PAL["grey"], lw=1.2),
                fontsize=9, ha="center", color=PAL["teal"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Forget Accuracy (%)")
    ax.set_title("Forget Accuracy Progression During Unlearning", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(-0.1, 2.6)
    ax.set_ylim(0, 108)
    ax.grid(alpha=0.2)

    fig.savefig(os.path.join(OUT, "fig3_fa_vs_epoch.png"))
    plt.close(fig)
    print("  Saved fig3_fa_vs_epoch.png")


# =========================================================================
# Figure 4: Ablation study — component removal impact
# =========================================================================
def fig4_ablation():
    configs = [
        "Full PURGE\n(kl_retain)",
        "GA\nobjective",
        "GA\n(longer)",
        "No intra-epoch\nstopping",
        "Entropy\ngating",
        "BN unfrozen",
    ]
    ta = [84.47, 89.28, 80.77, 72.65, 89.06, 21.56]
    fa = [8.38, 52.56, 0.02, 0.00, 58.64, 0.00]
    mia = [0.496, 0.250, 0.243, 0.001, 0.262, 0.0]

    colors = [PAL["blue"], PAL["orange"], PAL["red"], PAL["purple"], PAL["grey"], "#333333"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, vals, title, ideal_line in zip(
        axes,
        [ta, fa, mia],
        ["Test Accuracy (%)\n(higher = better)", "Forget Accuracy (%)\n(\u224810% = ideal)", "MIA AUROC\n(0.5 = ideal)"],
        [None, 10, 0.5],
    ):
        bars = ax.bar(range(len(configs)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, fontsize=7.5)
        ax.grid(axis="y", alpha=0.2)
        if ideal_line is not None:
            ax.axhline(y=ideal_line, color=PAL["red"], linestyle="--", alpha=0.5, linewidth=1.5)
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                fmt = f"{h:.3f}" if max(vals) <= 1 else f"{h:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2, h + max(vals) * 0.02,
                        fmt, ha="center", va="bottom", fontsize=7)

    fig.suptitle("Ablation Study: Impact of PURGE Components (CIFAR-10)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT, "fig4_ablation.png"))
    plt.close(fig)
    print("  Saved fig4_ablation.png")


# =========================================================================
# Figure 5: Before vs After unlearning — key metrics
# =========================================================================
def fig5_before_after():
    metrics = ["Test Acc", "Forget Acc", "Retain Acc"]
    before = [93.40, 98.44, 96.79]
    after = [84.47, 8.38, 96.31]

    x = np.arange(len(metrics))
    w = 0.30

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, before, w, label="Before Unlearning", color="#BDBDBD", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, after, w, label="After PURGE (kl_retain)", color=PAL["blue"], edgecolor="white", linewidth=0.5)

    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - w / 2, b + 1.5, f"{b:.1f}%", ha="center", fontsize=10, color="#555")
        ax.text(i + w / 2, a + 1.5, f"{a:.1f}%", ha="center", fontsize=10, fontweight="bold", color=PAL["blue"])

    delta_fa = before[1] - after[1]
    ax.annotate(f"\u2193{delta_fa:.0f}pp", xy=(1 + w / 2, after[1] + 5), fontsize=11,
                fontweight="bold", color=PAL["red"], ha="center")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CIFAR-10: Before vs After PURGE Unlearning (Class 0)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.2)

    fig.savefig(os.path.join(OUT, "fig5_before_after.png"))
    plt.close(fig)
    print("  Saved fig5_before_after.png")


# =========================================================================
# Figure 6: PURGE algorithm overview (block diagram)
# =========================================================================
def fig6_algorithm_overview():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color, textcolor="black", fontsize=9, bold=False):
        rect = mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                        boxstyle="round,pad=0.15", facecolor=color,
                                        edgecolor="#333", linewidth=1.2)
        ax.add_patch(rect)
        fw = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=textcolor, fontweight=fw, linespacing=1.4)

    draw_box(2, 7, 2.8, 0.9, "Retain Batch\n(x_r, y_r)", "#C8E6C9", bold=True)
    draw_box(7, 7, 2.8, 0.9, "Forget Batch\n(x_f, y_f)", "#FFCDD2", bold=True)
    draw_box(12, 7, 2.8, 0.9, "Frozen Original\nModel \u03B8_orig", "#BBDEFB", bold=True)

    draw_box(2, 5.2, 3.2, 1.0, "CE(retain)\n+ \u03B2\u00B7KD(out_r \u2225 \u03B8_orig)", "#E8F5E9")
    draw_box(7, 5.2, 3.0, 1.0, "Entropy Gate\nH(out_f) > \u03C4 ?", "#FFF3E0")
    draw_box(12, 5.2, 2.8, 0.8, "KD Targets\n(no grad)", "#E3F2FD")

    draw_box(2, 3.4, 2.0, 0.7, "g_retain", "#A5D6A7", bold=True, fontsize=11)
    draw_box(7, 3.4, 3.2, 1.2, "Forget Objective\n\u2212CE / KL\u2192uniform\n/ KL\u2192retain_confuse\n+ \u03BB\u00B7RepErasure", "#FFCDD2", fontsize=8)

    draw_box(7, 1.5, 2.0, 0.7, "g_forget", "#EF9A9A", bold=True, fontsize=11)

    draw_box(3.5, 1.5, 2.8, 0.9, "Projection\nif \u27E8g_f,g_r\u27E9 < 0:\ng_f \u2190 g_f \u2212 proj", "#FFF9C4", fontsize=8)

    draw_box(11, 1.5, 3.0, 0.9, "\u03B8 \u2190 \u03B8 \u2212 \u03B7\u00B7clip(g_f)\n+ stopping checks", "#FFE0B2", bold=True, fontsize=9)

    arrow = dict(arrowstyle="-|>", color="#555", lw=1.3, mutation_scale=12)
    for sx, sy, ex, ey in [
        (2, 6.55, 2, 5.7), (7, 6.55, 7, 5.7), (12, 6.55, 12, 5.6),
        (2, 4.7, 2, 3.75), (7, 4.7, 7, 4.0), (7, 2.8, 7, 1.85),
        (6.0, 1.5, 4.9, 1.5), (2, 3.05, 3.0, 1.95),
        (8.0, 1.5, 9.5, 1.5), (10.5, 5.2, 8.5, 5.2),
    ]:
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy), arrowprops=arrow)

    ax.set_title("PURGE: Algorithm Overview", fontsize=16, fontweight="bold", pad=15)

    fig.savefig(os.path.join(OUT, "fig6_algorithm_overview.png"))
    plt.close(fig)
    print("  Saved fig6_algorithm_overview.png")


# =========================================================================
# Figure 7: Representation-level feature cosine similarity across datasets
# =========================================================================
def fig7_representation():
    names = CROSS_DS["names"]
    cosines = CROSS_DS["cosine"]
    colors = [DS_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names[::-1], cosines[::-1], color=colors[::-1], edgecolor="white",
                   height=0.55, linewidth=0.5)

    ax.axvline(x=1.0, color="#999", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(0.99, -0.6, "before unlearning", fontsize=8, color="#999", ha="right")

    ax.set_xlabel("Feature Cosine Similarity (lower = more erased)")
    ax.set_title("Representation-Level Erasure Across Datasets", fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.2)

    for bar, val in zip(bars, cosines[::-1]):
        ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold")

    fig.savefig(os.path.join(OUT, "fig7_representation.png"))
    plt.close(fig)
    print("  Saved fig7_representation.png")


# =========================================================================
# Figure 8: Cross-dataset comparison — 4-panel bar chart (all 5 datasets)
# =========================================================================
def fig8_dataset_comparison():
    names = CROSS_DS["names"]
    ta = CROSS_DS["ta"]
    fa = CROSS_DS["fa"]
    ra = CROSS_DS["ra"]
    mia = CROSS_DS["mia"]
    colors = [DS_COLORS[n] for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for ax, vals, title, ideal, ylim in zip(
        axes,
        [ta, fa, ra, mia],
        ["Test Accuracy (%)", "Forget Accuracy (%) \u2193", "Retain Accuracy (%)", "MIA AUROC"],
        [None, None, None, 0.5],
        [(60, 100), (0, 15), (93, 101), (0.3, 0.7)],
    ):
        bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white", width=0.6, linewidth=0.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
        ax.set_ylim(ylim)
        ax.grid(axis="y", alpha=0.2)
        if ideal is not None:
            ax.axhline(y=ideal, color=PAL["red"], linestyle="--", alpha=0.5, linewidth=1.5)
        for bar, v in zip(bars, vals):
            fmt = f"{v:.3f}" if title == "MIA AUROC" else f"{v:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, v + (ylim[1] - ylim[0]) * 0.02,
                    fmt, ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("PURGE (kl_retain): Cross-Dataset Performance Summary", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(os.path.join(OUT, "fig8_dataset_comparison.png"))
    plt.close(fig)
    print("  Saved fig8_dataset_comparison.png")


# =========================================================================
# Figure 9: MNIST — Per-class metrics (dot/strip plot)
# =========================================================================
def fig9_mnist_per_class():
    d = MNIST_DATA
    classes = [str(c) for c in d["classes"]]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    metrics = [
        (d["ta"], "Test Accuracy (%)", PAL["blue"], (85, 93)),
        (d["fa"], "Forget Accuracy (%) \u2193", PAL["red"], (0, 12)),
        (d["ra"], "Retain Accuracy (%)", PAL["green"], (99.7, 100.0)),
        (d["mia"], "MIA AUROC", PAL["purple"], (0.4, 0.7)),
    ]

    for ax, (vals, title, color, ylim) in zip(axes.flat, metrics):
        ax.plot(classes, vals, "o-", color=color, linewidth=2, markersize=7, zorder=5)
        ax.fill_between(classes, [min(vals)] * len(classes), vals, alpha=0.08, color=color)
        avg = np.mean(vals)
        ax.axhline(y=avg, color=color, linestyle="--", alpha=0.4, linewidth=1)
        ax.text(9.3, avg, f"avg\n{avg:.2f}", fontsize=7, color=color, va="center")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Forget Class")
        ax.set_ylim(ylim)
        ax.grid(alpha=0.2)

        if title == "MIA AUROC":
            ax.axhline(y=0.5, color=PAL["grey"], linestyle=":", alpha=0.5)
            ax.text(0, 0.505, "ideal", fontsize=7, color=PAL["grey"])

    fig.suptitle("PURGE on MNIST \u2014 Per-Class Unlearning Results", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "fig9_mnist_per_class.png"))
    plt.close(fig)
    print("  Saved fig9_mnist_per_class.png")


# =========================================================================
# Figure 10: PathMNIST — Per-class metrics (horizontal bar chart)
# =========================================================================
def fig10_pathmnist_per_class():
    d = PATHMNIST_DATA
    labels = d["labels"]
    n = len(labels)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5.5))

    metrics = [
        (d["ta"], "Test Acc (%)", PAL["blue"]),
        (d["fa"], "Forget Acc (%) \u2193", PAL["red"]),
        (d["ra"], "Retain Acc (%)", PAL["green"]),
        (d["mia"], "MIA AUROC", PAL["purple"]),
    ]

    for ax, (vals, title, color) in zip(axes, metrics):
        y_pos = np.arange(n)
        bars = ax.barh(y_pos, vals, color=color, alpha=0.75, edgecolor="white",
                       height=0.6, linewidth=0.5)
        ax.set_yticks(y_pos)
        if ax == axes[0]:
            ax.set_yticklabels(labels, fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()

        for bar, v in zip(bars, vals):
            fmt = f"{v:.3f}" if title == "MIA AUROC" else f"{v:.1f}"
            ax.text(v + (max(vals) - min(vals)) * 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    fmt, va="center", fontsize=8, fontweight="bold")

        if title == "MIA AUROC":
            ax.axvline(x=0.5, color=PAL["grey"], linestyle="--", alpha=0.5)

    fig.suptitle("PURGE on PathMNIST \u2014 Per-Class Unlearning Results (9 Tissue Types)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "fig10_pathmnist_per_class.png"))
    plt.close(fig)
    print("  Saved fig10_pathmnist_per_class.png")


# =========================================================================
# Figure 11: Radar / spider chart — Cross-dataset profile
# =========================================================================
def fig11_radar_cross_dataset():
    categories = ["TA (%)", "100\u2212FA (%)", "RA (%)", "MIA\n(1\u2212|0.5\u2212x|\u00D72)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)

    for name, ta, fa, ra, mia in zip(
        CROSS_DS["names"], CROSS_DS["ta"], CROSS_DS["fa"], CROSS_DS["ra"], CROSS_DS["mia"]
    ):
        inv_fa = 100 - fa
        mia_score = (1 - abs(0.5 - mia) * 2) * 100
        values = [ta, inv_fa, ra, mia_score]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, markersize=5, label=name,
                color=DS_COLORS[name])
        ax.fill(angles, values, alpha=0.06, color=DS_COLORS[name])

    ax.set_ylim(60, 105)
    ax.set_rticks([70, 80, 90, 100])
    ax.set_rlabel_position(30)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=9, framealpha=0.9)
    ax.set_title("Cross-Dataset Performance Profile\n(all metrics scaled, higher = better)",
                 fontweight="bold", fontsize=13, pad=25)

    fig.savefig(os.path.join(OUT, "fig11_radar_cross_dataset.png"))
    plt.close(fig)
    print("  Saved fig11_radar_cross_dataset.png")


# =========================================================================
# Figure 12: RA preservation — how close each dataset stays to 100%
# =========================================================================
def fig12_ra_preservation():
    names = CROSS_DS["names"]
    ra = CROSS_DS["ra"]
    colors = [DS_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 4))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, ra, color=colors, edgecolor="white", height=0.5, linewidth=0.5)
    ax.set_xlim(93, 101)
    ax.axvline(x=100, color="#999", linestyle=":", alpha=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Retain Accuracy (%)")
    ax.set_title("Retain Accuracy Preservation Across Datasets\n(PURGE kl_retain \u2014 closer to 100% = less collateral damage)",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()

    for bar, v in zip(bars, ra):
        gap = 100 - v
        ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}%  (\u2212{gap:.2f}pp)", va="center", fontsize=10, fontweight="bold")

    fig.savefig(os.path.join(OUT, "fig12_ra_preservation.png"))
    plt.close(fig)
    print("  Saved fig12_ra_preservation.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_method_comparison()
    fig2_mia_comparison()
    fig3_fa_vs_epoch()
    fig4_ablation()
    fig5_before_after()
    fig6_algorithm_overview()
    fig7_representation()
    fig8_dataset_comparison()
    fig9_mnist_per_class()
    fig10_pathmnist_per_class()
    fig11_radar_cross_dataset()
    fig12_ra_preservation()
    print(f"\nAll 12 figures saved to {OUT}/")
