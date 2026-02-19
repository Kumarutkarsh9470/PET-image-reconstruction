"""
moeap_visualize.py
==================
Module 5: Visualisation & Results Analysis

Handles:
- Pareto front plots (Figure 5 equivalent)
- Population evolution plots (Figure 4 equivalent)
- Image grid display (Figure 6 equivalent)
- Reference metric correlation plots (Figure 8 equivalent)
- Convergence plot (Figure 12 equivalent)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# 1.  Pareto front comparison plot  (paper Figure 5)
# ---------------------------------------------------------------------------

def plot_pareto_comparison(pareto_scores,
                            em_results=None,
                            map_results=None,
                            title="Objective Space: Final Generation",
                            save_path=None):
    """
    Plot MOEAP Pareto front vs conventional baselines.

    Parameters
    ----------
    pareto_scores : list of (quant, detect) — MOEAP Pareto front
    em_results    : list of (image, (quant, detect)) from EM+smoothing
    map_results   : list of (image, (quant, detect)) from MAP
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # MOEAP solutions
    qs = [s[0] for s in pareto_scores]
    ds = [s[1] for s in pareto_scores]
    ax.plot(qs, ds, 'b-o', markersize=5, linewidth=1.5,
            label="MOEAP (Genetic Algorithm)", zorder=3)

    # Label endpoints
    if len(pareto_scores) >= 3:
        mid = len(pareto_scores) // 2
        for idx, lbl in [(0, "A\n(max detect)"),
                         (mid, "B\n(compromise)"),
                         (-1, "C\n(max quant)")]:
            ax.annotate(lbl, xy=(qs[idx], ds[idx]),
                        xytext=(10, 5), textcoords='offset points',
                        fontsize=8, color='blue')

    # EM + smoothing baseline
    if em_results is not None:
        em_qs = [r[1][0] for r in em_results]
        em_ds = [r[1][1] for r in em_results]
        ax.plot(em_qs, em_ds, 'g-+', markersize=7, linewidth=1.2,
                label="EM + Post-Smoothing")

    # MAP baseline
    if map_results is not None:
        map_qs = [r[1][0] for r in map_results]
        map_ds = [r[1][1] for r in map_results]
        ax.plot(map_qs, map_ds, 'k--^', markersize=6, linewidth=1.2,
                label="Penalised Likelihood (MAP)")

    ax.set_xlabel("Quantification Objective (Φ_quant)", fontsize=12)
    ax.set_ylabel("Detection Objective (Φ_detect)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 2.  Population evolution scatter (paper Figure 4)
# ---------------------------------------------------------------------------

def plot_population_evolution(history, generations_to_show=None,
                               save_path=None):
    """
    Show how the population evolves in objective space across generations.
    """
    if generations_to_show is None:
        n = len(history.gen)
        generations_to_show = [0,
                                n // 4,
                                n // 2,
                                n - 1]
        generations_to_show = sorted(set(
            min(g, n-1) for g in generations_to_show
        ))

    n_plots = len(generations_to_show)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4),
                              sharey=True)
    if n_plots == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_plots))

    for ax, gen_idx, col in zip(axes, generations_to_show, colors):
        pf_scores = history.pareto_front_scores[gen_idx]
        qs = [s[0] for s in pf_scores]
        ds = [s[1] for s in pf_scores]
        ax.scatter(qs, ds, c=[col], s=30, alpha=0.7, edgecolors='k', lw=0.3)
        ax.set_title(f"Gen {history.gen[gen_idx]+1}", fontsize=11)
        ax.set_xlabel("Φ_quant")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Φ_detect")
    fig.suptitle("Population Evolution in Objective Space", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 3.  Image grid display  (paper Figure 6)
# ---------------------------------------------------------------------------

def plot_image_grid(images, titles, ncols=3, cmap='hot',
                    suptitle="Reconstructed Images",
                    save_path=None):
    """
    Display a grid of reconstructed PET images.

    Parameters
    ----------
    images : list of np.ndarray (H, W)
    titles : list of str
    """
    n = len(images)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    # Global colour scale across all images
    vmin = min(img.min() for img in images)
    vmax = max(img.max() for img in images)

    for i, (img, title) in enumerate(zip(images, titles)):
        im = axes[i].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 4.  Reference metric correlation plots  (paper Figure 8)
# ---------------------------------------------------------------------------

def plot_reference_correlations(pareto_scores, ref_metrics_list,
                                 save_path=None):
    """
    Plot mean_error vs Φ_quant  and  SNR_known vs Φ_detect.

    Parameters
    ----------
    pareto_scores    : list of (quant, detect)
    ref_metrics_list : list of dicts from compute_reference_metrics()
    """
    quant_vals  = [s[0] for s in pareto_scores]
    detect_vals = [s[1] for s in pareto_scores]
    me_vals     = [m["mean_error_pct"]  for m in ref_metrics_list]
    snr_vals    = [m["snr_known_loc"]   for m in ref_metrics_list]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(quant_vals, me_vals, c='steelblue', s=40,
                    edgecolors='k', lw=0.4, alpha=0.8)
    axes[0].set_xlabel("Φ_quant (Poisson log-likelihood)", fontsize=11)
    axes[0].set_ylabel("Mean Error in Tumour (%)", fontsize=11)
    axes[0].set_title("Quantification Reference Test", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Trend line
    z = np.polyfit(quant_vals, me_vals, 1)
    p = np.poly1d(z)
    xs = np.linspace(min(quant_vals), max(quant_vals), 100)
    axes[0].plot(xs, p(xs), 'r--', alpha=0.6, label='trend')
    axes[0].legend(fontsize=9)

    axes[1].scatter(detect_vals, snr_vals, c='darkorange', s=40,
                    edgecolors='k', lw=0.4, alpha=0.8)
    axes[1].set_xlabel("Φ_detect (Scan-Statistic SNR)", fontsize=11)
    axes[1].set_ylabel("Known-Location SNR", fontsize=11)
    axes[1].set_title("Detection Reference Test", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    z2 = np.polyfit(detect_vals, snr_vals, 1)
    p2 = np.poly1d(z2)
    xs2 = np.linspace(min(detect_vals), max(detect_vals), 100)
    axes[1].plot(xs2, p2(xs2), 'r--', alpha=0.6, label='trend')
    axes[1].legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 5.  Convergence plot  (paper Figure 12)
# ---------------------------------------------------------------------------

def plot_convergence(history, save_path=None):
    """
    Plot max quant, max detect, and Pareto front size vs generation.
    """
    gens = history.gen

    max_quants  = [max(s[0] for s in pf) for pf in history.pareto_front_scores]
    max_detects = [max(s[1] for s in pf) for pf in history.pareto_front_scores]
    front_sizes = history.front_size

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(gens, max_quants, 'b-', linewidth=1.5)
    axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Max Φ_quant")
    axes[0].set_title("Quantification Convergence"); axes[0].grid(alpha=0.3)

    axes[1].plot(gens, max_detects, 'g-', linewidth=1.5)
    axes[1].set_xlabel("Generation"); axes[1].set_ylabel("Max Φ_detect")
    axes[1].set_title("Detection Convergence"); axes[1].grid(alpha=0.3)

    axes[2].plot(gens, front_sizes, 'r-', linewidth=1.5)
    axes[2].set_xlabel("Generation"); axes[2].set_ylabel("Pareto Front Size")
    axes[2].set_title("Front Size Evolution"); axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 6.  Convenience: Full results dashboard
# ---------------------------------------------------------------------------

def save_full_dashboard(pareto_images, pareto_scores, history,
                         x_true, tumor_mask, roi_mask,
                         em_results=None,
                         output_dir="/home/claude/moeap/results"):
    """
    Generate and save all plots in one call.
    """
    import os
    from pet_objectives import compute_reference_metrics

    os.makedirs(output_dir, exist_ok=True)

    # --- Pareto front ---
    plot_pareto_comparison(
        pareto_scores, em_results=em_results,
        save_path=f"{output_dir}/pareto_front.png"
    )

    # --- Evolution ---
    plot_population_evolution(
        history,
        save_path=f"{output_dir}/population_evolution.png"
    )

    # --- Key images ---
    n = len(pareto_images)
    mid = n // 2
    key_images = [pareto_images[0], pareto_images[mid], pareto_images[-1], x_true]
    key_titles = ["A: Max Detect", "B: Compromise", "C: Max Quant", "Ground Truth"]

    if em_results is not None:
        mid_em = len(em_results) // 2
        key_images.append(em_results[mid_em][0])
        key_titles.append("D: EM+Smooth")

    plot_image_grid(
        key_images, key_titles,
        suptitle="Key Reconstructed Images",
        save_path=f"{output_dir}/image_grid.png"
    )

    # --- Reference correlations ---
    ref_metrics = [
        compute_reference_metrics(img, x_true, tumor_mask)
        for img in pareto_images
    ]
    plot_reference_correlations(
        pareto_scores, ref_metrics,
        save_path=f"{output_dir}/reference_correlations.png"
    )

    # --- Convergence ---
    plot_convergence(history, save_path=f"{output_dir}/convergence.png")

    print(f"\nAll plots saved to: {output_dir}/")
    return ref_metrics
