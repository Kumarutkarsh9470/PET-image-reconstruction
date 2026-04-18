"""
moeap_doctor_view.py
====================
Generates the "clinical spectrum" visualization — the full Pareto front
laid out so a doctor can browse the complete detection↔quantification
tradeoff and pick the image best suited to their task.

Also includes improved reference correlation plots addressing the issues
identified in the result analysis.

Run standalone:
    python moeap_doctor_view.py --results_dir results/
Or called from save_full_dashboard automatically.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os


# ---------------------------------------------------------------------------
# 1.  DOCTOR SPECTRUM VIEW
#     Full Pareto front as a scrollable strip with clinical annotations
# ---------------------------------------------------------------------------

def plot_doctor_spectrum(pareto_images, pareto_scores,
                          x_true=None,
                          ref_metrics=None,
                          save_path=None,
                          max_cols=10):
    """
    The clinical spectrum plot: every image on the Pareto front displayed
    in a strip ordered from MAX DETECTION → MAX QUANTIFICATION.

    Each image is annotated with:
      - Its position on the front (rank)
      - Φ_detect value  (proxy for lesion visibility)
      - Φ_quant value   (proxy for measurement accuracy)
      - ME% and SNR if ref_metrics provided

    Parameters
    ----------
    pareto_images  : list of (H,W) arrays, sorted quant ascending (as returned by MOEAP)
    pareto_scores  : list of (quant, detect) tuples, matching order
    x_true         : (H,W) ground truth — appended at end if provided
    ref_metrics    : list of dicts from compute_reference_metrics, same length as pareto_images
    max_cols       : max images per row before wrapping
    """
    n      = len(pareto_images)
    # pareto_images is sorted quant-ascending → detect-descending
    # For doctor view: show detect-descending (most detectable first = left)
    images = list(reversed(pareto_images))
    scores = list(reversed(pareto_scores))
    refs   = list(reversed(ref_metrics)) if ref_metrics else None

    # Add ground truth at very end if provided
    if x_true is not None:
        images.append(x_true)
        scores.append(None)
        if refs is not None:
            refs.append(None)

    total  = len(images)
    ncols  = min(total, max_cols)
    nrows  = int(np.ceil(total / ncols))

    # Global colour scale
    vmin   = min(img.min() for img in pareto_images)
    vmax   = max(img.max() for img in pareto_images)

    # Figure sizing: each image cell is 2.2" wide, 3.2" tall (image + annotation)
    fig_w  = ncols * 2.2 + 0.8
    fig_h  = nrows * 3.4 + 1.2
    fig    = plt.figure(figsize=(fig_w, fig_h))

    # Title + direction arrow
    fig.suptitle(
        "Clinical Image Spectrum  —  Full Pareto Front\n"
        "← More Detectable   |   More Quantitatively Accurate →",
        fontsize=13, fontweight='bold', y=0.98
    )

    # Colour-code border by detect score for quick scanning
    detect_vals = [s[1] for s in scores if s is not None]
    d_min, d_max = min(detect_vals), max(detect_vals)
    cmap_border  = plt.cm.RdYlGn   # red = low detect, green = high detect

    for idx, (img, score) in enumerate(zip(images, scores)):
        row = idx // ncols
        col = idx % ncols

        # Axes position: leave room at top for title
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        ax.imshow(img, cmap='hot', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Colour-coded border
        if score is not None:
            norm_d  = (score[1] - d_min) / (d_max - d_min + 1e-10)
            border_color = cmap_border(norm_d)
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3.5)

        # Annotation below image
        if score is None:
            ax.set_title("Ground\nTruth", fontsize=7.5, color='white',
                         fontweight='bold', pad=2,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='navy', alpha=0.8))
        else:
            rank_from_detect = idx + 1   # 1 = most detectable
            title_lines = [f"#{rank_from_detect}  D={score[1]:.1f}"]
            if refs is not None and refs[idx] is not None:
                r = refs[idx]
                title_lines.append(f"ME={r['mean_error_pct']:+.0f}%  SNR={r['snr_known_loc']:.2f}")

            # Background colour from detect score
            fc = cmap_border(norm_d)
            ax.set_title("\n".join(title_lines),
                         fontsize=6.5, pad=2,
                         bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor=fc, alpha=0.7))

    # Colour bar for detect score
    sm  = ScalarMappable(cmap=cmap_border, norm=Normalize(vmin=d_min, vmax=d_max))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Φ_detect\n(detection score)", fontsize=9)

    plt.subplots_adjust(left=0.02, right=0.90, top=0.90, bottom=0.02,
                        wspace=0.08, hspace=0.35)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#1a1a1a')
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# 2.  IMPROVED PARETO FRONT PLOT
#     Addresses the issue seen in results: EM baseline barely overlaps
#     the MOEAP front. Better separation visualisation.
# ---------------------------------------------------------------------------

def plot_pareto_comparison_improved(pareto_scores,
                                     em_results=None,
                                     map_results=None,
                                     save_path=None):
    """
    Improved Pareto comparison with:
    - Dominance region shading
    - Per-point colour coding by position on front
    - EM baseline comparison with gap annotation
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor('#f8f8f8')

    qs = np.array([s[0] for s in pareto_scores])
    ds = np.array([s[1] for s in pareto_scores])

    # Colour points by their detect score (warm = high detect, cool = high quant)
    sc = ax.scatter(qs, ds, c=ds, cmap='plasma',
                    s=60, zorder=4, edgecolors='white', linewidths=0.5,
                    label='_nolegend_')
    ax.plot(qs, ds, 'b-', linewidth=1.2, alpha=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label='Φ_detect value', fraction=0.03, pad=0.02)

    # Annotate endpoints
    ax.annotate("← Max\nDetect", xy=(qs[0], ds[0]),
                xytext=(-45, 5), textcoords='offset points',
                fontsize=9, color='purple', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
    ax.annotate("Max\nQuant →", xy=(qs[-1], ds[-1]),
                xytext=(5, -30), textcoords='offset points',
                fontsize=9, color='darkorange', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

    # EM baseline
    if em_results is not None:
        em_qs = np.array([r[1][0] for r in em_results])
        em_ds = np.array([r[1][1] for r in em_results])
        ax.plot(em_qs, em_ds, 'g-o', markersize=6, linewidth=1.5,
                label='EM + Post-Smoothing', alpha=0.8, zorder=2)

        # Shade the region where MOEAP dominates EM
        # For overlapping quant range, show the detect gap
        q_overlap_min = max(qs.min(), em_qs.min())
        q_overlap_max = min(qs.max(), em_qs.max())
        if q_overlap_min < q_overlap_max:
            ax.axvspan(q_overlap_min, q_overlap_max, alpha=0.07,
                       color='blue', label='Overlap region')

    if map_results is not None:
        map_qs = [r[1][0] for r in map_results]
        map_ds = [r[1][1] for r in map_results]
        ax.plot(map_qs, map_ds, 'r--^', markersize=6, linewidth=1.5,
                label='MAP (Penalised Likelihood)', alpha=0.8)

    ax.set_xlabel("Quantification Objective (Φ_quant)\n← Less accurate data fit  |  More accurate →",
                  fontsize=11)
    ax.set_ylabel("Detection Objective (Φ_detect)\n← Harder to detect  |  Easier to detect →",
                  fontsize=11)
    ax.set_title("Pareto Front: Detection vs Quantification Tradeoff\n"
                 "Each point = one reconstructed PET image", fontsize=12)

    # Add legend entry for MOEAP manually (scatter doesn't auto-legend well)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=1.5, label='MOEAP (Genetic Algorithm)'),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_elements + handles, fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 3.  FIXED REFERENCE CORRELATION PLOTS
#     The paper expects ME to DECREASE as Φ_quant increases.
#     Our results show the opposite — flag this clearly.
# ---------------------------------------------------------------------------

def plot_reference_correlations_annotated(pareto_scores, ref_metrics_list,
                                           save_path=None):
    """
    Reference correlation plots with honest annotations about
    what the trend direction means and whether it matches the paper.
    """
    quant_vals  = [s[0] for s in pareto_scores]
    detect_vals = [s[1] for s in pareto_scores]
    me_vals     = [m["mean_error_pct"]  for m in ref_metrics_list]
    snr_vals    = [m["snr_known_loc"]   for m in ref_metrics_list]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Plot 1: Mean Error vs Φ_quant ---
    axes[0].scatter(quant_vals, me_vals, c='steelblue', s=40,
                    edgecolors='k', lw=0.4, alpha=0.8)
    z  = np.polyfit(quant_vals, me_vals, 1)
    p  = np.poly1d(z)
    xs = np.linspace(min(quant_vals), max(quant_vals), 100)
    axes[0].plot(xs, p(xs), 'r--', alpha=0.7, label=f'trend (slope={z[0]:.1e})')

    # Annotate trend direction
    direction = "↑ increasing" if z[0] > 0 else "↓ decreasing"
    expected  = "↓ decreasing"
    match     = "✓ Matches paper" if z[0] < 0 else "✗ Opposite to paper expectation"
    color     = 'green' if z[0] < 0 else 'red'
    axes[0].text(0.03, 0.97,
                 f"Trend: ME is {direction} with Φ_quant\n"
                 f"Paper expects: {expected}\n{match}",
                 transform=axes[0].transAxes,
                 fontsize=8.5, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 color=color)

    axes[0].set_xlabel("Φ_quant (Poisson log-likelihood)", fontsize=11)
    axes[0].set_ylabel("Mean Error in Tumour (%)", fontsize=11)
    axes[0].set_title("Quantification Reference Test\n"
                       "(Lower ME = better quantification)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Known-SNR vs Φ_detect ---
    axes[1].scatter(detect_vals, snr_vals, c='darkorange', s=40,
                    edgecolors='k', lw=0.4, alpha=0.8)
    z2  = np.polyfit(detect_vals, snr_vals, 1)
    p2  = np.poly1d(z2)
    xs2 = np.linspace(min(detect_vals), max(detect_vals), 100)
    axes[1].plot(xs2, p2(xs2), 'r--', alpha=0.7, label=f'trend (slope={z2[0]:.2e})')

    direction2 = "↑ increasing" if z2[0] > 0 else "↓ decreasing"
    expected2  = "↑ increasing"
    match2     = "✓ Matches paper" if z2[0] > 0 else "✗ Opposite to paper expectation"
    color2     = 'green' if z2[0] > 0 else 'red'
    axes[1].text(0.03, 0.97,
                 f"Trend: SNR is {direction2} with Φ_detect\n"
                 f"Paper expects: {expected2}\n{match2}",
                 transform=axes[1].transAxes,
                 fontsize=8.5, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 color=color2)

    axes[1].set_xlabel("Φ_detect (Scan-Statistic SNR)", fontsize=11)
    axes[1].set_ylabel("Known-Location SNR", fontsize=11)
    axes[1].set_title("Detection Reference Test\n"
                       "(Higher SNR = better detection)", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 4.  SINGLE-IMAGE CLINICAL REPORT
#     For when a doctor selects one image from the spectrum
# ---------------------------------------------------------------------------

def plot_clinical_report(image, score, ref_metric, rank, total,
                          x_true=None, save_path=None):
    """
    Single-page clinical report for one selected image from the Pareto front.
    Shows the image, its position on the front, and all metrics.
    """
    fig = plt.figure(figsize=(10, 7))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                             width_ratios=[2, 2, 1.5],
                             hspace=0.4, wspace=0.35)

    # Main image
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(image, cmap='hot')
    ax_main.set_title(f"Selected Image  #{rank}/{total}", fontsize=12)
    ax_main.axis('off')

    # Ground truth side by side
    if x_true is not None:
        ax_gt = fig.add_subplot(gs[0, 1])
        ax_gt.imshow(x_true, cmap='hot')
        ax_gt.set_title("Ground Truth", fontsize=11)
        ax_gt.axis('off')

    # Difference map
    if x_true is not None:
        ax_diff = fig.add_subplot(gs[1, 1])
        diff = image - x_true
        vabs = max(abs(diff.min()), abs(diff.max()))
        ax_diff.imshow(diff, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
        ax_diff.set_title("Difference (Recon − Truth)", fontsize=10)
        ax_diff.axis('off')

    # Metrics table
    ax_metrics = fig.add_subplot(gs[:, 2])
    ax_metrics.axis('off')

    detect_pct = rank / total * 100
    quant_pct  = (total - rank + 1) / total * 100

    rows = [
        ["Metric", "Value"],
        ["─" * 12, "─" * 10],
        ["Rank (detect)", f"#{rank} / {total}"],
        ["Φ_detect", f"{score[1]:.2f}"],
        ["Φ_quant", f"{score[0]:.0f}"],
        ["─" * 12, "─" * 10],
        ["Mean Error", f"{ref_metric['mean_error_pct']:+.1f}%"],
        ["Known SNR", f"{ref_metric['snr_known_loc']:.2f}"],
        ["Tumor mean", f"{ref_metric['tumor_mean']:.2f}"],
        ["BG mean", f"{ref_metric['bg_mean']:.2f}"],
        ["─" * 12, "─" * 10],
        ["Detect score", f"{detect_pct:.0f}th pctile"],
        ["Quant score", f"{quant_pct:.0f}th pctile"],
    ]

    for i, (col1, col2) in enumerate(rows):
        weight = 'bold' if i <= 1 else 'normal'
        color  = '#333333'
        if col1.startswith("─"):
            color = '#aaaaaa'
        ax_metrics.text(0.05, 1 - i * 0.075, col1,
                        transform=ax_metrics.transAxes,
                        fontsize=9, color=color, fontweight=weight)
        ax_metrics.text(0.55, 1 - i * 0.075, col2,
                        transform=ax_metrics.transAxes,
                        fontsize=9, color=color, fontweight=weight)

    fig.suptitle(f"Clinical Image Report  —  MOEAP Reconstruction\n"
                 f"Position on front: #{rank} of {total}  "
                 f"({'Max Detection End' if rank == 1 else 'Max Quantification End' if rank == total else 'Compromise'})",
                 fontsize=12, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 5.  Updated save_full_dashboard with all new plots
# ---------------------------------------------------------------------------

def save_full_dashboard_v2(pareto_images, pareto_scores, history,
                            x_true, tumor_mask, roi_mask,
                            em_results=None,
                            output_dir="results"):
    """
    Full dashboard including the new doctor spectrum and improved plots.
    """
    import os
    from pet_objectives import compute_reference_metrics

    os.makedirs(output_dir, exist_ok=True)

    # Compute reference metrics for all Pareto images
    print("Computing reference metrics for all Pareto images...")
    ref_metrics = [
        compute_reference_metrics(img, x_true, tumor_mask)
        for img in pareto_images
    ]

    # 1. Standard Pareto comparison
    from moeap_visualize import plot_pareto_comparison, plot_population_evolution, \
                                 plot_image_grid, plot_convergence

    plot_pareto_comparison(
        pareto_scores, em_results=em_results,
        save_path=f"{output_dir}/pareto_front.png"
    )

    # 2. Improved Pareto comparison
    plot_pareto_comparison_improved(
        pareto_scores, em_results=em_results,
        save_path=f"{output_dir}/pareto_front_improved.png"
    )

    # 3. Population evolution
    plot_population_evolution(history,
        save_path=f"{output_dir}/population_evolution.png")

    # 4. Key images (paper style)
    n   = len(pareto_images)
    mid = n // 2
    key_images = [pareto_images[0], pareto_images[mid], pareto_images[-1], x_true]
    key_titles = ["A: Max Detect", "B: Compromise", "C: Max Quant", "Ground Truth"]
    if em_results:
        key_images.append(em_results[len(em_results)//2][0])
        key_titles.append("D: EM+Smooth")
    plot_image_grid(key_images, key_titles,
                    suptitle="Key Images (Paper Figure 6 Style)",
                    save_path=f"{output_dir}/image_grid.png")

    # 5. Doctor spectrum — the full front
    print("Generating clinical spectrum (full Pareto front)...")
    plot_doctor_spectrum(
        pareto_images, pareto_scores,
        x_true=x_true,
        ref_metrics=ref_metrics,
        save_path=f"{output_dir}/doctor_spectrum.png",
        max_cols=10
    )

    # 6. Annotated reference correlations
    plot_reference_correlations_annotated(
        pareto_scores, ref_metrics,
        save_path=f"{output_dir}/reference_correlations_annotated.png"
    )

    # 7. Convergence
    plot_convergence(history, save_path=f"{output_dir}/convergence.png")

    # 8. Clinical reports for 3 key images
    reports_dir = f"{output_dir}/clinical_reports"
    os.makedirs(reports_dir, exist_ok=True)
    for label, idx in [("max_detect", 0),
                        ("compromise", n // 2),
                        ("max_quant",  n - 1)]:
        plot_clinical_report(
            pareto_images[idx], pareto_scores[idx],
            ref_metrics[idx],
            rank=n - idx,    # detect-descending rank
            total=n,
            x_true=x_true,
            save_path=f"{reports_dir}/report_{label}.png"
        )

    print(f"\nAll outputs saved to: {output_dir}/")
    print(f"  doctor_spectrum.png         -> full clinical spectrum")
    print(f"  pareto_front_improved.png   -> improved Pareto comparison")
    print(f"  reference_correlations_annotated.png -> with trend analysis")
    print(f"  clinical_reports/           -> per-image clinical reports")

    return ref_metrics
