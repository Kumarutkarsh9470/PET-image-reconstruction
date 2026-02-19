"""
run_moeap_torch.py
==================
Main entry point using the PyTorch-accelerated backend.

Usage:
    python run_moeap_torch.py                            # defaults
    python run_moeap_torch.py --phantom liver --N 100 --gens 200
    python run_moeap_torch.py --device cpu               # force CPU
"""

import argparse
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from pet_simulator import (make_phantom, radon_projector,
                            radon_backprojector, simulate_pet_data,
                            fbp_reconstruction)
from pet_objectives_torch import TorchPETProjector, DEVICE
from moeap_torch import MOEAPConfig, run_moeap_torch, run_em_baseline_torch
from moeap_doctor_view import save_full_dashboard_v2 as save_full_dashboard
from pet_objectives import compute_reference_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phantom",  default="heart", choices=["heart", "liver"])
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--n_angles", type=int, default=90)
    p.add_argument("--counts",   type=float, default=4e5)
    p.add_argument("--N",        type=int,   default=40)
    p.add_argument("--gens",     type=int,   default=50)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--device",   default="auto",
                   help="auto | cuda | mps | cpu")
    p.add_argument("--out",      default="/home/claude/moeap/results_torch")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  MOEAP (PyTorch backend)")
    print(f"  Phantom: {args.phantom}  |  {args.img_size}x{args.img_size}")
    print(f"  N={args.N}  Gens={args.gens}  Device={args.device}")
    print(f"{'='*60}")

    # 1. Build phantom and simulate
    x_true, tumor_mask, bg_mask = make_phantom(args.img_size, args.phantom)
    roi_np = (bg_mask | tumor_mask).astype(bool)

    fwd   = lambda img: radon_projector(img, n_angles=args.n_angles)
    y_np, _, _, _ = simulate_pet_data(x_true, fwd,
                                       n_total_counts=args.counts,
                                       seed=args.seed)
    x_fbp = fbp_reconstruction(y_np, args.img_size, args.n_angles)

    # 2. Build torch projector
    #    c, r: simple uniform corrections
    c_np = np.ones((args.n_angles, args.img_size), dtype=np.float32)
    r_np = (0.10 * y_np.mean()) * np.ones_like(c_np)

    device = DEVICE if args.device == "auto" else torch.device(args.device)
    c = torch.from_numpy(c_np).to(device)
    r = torch.from_numpy(r_np).to(device)
    projector = TorchPETProjector(args.img_size, args.n_angles, args.img_size,
                                   c, r, device)

    # 3. MOEAP
    config = MOEAPConfig(
        N=args.N,
        n_generations=args.gens,
        log_every=max(1, args.gens // 10),
        seed=args.seed,
        device=args.device
    )

    pareto_images, pareto_scores, history = run_moeap_torch(
        y_np, projector, roi_np, x_fbp,
        config=config, x_true=x_true, tumor_mask=tumor_mask
    )

    # 4. EM baseline
    fwhm_range = np.arange(0.35, 2.5, 0.35)
    em_results = run_em_baseline_torch(
        y_np, projector, roi_np, x_fbp, config,
        em_iters=200, fwhm_range_cm=fwhm_range
    )

    # 5. Save plots
    os.makedirs(args.out, exist_ok=True)

    # Need a numpy-based projector dummy for visualize compat
    # (visualize only uses compute_reference_metrics which is pure numpy)
    ref_metrics = save_full_dashboard(
        pareto_images, pareto_scores, history,
        x_true, tumor_mask, roi_np,
        em_results=em_results,
        output_dir=args.out
    )

    # 6. Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Image':<22} {'Φ_quant':>12} {'Φ_detect':>10} {'ME%':>8} {'SNR':>8}")
    print("-"*70)
    n = len(pareto_images)
    for label, idx in [("A: Max Detect", 0), ("B: Compromise", n//2), ("C: Max Quant", n-1)]:
        s = pareto_scores[idx]
        m = ref_metrics[idx]
        print(f"{label:<22} {s[0]:>12.1f} {s[1]:>10.4f} "
              f"{m['mean_error_pct']:>+8.1f} {m['snr_known_loc']:>8.2f}")
    print(f"\nResults saved to: {args.out}/")


if __name__ == "__main__":
    main()
