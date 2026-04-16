"""
polynomial_mutation.py
======================
Polynomial Mutation for PET Image Reconstruction
(coded from scratch, inspired by pymoo / Deb & Goyal 1996).

Provides:
  - polynomial_mutation()       : pixel-wise polynomial mutation
  - polynomial_mutation_torch() : PyTorch-accelerated batch version
  - hybrid_mutation()           : combines EM-directed + polynomial mutation

The polynomial mutation operator creates small perturbations around the
parent solution, controlled by:
  - eta_m : distribution index (higher → smaller perturbations)
  - P_m   : per-pixel mutation probability
  - bounds : [lower, upper] for pixel values

This complements the EM-directed mutation with a general-purpose
exploration operator, preventing premature convergence.
"""

import numpy as np

EPS = 1e-10


# ---------------------------------------------------------------------------
# 1.  Polynomial Mutation (NumPy)
# ---------------------------------------------------------------------------

def polynomial_mutation(x, rng,
                        eta_m=20.0,
                        P_m=None,
                        lower=0.0,
                        upper=None):
    """
    Polynomial mutation (Deb & Goyal, 1996) applied pixel-wise.

    Parameters
    ----------
    x      : np.ndarray (H, W)  parent image
    rng    : np.random.Generator
    eta_m  : float   distribution index (large → small perturbation)
    P_m    : float   mutation probability per pixel (default: 1/n_pixels)
    lower  : float   lower bound for pixel values
    upper  : float   upper bound (default: 2 * max pixel value)

    Returns
    -------
    x_mut : np.ndarray (H, W), float32
    """
    shape = x.shape
    x_flat = x.flatten().astype(np.float64)
    n = len(x_flat)

    if P_m is None:
        P_m = 1.0 / n

    if upper is None:
        upper = max(x_flat.max() * 2.0, 1.0)

    # Mutation mask
    mask = rng.random(n) < P_m

    # Random draw for delta computation
    u = rng.random(n)

    # Compute delta_q (polynomial perturbation)
    delta_1 = (x_flat - lower) / max(upper - lower, EPS)
    delta_2 = (upper - x_flat) / max(upper - lower, EPS)

    # Clip to avoid numerical issues
    delta_1 = np.clip(delta_1, 0, 1)
    delta_2 = np.clip(delta_2, 0, 1)

    mut_pow = 1.0 / (eta_m + 1.0)

    # Compute delta_q for u < 0.5 and u >= 0.5
    xy = 1.0 - delta_1
    val_low = 2.0 * u + (1.0 - 2.0 * u) * np.power(xy, eta_m + 1.0)
    delta_q_low = np.power(val_low, mut_pow) - 1.0

    xy2 = 1.0 - delta_2
    val_high = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * np.power(xy2, eta_m + 1.0)
    delta_q_high = 1.0 - np.power(val_high, mut_pow)

    delta_q = np.where(u < 0.5, delta_q_low, delta_q_high)

    # Apply mutation
    x_mut = x_flat.copy()
    x_mut[mask] = x_flat[mask] + delta_q[mask] * (upper - lower)

    # Clip to bounds
    x_mut = np.clip(x_mut, lower, upper)

    return x_mut.reshape(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# 2.  Polynomial Mutation (PyTorch batch)
# ---------------------------------------------------------------------------

def polynomial_mutation_torch(X, eta_m=20.0, P_m=None,
                               lower=0.0, upper=None):
    """
    Polynomial mutation for a batch of images using PyTorch.

    Parameters
    ----------
    X      : torch.Tensor (N, H, W)
    eta_m  : float
    P_m    : float  (default: 1 / (H * W))
    lower  : float
    upper  : float  (default: 2 * max pixel value)

    Returns
    -------
    X_mut : torch.Tensor (N, H, W)
    """
    import torch

    N, H, W = X.shape
    device = X.device
    n_pixels = H * W

    if P_m is None:
        P_m = 1.0 / n_pixels
    if upper is None:
        upper = max(X.max().item() * 2.0, 1.0)

    rang = upper - lower
    if rang < EPS:
        return X.clone()

    # Mutation mask
    mask = torch.rand(N, H, W, device=device) < P_m

    # Random draw
    u = torch.rand(N, H, W, device=device)

    delta_1 = (X - lower) / rang
    delta_2 = (upper - X) / rang
    delta_1 = torch.clamp(delta_1, 0.0, 1.0)
    delta_2 = torch.clamp(delta_2, 0.0, 1.0)

    mut_pow = 1.0 / (eta_m + 1.0)

    xy = 1.0 - delta_1
    val_low = 2.0 * u + (1.0 - 2.0 * u) * torch.pow(xy + EPS, eta_m + 1.0)
    val_low = torch.clamp(val_low, min=EPS)
    delta_q_low = torch.pow(val_low, mut_pow) - 1.0

    xy2 = 1.0 - delta_2
    val_high = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * torch.pow(xy2 + EPS, eta_m + 1.0)
    val_high = torch.clamp(val_high, min=EPS)
    delta_q_high = 1.0 - torch.pow(val_high, mut_pow)

    delta_q = torch.where(u < 0.5, delta_q_low, delta_q_high)

    X_mut = X.clone()
    X_mut[mask] = X[mask] + delta_q[mask] * rang

    X_mut = torch.clamp(X_mut, lower, upper)
    return X_mut


# ---------------------------------------------------------------------------
# 3.  Hybrid Mutation: EM-directed + polynomial
# ---------------------------------------------------------------------------

def hybrid_mutation(x, y, projector, rng,
                    p_em=0.7,
                    eta_m=20.0,
                    pixel_size_cm=0.35,
                    fwhm_min_cm=0.0,
                    fwhm_max_cm=2.0):
    """
    Hybrid mutation: combines EM-directed mutation (physics-informed)
    with polynomial mutation (exploration).

    - With probability p_em: apply EM-directed mutation
    - With probability (1 - p_em): apply polynomial mutation

    Parameters
    ----------
    x          : np.ndarray (H, W) current image
    y          : np.ndarray (n_angles, n_bins) sinogram
    projector  : PETProjector
    rng        : np.random.Generator
    p_em       : float  probability of using EM mutation (default: 0.7)
    eta_m      : float  polynomial mutation distribution index
    pixel_size_cm, fwhm_min_cm, fwhm_max_cm : EM mutation params

    Returns
    -------
    x_mut : np.ndarray (H, W), float32
    """
    from pet_objectives import directed_mutation

    if rng.random() < p_em:
        # EM-directed mutation (exploits PET physics)
        x_mut = directed_mutation(x, y, projector,
                                   pixel_size_cm=pixel_size_cm,
                                   fwhm_min_cm=fwhm_min_cm,
                                   fwhm_max_cm=fwhm_max_cm,
                                   rng=rng)
    else:
        # Polynomial mutation (general exploration)
        x_mut = polynomial_mutation(x, rng, eta_m=eta_m)

    return x_mut


def hybrid_mutation_batch_torch(X, y, projector,
                                 p_em=0.7,
                                 eta_m=20.0,
                                 pixel_size_cm=0.35,
                                 fwhm_min_cm=0.0,
                                 fwhm_max_cm=2.0):
    """
    Batch hybrid mutation on GPU.

    For each image, independently chooses EM or polynomial mutation.

    Parameters
    ----------
    X : torch.Tensor (N, H, W)
    y : torch.Tensor (n_angles, n_bins)

    Returns
    -------
    X_mut : torch.Tensor (N, H, W)
    """
    import torch
    from pet_objectives_torch import directed_mutation_batch

    N = X.shape[0]
    device = X.device

    # Step 1: Apply EM mutation to ALL images (batched, efficient)
    X_em = directed_mutation_batch(X, y, projector,
                                    pixel_size_cm=pixel_size_cm,
                                    fwhm_min_cm=fwhm_min_cm,
                                    fwhm_max_cm=fwhm_max_cm)

    # Step 2: Apply polynomial mutation to ALL images
    X_poly = polynomial_mutation_torch(X, eta_m=eta_m)

    # Step 3: For each image, choose EM or polynomial based on p_em
    em_mask = torch.rand(N, device=device) < p_em
    em_mask = em_mask.view(N, 1, 1)  # broadcast over (H, W)

    X_mut = torch.where(em_mask, X_em, X_poly)
    return X_mut


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Polynomial Mutation Sanity Check ===")

    rng = np.random.default_rng(42)

    # --- Test 1: Shape and bounds ---
    x = rng.uniform(0, 3, (64, 64)).astype(np.float32)
    x_mut = polynomial_mutation(x, rng, eta_m=20.0, lower=0.0, upper=5.0)

    assert x_mut.shape == x.shape, f"Shape mismatch: {x_mut.shape}"
    assert x_mut.min() >= 0.0, f"Below lower bound: {x_mut.min()}"
    assert x_mut.max() <= 5.0, f"Above upper bound: {x_mut.max()}"
    print(f"Shape: OK  Bounds: [{x_mut.min():.4f}, {x_mut.max():.4f}]")

    # --- Test 2: Mutation does change some pixels ---
    n_changed = np.sum(x_mut != x)
    n_total = x.size
    print(f"Changed pixels: {n_changed}/{n_total} "
          f"({100*n_changed/n_total:.2f}%)")
    assert n_changed > 0, "Mutation should change at least some pixels"

    # --- Test 3: Higher P_m → more changes ---
    x_low = polynomial_mutation(x, np.random.default_rng(0),
                                 P_m=0.001, eta_m=20.0)
    x_high = polynomial_mutation(x, np.random.default_rng(0),
                                  P_m=0.1, eta_m=20.0)
    changes_low = np.sum(x_low != x)
    changes_high = np.sum(x_high != x)
    print(f"P_m=0.001: {changes_low} changes,  P_m=0.1: {changes_high} changes")
    assert changes_high > changes_low, "Higher P_m should cause more changes"

    # --- Test 4: Large eta_m → smaller perturbations ---
    diffs_small_eta = np.abs(polynomial_mutation(
        x, np.random.default_rng(1), eta_m=5.0, P_m=0.5) - x)
    diffs_large_eta = np.abs(polynomial_mutation(
        x, np.random.default_rng(1), eta_m=100.0, P_m=0.5) - x)

    print(f"eta_m=5:   mean |delta| = {diffs_small_eta.mean():.4f}")
    print(f"eta_m=100: mean |delta| = {diffs_large_eta.mean():.4f}")
    assert diffs_large_eta.mean() < diffs_small_eta.mean(), \
        "Larger eta_m should produce smaller perturbations"

    # --- Test 5: PyTorch version ---
    try:
        import torch
        X_t = torch.from_numpy(x[None].repeat(8, axis=0))
        X_mut_t = polynomial_mutation_torch(X_t, eta_m=20.0)
        assert X_mut_t.shape == X_t.shape
        assert X_mut_t.min() >= 0
        print(f"PyTorch batch: shape {X_mut_t.shape}, "
              f"bounds [{X_mut_t.min():.3f}, {X_mut_t.max():.3f}]")
        print("polynomial_mutation_torch(): OK")
    except ImportError:
        print("PyTorch not available — skipping torch tests")

    print("=== ALL PASSED ===")
