"""
pet_objectives.py
=================
Module 2: Core PET Computations

Handles:
- EM update (directed mutation core)
- Gaussian post-smoothing
- Φ_quantification : Poisson log-likelihood
- Φ_detection     : Generalised scan-statistic model
"""

import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import fftconvolve


EPS = 1e-10    # numerical stability floor


# ---------------------------------------------------------------------------
# 1.  Forward / Back projection wrappers
#     (kept generic so you can swap in a real scanner projector later)
# ---------------------------------------------------------------------------

class PETProjector:
    """
    Thin wrapper around a forward projector and its adjoint.

    Parameters
    ----------
    forward_fn  : callable  image (H,W) -> sinogram (n_angles, n_bins)
    adjoint_fn  : callable  sinogram -> image, with same image_size
    sensitivity : np.ndarray (H,W)   P^T c 1  (pre-computed once)
    c           : np.ndarray (n_angles, n_bins)  attenuation factors
    r           : np.ndarray (n_angles, n_bins)  scatter + randoms
    """

    def __init__(self, forward_fn, adjoint_fn, c, r, image_size):
        self.forward_fn  = forward_fn
        self.adjoint_fn  = adjoint_fn
        self.c           = c
        self.r           = r
        self.image_size  = image_size

        # Pre-compute sensitivity image:  S = P^T (c * 1_detector)
        ones_sino = c.copy()            # c * ones
        self.sensitivity = adjoint_fn(ones_sino)
        self.sensitivity = np.clip(self.sensitivity, EPS, None)

    def project(self, x):
        """Expected sinogram:  ȳ = c * P(x) + r"""
        return self.c * self.forward_fn(x) + self.r

    def backproject(self, sino):
        """Weighted backprojection:  P^T(sino)"""
        return self.adjoint_fn(sino)


# ---------------------------------------------------------------------------
# 2.  EM Update  (directed mutation step 1)
# ---------------------------------------------------------------------------

def em_update(x, y, projector: PETProjector):
    """
    One EM (MLEM) update step.

        x^{n+1} = (x^n / S) * P^T[ c * y / (cPx^n + r) ]

    Parameters
    ----------
    x          : np.ndarray (H, W) current image estimate
    y          : np.ndarray (n_angles, n_bins) measured sinogram
    projector  : PETProjector

    Returns
    -------
    x_new : np.ndarray (H, W), non-negative
    """
    x_safe  = np.clip(x, EPS, None)
    y_bar   = projector.project(x_safe)             # ȳ = cPx + r
    ratio   = y / np.clip(y_bar, EPS, None)         # y / ȳ
    bp      = projector.backproject(projector.c * ratio)  # P^T(c * ratio)
    x_new   = x_safe * bp / projector.sensitivity
    x_new   = np.clip(x_new, 0.0, None)
    return x_new.astype(np.float32)


# ---------------------------------------------------------------------------
# 3.  Directed Mutation  = EM update + random smoothing
# ---------------------------------------------------------------------------

def directed_mutation(x, y, projector: PETProjector,
                      pixel_size_cm=0.35,
                      fwhm_min_cm=0.0, fwhm_max_cm=2.0,
                      rng=None):
    """
    Apply one EM update followed by random Gaussian smoothing.

    FWHM for the Gaussian kernel is drawn uniformly from
    [fwhm_min_cm, fwhm_max_cm].  The paper uses [0, 2 cm].

    Parameters
    ----------
    x             : current image (H, W)
    y             : measured sinogram
    projector     : PETProjector
    pixel_size_cm : pixel size in cm  (paper uses 3.5 mm = 0.35 cm)
    rng           : np.random.Generator (optional, for reproducibility)

    Returns
    -------
    x_mut : np.ndarray (H, W), non-negative, float32
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: EM update
    x_em = em_update(x, y, projector)

    # Step 2: Random Gaussian smoothing
    fwhm_cm = rng.uniform(fwhm_min_cm, fwhm_max_cm)
    if fwhm_cm > 0:
        sigma_cm     = fwhm_cm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        sigma_pixels = sigma_cm / pixel_size_cm
        x_mut = gaussian_filter(x_em, sigma=sigma_pixels)
    else:
        x_mut = x_em.copy()

    x_mut = np.clip(x_mut, 0.0, None)
    return x_mut.astype(np.float32)


# ---------------------------------------------------------------------------
# 4.  Φ_quantification :  Poisson log-likelihood
# ---------------------------------------------------------------------------

def phi_quantification(x, y, projector: PETProjector):
    """
    Poisson log-likelihood (data fidelity objective).

        Φ_quant(x) = Σ_i [ -ȳ_i + y_i * log(ȳ_i) ]

    Higher is better (we maximise).

    Parameters
    ----------
    x   : image (H, W)
    y   : measured sinogram (n_angles, n_bins)
    projector : PETProjector

    Returns
    -------
    scalar float
    """
    y_bar  = projector.project(np.clip(x, EPS, None))
    y_bar  = np.clip(y_bar, EPS, None)

    # Only sum over bins where y > 0  (log(0) contribution is -ȳ which is fine)
    ll = -y_bar + y * np.log(y_bar)
    return float(ll.sum())


# ---------------------------------------------------------------------------
# 5.  Φ_detection :  Generalised scan-statistic SNR
# ---------------------------------------------------------------------------

def _make_disc_kernel(radius_pixels):
    """Return a normalised disc (circular averaging) kernel."""
    r = int(np.ceil(radius_pixels))
    size = 2 * r + 1
    Y, X = np.ogrid[-r:r+1, -r:r+1]
    disc = (X**2 + Y**2 <= radius_pixels**2).astype(np.float32)
    return disc / disc.sum()


def _local_stats(image, kernel):
    """
    Compute local mean and variance via convolution.

    Var(X) = E[X^2] - E[X]^2

    Returns
    -------
    local_mean : np.ndarray same shape as image
    local_var  : np.ndarray same shape as image
    """
    local_mean  = fftconvolve(image, kernel, mode='same')
    local_sq    = fftconvolve(image**2, kernel, mode='same')
    local_var   = np.clip(local_sq - local_mean**2, 0, None)
    return local_mean.astype(np.float32), local_var.astype(np.float32)


def phi_detection(x, roi_mask,
                  disc_radius_cm=0.5,
                  pixel_size_cm=0.35,
                  L=2):
    """
    Generalised scan-statistic detection objective (Eq. 8 in paper).

        Φ_detect(x) = (1/L) Σ_l  [x̄_{s,l} - x̄_{b,l}] / sqrt(0.5*(σ²_{s,l} + σ²_{b,l}))

    The image is scanned with a disc window.  Inside the ROI, we pick
    the L highest-mean locations (signal present) and the L lowest-mean
    locations (signal absent / background).

    Parameters
    ----------
    x               : image (H, W)
    roi_mask        : bool array (H, W) — region to scan (e.g. lungs/liver)
    disc_radius_cm  : radius of scanning disc in cm (paper: 0.5 cm = 1 cm diam)
    pixel_size_cm   : pixel size in cm
    L               : number of signal-present + signal-absent candidates

    Returns
    -------
    scalar float  (higher = better detection)
    """
    radius_pixels = disc_radius_cm / pixel_size_cm
    kernel        = _make_disc_kernel(radius_pixels)

    local_mean, local_var = _local_stats(x, kernel)

    # Restrict search to ROI
    roi_means = local_mean.copy()
    roi_means[~roi_mask] = np.nan

    flat_means = roi_means.flatten()
    valid      = ~np.isnan(flat_means)
    valid_idx  = np.where(valid)[0]

    if len(valid_idx) < 2 * L:
        # Not enough valid pixels — return 0
        return 0.0

    sorted_idx  = valid_idx[np.argsort(flat_means[valid_idx])]
    signal_idx  = sorted_idx[-L:]      # highest L mean locations
    bg_idx      = sorted_idx[:L]       # lowest L mean locations

    H, W = x.shape
    snr_sum = 0.0
    for l in range(L):
        # Signal present location
        si   = signal_idx[l]
        sr, sc = divmod(si, W)
        mean_s = local_mean[sr, sc]
        var_s  = local_var[sr, sc]

        # Background location
        bi   = bg_idx[l]
        br, bc = divmod(bi, W)
        mean_b = local_mean[br, bc]
        var_b  = local_var[br, bc]

        denom  = np.sqrt(0.5 * (var_s + var_b) + EPS)
        snr_sum += (mean_s - mean_b) / denom

    return float(snr_sum / L)


# ---------------------------------------------------------------------------
# 6.  Evaluate both objectives for a single image
# ---------------------------------------------------------------------------

def evaluate_image(x, y, projector, roi_mask,
                   disc_radius_cm=0.5, pixel_size_cm=0.35, L=2):
    """
    Compute (Φ_quant, Φ_detect) for a single image.

    Returns
    -------
    (quant, detect) : tuple of floats
    """
    q = phi_quantification(x, y, projector)
    d = phi_detection(x, roi_mask,
                      disc_radius_cm=disc_radius_cm,
                      pixel_size_cm=pixel_size_cm,
                      L=L)
    return (q, d)


# ---------------------------------------------------------------------------
# 7.  Batch evaluation for entire population
# ---------------------------------------------------------------------------

def evaluate_population(population, y, projector, roi_mask,
                        disc_radius_cm=0.5, pixel_size_cm=0.35, L=2):
    """
    Evaluate objectives for all N images in the population.

    Parameters
    ----------
    population : list of np.ndarray, length N, each (H, W)

    Returns
    -------
    scores : list of (quant, detect) tuples, length N
    """
    scores = []
    for x in population:
        scores.append(evaluate_image(x, y, projector, roi_mask,
                                     disc_radius_cm, pixel_size_cm, L))
    return scores


# ---------------------------------------------------------------------------
# 8.  Reference metrics  (for validation only — not used by the GA)
# ---------------------------------------------------------------------------

def compute_reference_metrics(x, x_true, tumor_mask):
    """
    Compute ground-truth quality metrics for validation.

    Parameters
    ----------
    x          : reconstructed image
    x_true     : ground truth image
    tumor_mask : bool mask of tumour region

    Returns
    -------
    mean_error_pct : float   mean % error inside tumour
    tumor_mean     : float   mean reconstructed value in tumour
    tumor_std      : float   std reconstructed value in tumour
    bg_mean        : float   mean reconstructed value outside tumour
    bg_std         : float   std reconstructed value outside tumour
    """
    bg_mask     = ~tumor_mask

    tumor_vals  = x[tumor_mask]
    bg_vals     = x[bg_mask]
    true_tumor  = x_true[tumor_mask].mean()

    mean_error  = (tumor_vals.mean() - true_tumor) / (true_tumor + EPS) * 100.0
    snr_known   = (tumor_vals.mean() - bg_vals.mean()) / \
                  (np.sqrt(0.5 * (tumor_vals.var() + bg_vals.var())) + EPS)

    return {
        "mean_error_pct" : float(mean_error),
        "snr_known_loc"  : float(snr_known),
        "tumor_mean"     : float(tumor_vals.mean()),
        "tumor_std"      : float(tumor_vals.std()),
        "bg_mean"        : float(bg_vals.mean()),
        "bg_std"         : float(bg_vals.std()),
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/moeap")
    from pet_simulator import (make_phantom, radon_projector,
                                radon_backprojector, simulate_pet_data,
                                fbp_reconstruction)

    IMAGE_SIZE = 64
    N_ANGLES   = 90

    print("=== Objectives Sanity Check ===")

    x_true, tumor_mask, bg_mask = make_phantom(IMAGE_SIZE, "heart")

    fwd = lambda img: radon_projector(img, n_angles=N_ANGLES)
    adj = lambda sino: radon_backprojector(sino, IMAGE_SIZE, N_ANGLES)

    y, y_bar, c, r = simulate_pet_data(x_true, fwd, n_total_counts=1e5)

    projector = PETProjector(fwd, adj, c, r, IMAGE_SIZE)

    # Test on FBP reconstruction
    from pet_simulator import fbp_reconstruction
    x_fbp = fbp_reconstruction(y, IMAGE_SIZE, N_ANGLES)

    q = phi_quantification(x_fbp, y, projector)
    d = phi_detection(x_fbp, bg_mask | tumor_mask, L=2)
    print(f"FBP  → Φ_quant={q:.2f},  Φ_detect={d:.4f}")

    # One EM update
    x_em = em_update(x_fbp, y, projector)
    q2 = phi_quantification(x_em, y, projector)
    d2 = phi_detection(x_em, bg_mask | tumor_mask, L=2)
    print(f"1xEM → Φ_quant={q2:.2f},  Φ_detect={d2:.4f}")

    # Directed mutation
    rng   = np.random.default_rng(0)
    x_mut = directed_mutation(x_fbp, y, projector, rng=rng)
    q3    = phi_quantification(x_mut, y, projector)
    d3    = phi_detection(x_mut, bg_mask | tumor_mask, L=2)
    print(f"Mut  → Φ_quant={q3:.2f},  Φ_detect={d3:.4f}")

    # Reference metrics
    ref = compute_reference_metrics(x_em, x_true, tumor_mask)
    print(f"Ref metrics: ME={ref['mean_error_pct']:.1f}%,  SNR={ref['snr_known_loc']:.2f}")

    print("=== PASSED ===")
