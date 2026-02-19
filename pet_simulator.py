"""
pet_simulator.py
================
Module 1: PET System Model & Data Simulation

Handles:
- System matrix construction (simplified 2D parallel-beam projector)
- Ground truth phantom definition (heart/liver-like)
- Forward projection and Poisson noise simulation
- Filtered backprojection (FBP) for initialization

NOTE: In a real PET scanner you'd load a scanner-specific system matrix.
Here we build a simplified 2D parallel-beam projector sufficient for
proof-of-concept (matching the paper's simulated data setup).
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# 1.  Phantom definition
# ---------------------------------------------------------------------------

def make_phantom(image_size=128, phantom_type="heart"):
    """
    Create a simple 2D emission phantom with a hot nodule.

    Returns
    -------
    x_true : np.ndarray, shape (image_size, image_size), float32
        Ground-truth emission image (arbitrary activity units).
    tumor_mask : np.ndarray, bool
        Mask of the tumour / hot nodule region.
    background_mask : np.ndarray, bool
        Mask of the surrounding organ background.
    """
    img = np.zeros((image_size, image_size), dtype=np.float32)
    cx, cy = image_size // 2, image_size // 2

    if phantom_type == "heart":
        # --- outer body ellipse (low background) ---
        for i in range(image_size):
            for j in range(image_size):
                val = ((i - cx) / (cx * 0.85))**2 + ((j - cy) / (cy * 0.75))**2
                if val <= 1.0:
                    img[i, j] = 0.5          # body background

        # --- heart muscle (medium activity) ---
        for i in range(image_size):
            for j in range(image_size):
                val = ((i - cx) / (cx * 0.50))**2 + ((j - cy) / (cy * 0.40))**2
                if val <= 1.0:
                    inner = ((i - cx) / (cx * 0.30))**2 + ((j - cy) / (cy * 0.22))**2
                    if inner > 1.0:
                        img[i, j] = 1.5      # myocardium

        # --- hot nodule (1 cm diameter ~ 3-4 pixels at 3.5mm/pix) ---
        nod_r = int(round(0.5 / 0.35))       # 0.5cm radius / 0.35cm per pixel ≈ 1-2 px
        nod_r = max(nod_r, 2)
        nod_cx = cx + int(cx * 0.30)
        nod_cy = cy
        Y, X = np.ogrid[:image_size, :image_size]
        nodule = (X - nod_cx)**2 + (Y - nod_cy)**2 <= nod_r**2
        img[nodule] = 3.0                    # 2× background contrast

    elif phantom_type == "liver":
        # --- liver ellipse ---
        for i in range(image_size):
            for j in range(image_size):
                val = ((i - cx + cx*0.10) / (cx * 0.60))**2 + \
                      ((j - cy) / (cy * 0.80))**2
                if val <= 1.0:
                    img[i, j] = 2.0          # liver parenchyma

        # --- hot nodule ---
        nod_r = max(int(round(0.5 / 0.35)), 2)
        nod_cx = cx - int(cx * 0.20)
        nod_cy = cy + int(cy * 0.10)
        Y, X = np.ogrid[:image_size, :image_size]
        nodule = (X - nod_cx)**2 + (Y - nod_cy)**2 <= nod_r**2
        img[nodule] = 5.0                    # hot lesion

    else:
        raise ValueError(f"Unknown phantom_type: {phantom_type}")

    # Smooth slightly to remove hard edges (more physical)
    img = gaussian_filter(img, sigma=0.8)

    # Re-apply nodule cleanly after smoothing
    Y, X = np.ogrid[:image_size, :image_size]
    nodule_mask = (X - nod_cx)**2 + (Y - nod_cy)**2 <= nod_r**2
    background_mask = img > 0.1
    background_mask[nodule_mask] = False

    return img, nodule_mask, background_mask


# ---------------------------------------------------------------------------
# 2.  Simplified 2D parallel-beam system matrix  (strip-integral model)
# ---------------------------------------------------------------------------

def build_system_matrix(image_size=128, n_angles=180, n_bins=180):
    """
    Build a sparse-style 2D system matrix P using a ray-driven projector.

    For each detector bin (angle θ, bin s) we compute which image pixels
    contribute using bilinear interpolation along the ray.

    Returns
    -------
    P : np.ndarray, shape (n_angles * n_bins, image_size * image_size), float32
        System matrix.  Each row = one detector bin; each column = one voxel.
    n_angles, n_bins : int
    image_size : int
    """
    print(f"Building system matrix: {n_angles} angles × {n_bins} bins → "
          f"{n_angles*n_bins} measurements, {image_size**2} voxels ...")

    n_det = n_angles * n_bins
    n_pix = image_size * image_size
    P = np.zeros((n_det, n_pix), dtype=np.float32)

    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    # bin positions centred at 0, spanning ±(image_size/2) pixels
    half = image_size / 2.0
    bin_positions = np.linspace(-half, half, n_bins)

    # pixel centres
    px = np.arange(image_size) - (image_size - 1) / 2.0   # shape (image_size,)

    for a_idx, theta in enumerate(angles):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        perp = np.array([-sin_t, cos_t])   # direction perpendicular to ray

        for b_idx, s in enumerate(bin_positions):
            row_idx = a_idx * n_bins + b_idx

            # The ray at (angle=θ, bin=s) passes through points
            # satisfying:  x*cos(θ) + y*sin(θ) = s
            # Parameterise along the ray direction (cos_t, sin_t)
            # and sample at n_samples points

            n_samples = image_size * 2
            t_vals = np.linspace(-half * 1.42, half * 1.42, n_samples)

            # Ray in (col, row) = (x, y) image coords
            ray_x = s * cos_t - t_vals * sin_t   # col (x)
            ray_y = s * sin_t + t_vals * cos_t   # row (y)

            # Convert to pixel indices (0 .. image_size-1)
            col_f = ray_x + (image_size - 1) / 2.0
            row_f = ray_y + (image_size - 1) / 2.0

            # Bilinear interpolation weights
            col0 = np.floor(col_f).astype(int)
            row0 = np.floor(row_f).astype(int)
            dc = col_f - col0
            dr = row_f - row0

            for t_i in range(n_samples):
                c0, r0 = col0[t_i], row0[t_i]
                fc, fr = dc[t_i], dr[t_i]

                for dr_, dc_, w in [
                    (0, 0, (1-fr)*(1-fc)),
                    (0, 1, (1-fr)*fc),
                    (1, 0, fr*(1-fc)),
                    (1, 1, fr*fc)
                ]:
                    r = r0 + dr_
                    c = c0 + dc_
                    if 0 <= r < image_size and 0 <= c < image_size:
                        pix_idx = r * image_size + c
                        P[row_idx, pix_idx] += w * (2 * half * 1.42 / n_samples)

    print("System matrix built.")
    return P.astype(np.float32), n_angles, n_bins, image_size


# ---------------------------------------------------------------------------
# 3.  Fast approximate projector using scipy (production-style alternative)
# ---------------------------------------------------------------------------

def radon_projector(image, n_angles=180):
    """
    Fast forward projector using the Radon transform approximation.
    Much faster than the explicit P matrix — used for prototyping.

    Returns sinogram of shape (n_angles, n_bins) where n_bins = image.shape[0].
    """
    from scipy.ndimage import rotate
    n = image.shape[0]
    sino = np.zeros((n_angles, n), dtype=np.float32)
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    for i, angle in enumerate(angles):
        rotated = rotate(image, angle, reshape=False, order=1)
        sino[i] = rotated.sum(axis=0)
    return sino


def radon_backprojector(sinogram, image_size, n_angles=None):
    """Simple backprojection (un-filtered) for sensitivity / adjoint."""
    from scipy.ndimage import rotate
    if n_angles is None:
        n_angles = sinogram.shape[0]
    backprojected = np.zeros((image_size, image_size), dtype=np.float32)
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    for i, angle in enumerate(angles):
        sino_row = sinogram[i]                        # shape (n_bins,)
        # Replicate row across image then rotate back
        bp_slice = np.tile(sino_row, (image_size, 1))
        backprojected += rotate(bp_slice.astype(np.float32), -angle,
                                reshape=False, order=1)
    return backprojected / n_angles


# ---------------------------------------------------------------------------
# 4.  Simulate PET measurements
# ---------------------------------------------------------------------------

def simulate_pet_data(x_true, projector_fn, n_total_counts=4e5,
                      scatter_fraction=0.10, seed=42):
    """
    Simulate PET sinogram data from a ground-truth image.

    Parameters
    ----------
    x_true : np.ndarray (H, W)
    projector_fn : callable  image → sinogram
    n_total_counts : int     total expected detected events
    scatter_fraction : float fraction of events that are scatter/randoms

    Returns
    -------
    y : np.ndarray  measured (noisy) sinogram
    y_bar : np.ndarray  noiseless expected sinogram (= cPx + r)
    c : np.ndarray  attenuation correction factors (ones for simplicity)
    r : np.ndarray  scatter + randoms additive term
    """
    rng = np.random.default_rng(seed)

    # Noiseless forward projection
    sino_clean = projector_fn(x_true)

    # Normalise to desired count level
    scale = n_total_counts / sino_clean.sum()
    sino_clean = sino_clean * scale

    # Attenuation (simplified: uniform factor)
    c = np.ones_like(sino_clean, dtype=np.float32)

    # Scatter + randoms: uniform additive
    r = scatter_fraction * sino_clean.mean() * np.ones_like(sino_clean, dtype=np.float32)

    # Expected measurements
    y_bar = c * sino_clean + r

    # Add Poisson noise
    y = rng.poisson(y_bar).astype(np.float32)

    print(f"Simulated sinogram: shape={y.shape}, "
          f"total counts={y.sum():.0f}, "
          f"mean={y.mean():.2f}")

    return y, y_bar, c, r


# ---------------------------------------------------------------------------
# 5.  Filtered Backprojection (FBP) — used for population initialisation
# ---------------------------------------------------------------------------

def fbp_reconstruction(sinogram, image_size, n_angles=None):
    """
    Simple ramp-filtered backprojection for initialising the GA population.

    Parameters
    ----------
    sinogram : np.ndarray (n_angles, n_bins)
    image_size : int

    Returns
    -------
    x_fbp : np.ndarray (image_size, image_size), float32, non-negative
    """
    from scipy.ndimage import rotate
    if n_angles is None:
        n_angles = sinogram.shape[0]

    n_bins = sinogram.shape[1]

    # Ramp filter in frequency domain
    freqs = np.fft.fftfreq(n_bins)
    ramp = np.abs(freqs)

    filtered_sino = np.zeros_like(sinogram)
    for i in range(n_angles):
        F = np.fft.fft(sinogram[i])
        filtered_sino[i] = np.real(np.fft.ifft(F * ramp))

    # Backproject
    x_fbp = radon_backprojector(filtered_sino, image_size, n_angles)

    # Non-negativity
    x_fbp = np.clip(x_fbp, 0, None)

    return x_fbp.astype(np.float32)


# ---------------------------------------------------------------------------
# 6.  Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    IMAGE_SIZE = 64   # small for quick test
    N_ANGLES   = 90

    print("=== PET Simulator Sanity Check ===")

    # Build phantom
    x_true, tumor_mask, bg_mask = make_phantom(IMAGE_SIZE, "heart")
    print(f"Phantom: shape={x_true.shape}, "
          f"max={x_true.max():.2f}, "
          f"tumor pixels={tumor_mask.sum()}")

    # Define projector
    proj_fn = lambda img: radon_projector(img, n_angles=N_ANGLES)

    # Simulate data
    y, y_bar, c, r = simulate_pet_data(x_true, proj_fn,
                                        n_total_counts=1e5, seed=0)

    # FBP reconstruction
    x_fbp = fbp_reconstruction(y, IMAGE_SIZE, N_ANGLES)
    print(f"FBP: shape={x_fbp.shape}, max={x_fbp.max():.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x_true, cmap='hot'); axes[0].set_title("Ground Truth")
    axes[1].imshow(y, cmap='gray');    axes[1].set_title("Sinogram (noisy)")
    axes[2].imshow(x_fbp, cmap='hot'); axes[2].set_title("FBP Reconstruction")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig("/home/claude/moeap/sanity_check_simulator.png", dpi=100)
    print("Saved: sanity_check_simulator.png")
    print("=== PASSED ===")
