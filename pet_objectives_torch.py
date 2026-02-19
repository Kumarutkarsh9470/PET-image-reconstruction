"""
pet_objectives_torch.py
========================
PyTorch-accelerated versions of:
  - Forward / back projection  (batched over population)
  - EM update                  (batched)
  - Directed mutation          (batched)
  - Φ_quantification           (batched)
  - Φ_detection                (batched, via 2D conv)

The key speedup: instead of looping over N=100 images one at a time,
we treat the population as a batch tensor of shape (N, H, W) and run
everything in parallel on GPU.

Requirements:  pip install torch torchvision
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

EPS = 1e-10


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[torch] Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[torch] Using Apple MPS")
    else:
        dev = torch.device("cpu")
        print("[torch] Using CPU  (no GPU found — still faster than pure NumPy)")
    return dev


DEVICE = get_device()


# ---------------------------------------------------------------------------
# 1.  Batched Radon Transform  (forward projector)
#
#     Uses PyTorch's grid_sample for differentiable bilinear interpolation.
#     Operates on a batch of images simultaneously.
# ---------------------------------------------------------------------------

def _make_radon_grid(image_size: int, n_angles: int, n_bins: int,
                     device: torch.device) -> torch.Tensor:
    """
    Pre-compute the sampling grid for the Radon transform.

    Returns
    -------
    grid : Tensor (n_angles * n_bins, image_size, 2)
        grid_sample coordinates in [-1, 1] for each ray sample.
    """
    angles     = torch.linspace(0, torch.pi, n_angles,
                                device=device, dtype=torch.float32)
    cos_t      = torch.cos(angles)   # (n_angles,)
    sin_t      = torch.sin(angles)

    # Bin positions in [-1, 1]  (normalised image coordinates)
    bins       = torch.linspace(-1, 1, n_bins, device=device, dtype=torch.float32)

    # Sample t along ray: n_samples points
    n_samples  = image_size
    t_vals     = torch.linspace(-1.42, 1.42, n_samples,
                                device=device, dtype=torch.float32)

    # For each (angle, bin): ray points = s*[cos,sin] + t*[-sin,cos]
    # shapes: angles (A,) bins (B,) t_vals (T,)
    # broadcast → (A, B, T)
    A, B, T    = n_angles, n_bins, n_samples

    cos_t_  = cos_t.view(A, 1, 1).expand(A, B, T)
    sin_t_  = sin_t.view(A, 1, 1).expand(A, B, T)
    bins_   = bins.view(1, B, 1).expand(A, B, T)
    t_      = t_vals.view(1, 1, T).expand(A, B, T)

    ray_x   = bins_ * cos_t_ - t_ * sin_t_   # (A, B, T)  col (x)
    ray_y   = bins_ * sin_t_ + t_ * cos_t_   # (A, B, T)  row (y)

    # grid_sample expects (N, H_out, W_out, 2) with coords in [-1,1]
    # We treat each detector bin as one "row" of the output
    # Flatten (A,B) → A*B, keep T as the width dimension
    grid = torch.stack([ray_x, ray_y], dim=-1)   # (A, B, T, 2)
    grid = grid.view(A * B, T, 1, 2)             # (A*B, T, 1, 2)

    return grid   # we'll store this once and reuse


class TorchPETProjector:
    """
    Batched PET forward/back projector using PyTorch.

    All operations run on `device` and accept/return torch.Tensors.

    Parameters
    ----------
    image_size : int
    n_angles   : int
    n_bins     : int   (sinogram bins per angle)
    c          : Tensor (n_angles, n_bins)  attenuation
    r          : Tensor (n_angles, n_bins)  scatter+randoms
    device     : torch.device
    """

    def __init__(self, image_size: int, n_angles: int, n_bins: int,
                 c: torch.Tensor, r: torch.Tensor,
                 device: torch.device = DEVICE):

        self.image_size = image_size
        self.n_angles   = n_angles
        self.n_bins     = n_bins
        self.device     = device
        self.c          = c.to(device)   # (n_angles, n_bins)
        self.r          = r.to(device)

        # Pre-compute radon sampling grid  (A*B, T, 1, 2)
        self._grid = _make_radon_grid(image_size, n_angles, n_bins, device)

        # Scale factor: arc-length element
        self._scale = 2.0 * 1.42 / image_size

        # Pre-compute sensitivity image  S = P^T(c * 1)
        ones_sino    = self.c.clone()
        self.sensitivity = self._backproject_single(ones_sino)
        self.sensitivity = torch.clamp(self.sensitivity, min=EPS)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward project one image.

        Parameters
        ----------
        x : Tensor (H, W)

        Returns
        -------
        sino : Tensor (n_angles, n_bins)
        """
        # grid_sample expects (N, C, H, W)
        x4d   = x.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

        # Sample image along all rays simultaneously
        # grid: (A*B, T, 1, 2) → treat as batch of A*B "images"
        # Expand x to match batch: (A*B, 1, H, W)
        A_B   = self.n_angles * self.n_bins
        x_exp = x4d.expand(A_B, -1, -1, -1)

        sampled = F.grid_sample(x_exp, self._grid,
                                mode='bilinear',
                                padding_mode='zeros',
                                align_corners=True)
        # sampled: (A*B, 1, T, 1)  → squeeze → (A*B, T)
        sampled = sampled.squeeze(1).squeeze(-1)   # (A*B, T)

        # Integrate along ray (sum over T, multiply by scale)
        sino = sampled.sum(dim=1) * self._scale    # (A*B,)
        sino = sino.view(self.n_angles, self.n_bins)
        return sino

    def _backproject_single(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Backproject one sinogram into image space.

        Parameters
        ----------
        sino : Tensor (n_angles, n_bins)

        Returns
        -------
        bp : Tensor (H, W)
        """
        # Treat each ray as scattering sino value back along its path
        # using the transpose of grid_sample
        A_B   = self.n_angles * self.n_bins
        w     = sino.view(A_B)       # (A*B,)

        # Create weight volume: each ray contributes `w` uniformly along T
        T     = self._grid.shape[1]
        weights = w.view(A_B, 1, 1).expand(A_B, T, 1)   # (A*B, T, 1)

        # Scatter weights back onto image grid using transpose (adjoint) of
        # grid_sample — implemented via explicit scatter add
        H = W = self.image_size
        bp = torch.zeros(1, 1, H, W, device=self.device, dtype=torch.float32)

        # Get grid coordinates  (A*B, T, 1, 2) → (A*B, T, 2)
        grid_pts = self._grid.view(A_B, T, 2)   # x,y in [-1,1]

        # Convert to pixel coords
        px = ((grid_pts[..., 0] + 1) / 2 * (W - 1)).long()  # (A*B, T)
        py = ((grid_pts[..., 1] + 1) / 2 * (H - 1)).long()

        # Mask valid
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)

        px_v = px[valid]
        py_v = py[valid]
        w_v  = weights.squeeze(-1)[valid]   # (n_valid,)

        # Flat indices
        flat_idx = py_v * W + px_v
        bp_flat  = bp.view(-1)
        bp_flat.scatter_add_(0, flat_idx, w_v * self._scale)

        return bp.view(H, W)

    # ------------------------------------------------------------------
    # Batched versions (operate on population tensor)
    # ------------------------------------------------------------------

    def forward_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward project a batch of images.

        Parameters
        ----------
        X : Tensor (N, H, W)

        Returns
        -------
        sinos : Tensor (N, n_angles, n_bins)
        """
        N = X.shape[0]
        sinos = torch.stack([self._forward_single(X[i]) for i in range(N)])
        return sinos

    def project_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Expected sinograms:  ȳ = c * P(X) + r  for batch X (N, H, W)

        Returns Tensor (N, n_angles, n_bins)
        """
        Px   = self.forward_batch(X)                       # (N, A, B)
        cPx  = self.c.unsqueeze(0) * Px                    # broadcast c
        return cPx + self.r.unsqueeze(0)

    def backproject_batch(self, sinos: torch.Tensor) -> torch.Tensor:
        """
        Batched backprojection.

        Parameters
        ----------
        sinos : Tensor (N, n_angles, n_bins)

        Returns Tensor (N, H, W)
        """
        N = sinos.shape[0]
        return torch.stack([self._backproject_single(sinos[i]) for i in range(N)])


# ---------------------------------------------------------------------------
# 2.  Batched EM update
# ---------------------------------------------------------------------------

def em_update_batch(X: torch.Tensor, y: torch.Tensor,
                    projector: TorchPETProjector) -> torch.Tensor:
    """
    One EM update for the entire population simultaneously.

        X^{n+1} = (X^n / S) * P^T[ c * y / (cPX^n + r) ]

    Parameters
    ----------
    X : Tensor (N, H, W)   population
    y : Tensor (n_angles, n_bins)  measured sinogram

    Returns Tensor (N, H, W)
    """
    X_safe  = torch.clamp(X, min=EPS)
    y_bar   = projector.project_batch(X_safe)              # (N, A, B)
    ratio   = y.unsqueeze(0) / torch.clamp(y_bar, min=EPS) # (N, A, B)
    c_ratio = projector.c.unsqueeze(0) * ratio             # (N, A, B)
    BP      = projector.backproject_batch(c_ratio)         # (N, H, W)
    X_new   = X_safe * BP / projector.sensitivity.unsqueeze(0)
    return torch.clamp(X_new, min=0.0)


# ---------------------------------------------------------------------------
# 3.  Batched directed mutation
# ---------------------------------------------------------------------------

def directed_mutation_batch(X: torch.Tensor, y: torch.Tensor,
                             projector: TorchPETProjector,
                             pixel_size_cm: float = 0.35,
                             fwhm_min_cm: float = 0.0,
                             fwhm_max_cm: float = 2.0) -> torch.Tensor:
    """
    Apply EM update + random per-image Gaussian smoothing to entire population.

    Each image gets its own independently sampled FWHM — vectorised via
    depthwise convolution with per-image kernels.

    Parameters
    ----------
    X : Tensor (N, H, W)

    Returns Tensor (N, H, W)
    """
    N, H, W = X.shape
    device  = X.device

    # Step 1: batched EM update
    X_em    = em_update_batch(X, y, projector)

    # Step 2: per-image random Gaussian smoothing
    fwhm_vals = torch.rand(N, device=device) * (fwhm_max_cm - fwhm_min_cm) \
                + fwhm_min_cm                       # (N,)
    sigma_pix = fwhm_vals / (pixel_size_cm * 2.0 * (2.0 * torch.log(
                    torch.tensor(2.0, device=device))).sqrt())  # (N,)

    # Build one kernel per image, then apply via a proper grouped conv.
    # Input shape for grouped conv:  (1, N, H, W)  — batch=1, channels=N
    # Weight shape:                  (N, 1, k, k)  — out_ch=N, in_ch/group=1
    # groups=N  ⟹ each of the N channels uses its own (1,k,k) kernel.
    MAX_SIGMA = sigma_pix.max().item()
    k_size    = max(3, 2 * int(3 * MAX_SIGMA) + 1)
    k_size    = k_size if k_size % 2 == 1 else k_size + 1
    half      = k_size // 2

    coords  = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    yy, xx  = torch.meshgrid(coords, coords, indexing='ij')   # (k, k)
    r2      = (xx**2 + yy**2)                                  # (k, k)

    sigma2  = (sigma_pix**2).view(N, 1, 1).clamp(min=EPS)     # (N, 1, 1)
    # kernels shape: (N, k, k)
    kernels = torch.exp(-r2.unsqueeze(0) / (2 * sigma2))
    kernels = kernels / kernels.sum(dim=(-2, -1), keepdim=True)
    # Reshape to (N, 1, k, k) — required weight layout for groups=N
    kernels = kernels.unsqueeze(1)                             # (N, 1, k, k)

    # Input: (N, H, W) → (1, N, H, W)
    X_4d     = X_em.unsqueeze(0)                               # (1, N, H, W)
    X_smooth = F.conv2d(X_4d, kernels,
                        padding=half,
                        groups=N)                              # (1, N, H, W)
    X_smooth = X_smooth.squeeze(0)                             # (N, H, W)

    # For images with fwhm≈0, keep the unsmoothed EM result
    no_smooth = (fwhm_vals < 1e-3).view(N, 1, 1)
    X_out     = torch.where(no_smooth, X_em, X_smooth)

    return torch.clamp(X_out, min=0.0)


# ---------------------------------------------------------------------------
# 4.  Batched Φ_quantification
# ---------------------------------------------------------------------------

def phi_quant_batch(X: torch.Tensor, y: torch.Tensor,
                    projector: TorchPETProjector) -> torch.Tensor:
    """
    Poisson log-likelihood for entire population.

    Parameters
    ----------
    X : Tensor (N, H, W)
    y : Tensor (n_angles, n_bins)

    Returns
    -------
    scores : Tensor (N,)
    """
    X_safe  = torch.clamp(X, min=EPS)
    y_bar   = projector.project_batch(X_safe)                    # (N, A, B)
    y_bar   = torch.clamp(y_bar, min=EPS)
    y_exp   = y.unsqueeze(0).expand_as(y_bar)
    ll      = -y_bar + y_exp * torch.log(y_bar)
    return ll.sum(dim=(-2, -1))                                  # (N,)


# ---------------------------------------------------------------------------
# 5.  Batched Φ_detection  (scan-statistic)
# ---------------------------------------------------------------------------

def _make_disc_kernel_torch(radius_pixels: float,
                             device: torch.device) -> torch.Tensor:
    """Circular averaging kernel, shape (1, 1, k, k)."""
    r    = int(np.ceil(radius_pixels))
    size = 2 * r + 1
    Y, X = torch.meshgrid(
        torch.arange(-r, r + 1, dtype=torch.float32, device=device),
        torch.arange(-r, r + 1, dtype=torch.float32, device=device),
        indexing='ij'
    )
    disc = (X**2 + Y**2 <= radius_pixels**2).float()
    disc = disc / disc.sum()
    return disc.unsqueeze(0).unsqueeze(0)   # (1, 1, k, k)


def phi_detect_batch(X: torch.Tensor,
                     roi_mask: torch.Tensor,
                     disc_radius_cm: float = 0.5,
                     pixel_size_cm: float = 0.35,
                     L: int = 2) -> torch.Tensor:
    """
    Batched detection objective for entire population.

    Parameters
    ----------
    X        : Tensor (N, H, W)
    roi_mask : Tensor (H, W) bool  — region to search
    L        : number of signal/background candidate pairs

    Returns
    -------
    scores : Tensor (N,)
    """
    N, H, W    = X.shape
    device     = X.device

    radius_pix = disc_radius_cm / pixel_size_cm
    kernel     = _make_disc_kernel_torch(radius_pix, device)  # (1, 1, k, k)
    pad        = kernel.shape[-1] // 2

    # Apply same disc kernel to all N images using standard (N,1,H,W) input.
    # No grouped conv trick needed — F.conv2d naturally loops over batch.
    X_4d       = X.unsqueeze(1)                              # (N, 1, H, W)
    local_mean = F.conv2d(X_4d, kernel, padding=pad)         # (N, 1, H, W)
    local_mean = local_mean.squeeze(1)                       # (N, H, W)

    X2_4d      = (X ** 2).unsqueeze(1)                       # (N, 1, H, W)
    local_sq   = F.conv2d(X2_4d, kernel, padding=pad).squeeze(1)  # (N, H, W)
    local_var  = torch.clamp(local_sq - local_mean ** 2, min=0.0)

    # Apply ROI mask
    roi = roi_mask.unsqueeze(0).float()          # (1, H, W)
    # Set non-ROI to large negative / large positive for argsorting
    masked_mean_hi = local_mean.clone()
    masked_mean_hi[:, ~roi_mask] = -1e9          # for finding max
    masked_mean_lo = local_mean.clone()
    masked_mean_lo[:, ~roi_mask] = 1e9           # for finding min

    # Flatten spatial dims
    flat_hi = masked_mean_hi.view(N, -1)         # (N, H*W)
    flat_lo = masked_mean_lo.view(N, -1)
    flat_m  = local_mean.view(N, -1)
    flat_v  = local_var.view(N, -1)

    # Top-L signal locations and bottom-L background locations
    _, top_idx = torch.topk(flat_hi, L, dim=1)   # (N, L)
    _, bot_idx = torch.topk(flat_lo, L, dim=1, largest=False)

    # Gather stats
    mean_s = flat_m.gather(1, top_idx)           # (N, L)
    var_s  = flat_v.gather(1, top_idx)
    mean_b = flat_m.gather(1, bot_idx)
    var_b  = flat_v.gather(1, bot_idx)

    denom  = torch.sqrt(0.5 * (var_s + var_b) + EPS)
    snr    = (mean_s - mean_b) / denom           # (N, L)
    return snr.mean(dim=1)                        # (N,)


# ---------------------------------------------------------------------------
# 6.  Evaluate full population in one call
# ---------------------------------------------------------------------------

def evaluate_population_torch(X: torch.Tensor, y: torch.Tensor,
                               projector: TorchPETProjector,
                               roi_mask: torch.Tensor,
                               disc_radius_cm: float = 0.5,
                               pixel_size_cm: float = 0.35,
                               L: int = 2):
    """
    Compute (Φ_quant, Φ_detect) for entire population in one batched pass.

    Returns
    -------
    scores : list of (float, float) tuples, length N
    """
    with torch.no_grad():
        q_batch = phi_quant_batch(X, y, projector)
        d_batch = phi_detect_batch(X, roi_mask,
                                   disc_radius_cm=disc_radius_cm,
                                   pixel_size_cm=pixel_size_cm,
                                   L=L)

    q_list = q_batch.cpu().tolist()
    d_list = d_batch.cpu().tolist()
    return list(zip(q_list, d_list))


# ---------------------------------------------------------------------------
# 7.  Convenience: convert numpy population ↔ torch tensor
# ---------------------------------------------------------------------------

def pop_to_tensor(population: List[np.ndarray],
                  device: torch.device = DEVICE) -> torch.Tensor:
    """List of (H,W) numpy arrays → Tensor (N, H, W) on device."""
    return torch.from_numpy(np.stack(population)).to(device)


def tensor_to_pop(X: torch.Tensor) -> List[np.ndarray]:
    """Tensor (N, H, W) → list of (H,W) numpy arrays."""
    return [X[i].cpu().numpy() for i in range(X.shape[0])]


# ---------------------------------------------------------------------------
# Quick benchmark: numpy vs torch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    IMAGE_SIZE = 64
    N_ANGLES   = 90
    N_POP      = 20   # population size for timing

    print(f"\n=== PyTorch Objectives Benchmark (img={IMAGE_SIZE}, N={N_POP}) ===\n")

    # Build projector
    c_np = np.ones((N_ANGLES, IMAGE_SIZE), dtype=np.float32)
    r_np = np.ones((N_ANGLES, IMAGE_SIZE), dtype=np.float32) * 0.1

    c = torch.from_numpy(c_np).to(DEVICE)
    r = torch.from_numpy(r_np).to(DEVICE)

    projector = TorchPETProjector(IMAGE_SIZE, N_ANGLES, IMAGE_SIZE, c, r, DEVICE)

    # Dummy population and sinogram
    rng   = np.random.default_rng(0)
    X_np  = rng.uniform(0, 2, (N_POP, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    y_np  = rng.poisson(50, (N_ANGLES, IMAGE_SIZE)).astype(np.float32)

    X     = torch.from_numpy(X_np).to(DEVICE)
    y     = torch.from_numpy(y_np).to(DEVICE)
    roi   = torch.ones(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool, device=DEVICE)

    # Warmup
    _ = evaluate_population_torch(X, y, projector, roi, L=2)

    # Time batched evaluation
    t0 = time.time()
    for _ in range(3):
        scores = evaluate_population_torch(X, y, projector, roi, L=2)
    t_torch = (time.time() - t0) / 3

    print(f"Torch batched evaluation ({N_POP} images): {t_torch:.3f}s per call")
    print(f"Projected per-generation time (N=100): ~{t_torch * 5:.1f}s")

    # Time mutation
    t0 = time.time()
    for _ in range(3):
        X_mut = directed_mutation_batch(X, y, projector)
    t_mut = (time.time() - t0) / 3
    print(f"Torch batched mutation   ({N_POP} images): {t_mut:.3f}s per call")

    print(f"\nSample scores (first 3): {scores[:3]}")
    print("=== BENCHMARK DONE ===")
