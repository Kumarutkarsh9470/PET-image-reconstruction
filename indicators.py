"""
indicators.py
Pareto Front Quality Indicators (coded from scratch, inspired by pymoo).

Provides:
  - Hypervolume (2D exact sweep-line)
  - Hypervolume Contribution (per-point)
  - Inverted Generational Distance (IGD)
  - Spacing (uniformity of front distribution)

All functions work on (N, 2) numpy arrays where each row is (obj1, obj2).

NOTE: Our objectives are MAXIMISED (higher = better).
      For HV computation we negate internally so the standard
      "dominated region below ref_point" formulation works correctly.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist


EPS = 1e-10

# 1.  Hypervolume  (2D exact, O(N log N) sweep-line)

def hypervolume_2d(F, ref_point):
    """
    Exact 2D hypervolume via sweep-line.

    Computes the area of objective space dominated by the front F
    and bounded by ref_point.

    Parameters
    ----------
    F         : np.ndarray (N, 2)  objective values (MAXIMISED)
    ref_point : np.ndarray (2,)    reference point (must be dominated
                                   by all front members, i.e. WORSE than all)

    Returns
    -------
    hv : float   hypervolume value (higher = better front)

    Algorithm
    ---------
    Since we maximise, we want the area of the region that is dominated
    by F and bounded below by ref_point.

    1. Clip F to ref_point (discard points worse than ref in any obj)
    2. Sort by obj1 ascending
    3. Sweep left-to-right, accumulating rectangles
    """
    if len(F) == 0:
        return 0.0

    ref = np.asarray(ref_point, dtype=np.float64)
    pts = np.asarray(F, dtype=np.float64)

    # Keep only points that dominate the reference (better in BOTH objectives)
    mask = np.all(pts > ref, axis=1)
    pts = pts[mask]

    if len(pts) == 0:
        return 0.0

    # Sort by obj1 ascending  (then obj2 descending for tie-break)
    order = np.lexsort((-pts[:, 1], pts[:, 0]))
    pts = pts[order]

    # Sweep-line: accumulate area of rectangles
    hv = 0.0
    prev_obj2 = ref[1]  # lower bound for obj2

    for i in range(len(pts) - 1, -1, -1):
        # Rectangle from this point to the next (or to ref)
        # Width along obj1: from pts[i, 0] to (pts[i+1, 0] or ref[0] — not needed)
        # Actually, the standard 2D HV sweep works differently for maximisation:
        pass

    # Simpler formulation: sort by obj1 descending
    pts_desc = pts[np.argsort(-pts[:, 0])]

    hv = 0.0
    max_obj2 = ref[1]   # running max of obj2 seen so far (from right)

    # Walk from highest obj1 to lowest
    # Each point contributes a rectangle:
    #   width  = pts[i, 0] - pts[i+1, 0]   (or pts[i, 0] - ref[0] for last)
    #   height = max(pts[i, 1], max_obj2) - ref[1]
    # But actually the clean way for maximisation HV:

    # Re-sort by obj1 ascending
    pts_asc = pts[np.argsort(pts[:, 0])]

    hv = 0.0
    current_max_obj2 = ref[1]

    for i in range(len(pts_asc)):
        if pts_asc[i, 1] > current_max_obj2:
            # This point extends the front upward in obj2
            # Width of rectangle: next_obj1 - this_obj1  (or ref[0] if last)
            if i + 1 < len(pts_asc):
                width = pts_asc[i + 1, 0] - pts_asc[i, 0]
            else:
                # No — we need to think about this differently
                pass
            current_max_obj2 = pts_asc[i, 1]
    # 1. Sort non-dominated points by obj1 ascending
    # 2. The front is a staircase going right and down
    # 3. Area = sum of rectangles under each step

    # Filter to non-dominated only
    pts_sorted = pts[np.argsort(pts[:, 0])]  # sort by obj1 asc

    # Remove dominated points (keep only non-dominated)
    nd = []
    best_obj2 = -np.inf
    for i in range(len(pts_sorted) - 1, -1, -1):
        if pts_sorted[i, 1] > best_obj2:
            nd.append(pts_sorted[i])
            best_obj2 = pts_sorted[i, 1]
    nd = np.array(nd[::-1])  # now sorted by obj1 ascending, obj2 descending

    # Compute HV as sum of rectangles
    # Each point i creates a rectangle:
    #   left  = nd[i, 0]
    #   right = nd[i+1, 0]  (or we go to infinity — but bounded by ref? No, ref is lower-left)
    #   bottom = ref[1]
    #   top   = nd[i, 1]

    hv = 0.0
    for i in range(len(nd)):
        height = nd[i, 1] - ref[1]
        if i + 1 < len(nd):
            width = nd[i + 1, 0] - nd[i, 0]
        else:
            # Correction: Standard HV for minimisation uses ref as upper-right.
            # For maximisation, we should negate and use standard formulation.
            width = 0
        hv += height * width
    return _hypervolume_2d_minimisation(-F, -ref_point)

def _hypervolume_2d_minimisation(F, ref_point):
    """
    Standard 2D hypervolume for MINIMISATION problems.

    Computes the area dominated by the front F and bounded above
    by ref_point.  Points must satisfy F[i] <= ref_point componentwise
    to contribute.

    Parameters
    ----------
    F         : (N, 2) objective values (minimised)
    ref_point : (2,)   upper-right reference point
    """
    if len(F) == 0:
        return 0.0

    ref = np.asarray(ref_point, dtype=np.float64)
    pts = np.asarray(F, dtype=np.float64).copy()
    mask = np.all(pts < ref, axis=1)
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0
    order = np.lexsort((-pts[:, 1], pts[:, 0]))
    pts = pts[order]

    # Sweep-line: walk left to right, track minimum obj2 seen
    # Non-dominated filtering (since we sorted by obj1 asc,
    # a point is dominated if its obj2 >= a previous point's obj2)
    nd = [pts[0]]
    for i in range(1, len(pts)):
        if pts[i, 1] < nd[-1][1]:
            nd.append(pts[i])
    nd = np.array(nd)
    # Each point i contributes rectangle:
    #   width  = nd[i+1, 0] - nd[i, 0]   (last point: ref[0] - nd[-1, 0])
    #   height = ref[1] - nd[i, 1]
    hv = 0.0
    for i in range(len(nd)):
        if i + 1 < len(nd):
            width = nd[i + 1, 0] - nd[i, 0]
        else:
            width = ref[0] - nd[i, 0]
        height = ref[1] - nd[i, 1]
        hv += width * height
    return hv
def hypervolume(F, ref_point):
    """
    Compute hypervolume indicator for a Pareto front.

    For 2-objective problems uses exact sweep-line.
    Objectives are MAXIMISED (higher = better).
    ref_point should be WORSE than all front members.

    Parameters
    ----------
    F         : np.ndarray (N, 2)  or list of (obj1, obj2) tuples
    ref_point : array-like (2,)

    Returns
    -------
    hv : float
    """
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    ref = np.asarray(ref_point, dtype=np.float64)

    assert F.shape[1] == 2, "Only 2-objective HV is supported."
    return hypervolume_2d(F, ref)

# 2.  Hypervolume Contribution (per-point)

def hypervolume_contribution(F, ref_point):
    """
    Compute each point's exclusive hypervolume contribution.

    HVC(i) = HV(F) - HV(F without point i)

    Parameters
    ----------
    F         : (N, 2)
    ref_point : (2,)

    Returns
    -------
    hvc : np.ndarray (N,)
    """
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    ref = np.asarray(ref_point, dtype=np.float64)

    total_hv = hypervolume(F, ref)
    n = len(F)
    hvc = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        hvc[i] = total_hv - hypervolume(F[mask], ref)

    return hvc

# 3.  Inverted Generational Distance (IGD)

def igd(F, pf_reference, normalize=False):
    """
    Inverted Generational Distance.

    Measures how close a reference Pareto front is to the obtained front F.
    For each point in pf_reference, finds the nearest point in F,
    then returns the mean of these distances.

    Lower IGD = better approximation.

    Parameters
    ----------
    F            : (N, 2)   obtained front
    pf_reference : (M, 2)   reference / true Pareto front
    normalize    : bool      normalize objectives to [0, 1]

    Returns
    -------
    igd_value : float
    """
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    pf = np.atleast_2d(np.asarray(pf_reference, dtype=np.float64))

    if normalize:
        combined = np.vstack([F, pf])
        lo = combined.min(axis=0)
        hi = combined.max(axis=0)
        rng = hi - lo
        rng[rng < EPS] = 1.0
        F = (F - lo) / rng
        pf = (pf - lo) / rng

    # Distance from each pf point to nearest F point
    D = cdist(pf, F, metric='euclidean')
    min_dists = D.min(axis=1)

    return float(min_dists.mean())


def igd_plus(F, pf_reference, normalize=False):
    """
    IGD+ (modified distance that only penalises in dominated directions).

    For each reference point z, the distance to a front point a is:
        d+(z, a) = sqrt( sum( max(a_i - z_i, 0)^2 ) )   for minimisation

    Since we maximise, the "worse" direction is when a < z:
        d+(z, a) = sqrt( sum( max(z_i - a_i, 0)^2 ) )

    Parameters
    F, pf_reference : (N, 2) and (M, 2) objective values (MAXIMISED)

    Returns
    igd_plus_value : float
    """
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    pf = np.atleast_2d(np.asarray(pf_reference, dtype=np.float64))

    if normalize:
        combined = np.vstack([F, pf])
        lo = combined.min(axis=0)
        hi = combined.max(axis=0)
        rng = hi - lo
        rng[rng < EPS] = 1.0
        F = (F - lo) / rng
        pf = (pf - lo) / rng

    # For each pf point, compute d+ to each F point
    # d+(z, a) = sqrt(sum(max(z - a, 0)^2)) for maximisation
    n_pf = len(pf)
    n_f = len(F)

    min_dists = np.zeros(n_pf)
    for i in range(n_pf):
        diffs = np.maximum(pf[i] - F, 0)   # (n_f, n_obj)
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        min_dists[i] = dists.min()

    return float(min_dists.mean())

# 4.  Spacing Indicator

def spacing(F):
    """
    Spacing indicator — measures uniformity of point distribution.

    Computes the standard deviation of nearest-neighbour L1 distances.
    Lower spacing = more uniform front.

    Parameters
    ----------
    F : (N, 2) objective values

    Returns
    S : float   spacing value (0 = perfectly uniform)
    """
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    n = len(F)

    if n <= 1:
        return 0.0

    # Pairwise L1 (cityblock) distances
    D = squareform(pdist(F, metric='cityblock'))

    # Nearest neighbour distance for each point
    np.fill_diagonal(D, np.inf)
    d = D.min(axis=1)
    dm = d.mean()

    S = np.sqrt(np.sum((d - dm) ** 2) / n)
    return float(S)

# 5.  Convenience: compute all indicators at once

def compute_all_indicators(F, ref_point, pf_reference=None):
    """
    Compute all available indicators for a Pareto front.

    Parameters
    F            : (N, 2) or list of tuples
    ref_point    : (2,)
    pf_reference : (M, 2) optional reference front for IGD

    Returns
    dict with keys: 'hv', 'spacing', and optionally 'igd'
    """
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    ref = np.asarray(ref_point, dtype=np.float64)

    result = {
        'hv': hypervolume(F, ref),
        'spacing': spacing(F),
        'front_size': len(F),
    }

    if pf_reference is not None:
        result['igd'] = igd(F, pf_reference)

    return result

# Sanity check

if __name__ == "__main__":
    print("=== Indicators Sanity Check ===")

    # Test 1: Hypervolume on known front
    # Square front: (0,1), (1,0) with ref (-1, -1)  → maximisation
    # After negation: minimise (0,-1), (-1,0) with ref (1,1)
    # HV = area = 1*1 = 1  ... let's compute manually
    #
    # Actually: front = [(0,1), (1,0)], ref = (-1,-1)
    # Negate: F_min = [(0,-1), (-1,0)], ref_min = (1,1)
    # Non-dom in minimisation: both are non-dominated
    # Sort by obj1: [(-1,0), (0,-1)]
    # Rect 1: width=(0-(-1))=1, height=(1-0)=1 → area=1
    # Rect 2: width=(1-0)=1, height=(1-(-1))=2 → area=2
    # Total   HV = 3
    F = np.array([[0.0, 1.0], [1.0, 0.0]])
    ref = np.array([-1.0, -1.0])
    hv = hypervolume(F, ref)
    print(f"HV test 1: {hv:.4f}  (expected: 3.0)")
    assert abs(hv - 3.0) < 0.01, f"HV test 1 failed: {hv}"

    # Test 2: Single point HV
    F2 = np.array([[2.0, 3.0]])
    ref2 = np.array([0.0, 0.0])
    hv2 = hypervolume(F2, ref2)
    # Negate: F_min = [(-2, -3)], ref_min = (0, 0)
    # Rect: width = (0-(-2))=2, height = (0-(-3))=3 → area=6
    print(f"HV test 2: {hv2:.4f}  (expected: 6.0)")
    assert abs(hv2 - 6.0) < 0.01, f"HV test 2 failed: {hv2}"

    # Test 3: HV contribution
    hvc = hypervolume_contribution(F, ref)
    print(f"HVC: {hvc}  (sum should be <= HV={hv:.1f})")
    # Each point dominates some exclusive region
    assert all(hvc >= 0), "HVC should be non-negative"

    # Test 4: Spacing on uniform vs clustered
    uniform = np.array([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], dtype=float)
    s_uniform = spacing(uniform)

    clustered = np.array([[0, 4], [0.1, 3.9], [0.2, 3.8], [4, 0]], dtype=float)
    s_clustered = spacing(clustered)

    print(f"Spacing (uniform):   {s_uniform:.4f}")
    print(f"Spacing (clustered): {s_clustered:.4f}")
    assert s_uniform < s_clustered, "Uniform front should have lower spacing"

    # Test 5: IGD 
    ref_front = np.array([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], dtype=float)
    approx = np.array([[0.5, 3.5], [2.5, 1.5]], dtype=float)
    igd_val = igd(approx, ref_front)
    print(f"IGD: {igd_val:.4f}  (should be > 0)")
    assert igd_val > 0, "IGD should be positive"

    # Test 6: IGD+ 
    igdp_val = igd_plus(approx, ref_front)
    print(f"IGD+: {igdp_val:.4f}  (should be >= 0)")
    assert igdp_val >= 0

    # Test 7: compute_all_indicators 
    all_ind = compute_all_indicators(uniform, np.array([-1.0, -1.0]),
                                      pf_reference=uniform)
    print(f"All indicators: {all_ind}")
    assert 'hv' in all_ind and 'spacing' in all_ind and 'igd' in all_ind

    print("ALL PASSED")
