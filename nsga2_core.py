"""
nsga2_core.py
=============
Module 3: NSGA-II Core Operations

Handles:
- Dominance checking
- Fast non-dominated sorting  (O(MN²) — sufficient for N=100)
- Crowding distance computation
- Tournament selection
- Simulated Binary Crossover (SBC)
"""

import numpy as np


EPS = 1e-10


# ---------------------------------------------------------------------------
# 1.  Dominance
# ---------------------------------------------------------------------------

def dominates(score_a, score_b):
    """
    Return True if solution a dominates solution b.

    a dominates b iff:
      - a is at least as good as b in ALL objectives, AND
      - a is strictly better in AT LEAST ONE objective.

    Both objectives are maximised.

    Parameters
    ----------
    score_a, score_b : tuple (quant, detect)
    """
    at_least_as_good = all(a >= b for a, b in zip(score_a, score_b))
    strictly_better  = any(a >  b for a, b in zip(score_a, score_b))
    return at_least_as_good and strictly_better


# ---------------------------------------------------------------------------
# 2.  Fast non-dominated sorting
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(scores):
    """
    Deb et al. NSGA-II non-dominated sorting.

    Parameters
    ----------
    scores : list of (quant, detect) tuples, length N

    Returns
    -------
    fronts : list of lists
        fronts[0] = indices of Pareto-optimal (rank-1) solutions
        fronts[1] = rank-2 solutions, etc.
    ranks  : np.ndarray (N,) int
        Pareto rank of each solution (1-indexed).
    """
    N = len(scores)

    # S[i]  = set of solutions dominated by i
    # n[i]  = number of solutions that dominate i
    S = [[] for _ in range(N)]
    n = np.zeros(N, dtype=int)

    fronts = [[]]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if dominates(scores[i], scores[j]):
                S[i].append(j)
            elif dominates(scores[j], scores[i]):
                n[i] += 1

        if n[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    fronts = [f for f in fronts if f]   # remove empty trailing front

    # Build rank array
    ranks = np.zeros(N, dtype=int)
    for rank_idx, front in enumerate(fronts):
        for i in front:
            ranks[i] = rank_idx + 1    # 1-indexed

    return fronts, ranks


# ---------------------------------------------------------------------------
# 3.  Crowding distance
# ---------------------------------------------------------------------------

def crowding_distance(scores, front_indices):
    """
    Compute crowding distance for solutions in a single front.

    Parameters
    ----------
    scores        : list of all (quant, detect) tuples (length = total pop)
    front_indices : list of indices belonging to this front

    Returns
    -------
    cd : dict  {index: crowding_distance_value}
    """
    if len(front_indices) <= 2:
        return {i: np.inf for i in front_indices}

    n_obj = len(scores[0])
    cd    = {i: 0.0 for i in front_indices}

    front_scores = np.array([scores[i] for i in front_indices])   # (|F|, n_obj)

    for m in range(n_obj):
        obj_vals = front_scores[:, m]
        sorted_order = np.argsort(obj_vals)

        # Boundary solutions get infinite distance
        cd[front_indices[sorted_order[0]]]  = np.inf
        cd[front_indices[sorted_order[-1]]] = np.inf

        obj_range = obj_vals[sorted_order[-1]] - obj_vals[sorted_order[0]]
        if obj_range < EPS:
            continue

        for k in range(1, len(front_indices) - 1):
            idx = front_indices[sorted_order[k]]
            prev_val = obj_vals[sorted_order[k - 1]]
            next_val = obj_vals[sorted_order[k + 1]]
            cd[idx] += (next_val - prev_val) / obj_range

    return cd


def compute_all_crowding(scores, fronts):
    """
    Compute crowding distances for all solutions across all fronts.

    Returns
    -------
    all_cd : np.ndarray (N,)  crowding distance for each solution
    """
    N      = len(scores)
    all_cd = np.zeros(N, dtype=float)
    for front in fronts:
        cd = crowding_distance(scores, front)
        for i, val in cd.items():
            all_cd[i] = val
    return all_cd


# ---------------------------------------------------------------------------
# 4.  Tournament selection
# ---------------------------------------------------------------------------

def tournament_selection(ranks, crowding, rng, k=2):
    """
    Binary tournament selection.

    Selects k random candidates; winner = lower rank.
    Tie-break: higher crowding distance (more spread out = preferred).

    Parameters
    ----------
    ranks    : np.ndarray (N,) — Pareto rank (1-indexed; lower is better)
    crowding : np.ndarray (N,) — crowding distance
    rng      : np.random.Generator
    k        : tournament size

    Returns
    -------
    winner_index : int
    """
    N         = len(ranks)
    candidates = rng.choice(N, size=k, replace=False)
    winner     = candidates[0]

    for c in candidates[1:]:
        if ranks[c] < ranks[winner]:
            winner = c
        elif ranks[c] == ranks[winner] and crowding[c] > crowding[winner]:
            winner = c

    return int(winner)


# ---------------------------------------------------------------------------
# 5.  Simulated Binary Crossover  (SBC)
# ---------------------------------------------------------------------------

def simulated_binary_crossover(p1, p2, rng,
                                eta_c=20.0, P_c=0.95):
    """
    Simulated Binary Crossover (Deb & Agrawal, 1995).

    Operates pixel-wise on flattened images.

    Parameters
    ----------
    p1, p2  : np.ndarray (H, W)   parent images
    rng     : np.random.Generator
    eta_c   : distribution index  (large → offspring near parents)
    P_c     : probability of crossover at each pixel

    Returns
    -------
    q1, q2 : np.ndarray (H, W) offspring images, non-negative
    """
    shape     = p1.shape
    p1_flat   = p1.flatten().astype(np.float64)
    p2_flat   = p2.flatten().astype(np.float64)
    n         = len(p1_flat)

    q1_flat = p1_flat.copy()
    q2_flat = p2_flat.copy()

    # Crossover mask
    mask = rng.random(n) < P_c

    # Random uniform draw for beta
    u  = rng.random(n)

    beta = np.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta_c + 1.0)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
    )

    # Apply crossover only where mask is True
    q1_flat = np.where(
        mask,
        0.5 * ((1.0 + beta) * p1_flat + (1.0 - beta) * p2_flat),
        p1_flat
    )
    q2_flat = np.where(
        mask,
        0.5 * ((1.0 - beta) * p1_flat + (1.0 + beta) * p2_flat),
        p2_flat
    )

    # Non-negativity
    q1 = np.clip(q1_flat.reshape(shape), 0.0, None).astype(np.float32)
    q2 = np.clip(q2_flat.reshape(shape), 0.0, None).astype(np.float32)

    return q1, q2


# ---------------------------------------------------------------------------
# 6.  Population selection for next generation
# ---------------------------------------------------------------------------

def select_next_generation(combined_pop, combined_scores, N):
    """
    Select N individuals from combined population (size 2N) using
    non-dominated sorting + crowding distance tie-breaking.

    Parameters
    ----------
    combined_pop    : list of images, length 2N
    combined_scores : list of (quant, detect) tuples, length 2N
    N               : target population size

    Returns
    -------
    next_pop    : list of N images
    next_scores : list of N score tuples
    """
    fronts, ranks = fast_non_dominated_sort(combined_scores)
    all_cd        = compute_all_crowding(combined_scores, fronts)

    selected = []

    for front in fronts:
        if len(selected) + len(front) <= N:
            selected.extend(front)
        else:
            # Need to partially include this front
            remaining = N - len(selected)
            # Sort by crowding distance descending (more spread = preferred)
            front_sorted = sorted(front,
                                  key=lambda i: all_cd[i],
                                  reverse=True)
            selected.extend(front_sorted[:remaining])
            break

    next_pop    = [combined_pop[i]    for i in selected]
    next_scores = [combined_scores[i] for i in selected]

    return next_pop, next_scores


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=== NSGA-II Core Sanity Check ===")

    rng = np.random.default_rng(42)

    # --- Test dominance ---
    assert dominates((2.0, 3.0), (1.0, 2.0))   # a dominates b
    assert not dominates((1.0, 3.0), (2.0, 2.0)) # not dominating
    assert not dominates((1.0, 1.0), (1.0, 1.0)) # equal → no dominance
    print("dominates(): OK")

    # --- Test non-dominated sorting on toy 2D problem ---
    # True Pareto front: points on y = -x + 5  (maximise both)
    toy_scores = [
        (1.0, 4.0),   # 0  front 1
        (2.0, 3.0),   # 1  front 1
        (4.0, 1.0),   # 2  front 1
        (1.5, 2.0),   # 3  front 2 (dominated by 1)
        (0.5, 0.5),   # 4  front 3 (dominated by many)
        (3.0, 2.0),   # 5  front 1
    ]
    fronts, ranks = fast_non_dominated_sort(toy_scores)
    print(f"Fronts: {fronts}")
    print(f"Ranks:  {ranks}")
    assert set(fronts[0]) == {0, 1, 2, 5}, f"Expected front1={{0,1,2,5}}, got {fronts[0]}"
    assert 3 in fronts[1], f"Expected 3 in front2"
    assert 4 in fronts[2], f"Expected 4 in front3"
    print("fast_non_dominated_sort(): OK")

    # --- Test crowding distance ---
    cd = crowding_distance(toy_scores, fronts[0])
    print(f"Crowding distances (front 1): {cd}")
    # Boundary solutions should be inf
    assert cd[0] == np.inf or cd[2] == np.inf, "Boundary solutions should have inf CD"
    print("crowding_distance(): OK")

    # --- Test SBC crossover ---
    p1 = rng.random((8, 8)).astype(np.float32) * 2
    p2 = rng.random((8, 8)).astype(np.float32) * 2
    q1, q2 = simulated_binary_crossover(p1, p2, rng, eta_c=20, P_c=0.95)
    assert q1.shape == p1.shape
    assert (q1 >= 0).all() and (q2 >= 0).all()
    print(f"SBC: parent means ({p1.mean():.3f}, {p2.mean():.3f}) → "
          f"offspring means ({q1.mean():.3f}, {q2.mean():.3f})")
    print("simulated_binary_crossover(): OK")

    # --- Test tournament selection ---
    N_test   = 20
    r_test   = np.random.randint(1, 5, size=N_test)
    cd_test  = np.random.rand(N_test)
    winners  = [tournament_selection(r_test, cd_test, rng) for _ in range(100)]
    avg_rank = r_test[winners].mean()
    print(f"Tournament selection avg winning rank: {avg_rank:.2f} "
          f"(should be < overall avg {r_test.mean():.2f})")
    assert avg_rank < r_test.mean(), "Tournament should prefer lower ranks"
    print("tournament_selection(): OK")

    print("=== ALL PASSED ===")
