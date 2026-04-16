"""
termination.py
==============
Adaptive Termination Criteria for MOEAP (coded from scratch, inspired by pymoo).

Provides:
  - ConvergenceTermination: stops when Pareto front stops improving
    Tracks 3 signals over a sliding window:
    1. Delta ideal point (max obj change)
    2. Delta nadir point (max obj change)
    3. Delta IGD between consecutive fronts

  - MaxGenTermination: simple fixed-generation limit (baseline)
  - CombinedTermination: earliest of multiple criteria

Usage in MOEAP loop:
    term = ConvergenceTermination(tol=0.005, n_patience=10)
    for gen in range(max_gens):
        ...
        term.update(scores)
        if term.has_terminated():
            print(f"Converged: {term.reason}")
            break
"""

import numpy as np
from indicators import igd


# ---------------------------------------------------------------------------
# 1.  Base Termination
# ---------------------------------------------------------------------------

class Termination:
    """Base class for termination criteria."""

    def __init__(self):
        self._terminated = False
        self.reason = ""

    def has_terminated(self):
        return self._terminated

    def update(self, scores, gen=None):
        raise NotImplementedError

    def reset(self):
        self._terminated = False
        self.reason = ""


# ---------------------------------------------------------------------------
# 2.  Max Generation Termination
# ---------------------------------------------------------------------------

class MaxGenTermination(Termination):
    """Terminate after a fixed number of generations."""

    def __init__(self, n_max_gen):
        super().__init__()
        self.n_max_gen = n_max_gen

    def update(self, scores, gen=None):
        if gen is not None and gen >= self.n_max_gen - 1:
            self._terminated = True
            self.reason = f"max_gen={self.n_max_gen} reached"


# ---------------------------------------------------------------------------
# 3.  Convergence Termination  (multi-objective f-tol)
# ---------------------------------------------------------------------------

class ConvergenceTermination(Termination):
    """
    Adaptive termination based on Pareto front convergence.

    Monitors the change in the Pareto front across generations.
    Terminates when all 3 delta signals stay below `tol` for
    `n_patience` consecutive generations.

    Signals tracked:
      - delta_ideal : max normalised change in ideal point
      - delta_nadir : max normalised change in nadir point
      - delta_igd   : IGD between consecutive generation fronts

    Parameters
    ----------
    tol        : float   convergence tolerance (default: 0.005)
    n_patience : int     number of consecutive gens below tol (default: 10)
    """

    def __init__(self, tol=0.005, n_patience=10):
        super().__init__()
        self.tol = tol
        self.n_patience = n_patience

        self._prev_ideal = None
        self._prev_nadir = None
        self._prev_F = None
        self._patience_counter = 0

        # History for analysis
        self.history = {
            'delta_ideal': [],
            'delta_nadir': [],
            'delta_igd': [],
            'converged_gen': None,
        }

    def update(self, scores, gen=None):
        """
        Update termination state with current generation's scores.

        Parameters
        ----------
        scores : list of (obj1, obj2) tuples  — current population scores
        gen    : int, current generation number (optional)
        """
        F = np.array(scores, dtype=np.float64)

        # Extract Pareto-front-only members for cleaner signal
        # (use all scores — the front members drive ideal/nadir anyway)
        ideal = F.min(axis=0)
        nadir = F.max(axis=0)

        if self._prev_ideal is None:
            # First generation — just store and return
            self._prev_ideal = ideal
            self._prev_nadir = nadir
            self._prev_F = F
            return

        # Normalisation range
        norm = nadir - ideal
        norm[norm < 1e-32] = 1.0

        # Delta ideal
        delta_ideal = np.max(np.abs(ideal - self._prev_ideal) / norm)

        # Delta nadir
        delta_nadir = np.max(np.abs(nadir - self._prev_nadir) / norm)

        # Delta IGD: how much did the front move?
        # Normalise both current and previous front
        F_norm = (F - ideal) / norm
        prev_norm = (self._prev_F - ideal) / norm

        delta_f = igd(F_norm, prev_norm, normalize=False)

        # Store history
        self.history['delta_ideal'].append(float(delta_ideal))
        self.history['delta_nadir'].append(float(delta_nadir))
        self.history['delta_igd'].append(float(delta_f))

        # Check convergence
        max_delta = max(delta_ideal, delta_nadir, delta_f)

        if max_delta < self.tol:
            self._patience_counter += 1
        else:
            self._patience_counter = 0

        if self._patience_counter >= self.n_patience:
            self._terminated = True
            self.reason = (f"converged (delta={max_delta:.6f} < tol={self.tol} "
                           f"for {self.n_patience} consecutive gens)")
            self.history['converged_gen'] = gen

        # Update state
        self._prev_ideal = ideal.copy()
        self._prev_nadir = nadir.copy()
        self._prev_F = F.copy()

    def reset(self):
        super().reset()
        self._prev_ideal = None
        self._prev_nadir = None
        self._prev_F = None
        self._patience_counter = 0
        self.history = {
            'delta_ideal': [],
            'delta_nadir': [],
            'delta_igd': [],
            'converged_gen': None,
        }


# ---------------------------------------------------------------------------
# 4.  Combined Termination
# ---------------------------------------------------------------------------

class CombinedTermination(Termination):
    """
    Combines multiple termination criteria (first to trigger wins).

    Usage:
        term = CombinedTermination([
            MaxGenTermination(200),
            ConvergenceTermination(tol=0.005, n_patience=10),
        ])
    """

    def __init__(self, criteria):
        super().__init__()
        self.criteria = criteria

    def update(self, scores, gen=None):
        for c in self.criteria:
            c.update(scores, gen=gen)
            if c.has_terminated():
                self._terminated = True
                self.reason = c.reason
                break

    def has_terminated(self):
        return self._terminated

    def reset(self):
        super().reset()
        for c in self.criteria:
            c.reset()


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Termination Sanity Check ===")

    rng = np.random.default_rng(42)

    # --- Test 1: MaxGenTermination ---
    term = MaxGenTermination(10)
    for g in range(15):
        term.update([(1, 2), (3, 4)], gen=g)
        if term.has_terminated():
            print(f"MaxGen terminated at gen {g}: {term.reason}")
            assert g == 9, f"Expected gen 9, got {g}"
            break
    print("MaxGenTermination: OK")

    # --- Test 2: ConvergenceTermination on stationary front ---
    term2 = ConvergenceTermination(tol=0.01, n_patience=5)
    fixed_scores = [(float(i), float(10 - i)) for i in range(10)]

    converged = False
    for g in range(100):
        # Add tiny noise
        noisy = [(s[0] + rng.normal(0, 0.001),
                  s[1] + rng.normal(0, 0.001)) for s in fixed_scores]
        term2.update(noisy, gen=g)
        if term2.has_terminated():
            print(f"Convergence terminated at gen {g}: {term2.reason}")
            converged = True
            break

    assert converged, "Should have detected convergence on near-stationary front"
    print("ConvergenceTermination (stationary): OK")

    # --- Test 3: ConvergenceTermination should NOT trigger on improving front ---
    term3 = ConvergenceTermination(tol=0.01, n_patience=5)
    for g in range(50):
        # Front is clearly improving each generation
        scores = [(float(i + g * 0.5), float(50 - i + g * 0.5))
                  for i in range(10)]
        term3.update(scores, gen=g)
    assert not term3.has_terminated(), \
        "Should NOT have terminated on consistently improving front"
    print("ConvergenceTermination (improving): OK")

    # --- Test 4: CombinedTermination ---
    combined = CombinedTermination([
        MaxGenTermination(20),
        ConvergenceTermination(tol=0.001, n_patience=100),  # won't trigger
    ])
    for g in range(30):
        combined.update([(g, 30 - g)], gen=g)
        if combined.has_terminated():
            print(f"Combined terminated at gen {g}: {combined.reason}")
            assert "max_gen" in combined.reason
            break
    print("CombinedTermination: OK")

    print("=== ALL PASSED ===")
