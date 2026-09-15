# Vendored unmodified from the numerics of arXiv:2608.13862
# (https://github.com/arulrhikm/numerics, MIT licence); see LICENSE-richardson.
"""Randomized Richardson extrapolation of Trotter formulas.

Public API consumed by the plotting scripts:

* ``compute_min_samples`` — search Richardson grids and pick the best schedule
  per target precision.
* ``get_richardson_coefficients`` — p-specific Vandermonde (optimized search).
* ``get_wc_richardson_coefficients`` — even-power WC Vandermonde (closed form).
* ``b_norm1`` / ``b_suppressed_norm`` — coefficient norms (`‖b̃‖₁` per Eqs. 220--223).
* ``compute_lambda_scale`` — λ-comm ratio (``lemma57_fixed`` / ``legacy``).
* ``compute_steps_plane_wave`` / ``compute_steps_gate_depth`` — step counts.
* ``gate_overhead`` — ``C_p`` (Suzuki exponentials per Trotter step).
* ``richardson_b_over_eps_prefactor`` — ``K`` in ``(K ‖b‖₁ / ε)^(1/(σ(m-1)+p))``.
* ``set_lambda_scale_mode`` — toggle the λ-comm model.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

# Static prefactor K per Trotter order (used unless the order is not listed,
# in which case ``RICHARDSON_K_DEFAULT`` is returned).
RICHARDSON_K_BY_P: dict[int, float] = {
    1: 2.7232,
    2: 3.627,
    4: 5.15485,
}
RICHARDSON_K_DEFAULT: float = (4.0 / 3.0) * (math.e - 1.0)

# Brute-force search rejects grids whose ‖b‖₁² exceeds this default cap when
# ``compute_min_samples(..., brute_force_b_norm1_sq_max=None)``. Plot scripts
# pass an explicit cap via ``--brute-bnorm-sq-max`` instead.
BRUTE_FORCE_B_NORM1_SQ_MAX_DEFAULT: float = 1e6

# Pre-tabulated Lemma 57 geometric ratios for the orders we evaluate often.
LEMMA57_GEOMETRIC_RATIO_BY_P: dict[int, float] = {
    1: 1.5035,
    2: 1.0968,
    4: 1.0445,
}

LAMBDA_SCALE_MODE: str = "lemma57_fixed"


def richardson_b_over_eps_prefactor(p: int) -> float:
    """Prefactor ``K`` in ``(K · ‖b‖₁ / ε)^(1 / (σ(m-1) + p))``."""
    return float(RICHARDSON_K_BY_P.get(int(p), RICHARDSON_K_DEFAULT))


def set_lambda_scale_mode(mode: str) -> None:
    """Select the λ-comm model: ``'lemma57_fixed'`` (default) or ``'legacy'``."""
    global LAMBDA_SCALE_MODE
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"lemma57_fixed", "legacy"}:
        raise ValueError("mode must be 'lemma57_fixed' or 'legacy'")
    LAMBDA_SCALE_MODE = mode_norm


def sigma_parity(p: int) -> int:
    """σ = 1 for odd ``p``, σ = 2 for even ``p``."""
    return 1 if int(p) % 2 else 2


def richardson_stepcount_eps_denominator(m: int, p: int) -> float:
    """Exponent denominator ``σ(m-1) + p`` for the Richardson step bound."""
    return float(sigma_parity(p) * (int(m) - 1) + int(p))


def richardson_lambda_comm_exponent(p: int) -> float:
    """Power of λ_comm in Richardson step counts: ``1 + 1/p``."""
    return 1.0 + 1.0 / int(p)


def b_norm1(b) -> float:
    """1-norm ``Σ |b_k|`` of the Richardson coefficients."""
    return float(np.sum(np.abs(np.asarray(b, dtype=float))))


def q_refinement_ratio(q_integers) -> float:
    """Factor ``q_max / q_min`` on Richardson Trotter step counts (Eqs. 124, 140)."""
    q = np.asarray(q_integers, dtype=float)
    q_min = float(np.min(q))
    if q_min <= 0:
        return float(np.max(q))
    return float(np.max(q) / q_min)


def b_suppressed_norm(
    b,
    q_integers,
    m: int,
    p: int,
    *,
    well_conditioned: bool = False,
) -> float:
    """Norm ``Σ |b̃_i|`` with ``b̃_i = b_i (q_min / q_i)^e`` (section.tex Eq. 48).

    The reference point is ``q_1 = q_min`` (largest step ``s_1``), so the ratio
    ``q_min / q_i ≤ 1`` and the suppressed norm is *smaller* than ``‖b‖₁`` — the
    refinement of the error-series remainder amplification. ``e = 1`` on
    well-conditioned grids (``‖b^(1)‖``); ``e = p`` on brute-force optimized
    grids (``‖b^(p)‖``).
    """
    del m
    b = np.asarray(b, dtype=float)
    q = np.asarray(q_integers, dtype=float)
    q_min = float(np.min(q))
    exp = 1 if well_conditioned else int(p)
    scaled = b * ((q_min / q) ** exp)
    return float(np.sum(np.abs(scaled)))


def safe_power(base: float, m: float) -> float:
    """Numerically stable ``base ** (1 / m)``; returns ``0.0`` when ``base ≤ 0``."""
    if base <= 0:
        return 0.0
    return float(np.exp(np.log(base) / m))


def get_sample_points(m: int) -> tuple[list[float], list[int]]:
    """Well-conditioned grid of refinement factors ``q_k`` (Low–Kliuchnikov–Wiebe).

    Returns ``(s_k, q_k)`` with ``s_k = 1 / q_k`` and the ``q_k`` made distinct
    by bumping any duplicates upward. For ``m = 1``, returns the degenerate
    single-point grid ``q = [1]`` (plain Trotter at full resolution).
    """
    m = int(m)
    if m == 1:
        return [1.0], [1]
    numerator = (np.sqrt(8) * m) / np.pi
    q_integers: list[int] = []
    for k in range(1, m + 1):
        denominator = np.sin(np.pi * (2 * k - 1) / (8 * m))
        q_integers.append(int(np.ceil(numerator / denominator)))

    seen: set[int] = set()
    for i, q in enumerate(q_integers):
        while q in seen:
            q += 1
        q_integers[i] = q
        seen.add(q)

    s_list = [1.0 / q for q in q_integers]
    return s_list, q_integers


def setup_vandermonde_matrix(s_list, p: int, m: int) -> np.ndarray:
    """Optimized extrapolation Vandermonde tailored to Trotter order ``p``.

    Row ``j`` uses exponent ``0`` for ``j = 0``, else ``p + σ(j−1)`` with
    ``σ = sigma_parity(p)``. This matches the leading error powers of order-``p``
    Suzuki product formulas (``p, p+σ, p+2σ, …``).
    """
    sigma = sigma_parity(p)
    V = np.zeros((m, m))
    for j in range(m):
        power = 0 if j == 0 else p + sigma * (j - 1)
        for k in range(m):
            V[j, k] = s_list[k] ** power
    return V


def setup_wc_vandermonde_matrix(s_list, m: int) -> np.ndarray:
    """Well-conditioned (LKW) Vandermonde: row ``j`` has exponent ``2j``.

    Prescriptive even error series starting at ``s²``; cancels ``s², s⁴, …, s^{2(m−1)}``
    so the leading remainder is ``s^{2m}``. Independent of Trotter order ``p`` — use
    only when the underlying error series is even (``p ∈ {2, 4, …}``).
    """
    m = int(m)
    V = np.zeros((m, m))
    for j in range(m):
        power = 2 * j
        for k in range(m):
            V[j, k] = s_list[k] ** power
    return V


def get_wc_richardson_coefficients_closed_form(s_list) -> np.ndarray:
    """Closed-form WC weights via Lagrange extrapolation to ``t = 0``.

    For nodes ``t_k = s_k²`` the unique degree-``m−1`` polynomial with
    ``P(0) = 1`` and ``P(t_k) = 0`` for ``k ≥ 2`` in the moment system gives
    ``b_k = ∏_{j≠k} (−t_j) / (t_k − t_j)`` (Eq. 62 in the Trotter-mitigation note).
    """
    s = np.asarray(s_list, dtype=float)
    t = s * s
    m = len(t)
    b = np.empty(m, dtype=float)
    for k in range(m):
        weight = 1.0
        for j in range(m):
            if j != k:
                weight *= (-t[j]) / (t[k] - t[j])
        b[k] = weight
    return b


def get_wc_richardson_coefficients(s_list, m: int) -> tuple[np.ndarray, np.ndarray]:
    """WC Richardson coefficients ``(b, V)`` using the closed-form weights."""
    V = setup_wc_vandermonde_matrix(s_list, m)
    b = get_wc_richardson_coefficients_closed_form(s_list)
    return b, V


def get_richardson_coefficients(s_list, p: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``V b = ê₁`` for optimized (p-specific) Richardson coefficients."""
    V = setup_vandermonde_matrix(s_list, p, m)
    e1 = np.zeros(m)
    e1[0] = 1.0
    try:
        b = np.linalg.solve(V, e1)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"Singular Vandermonde for p={p}, m={m}, s_list={list(s_list)} "
            f"(cond(V) = {np.linalg.cond(V):.2e})"
        ) from exc
    return b, V


def wc_extrapolation_valid_for_p(p: int) -> bool:
    """True when order ``p`` has an even error series suitable for WC extrapolation."""
    return sigma_parity(int(p)) == 2


def lemma57_geometric_ratio(p: int, num_points: int = 20000) -> float:
    """``sup_{0 < k ≤ 1/p} ( e/(p+1)² · (1/k − p) )^k`` (corrected Lemma 57 ratio)."""
    p = int(p)
    if p <= 0:
        return 1.0
    k_upper = 1.0 / float(p)
    k_eps = min(1e-12, 0.5 * k_upper)
    if k_upper <= k_eps:
        return 1.0
    k_vals = np.linspace(k_eps, k_upper, max(64, num_points), endpoint=False)
    inside = np.maximum(
        (math.e / ((p + 1) ** 2)) * ((1.0 / k_vals) - p),
        np.finfo(float).tiny,
    )
    log_vals = k_vals * np.log(inside)
    return float(np.exp(np.max(np.concatenate(([0.0], log_vals)))))


def _legacy_lambda_scale(m: int, p: int, num_points: int) -> float:
    """Pre-Lemma-57 (``legacy``) m-dependent supremum, capped at 1.764."""
    sigma = sigma_parity(p)
    denom = sigma * int(m)
    if denom <= 0:
        return 1.0
    k_upper = (denom - p) / (denom * (p + 1))
    if k_upper <= 0:
        return 1.0
    k_eps = min(1e-12, 0.5 * k_upper)
    if k_upper <= k_eps:
        return 1.0
    k_vals = np.linspace(k_eps, k_upper, max(64, num_points))
    log_vals = k_vals * np.log((math.e / k_vals) - math.e)
    return min(float(np.exp(np.max(log_vals))), 1.764)


def compute_lambda_scale(A: float, m: int, p: int, n: int = 1, num_points: int = 20000) -> float:
    """λ-comm ratio used in the Richardson step bound.

    ``A`` and ``n`` are accepted for API symmetry but unused under both modes
    (the bound currently depends only on ``p`` and, for ``legacy``, ``m``).
    """
    del A, n
    p = int(p)
    if LAMBDA_SCALE_MODE == "legacy":
        return _legacy_lambda_scale(m, p, num_points)
    return float(
        LEMMA57_GEOMETRIC_RATIO_BY_P.get(p, lemma57_geometric_ratio(p, num_points=num_points))
    )


def gate_overhead(p: int) -> int:
    """Suzuki exponentials per Trotter step: ``C_1 = 1``, ``C_2 = 2``, ``C_{2k} = 2·5^{k-1}``."""
    p = int(p)
    if p == 1:
        return 1
    if p == 2:
        return 2
    if p % 2 == 0:
        return 2 * (5 ** (p // 2 - 1))
    return p


def compute_steps_plane_wave(error: float, m: int, m_expr: float, p: int,
                             A: float = 1.0, n: int = 1) -> tuple[float, float]:
    """Plane-wave-style ``(trotter_steps, richardson_steps)`` (shared prefactors omitted)."""
    trotter = (error ** (-1.0 / p)) * ((2.0 / (1.0 + p)) ** (1.0 / p))
    lam = compute_lambda_scale(A, m, p, n=n)
    richardson = (lam ** richardson_lambda_comm_exponent(p)) * m_expr
    return trotter, richardson


def compute_steps_gate_depth(error: float, m: int, m_expr: float, p: int,
                             A: float = 1.0, n: int = 100) -> tuple[float, float]:
    """Total gate depth ``(trotter, richardson)``, i.e. step counts multiplied by ``C_p``."""
    c = gate_overhead(p)
    trotter = c ** (1 + 1 / p) * (error ** (-1.0 / p)) * ((2.0 / (1.0 + p)) ** (1.0 / p))
    lam = compute_lambda_scale(A, m, p, n=n)
    richardson = (c * lam) ** (1 + 1 / p) * m_expr
    return trotter, richardson


def _richardson_coefficients_for_grid(
    s_list, *, p: int, m: int, well_conditioned: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if well_conditioned:
        return get_wc_richardson_coefficients(s_list, m)
    return get_richardson_coefficients(s_list, p, m)


def _score_grid(
    *,
    p: int,
    m: int,
    q_grid: list[int],
    error: float,
    A: float,
    n: int,
    b_list: np.ndarray | None = None,
    well_conditioned: bool = False,
) -> tuple[float, float, np.ndarray]:
    """Return ``(base_val, full_val, b_list)`` for a single Richardson grid.

    ``base_val`` is the Richardson cost without the ``λ^{1+1/p}`` factor
    (matches the ``m_expr`` stored downstream); ``full_val`` includes it.
    """
    if b_list is None:
        s_list = [1.0 / q for q in q_grid]
        b_list, _ = _richardson_coefficients_for_grid(
            s_list, p=p, m=m, well_conditioned=well_conditioned,
        )
    b_tilde = b_suppressed_norm(
        b_list, q_grid, m, p, well_conditioned=well_conditioned,
    )
    den_eps = richardson_stepcount_eps_denominator(m, p)
    K = richardson_b_over_eps_prefactor(p)
    base_val = q_refinement_ratio(q_grid) * safe_power(K * b_tilde / error, den_eps)
    lam = compute_lambda_scale(A, m, p, n=n)
    full_val = (lam ** richardson_lambda_comm_exponent(p)) * base_val
    return base_val, full_val, b_list


def compute_min_samples(
    errors,
    *,
    p: int,
    m_max: int = 20,
    q_min: int = 1,
    q_max: int | None = None,
    well_conditioned_formula: bool = False,
    brute_permutations: bool = False,
    brute_permutations_max_count: int = 2_000_000,
    brute_force_b_norm1_sq_max: float | None = None,
    A: float = 1.0,
    n: int = 1,
    verbose: bool = False,
) -> tuple[list[int], list[float], list[list[int]]]:
    """Pick the cheapest Richardson schedule per target precision.

    For each ``ε`` in ``errors`` the search enumerates Richardson depths
    ``m ∈ [1, m_max]`` and, for each ``m``, either:

      * the well-conditioned grid ``q_k`` from ``get_sample_points`` (when
        ``well_conditioned_formula=True``; requires even ``p``; uses the
        prescriptive even-power Vandermonde and closed-form ``b``; ``m = 1``
        uses ``q = [1]``), or
      * every ``m``-subset of ``{q_min, …, q_max}`` for ``m ≥ 2`` (``itertools.combinations``),
        optionally permuted when ``brute_permutations`` is set and the
        permutation count is within ``brute_permutations_max_count``; for ``m = 1``
        only ``q = [1]`` is considered (no other single-``q`` grids).

    Grids whose ``‖b‖₁² > brute_force_b_norm1_sq_max`` are rejected before
    scoring (brute-force branch only). The winning grid minimizes the full
    Richardson cost ``λ^{1+1/p} · base_val``.

    Returns parallel lists ``(m_per_ε, base_per_ε, q_grid_per_ε)``.
    """
    if well_conditioned_formula and not wc_extrapolation_valid_for_p(p):
        raise ValueError(
            f"Well-conditioned extrapolation requires an even error series "
            f"(even p); got p={p}. Use optimized search instead."
        )

    sq_cap = float(brute_force_b_norm1_sq_max) if brute_force_b_norm1_sq_max is not None \
        else BRUTE_FORCE_B_NORM1_SQ_MAX_DEFAULT
    q_hi = int(q_max) if q_max is not None else int(m_max)
    q_lo = int(q_min)
    if not well_conditioned_formula and not (1 <= q_lo <= q_hi):
        raise ValueError(f"require 1 <= q_min ({q_lo}) <= q_max ({q_hi}) for brute-force search")
    q_range = list(range(q_lo, q_hi + 1))

    out_m: list[int] = []
    out_base: list[float] = []
    out_q: list[list[int]] = []

    for error in errors:
        best_full = float("inf")
        best: tuple[int, float, list[int]] | None = None

        for m in range(1, m_max + 1):
            if well_conditioned_formula:
                _, q_grid = get_sample_points(m)
                base, full, _ = _score_grid(
                    p=p, m=m, q_grid=q_grid, error=error, A=A, n=n,
                    well_conditioned=True,
                )
                if full < best_full:
                    best_full = full
                    best = (m, base, list(q_grid))
                continue

            if m == 1:
                combos = ((1,),)
            else:
                if len(q_range) < m:
                    break
                use_perm = (
                    brute_permutations
                    and math.perm(len(q_range), m) <= brute_permutations_max_count
                )
                combos = (
                    itertools.permutations(q_range, m) if use_perm
                    else itertools.combinations(q_range, m)
                )

            for combo in combos:
                q_grid = list(combo)
                s_list = [1.0 / q for q in q_grid]
                b_list, _ = get_richardson_coefficients(s_list, p, m)
                if b_norm1(b_list) ** 2 > sq_cap:
                    continue
                base, full, _ = _score_grid(
                    p=p, m=m, q_grid=q_grid, error=error, A=A, n=n, b_list=b_list,
                )
                if full < best_full:
                    best_full = full
                    best = (m, base, q_grid)

            if verbose and best is not None:
                print(f"  ε={error:.3e} m={m}: best so far m={best[0]} q={best[2]}")

        if best is None:
            raise ValueError(
                f"No admissible Richardson grid for ε={error:g} "
                f"(m_max={m_max}, q∈[{q_lo}, {q_hi}], "
                f"well_conditioned={well_conditioned_formula})."
            )

        out_m.append(best[0])
        out_base.append(best[1])
        out_q.append(best[2])

    return out_m, out_base, out_q
