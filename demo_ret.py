"""End-to-end Randomized Extrapolation Trotterization (RET, arXiv:2608.13862) on a
tiny TGV instance: estimate the kinetic-energy ratio KE(T)/KE(0) by sampling
Hadamard tests of Richardson-extrapolated second-order Trotter circuits of the
Hermitian dilation of the space-time system, with f(x) = 1/x from the CKS Fourier
series -- Algorithm 2 / Task 1(ii) of the paper, run on a statevector.

Instance: N = 4, N_C = 1, Re = 10, Nt = 3  ->  L is 64 x 64, dilated H is 128 x 128
(7 system qubits + 1 Hadamard ancilla).  Estimated quantity (Task 1(i), Algorithm 1):
the overlap <phi| L^{-1} b> of the solution with the exact-solution vector, i.e. the
L2-error metric of the challenge.  Everything is exact-arithmetic except the
Bernoulli outcomes of the Hadamard tests, which are drawn from the exact
expectation values -- i.e. the quantum computer is simulated exactly.

Outputs results/demo_ret.json: exact target, RET estimate vs samples, Fourier-,
Richardson- and shot-error budget, and the plain-Trotter comparison.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tgv_operator import TGV  # noqa: E402
from resources import pieces, inverse_fourier_series  # noqa: E402
import richardson as rich  # noqa: E402

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def dense(apply, dim):
    return np.column_stack([apply(e) for e in np.eye(dim, dtype=complex)])


def main(N=4, Re=10.0, NC=1, eps=0.1, n_samples=1_000_000, seed=0, prune=1e-5):
    """Task 1(i): overlap of the solution with the exact-solution vector,
    <1,phi| f(H) |0,b> = ||L|| <phi| L^{-1} b>, phi = normalised exact Carleman vector of
    omega(T) in the last slice.  One Hadamard-test circuit per sample."""
    t0 = time.time()
    g = TGV(N=N, Re=Re, NC=NC)
    ps = pieces(g)
    d = g.dim
    L = dense(g.L_apply, d)
    nL = np.linalg.norm(L, 2)
    H_terms = []
    for p in ps:
        A = dense(p.apply, d) / nL
        H_terms.append(np.block([[np.zeros((d, d)), A], [A.conj().T, np.zeros((d, d))]]))
    H = sum(H_terms)
    evals = np.linalg.eigvalsh(H)
    kappa = 1.0 / np.min(np.abs(evals))
    b = g.rhs_global(g.initial_state()); b = b / np.linalg.norm(b)
    psi = np.concatenate([b, np.zeros(d)])                  # |0>|b>
    phi_c = np.zeros(d, dtype=complex)
    phi_c[g.Nt * g.dim_c: g.Nt * g.dim_c + g.n2] = g.omega_exact_hat(g.T)
    phi_c /= np.linalg.norm(phi_c)
    phi = np.concatenate([np.zeros(d), phi_c])              # |1>|phi>
    Hinv = np.linalg.inv(H)
    target = complex(phi.conj() @ Hinv @ psi)               # = ||L|| <phi|L^{-1} b>
    # classical cross-check: <phi|L^{-1}b> via the march
    xfull = g.L_solve(b)
    assert abs(complex(phi_c.conj() @ xfull) * nL - target) < 1e-8 * abs(target)

    # Fourier series for 1/x  (Task 1(i): eps_F = eps/3)
    eps_F = eps / 3
    fs = inverse_fourier_series(kappa, eps_F)
    Y, Z, dy, dz = fs["Y"], fs["Z"], fs["dy"], fs["dz"]
    M = int(math.ceil(Y / (2 * dy))); J = 2 * M + 1
    zs = np.arange(-Z, Z + 1e-12, dz)
    wz = dz * zs * np.exp(-zs**2 / 2)
    simpson = np.full(J, 2.0); simpson[1::2] = 4.0; simpson[0] = simpson[-1] = 1.0
    simpson *= dy / 3
    cj = (1j / math.sqrt(2 * math.pi)) * np.outer(simpson, wz).ravel()
    tj = np.outer(np.arange(J) * dy, zs).ravel()
    c_full = float(np.sum(np.abs(cj)))
    keep = np.abs(cj) > prune * np.max(np.abs(cj))
    cj, tj = cj[keep], tj[keep]
    c_l1 = float(np.sum(np.abs(cj)))
    fH = sum(c * _expm_h(H, t) for c, t in zip(cj, tj))
    fourier_err = float(np.linalg.norm(fH - Hinv, 2))

    eps_R = eps / (3 * c_l1)
    m, base, qgrid = rich.compute_min_samples([eps_R], p=2, m_max=8, q_max=12, well_conditioned_formula=True)
    m, base, qgrid = m[0], base[0], qgrid[0]
    bcoef, _ = rich.get_wc_richardson_coefficients([1.0 / q for q in qgrid], m)
    b1 = float(np.sum(np.abs(bcoef)))
    tmax = float(np.max(np.abs(tj)))
    r0 = _choose_r0(H, H_terms, tmax, qgrid, bcoef, eps_R)

    # one scalar per circuit: <phi| P^{1/s_j}(s_j t_k) |psi>
    vals = np.empty((len(tj), len(qgrid)), dtype=complex)
    steps = np.empty((len(tj), len(qgrid)), dtype=int)
    # group by Trotter step count to reuse matrix powers
    for j, q in enumerate(qgrid):
        r = int(math.ceil(r0 / q))
        for k, t in enumerate(tj):
            nstep = max(1, int(math.ceil(r * abs(t) / tmax)))
            steps[k, j] = nstep
            vals[k, j] = phi.conj() @ (_trotter2(H_terms, t, nstep) @ psi)
    coef = np.outer(cj, bcoef)                                  # (k, j)
    S = c_l1 * b1
    est_exact = complex(np.sum(coef * vals))                    # Fourier-Richardson value
    rng = np.random.default_rng(seed)
    p = (np.abs(coef) / S).ravel()
    idx = rng.choice(coef.size, size=n_samples, p=p)
    ph = (coef.ravel()[idx] / np.abs(coef.ravel()[idx]))
    gv = vals.ravel()[idx] * ph
    xr = rng.random(n_samples) < (1 + np.real(gv)) / 2
    xi = rng.random(n_samples) < (1 + np.imag(gv)) / 2
    mu = S * ((2 * xr - 1) + 1j * (2 * xi - 1))
    running = np.cumsum(mu) / np.arange(1, n_samples + 1)
    checkpoints = [10**k for k in range(1, int(math.log10(n_samples)) + 1)]
    max_steps = int(steps.max())
    out = dict(N=N, Re=Re, NC=NC, Nt=g.Nt, dim=d, n_qubits=int(math.log2(2 * d)) + 1, kappa=float(kappa),
               normL=float(nL), eps=eps, eps_F=eps_F, eps_R=eps_R, c_l1=c_l1, c_l1_unpruned=c_full, t_max=tmax,
               K_fourier=int(len(cj)), fourier_operator_error=fourier_err,
               richardson=dict(m=m, qgrid=qgrid, b=bcoef.tolist(), b_norm1=b1, r0=r0, max_trotter_steps=max_steps,
                               operator_error_at_tmax=_RICH_ERR.get('err')),
               S=S, target_re=target.real, target_im=target.imag, target_abs=abs(target),
               overlap_exact=abs(target) / nL,                   # |<phi|L^{-1}b>|
               fourier_richardson_value=[est_exact.real, est_exact.imag],
               fourier_richardson_error=abs(est_exact - target),
               estimate_vs_samples={str(n): [float(running[n - 1].real), float(running[n - 1].imag)] for n in checkpoints},
               error_vs_samples={str(n): float(abs(running[n - 1] - target)) for n in checkpoints},
               hoeffding_samples_for_eps=2 * S**2 * math.log(40) / eps**2,
               seconds=time.time() - t0)
    with open(os.path.join(RESULTS, "demo_ret.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("richardson",)}, indent=1, default=float))
    print("richardson:", out["richardson"])
    return out


_EIG = {}


def _expm_h(Hm, t):
    key = id(Hm)
    if key not in _EIG:
        _EIG[key] = np.linalg.eigh(Hm)
    w, V = _EIG[key]
    return (V * np.exp(-1j * w * t)) @ V.conj().T


def _trotter2(H_terms, t, r):
    """Second-order (Strang) product formula for exp(-iHt) with r steps."""
    dt = t / r
    halves = [_expm_h(Hk, dt / 2) for Hk in H_terms]
    step = np.eye(H_terms[0].shape[0], dtype=complex)
    for U in halves:
        step = U @ step
    for U in reversed(halves):
        step = U @ step
    return np.linalg.matrix_power(step, r)


_RICH_ERR = {}


def _choose_r0(H, H_terms, tmax, qgrid, bcoef, eps_R):
    """Smallest base step number whose extrapolated Strang circuit meets eps_R at t_max
    (measured operator-norm error, recorded in _RICH_ERR)."""
    exact = _expm_h(H, tmax)
    r0 = 2
    while r0 <= 65536:
        approx = sum(bj * _trotter2(H_terms, tmax, max(1, int(math.ceil(r0 / q)))) for bj, q in zip(bcoef, qgrid))
        err = float(np.linalg.norm(approx - exact, 2))
        _RICH_ERR["r0"], _RICH_ERR["err"] = r0, err
        if err < eps_R:
            return r0
        r0 *= 2
    return r0


if __name__ == "__main__":
    main()
