"""Resource model: RET (arXiv:2608.13862) with f(x) = 1/x on the Hermitian dilation
of the space-time system L of tgv_operator.py.

Pipeline per (N, N_C, Re):
  1. Term decomposition  L = sum_g coef_g * Piece_g  (Gamma pieces, each a
     structured operator whose dilated exponential is efficiently implementable).
  2. Spectral data: ||L||, ||L^{-1}||, kappa;  Lambda = sum ||H_g||;
     alpha_comm^(2), alpha_comm^(3) from the dilation identities
         [H_A, H_B]        = diag(A B^+ - B A^+,  A^+ B - B^+ A)
         [H_A, [H_B, H_C]] = offdiag(A K2 - K1 A),  K1 = B C^+ - C B^+, K2 = B^+ C - C^+ B
     so every norm is a norm of a composite operator on the UNdilated space.
  3. Fourier series for 1/x on [1/kappa, 1] (Childs-Kothari-Somma form) with
     explicit constants -> c = sum|c_k|, t_max.
  4. RET cost (Theorem 26 structure with the explicit constants of Lemma 15 /
     Theorem 20 of the paper, and the fixed-order bound of Theorem 38 so that only
     alpha^(3) and Lambda are needed):
         r_max   = base(eps_R) * max{1, (a_max Ups)^{3/2} sqrt(alpha3) T^{3/2} + a_max Ups Lambda T}
         exps    = Ups * Gamma * r_max            (Trotter exponentials per sample)
         samples = 2 (c ||b||_1)^{2q} log(2/delta) / eps^2 ,  q = 1 (Task 1(i)) or 2 (Task 1(ii))
  5. Gate model (stated assumption): each dilated-term exponential costs
         T_exp = 4 * (8 n_reg + b_arith^2) + 3 log2(1/eps_rot)   T gates
     (two controlled structure-unitaries with b-bit fixed-point arithmetic for the
     k-dependent symbol, one synthesised rotation).

All numbers land in results/<tag>.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tgv_operator import TGV  # noqa: E402
import richardson as rich      # noqa: E402  (the paper's step-count code, vendored)

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


# --------------------------------------------------------------------------
# 1. term decomposition
# --------------------------------------------------------------------------
class Piece:
    """coef * (time-structure) (x) (Carleman-space operator)."""

    def __init__(self, g: TGV, name: str, coef: float, where: str, op, op_adj):
        self.g, self.name, self.coef, self.where = g, name, coef, where
        self.op, self.op_adj = op, op_adj  # act on one Carleman slice

    def apply(self, x):
        g = self.g
        X = g.slices(x)
        Y = np.zeros_like(X)
        if self.where == "all":            # identity on every slice
            return self.coef * x
        for k in range(1, g.Nt + 1):
            src = X[k] if self.where == "diag" else X[k - 1]
            Y[k] = self.op(src)
        return self.coef * Y.ravel()

    def apply_adj(self, x):
        g = self.g
        X = g.slices(x)
        Y = np.zeros_like(X)
        if self.where == "all":
            return np.conj(self.coef) * x
        for k in range(1, g.Nt + 1):
            dst = k if self.where == "diag" else k - 1
            Y[dst] += self.op_adj(X[k])
        return np.conj(self.coef) * Y.ravel()


def pieces(g: TGV) -> list[Piece]:
    th, dt = g.theta, g.dt
    f1 = g.f1.ravel()
    n2 = g.n2

    def only1(fn):  # act on omega block, zero elsewhere
        def h(x):
            y = np.zeros_like(x); y[:n2] = fn(x[:n2]); return y
        return h

    def F1(x):  return only1(lambda w: f1 * w)(x)
    def F1a(x): return only1(lambda w: np.conj(f1) * w)(x)

    ps = [Piece(g, "I", 1.0, "all", None, None),
          Piece(g, "E_diag(F1)", -th * dt, "diag", F1, F1a),
          Piece(g, "E_sub(I)", -1.0, "sub", lambda x: x.copy(), lambda x: x.copy()),
          Piece(g, "E_sub(F1)", -(1 - th) * dt, "sub", F1, F1a)]
    if g.NC == 2:
        def F2x(x):
            y = np.zeros_like(x); y[:n2] = g.F2x(x[n2:].reshape(n2, n2)); return y
        def F2xa(x):
            y = np.zeros_like(x); y[n2:] = g.F2x_adj(x[:n2]); return y
        def F2y(x):
            y = np.zeros_like(x); y[:n2] = g.F2y(x[n2:].reshape(n2, n2)); return y
        def F2ya(x):
            y = np.zeros_like(x); y[n2:] = g.F2y_adj(x[:n2]); return y
        ks = (f1[:, None] + f1[None, :]).ravel()
        def KS(x):
            y = np.zeros_like(x); y[n2:] = ks * x[n2:]; return y
        def KSa(x):
            y = np.zeros_like(x); y[n2:] = np.conj(ks) * x[n2:]; return y
        for where, c in (("diag", -th * dt), ("sub", -(1 - th) * dt)):
            tag = "E_diag" if where == "diag" else "E_sub"
            ps += [Piece(g, f"{tag}(F2x)", c, where, F2x, F2xa),
                   Piece(g, f"{tag}(F2y)", c, where, F2y, F2ya),
                   Piece(g, f"{tag}(F1+F1)", c, where, KS, KSa)]
    return ps


def check_decomposition(g: TGV, ps: list[Piece], rng) -> float:
    x = rng.standard_normal(g.dim) + 1j * rng.standard_normal(g.dim)
    y = sum(p.apply(x) for p in ps)
    return float(np.linalg.norm(y - g.L_apply(x)) / np.linalg.norm(x))


# --------------------------------------------------------------------------
# 2. norms
# --------------------------------------------------------------------------
def opnorm(apply, adj, dim, tol=2e-2, maxiter=60, seed=0) -> float:
    """Spectral norm by power iteration on A^+ A (accurate to ~tol relative)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    last = 0.0
    for it in range(maxiter):
        w = adj(apply(v))
        val = float(np.linalg.norm(w))
        if val == 0.0:
            return 0.0
        if it > 3 and abs(val - last) < tol * val:
            last = val
            break
        last = val
        v = w / val
    return math.sqrt(last)


def commutator_norms(g: TGV, ps: list[Piece], order3: bool = True, tol=2e-2, log=print):
    """Lambda, alpha^(2), alpha^(3) for the dilated, normalised terms H_g."""
    nL = opnorm(g.L_apply, g.L_adj, g.dim, tol=tol)
    nLinv = opnorm(g.L_solve, g.L_solve_adj, g.dim, tol=tol)
    kappa = nL * nLinv
    G = len(ps)
    norms = [opnorm(p.apply, p.apply_adj, g.dim, tol=tol) for p in ps]
    Lam = sum(norms) / nL

    def K_ops(a: Piece, b: Piece):
        # K1 = A B^+ - B A^+   (on V),  K2 = A^+ B - B^+ A
        K1 = lambda x: a.apply(b.apply_adj(x)) - b.apply(a.apply_adj(x))
        K2 = lambda x: a.apply_adj(b.apply(x)) - b.apply_adj(a.apply(x))
        return K1, K2

    alpha2 = 0.0
    pair_norm = {}
    for i, j in itertools.combinations(range(G), 2):
        K1, K2 = K_ops(ps[i], ps[j])
        # K1, K2 are anti-Hermitian: rmatvec = -matvec
        n1 = opnorm(K1, lambda x: -K1(x), g.dim, tol=tol)
        n2_ = opnorm(K2, lambda x: -K2(x), g.dim, tol=tol)
        pair_norm[(i, j)] = max(n1, n2_)
        alpha2 += 2 * pair_norm[(i, j)]        # (i,j) and (j,i)
    alpha2 /= nL**2
    log(f"  ||L||={nL:.4g} ||L^-1||={nLinv:.4g} kappa={kappa:.4g} Lambda={Lam:.4g} alpha2={alpha2:.4g}")

    alpha3 = None
    alpha3_bound = 0.0
    # cheap rigorous bound: ||[A,[B,C]]|| <= 2 ||A|| ||[B,C]||
    for (i, j), v in pair_norm.items():
        alpha3_bound += 2 * sum(norms) * 2 * v      # sum over a of 2||H_a|| * (both orders)
    alpha3_bound /= nL**3
    if order3:
        alpha3 = 0.0
        t0 = time.time()
        for a in range(G):
            for i, j in itertools.combinations(range(G), 2):
                if pair_norm[(i, j)] == 0.0:
                    continue
                K1, K2 = K_ops(ps[i], ps[j])
                A = ps[a]
                M = lambda x, A=A, K1=K1, K2=K2: A.apply(K2(x)) - K1(A.apply(x))
                Ma = lambda x, A=A, K1=K1, K2=K2: -K2(A.apply_adj(x)) + A.apply_adj(K1(x))
                nrm = opnorm(M, Ma, g.dim, tol=tol)
                alpha3 += 2 * nrm                    # (i,j) and (j,i)
        alpha3 /= nL**3
        log(f"  alpha3={alpha3:.4g} (bound {alpha3_bound:.4g}) in {time.time()-t0:.0f}s")
    return dict(normL=nL, normLinv=nLinv, kappa=kappa, Lambda=Lam, alpha2=alpha2,
                alpha3=alpha3, alpha3_bound=alpha3_bound, Gamma=G,
                piece_norms={p.name: n / nL for p, n in zip(ps, norms)})


# --------------------------------------------------------------------------
# 3. Fourier series for 1/x  (Childs-Kothari-Somma, Lemma 11 form)
#    1/x = (i/sqrt(2 pi)) int_0^inf dy int_-inf^inf dz  z e^{-z^2/2} e^{-i x y z}
# --------------------------------------------------------------------------
_FS_CACHE = {}


def inverse_fourier_series(kappa: float, eps: float):
    """Return (c_l1, t_max, K_terms, achieved_error, params) for the smallest grid
    found by a coarse search that approximates 1/x to sup-error eps on
    [1/kappa, 1] (and by oddness on [-1, -1/kappa]).  Since c ~ Y and t_max = Y Z,
    the search walks (Y, Z) upward and stops at the first pair that meets eps."""
    key = (round(float(kappa), 3), float(f"{eps:.3g}"))
    if key in _FS_CACHE:
        return _FS_CACHE[key]
    xs = np.concatenate([np.geomspace(1 / kappa, 1, 300), np.linspace(1 / kappa, 1, 300)])
    best = None
    for Yfac in (1.0, 1.25, 1.5, 2.0):
        if best is not None:
            break
        Y = Yfac * kappa * math.sqrt(2 * math.log(kappa / eps))
        for Zfac in (1.0, 1.25, 1.5):
            if best is not None:
                break
            # z-tail must be cut at eps / Y because the y-integral has length ~Y
            Z = Zfac * math.sqrt(2 * math.log(4 * Y / eps))
            for dzfac in (1.0, 0.5, 0.25):
                if best is not None:
                    break
                dz = dzfac * math.pi / (Y + 1.0)   # resolve e^{-i x y z} with |xy| <= Y
                for dy_fac in (1.0, 0.5, 0.25, 0.125, 1 / 16, 1 / 32):
                    dy = dy_fac * min(math.pi / (2 * Z), 1.0)
                    M = int(math.ceil(Y / (2 * dy)))                # Simpson: J = 2M+1 points
                    J = 2 * M + 1
                    zs = np.arange(-Z, Z + 1e-12, dz)
                    w = dz * zs * np.exp(-zs**2 / 2)
                    # f(x) = i/sqrt(2pi) * sum_k w_k * sum_j c_j exp(-i x z_k j dy),
                    # Simpson weights c_j = (dy/3)(1,4,2,4,...,4,1); both partial sums are
                    # geometric series -> closed form
                    a = np.outer(xs, zs) * dy
                    with np.errstate(divide="ignore", invalid="ignore"):
                        S_all = (1 - np.exp(-1j * a * J)) / (1 - np.exp(-1j * a))
                        S_odd = np.exp(-1j * a) * (1 - np.exp(-2j * a * M)) / (1 - np.exp(-2j * a))
                    small = np.abs(a) < 1e-12
                    S_all = np.where(small, J, S_all)
                    S_odd = np.where(small, M, S_odd)
                    Gs = (dy / 3) * (2 * S_all + 2 * S_odd - 1 - np.exp(-1j * a * (J - 1)))
                    approx = (1j / math.sqrt(2 * math.pi)) * (Gs * w[None, :]).sum(1)
                    err = float(np.max(np.abs(approx - 1 / xs)))
                    if err <= eps:
                        c = float((dy / 3) * (6 * M) * np.sum(np.abs(w)) / math.sqrt(2 * math.pi))
                        tmax = float((J - 1) * dy * Z)
                        K = int(J * len(zs))
                        cand = (c * tmax, dict(c=c, t_max=tmax, K=K, err=err, Y=Y, Z=Z, dy=dy, dz=dz))
                        if best is None or cand[0] < best[0]:
                            best = cand
                        break  # finer dy not needed
    if best is None:
        raise RuntimeError("no Fourier grid met the target")
    _FS_CACHE[key] = best[1]
    return best[1]


# --------------------------------------------------------------------------
# 4-5. RET cost assembly
# --------------------------------------------------------------------------
GATE_MODEL = dict(b_arith=16, eps_rot_factor=1e-2, T_per_toffoli=4, t_gate_seconds=1e-5,
                  p=2, a_max=0.5, Upsilon=2, delta=0.05)


def ret_cost(spec: dict, n_qubits_system: int, eps: float, task: int = 2, gm: dict = GATE_MODEL):
    p, a_max, Ups = gm["p"], gm["a_max"], gm["Upsilon"]
    kappa, Lam, alpha3, G = spec["kappa"], spec["Lambda"], spec["alpha3"] or spec["alpha3_bound"], spec["Gamma"]
    ftil = max(1.0, kappa)                           # ||f(H)|| for f = 1/x on the normalised spectrum
    q = 2 if task == 2 else 1
    eps_F = eps / (9 * ftil) if task == 2 else eps / 3
    fs = inverse_fourier_series(kappa, eps_F)
    c, T = fs["c"], fs["t_max"]
    eps_R = eps / (9 * ftil * c) if task == 2 else eps / (3 * c)
    # Richardson schedule (paper's search code; p = 2 avoids the README-flagged R_p bug)
    m, base, qgrid = rich.compute_min_samples([eps_R], p=p, m_max=12, q_max=12,
                                              well_conditioned_formula=True)
    m, base, qgrid = m[0], base[0], qgrid[0]
    b, _ = rich.get_wc_richardson_coefficients([1.0 / qq for qq in qgrid], m)
    b1 = rich.b_norm1(b)
    # fixed-order (Theorem 38) replacement of (lambda_comm T)^{1+1/p}
    growth = (a_max * Ups) ** (1 + 1 / p) * math.sqrt(alpha3) * T ** (1 + 1 / p) + a_max * Ups * Lam * T
    r_max = base * max(1.0, growth)
    exps_per_sample = Ups * G * r_max
    S = c * b1
    samples = 2 * S ** (2 * q) * math.log(2 / gm["delta"]) / eps**2
    # gate model
    n_reg = n_qubits_system
    eps_rot = gm["eps_rot_factor"] * eps_R / max(1.0, exps_per_sample)   # per-rotation synthesis error budget
    T_exp = gm["T_per_toffoli"] * (8 * n_reg + gm["b_arith"] ** 2) + 3 * math.log2(1 / eps_rot)
    T_per_sample = exps_per_sample * T_exp
    T_total = T_per_sample * samples
    qubits = n_reg + 2 + 3 * gm["b_arith"]            # + dilation + Hadamard ancilla + arithmetic workspace
    return dict(task=task, eps=eps, eps_F=eps_F, eps_R=eps_R, fourier=fs, m=m, qgrid=qgrid,
                b_norm1=b1, richardson_base=base, growth=growth, r_max=r_max,
                exps_per_sample=exps_per_sample, samples=samples, T_exp=T_exp,
                T_per_sample=T_per_sample, T_total=T_total, logical_qubits=qubits,
                depth_T_per_sample=T_per_sample,
                time_serial_s=T_total * gm["t_gate_seconds"],
                time_parallel_1000_s=T_total * gm["t_gate_seconds"] / 1000)


# --------------------------------------------------------------------------
def run(N: int, NC: int, Re: float, order3: bool, tol: float, eps_list, tag: str | None = None, do_ret: bool = True):
    g = TGV(N=N, Re=Re, NC=NC)
    rng = np.random.default_rng(0)
    ps = pieces(g)
    res = dict(N=N, NC=NC, Re=Re, Nt=g.Nt, dt=g.dt, nu=g.nu, dim_carleman=g.dim_c, dim=g.dim,
               n_qubits_system=int(math.ceil(math.log2(g.dim))),   # space-time Carleman register (the
               # dilation and Hadamard ancillas are added on top of this in ret_cost)
               decomposition_residual=check_decomposition(g, ps, rng))
    print(f"[N={N} NC={NC} Re={Re}] dim={g.dim} Nt={g.Nt} n_sys={res['n_qubits_system']} "
          f"decomp-residual={res['decomposition_residual']:.1e}", flush=True)
    t0 = time.time()
    spec = commutator_norms(g, ps, order3=order3, tol=tol)
    spec["seconds"] = time.time() - t0
    res["spectral"] = spec
    res["ret"] = {}
    for eps in (eps_list if do_ret else []):
        for task in (1, 2):
            res["ret"][f"eps{eps:g}_task{task}"] = ret_cost(spec, res["n_qubits_system"], eps, task)
    tag = tag or f"N{N}_NC{NC}_Re{Re:g}"
    with open(os.path.join(RESULTS, tag + ".json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"  wrote {tag}.json in {time.time()-t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--NC", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--Re", type=float, nargs="+", default=[10.0, 100.0])
    ap.add_argument("--no-order3", action="store_true")
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-2, 1e-3])
    ap.add_argument("--no-ret", action="store_true", help="spectral quantities only (RET costs are assembled in fit_and_plot.py)")
    a = ap.parse_args()
    for N in a.N:
        for NC in a.NC:
            for Re in a.Re:
                run(N, NC, Re, order3=not a.no_order3, tol=a.tol, eps_list=a.eps, do_ret=not a.no_ret)
