"""Error scaling of the discretised, Carleman-truncated TGV solver against the
exact solution -- the two error metrics named by the Airbus statement (sec. 4.1):
  (i)  L2 error of the velocity field,
  (ii) kinetic-energy decay,
plus the Carleman-truncation error on a *perturbed* TGV (the pure TGV has
J(psi, omega) = 0 identically, so N_C = 1 is already exact for it).

Grid rule (stated assumption, Jennings et al. 2512.03758 beta = 3/4):
    N(Re) = 2^ceil(log2(eta Re^{3/4})),  eta = 8 / 10^{3/4}  ->  N = 8, 64, 256 at Re = 10, 100, 1000.
Writes results/error_scaling.json.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tgv_operator import TGV  # noqa: E402

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
ETA = 8 / 10 ** 0.75


def grid_for(Re: float) -> int:
    return int(2 ** math.ceil(math.log2(ETA * Re ** 0.75)))


def velocity_hat(g: TGV, om_hat: np.ndarray):
    """u_hat, v_hat from omega_hat via psi = -lap^{-1} omega, u = psi_y, v = -psi_x."""
    w = om_hat.reshape(g.N, g.N)
    lam_safe = np.where(g.lam > 0, g.lam, np.inf)
    psi = w / lam_safe
    return (1j * g.sy * psi), (-1j * g.sx * psi)


def l2_velocity_error(g: TGV, om_hat: np.ndarray, t: float) -> float:
    u, v = velocity_hat(g, om_hat)
    ue, ve = velocity_hat(g, g.omega_exact_hat(t))
    num = np.sum(np.abs(u - ue) ** 2 + np.abs(v - ve) ** 2)
    den = np.sum(np.abs(ue) ** 2 + np.abs(ve) ** 2)
    return math.sqrt(float(num / den))


def run_case(Re: float, N: int, NC: int, perturb: float, cfl: float = 0.5, T: float = 1.0):
    g = TGV(N=N, Re=Re, NC=NC, cfl=cfl, T=T)
    x0 = g.initial_state(perturb=perturb)
    x, hist = g.march(x0, store=True)
    ke0 = g.kinetic_energy(x0[: g.n2])
    ke_t = [g.kinetic_energy(h[: g.n2]) / ke0 for h in hist]
    times = [k * g.dt for k in range(g.Nt + 1)]
    out = dict(Re=Re, N=N, NC=NC, Nt=g.Nt, dt=g.dt, nu=g.nu, perturb=perturb,
               ke_ratio=ke_t, times=times,
               ke_ratio_exact=[math.exp(-4 * g.nu * t) for t in times])
    if perturb == 0.0:
        out["l2_err_T"] = l2_velocity_error(g, x[: g.n2], T)
        out["ke_err_T"] = abs(ke_t[-1] - math.exp(-4 * g.nu * T))
        # fitted decay rate from the log-slope of KE
        slope = -np.polyfit(times[1:], np.log(ke_t[1:]), 1)[0]
        out["decay_rate_fit"] = float(slope)
        out["decay_rate_exact"] = 4 * g.nu
    return out


def reference_solution(Re: float, N: int, perturb: float, T: float = 1.0):
    """High-resolution reference for the perturbed case: same grid, N_C = 2, but with a
    pseudo-spectral nonlinear march (exact quadratic term on the truncated basis)."""
    g = TGV(N=N, Re=Re, NC=1, cfl=0.1, T=T)
    w = g.initial_state(perturb=perturb)[: g.n2]
    th, dt = g.theta, g.dt
    f1 = g.f1.ravel()
    for _ in range(g.Nt):
        # Crank-Nicolson on the linear part, explicit (Heun) on the nonlinear term
        nl = g.F2_pair(w, w)
        w_star = (w * (1 + (1 - th) * dt * f1) + dt * nl) / (1 - th * dt * f1)
        nl2 = g.F2_pair(w_star, w_star)
        w = (w * (1 + (1 - th) * dt * f1) + 0.5 * dt * (nl + nl2)) / (1 - th * dt * f1)
    return g, w


def main():
    res = dict(eta=ETA, grid_rule="N = 2^ceil(log2(eta Re^0.75))", cases=[], perturbed=[])
    # (i)+(ii): pure TGV at the Re-dependent grid, N_C = 1 (exact nonlinearity), and a
    # refinement sweep to expose the spatial/temporal error orders
    for Re in (10.0, 100.0, 1000.0):
        N = grid_for(Re)
        for NN in sorted({N, max(8, N // 2), min(256, 2 * N)}):
            c = run_case(Re, NN, 1, 0.0)
            c["at_grid_rule"] = (NN == N)
            res["cases"].append(c)
            print(f"Re={Re:g} N={NN:3d} Nt={c['Nt']:3d} L2err={c['l2_err_T']:.3e} KEerr={c['ke_err_T']:.3e} "
                  f"rate {c['decay_rate_fit']:.4f} vs {c['decay_rate_exact']:.4f}", flush=True)
    # Carleman truncation error on a perturbed TGV: N_C = 1 vs 2 against a nonlinear reference
    for Re in (10.0, 100.0):
        for N in (8, 16):
            for pert in (0.05, 0.2):
                gref, wref = reference_solution(Re, N, pert)
                row = dict(Re=Re, N=N, perturb=pert)
                for NC in (1, 2):
                    g = TGV(N=N, Re=Re, NC=NC, cfl=0.5)
                    x = g.march(g.initial_state(perturb=pert))
                    w = x[: g.n2]
                    row[f"NC{NC}_rel_l2"] = float(np.linalg.norm(w - wref) / np.linalg.norm(wref))
                    row[f"NC{NC}_ke_ratio"] = g.kinetic_energy(w) / g.kinetic_energy(g.initial_state(perturb=pert)[: g.n2])
                row["ref_ke_ratio"] = gref.kinetic_energy(wref) / gref.kinetic_energy(gref.initial_state(perturb=pert)[: gref.n2])
                res["perturbed"].append(row)
                print(f"perturbed Re={Re:g} N={N} eps_pert={pert}: NC1 err={row['NC1_rel_l2']:.3e} NC2 err={row['NC2_rel_l2']:.3e}", flush=True)
    with open(os.path.join(RESULTS, "error_scaling.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote results/error_scaling.json")


if __name__ == "__main__":
    main()
