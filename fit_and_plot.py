"""Collect results/*.json, fit the N-dependence of the spectral quantities, build
the Reynolds-scaling estimates at the grid rule N(Re), draw the two proposal
figures and emit numbers.tex (LaTeX macros) + results/summary.json.

fig1: the Airbus plot -- (a) T-count / time-to-solution vs Re, (b) logical qubits
      vs Re, (c) error vs Re (L2 velocity, KE decay, RET budget).
fig2: crossover -- RET total T-count vs block-encoding QLSA (Jennings et al.,
      10^8 gates/query, q_Q = alpha_A kappa_A log(1/eps), x 1/eps amplitude
      estimation) vs the classical FV solver, as a function of Re, both tasks.
"""
from __future__ import annotations

import glob
import re
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "legend.fontsize": 7, "axes.titlesize": 9, "axes.labelsize": 8})

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from resources import ret_cost, GATE_MODEL  # noqa: E402
from error_scaling import grid_for  # noqa: E402

RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "..", "figs")
os.makedirs(FIGS, exist_ok=True)


def load():
    rows = []
    for f in glob.glob(os.path.join(RESULTS, "N*_NC*_Re*.json")):
        rows.append(json.load(open(f)))
    return rows


def powerfit(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 2 or np.any(ys <= 0):
        return float("nan"), float("nan")
    p = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(p[0]), float(math.exp(p[1]))


def extrapolate_spec(rows, NC, Re, N_target):
    """Fit kappa, Lambda, alpha3 ~ A N^b from the computed rows at this (NC, Re)
    and evaluate at N_target; return a spec dict for ret_cost and the fits."""
    sub = sorted([r for r in rows if r["NC"] == NC and r["Re"] == Re], key=lambda r: r["N"])
    Ns = [r["N"] for r in sub]
    out, fits = {}, {}
    for key in ("kappa", "Lambda", "alpha3", "alpha2"):
        vals = [r["spectral"][key] if r["spectral"][key] is not None else r["spectral"]["alpha3_bound"] for r in sub]
        b, A = powerfit(Ns, vals)
        fits[key] = dict(exponent=b, prefactor=A, N=Ns, values=vals)
        exact = [r for r in sub if r["N"] == N_target]
        out[key] = exact[0]["spectral"][key] if exact and exact[0]["spectral"][key] is not None else A * N_target**b
    out["alpha3_bound"] = out["alpha3"]
    out["Gamma"] = sub[0]["spectral"]["Gamma"]
    # Carleman-generator norm Lambda_C = sum ||F-pieces|| (undo the theta*dt coefficient and ||L|| normalisation)
    lamC = []
    for r in sub:
        pn, nL, dt = r["spectral"]["piece_norms"], r["spectral"]["normL"], r["dt"]
        lamC.append(sum(v * nL / (0.5 * dt) for k, v in pn.items() if k.startswith("E_diag(F")))
    b, A = powerfit(Ns, lamC)
    fits["LambdaC"] = dict(exponent=b, prefactor=A, N=Ns, values=lamC)
    out["LambdaC"] = lamC[Ns.index(N_target)] if N_target in Ns else A * N_target**b
    # dimension at the target grid: Carleman dim x (Nt+1) x 2 (dilation)
    n2 = N_target**2
    dim_c = n2 + (N_target**4 if NC == 2 else 0)
    Nt = int(math.ceil(1.0 / (0.5 * (2 * math.pi / N_target) / 2)))   # cfl=0.5, |Uc|+V0 = 2, T=1
    n_sys = int(math.ceil(math.log2(2 * dim_c * (Nt + 1))))
    out["Nt"], out["n_sys"], out["dim"] = Nt, n_sys, dim_c * (Nt + 1)
    out["extrapolated"] = N_target not in Ns
    return out, fits


def classical_cost(N, NC, Nt):
    """Flops of the same theta-scheme classically: per step ~ 20 N^2 log2 N (FFT-based
    Fourier-multiplier solves, N_C = 1) or ~ 20 N^4 log2 N (Carleman N_C = 2 emulation)."""
    per_step = 20 * (N**2 if NC == 1 else N**4) * math.log2(N)
    return per_step * Nt


def qlsa_cost(kappa, eps, Lambda_norm, gates_per_query=1e8):
    """Block-encoding QLSA comparator (Jennings et al. 2512.03758): q_Q = alpha_A kappa log(1/eps)
    queries, 1e8 gates per query in D = 2, times 1/eps amplitude-estimation repetitions for
    a scalar observable."""
    return Lambda_norm * kappa * math.log(1 / eps) * gates_per_query / eps


def schro_cost(LambdaC, n_sys, eps, Gamma, gm=GATE_MODEL, T=1.0):
    """Bound-level estimate of the inversion-free arm: RET with f = exp on the
    Schrodingerised Carleman generator. c = 1, t_max = T ||C||-scale, generic
    sqrt(alpha3) <= 2 Lambda^{3/2}; quadratic observable (q = 2) with S = ||b||_1."""
    import richardson as rich
    eps_R = eps / 9
    m, base, qgrid = rich.compute_min_samples([eps_R], p=2, m_max=12, q_max=12, well_conditioned_formula=True)
    m, base, qgrid = m[0], base[0], qgrid[0]
    bco, _ = rich.get_wc_richardson_coefficients([1.0 / q for q in qgrid], m)
    b1 = float(np.sum(np.abs(bco)))
    growth = 2 * LambdaC ** 1.5 * T ** 1.5 + LambdaC * T
    r_max = base * max(1.0, growth)
    exps = 2 * Gamma * r_max
    samples = 2 * b1 ** 4 * math.log(2 / gm["delta"]) / eps**2
    n_reg = n_sys + int(math.ceil(math.log2(1 / eps))) + 4      # auxiliary Schrodingerisation register
    eps_rot = gm["eps_rot_factor"] * eps_R / max(1.0, exps)
    T_exp = gm["T_per_toffoli"] * (8 * n_reg + gm["b_arith"] ** 2) + 3 * math.log2(1 / eps_rot)
    T_total = exps * T_exp * samples
    return dict(T_total=T_total, samples=samples, exps_per_sample=exps,
                logical_qubits=n_reg + 2 + 3 * gm["b_arith"], r_max=r_max,
                T_per_sample=exps * T_exp,
                time_serial_s=T_total * gm["t_gate_seconds"],
                time_parallel_1000_s=T_total * gm["t_gate_seconds"] / 1000)


def main():
    rows = load()
    err = json.load(open(os.path.join(RESULTS, "error_scaling.json")))
    demo = json.load(open(os.path.join(RESULTS, "demo_ret.json"))) if os.path.exists(os.path.join(RESULTS, "demo_ret.json")) else None
    Res = [10.0, 100.0, 1000.0]
    eps_list = [1e-2, 1e-3]
    summary = dict(gate_model=GATE_MODEL, points=[], fits={}, error=err, demo=demo)
    for NC in (1, 2):
        for Re in Res:
            N = grid_for(Re)
            have = [r for r in rows if r["NC"] == NC and r["Re"] == Re]
            if not have:
                continue
            spec, fits = extrapolate_spec(rows, NC, Re, N)
            summary["fits"][f"NC{NC}_Re{Re:g}"] = fits
            for eps in eps_list:
                for task in (1, 2):
                    rc = ret_cost(spec, spec["n_sys"], eps, task)
                    pt = dict(NC=NC, Re=Re, N=N, Nt=spec["Nt"], n_sys=spec["n_sys"], kappa=spec["kappa"],
                              Lambda=spec["Lambda"], alpha3=spec["alpha3"], extrapolated=spec["extrapolated"],
                              eps=eps, task=task, T_total=rc["T_total"], T_per_sample=rc["T_per_sample"],
                              samples=rc["samples"], exps_per_sample=rc["exps_per_sample"],
                              logical_qubits=rc["logical_qubits"], time_serial_s=rc["time_serial_s"],
                              time_parallel_1000_s=rc["time_parallel_1000_s"], c=rc["fourier"]["c"],
                              t_max=rc["fourier"]["t_max"], m=rc["m"], b_norm1=rc["b_norm1"],
                              classical_flops=classical_cost(N, NC, spec["Nt"]),
                              qlsa_gates=qlsa_cost(spec["kappa"], eps, spec["Lambda"]),
                              LambdaC=spec["LambdaC"],
                              **{"schro_" + k: v for k, v in schro_cost(spec["LambdaC"], spec["n_sys"], eps, spec["Gamma"]).items()})
                    summary["points"].append(pt)
                    print(f"NC={NC} Re={Re:g} N={N} eps={eps:g} task={task}: kappa={spec['kappa']:.3g} "
                          f"T_total={rc['T_total']:.2e} qubits={rc['logical_qubits']} samples={rc['samples']:.2e} "
                          f"{'(extrap)' if spec['extrapolated'] else ''}", flush=True)

    # ---------------- figure 1: the Airbus plot ----------------
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.9))
    TASK_LABEL = {1: "overlap ($L_2$ error)", 2: "observable (KE decay)"}
    SHORT = {1: "$L_2$", 2: "KE"}
    for NC, ls in ((1, "--"), (2, "-")):
        for task, mk in ((1, "o"), (2, "s")):
            pts = sorted([p for p in summary["points"] if p["NC"] == NC and p["task"] == task and p["eps"] == 1e-2],
                         key=lambda p: p["Re"])
            if not pts:
                continue
            xs = [p["Re"] for p in pts]; ys = [p["T_total"] for p in pts]
            b, A = powerfit(xs, ys)
            line, = ax[0].loglog(xs, ys, ls, marker=mk, markerfacecolor="white",
                                 label=f"inv. $N_C$={NC}, {SHORT[task]} (Re$^{{{b:.2f}}}$)")
            comp = [p for p in pts if not p["extrapolated"]]
            ax[0].loglog([p["Re"] for p in comp], [p["T_total"] for p in comp], ls="none",
                         marker=mk, color=line.get_color())
            summary["fits"][f"Tscaling_NC{NC}_task{task}"] = dict(exponent=b, prefactor=A)
        ps = sorted([p for p in summary["points"] if p["NC"] == NC and p["task"] == 2 and p["eps"] == 1e-2],
                    key=lambda p: p["Re"])
        if ps:
            ax[0].loglog([p["Re"] for p in ps], [p["schro_T_total"] for p in ps], ls, color="m", alpha=0.8,
                         label=f"Schröd. $N_C$={NC} (bound)")
    ax[0].set_xlabel("Re"); ax[0].set_ylabel("total T-gates ($\\varepsilon=10^{-2}$)")
    ax[0].legend(fontsize=7, loc="lower right", labelspacing=0.3, borderpad=0.3, framealpha=0.85)
    ax[0].set_title("(a) time-to-solution")
    axt = ax[0].twinx()                       # serial wall clock at the stated T-gate rate
    lo, hi = ax[0].get_ylim()
    axt.set_yscale("log"); axt.set_ylim(lo * GATE_MODEL["t_gate_seconds"] / 3.15e7, hi * GATE_MODEL["t_gate_seconds"] / 3.15e7)
    axt.set_ylabel("serial run time (years)", fontsize=8); axt.tick_params(labelsize=7)
    for NC, ls in ((1, "--"), (2, "-")):
        p2 = sorted([p for p in summary["points"] if p["NC"] == NC and p["task"] == 1 and p["eps"] == 1e-2], key=lambda p: p["Re"])
        if p2:
            ax[1].semilogx([p["Re"] for p in p2], [p["logical_qubits"] for p in p2], ls, marker="o", label=f"total logical, $N_C$={NC}")
            ax[1].semilogx([p["Re"] for p in p2], [p["n_sys"] for p in p2], ls, marker="^", alpha=0.6, label=f"system register, $N_C$={NC}")
    ax[1].set_xlabel("Re"); ax[1].set_ylabel("qubits"); ax[1].legend(); ax[1].set_title("(b) memory")
    cases = [c for c in err["cases"] if c["at_grid_rule"]]
    ax[2].loglog([c["Re"] for c in cases], [c["l2_err_T"] for c in cases], "o-", label="$L_2$ velocity error (N(Re))")
    ax[2].loglog([c["Re"] for c in cases], [c["ke_err_T"] for c in cases], "s-", label="KE-decay error (N(Re))")
    allc = sorted(err["cases"], key=lambda c: (c["Re"], c["N"]))
    for Re in Res:
        cc = [c for c in allc if c["Re"] == Re]
        ax[2].loglog([Re] * len(cc), [c["l2_err_T"] for c in cc], "k.", alpha=0.4)
    ax[2].axhline(1e-2, color="gray", ls=":", label="RET $\\varepsilon$ budget")
    ax[2].set_xlabel("Re"); ax[2].set_ylabel("relative error at $t=1$"); ax[2].legend()
    ax[2].set_title("(c) error scaling")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig1_airbus_scaling.pdf")); fig.savefig(os.path.join(FIGS, "fig1_airbus_scaling.png"), dpi=160)

    # ---------------- figure 2: crossover ----------------
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    for i, eps in enumerate(eps_list):
        for NC, ls in ((1, "--"), (2, "-")):
            for task, mk in ((1, "o"), (2, "s")):
                pts = sorted([p for p in summary["points"] if p["NC"] == NC and p["task"] == task and p["eps"] == eps], key=lambda p: p["Re"])
                if pts:
                    ax[i].loglog([p["Re"] for p in pts], [p["T_total"] for p in pts], ls, marker=mk,
                                 label=f"inversion $N_C$={NC}, {TASK_LABEL[task]}")
            pts = sorted([p for p in summary["points"] if p["NC"] == NC and p["task"] == 1 and p["eps"] == eps], key=lambda p: p["Re"])
            if pts:
                ax[i].loglog([p["Re"] for p in pts], [p["qlsa_gates"] for p in pts], ls, color="k", alpha=0.6, label=f"block-encoding QLSA $N_C$={NC}")
                ax[i].loglog([p["Re"] for p in pts], [p["classical_flops"] for p in pts], ls, color="g", label=f"classical FV flops $N_C$={NC}")
                ax[i].loglog([p["Re"] for p in pts], [p["schro_T_total"] for p in pts], ls, color="m", alpha=0.8, label=f"Schrödingerisation arm $N_C$={NC} (bound)")
        ax[i].set_xlabel("Re"); ax[i].set_ylabel("gates / flops"); ax[i].set_title(f"$\\varepsilon = 10^{{{int(math.log10(eps))}}}$")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.18, 1, 1)); fig.savefig(os.path.join(FIGS, "fig2_crossover.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIGS, "fig2_crossover.png"), dpi=160)

    # ---------------- figure 3: RET demo ----------------
    if demo:
        fig, ax = plt.subplots(figsize=(4, 3))
        ns = sorted(int(k) for k in demo["error_vs_samples"])
        ax.loglog(ns, [demo["error_vs_samples"][str(n)] for n in ns], "o-", label="|estimate $-$ exact|")
        ax.loglog(ns, [demo["S"] * math.sqrt(2 * math.log(40) / n) for n in ns], "--", label="Hoeffding bound")
        ax.axhline(demo["fourier_richardson_error"], color="gray", ls=":", label="Fourier+Richardson bias")
        ax.set_xlabel("samples"); ax.set_ylabel(r"error in $\|L\|\,\langle\phi|L^{-1}b\rangle$"); ax.legend()
        ax.set_title(f"RET on the {demo['n_qubits']}-qubit TGV instance", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig3_ret_demo.pdf")); fig.savefig(os.path.join(FIGS, "fig3_ret_demo.png"), dpi=160)

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    write_numbers(summary)


def ratio(x):
    """Format a dimensionless ratio: plain integer below 1000, else 10^k style."""
    if not np.isfinite(x) or x <= 0:
        return "--"
    return f"{x:.0f}" if x < 1000 else sci(x, 0)


def sci(x, digits=1):
    if x == 0 or not np.isfinite(x):
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    if digits == 0:
        return f"10^{{{e}}}" if abs(m - 1) < 0.05 else f"{m:.0f}\\times 10^{{{e}}}"
    return f"{m:.{digits}f}\\times 10^{{{e}}}"


W_NC = {1: "NCone", 2: "NCtwo"}
W_RE = {10: "ReTen", 100: "ReHund", 1000: "ReThou"}
W_EPS = {2: "EpsTwo", 3: "EpsThree"}
W_TASK = {1: "TaskOne", 2: "TaskTwo"}
W_N = {4: "NFour", 8: "NEight", 16: "NSixteen", 32: "NThirtyTwo", 64: "NSixtyFour", 128: "NOneTwoEight", 256: "NTwoFiveSix"}
W_P = {5: "PFive", 20: "PTwenty"}


def write_numbers(summary):
    lines = ["% generated by fit_and_plot.py -- do not edit"]
    def put(name, val):
        assert re.fullmatch(r"[A-Za-z]+", name), name
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    for p in summary["points"]:
        tag = W_NC[p["NC"]] + W_RE[int(p["Re"])] + W_EPS[int(-math.log10(p["eps"]))] + W_TASK[p["task"]]
        put("Ttot" + tag, sci(p["T_total"]))
        put("samples" + tag, sci(p["samples"]))
        put("Tsample" + tag, sci(p["T_per_sample"]))
        put("exps" + tag, sci(p["exps_per_sample"]))
        put("qubits" + tag, str(p["logical_qubits"]))
        put("nsys" + tag, str(p["n_sys"]))
        put("kappa" + tag, f"{p['kappa']:.3g}")
        put("Lam" + tag, f"{p['Lambda']:.3g}")
        put("alphathree" + tag, f"{p['alpha3']:.3g}")
        put("cnorm" + tag, f"{p['c']:.3g}")
        put("tmax" + tag, f"{p['t_max']:.3g}")
        put("qlsa" + tag, sci(p["qlsa_gates"]))
        put("classical" + tag, sci(p["classical_flops"]))
        put("Nt" + tag, str(p["Nt"]))
        put("Ngrid" + tag, str(p["N"]))
        put("timeser" + tag, sci(p["time_serial_s"]))
        put("timepar" + tag, sci(p["time_parallel_1000_s"]))
        put("extrap" + tag, "estimated" if p["extrapolated"] else "computed")
        put("LamC" + tag, f"{p['LambdaC']:.3g}")
        put("schroT" + tag, sci(p["schro_T_total"]))
        put("schrosamples" + tag, sci(p["schro_samples"]))
        put("schroqubits" + tag, str(p["schro_logical_qubits"]))
        put("schroTsample" + tag, sci(p["schro_T_per_sample"]))
        put("schrotimeser" + tag, sci(p["schro_time_serial_s"]))
        put("schrotimepar" + tag, sci(p["schro_time_parallel_1000_s"]))
        put("schrooverqlsa" + tag, ratio(p["schro_T_total"] / p["qlsa_gates"]))
        put("retoverqlsa" + tag, ratio(p["T_total"] / p["qlsa_gates"]))
    for k, v in summary["fits"].items():
        if "exponent" in v:                       # Tscaling_NC{nc}_task{t}
            nc, t = int(k.split("NC")[1][0]), int(k.split("task")[1])
            put("expTscaling" + W_NC[nc] + W_TASK[t], f"{v['exponent']:.2f}")
        else:                                     # NC{nc}_Re{re}: {key: fit}
            nc = int(k.split("NC")[1][0]); re_ = int(float(k.split("Re")[1]))
            for kk, vv in v.items():
                word = {"kappa": "kappa", "Lambda": "Lambda", "alpha3": "alphathree", "alpha2": "alphatwo", "LambdaC": "LambdaC"}[kk]
                put("exp" + W_NC[nc] + W_RE[re_] + word, f"{vv['exponent']:.2f}")
    err = summary["error"]
    for c in err["cases"]:
        tag = W_RE[int(c["Re"])] + W_N[c["N"]]
        put("lerr" + tag, sci(c["l2_err_T"]))
        put("keerr" + tag, sci(c["ke_err_T"]))
        put("rate" + tag, f"{c['decay_rate_fit']:.4f}")
        put("rateexact" + tag, f"{c['decay_rate_exact']:.4f}")
        put("Nt" + tag, str(c["Nt"]))
    for r in err["perturbed"]:
        tag = W_RE[int(r["Re"])] + W_N[r["N"]] + W_P[int(round(r["perturb"] * 100))]
        put("pertNCone" + tag, sci(r["NC1_rel_l2"]))
        put("pertNCtwo" + tag, sci(r["NC2_rel_l2"]))
    d = summary["demo"]
    if not d:   # placeholders so the documents compile before the demonstrator has run
        for nm in ("demoqubits", "demokappa", "demoK", "democ", "demoS", "demotarget", "demobias", "demonmax",
                   "demoerrnmax", "demorzero", "demom", "demoeps", "demoNt", "demofourierr", "demosteps"):
            put(nm, "\\textbf{??}")
    if d:
        put("demoqubits", str(d["n_qubits"])); put("demokappa", f"{d['kappa']:.3g}")
        put("demoK", sci(d["K_fourier"], 0)); put("democ", f"{d['c_l1']:.3g}"); put("demoS", f"{d['S']:.3g}")
        put("demotarget", f"{d['overlap_exact']:.4f}")
        put("demosteps", str(d["richardson"]["max_trotter_steps"]))
        put("demobias", sci(d["fourier_richardson_error"]))
        nmax = max(int(k) for k in d["error_vs_samples"])
        put("demonmax", sci(nmax, 0)); put("demoerrnmax", sci(d["error_vs_samples"][str(nmax)]))
        put("demorzero", str(d["richardson"]["r0"])); put("demom", str(d["richardson"]["m"]))
        put("demoeps", f"{d['eps']:g}"); put("demoNt", str(d["Nt"]))
        put("demofourierr", sci(d["fourier_operator_error"]) if d["fourier_operator_error"] is not None else "--")
    gm = summary["gate_model"]
    put("gmbits", str(gm["b_arith"])); put("gmtgate", sci(gm["t_gate_seconds"], 0))
    with open(os.path.join(HERE, "..", "numbers.tex"), "w") as f:
        f.write(chr(10).join(lines) + chr(10))
    print(f"wrote numbers.tex with {len(lines)-1} macros")
    write_tables(summary)


def write_tables(summary):
    """Full resource table for the appendix."""
    rows = sorted(summary["points"], key=lambda p: (p["NC"], p["Re"], p["eps"], p["task"]))
    out = ["% generated by fit_and_plot.py -- do not edit",
           "\\begin{tabular}{cccc|cccc|ccc|cc}", "\\toprule",
           "$N_C$ & Re & $\\varepsilon$ & metric & $N$ & $N_t$ & $\\kappa$ & $\\Lambda$ & exps/sample & samples & T-gates & qubits & status\\\\",
           "\\midrule"]
    for p in rows:
        out.append(f"{p['NC']} & {int(p['Re'])} & $10^{{{int(math.log10(p['eps']))}}}$ & {'$L_2$' if p['task'] == 1 else 'KE'} & {p['N']} & {p['Nt']} & "
                   f"{p['kappa']:.3g} & {p['Lambda']:.3g} & ${sci(p['exps_per_sample'])}$ & ${sci(p['samples'])}$ & "
                   f"${sci(p['T_total'])}$ & {p['logical_qubits']} & {'est.' if p['extrapolated'] else 'comp.'}\\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(HERE, "..", "tables.tex"), "w") as f:
        f.write(chr(10).join(out) + chr(10))
    # spectral fits table
    out = ["% generated", "\\begin{tabular}{ccccc}", "\\toprule",
           "$N_C$ & Re & $\\kappa \\sim N^{b}$ & $\\Lambda \\sim N^{b}$ & $\\alpha^{(3)} \\sim N^{b}$\\\\", "\\midrule"]
    for k, v in summary["fits"].items():
        if "exponent" in v:
            continue
        nc = int(k.split("NC")[1][0]); re_ = int(float(k.split("Re")[1]))
        out.append(f"{nc} & {re_} & {v['kappa']['exponent']:.2f} & {v['Lambda']['exponent']:.2f} & {v['alpha3']['exponent']:.2f}\\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(HERE, "..", "tables_fits.tex"), "w") as f:
        f.write(chr(10).join(out) + chr(10))


if __name__ == "__main__":
    main()
