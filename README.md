# Resource estimates for the 2D convecting Taylor–Green vortex

Code and data behind the Phase 1 concept proposal *Explicit-constant resource estimates for the
2D convecting Taylor–Green vortex with a one-ancilla, block-encoding-free quantum solver*
(2026 Global Quantum + AI Challenge, Airbus problem statement "Quantum Solvers: Enhancing
Predictive Aerodynamic Modeling Capabilities").

The solver is a Carleman-linearised, Crank–Nicolson finite-volume discretisation of the 2D
convecting Taylor–Green vortex, assembled into one space-time linear system and solved with
Randomized Extrapolation Trotterization (RET, [arXiv:2608.13862](https://arxiv.org/abs/2608.13862))
as the matrix-inverse primitive. Every constant in the resource model is computed on the
assembled matrices: condition number, Trotter-term norms, and nested-commutator norms.

## Run it

```
pip install -r requirements.txt

python tgv_operator.py                          # self-tests: adjoints, inverses, F2 = 0 on the exact mode
python resources.py --N 4 8 --NC 1 2 --Re 10 100
python resources.py --N 16 32 64 --NC 1 --Re 10 100 1000
python resources.py --N 16 --NC 2 --Re 10 100 --no-order3
python error_scaling.py                         # L2 and kinetic-energy error vs the analytic solution
python demo_ret.py                              # end-to-end RET run on the 8-qubit instance
python fit_and_plot.py                          # -> figs/, and the LaTeX macros used in the proposal
```

`results/` holds the JSON output of every run above, so `fit_and_plot.py` reproduces the figures
and every number in the proposal without re-running the spectral sweeps (those take a few hours
at `N = 64`).

## What is where

| File | What |
|---|---|
| `tgv_operator.py` | vorticity discretisation, Carleman lift, space-time operator `L`, spectral quantities |
| `resources.py` | Trotter decomposition, CKS Fourier series for 1/x, RET cost model, gate model |
| `error_scaling.py` | classical emulation of the same scheme against the exact solution, including the perturbed vortex |
| `demo_ret.py` | the end-to-end demonstrator: CKS series, Richardson schedule, Hadamard-test sampling |
| `fit_and_plot.py` | N-extrapolation, the three Reynolds-scaling curves, the crossover plot, LaTeX macros |
| `richardson.py` | Richardson step-count code, vendored from the RET paper's own numerics |
| `results/` | all JSON and logs |
| `figs/` | generated figures |

## Licence

MIT, see `LICENSE`. `richardson.py` comes from the numerics of arXiv:2608.13862
([github.com/arulrhikm/numerics](https://github.com/arulrhikm/numerics)) and keeps its own
copyright line in `LICENSE-richardson`; it is MIT too.
