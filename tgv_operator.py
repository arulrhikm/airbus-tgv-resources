"""2D convecting Taylor-Green vortex: finite-volume vorticity discretization,
Carleman lift, theta-method time stepping, and the global space-time linear
system L x = b that the quantum linear-systems primitive inverts.

Conventions (see kb/challenges/airbus.md section 4a):
  domain [0, 2*pi]^2 periodic, N x N cells, h = 2*pi/N
  V0 = Uc = rho = 1, Vc = p0 = 0, trig length L = 1, nu = V0 * (2*pi) / Re
  exact vorticity  omega(x,y,t) = 2 V0 sin(x - Uc t) sin(y) exp(-2 nu t)

State: vorticity omega on the div-free subspace (pressure eliminated exactly by
the discrete Leray projector, which on a periodic grid is the streamfunction
relation  lap_h psi = -omega).  Second-order central finite-volume fluxes on the
uniform periodic grid coincide with central differences, so every linear piece is
a Fourier multiplier; the quadratic advection J(psi, omega) is a local product
in real space.  All operators are exposed as matvec / rmatvec closures so the
N^4-dimensional Carleman block is never materialised.

Carleman order N_C = 1: state omega (N^2)         -> nonlinearity dropped
Carleman order N_C = 2: state (omega, omega (x) omega)  (N^2 + N^4)
  C = [[F1, F2], [0, F1 (+) F1]]     (Kronecker sum on the second block)

theta-method:  A_plus x_{k+1} = A_minus x_k,  A_plus = I - theta dt C,
               A_minus = I + (1-theta) dt C.  theta = 1/2 is Crank-Nicolson.
Global system on slices 0..Nt:  row 0: x_0 = b;  row k: -A_minus x_{k-1} + A_plus x_k = 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, svds


# --------------------------------------------------------------------------
# grid and symbols
# --------------------------------------------------------------------------
@dataclass
class TGV:
    N: int
    Re: float
    NC: int = 2
    T: float = 1.0
    cfl: float = 0.5
    theta: float = 0.5
    V0: float = 1.0
    Uc: float = 1.0
    Vc: float = 0.0
    Lbox: float = 2 * math.pi
    Nt: int | None = None
    # derived
    h: float = field(init=False)
    nu: float = field(init=False)
    dt: float = field(init=False)
    kx: np.ndarray = field(init=False)
    ky: np.ndarray = field(init=False)
    sx: np.ndarray = field(init=False)   # central-difference symbol i*sx
    sy: np.ndarray = field(init=False)
    lam: np.ndarray = field(init=False)  # -laplacian symbol (>= 0)
    f1: np.ndarray = field(init=False)   # F1 diagonal (Fourier)
    kxpsi: np.ndarray = field(init=False)  # symbol of d_x psi from omega : i*sx/lam
    kypsi: np.ndarray = field(init=False)

    def __post_init__(self):
        N = self.N
        self.h = self.Lbox / N
        self.nu = self.V0 * self.Lbox / self.Re
        k = np.fft.fftfreq(N, d=1.0 / N)  # integer wavenumbers
        self.kx, self.ky = np.meshgrid(k, k, indexing="ij")
        self.sx = np.sin(self.kx * self.h) / self.h
        self.sy = np.sin(self.ky * self.h) / self.h
        self.lam = (2 - 2 * np.cos(self.kx * self.h)) / self.h**2 + (2 - 2 * np.cos(self.ky * self.h)) / self.h**2
        # F1 = nu lap_h - Uc d_x - Vc d_y  (Fourier symbol)
        self.f1 = -(self.nu * self.lam) - 1j * (self.Uc * self.sx + self.Vc * self.sy)
        lam_safe = np.where(self.lam > 0, self.lam, np.inf)
        # psi = -lap^{-1} omega  ->  psi_hat = omega_hat / lam ;  d_x psi -> i sx psi
        self.kxpsi = 1j * self.sx / lam_safe
        self.kypsi = 1j * self.sy / lam_safe
        # time step: advective CFL on the (Uc + V0) velocity scale
        if self.Nt is None:
            dt = self.dt_override if self.dt_override else self.cfl * self.h / (abs(self.Uc) + abs(self.Vc) + self.V0)
            self.Nt = int(math.ceil(self.T / dt))
        self.dt = self.T / self.Nt
        self.n2 = N * N
        self.n4 = N**4 if self.NC == 2 else 0
        self.dim_c = self.n2 + self.n4          # Carleman state dimension
        self.dim = self.dim_c * (self.Nt + 1)   # global space-time dimension

    dt_override: float | None = None

    # ---------------- exact solution ----------------
    def omega_exact_hat(self, t: float) -> np.ndarray:
        """Fourier coefficients (unnormalised fft2 convention) of the exact vorticity
        sampled at cell centres."""
        x = (np.arange(self.N) + 0.5) * self.h
        X, Y = np.meshgrid(x, x, indexing="ij")
        om = 2 * self.V0 * np.sin(X - self.Uc * t) * np.sin(Y - self.Vc * t) * math.exp(-2 * self.nu * t)
        return np.fft.fft2(om).ravel()

    def kinetic_energy(self, om_hat: np.ndarray) -> float:
        """Fluctuation kinetic energy 1/2 <|u|^2> from the vorticity spectrum
        (|u_k|^2 = |omega_k|^2 / lam_k with the discrete symbols)."""
        w = om_hat.reshape(self.N, self.N)
        lam_safe = np.where(self.lam > 0, self.lam, np.inf)
        return 0.5 * float(np.sum(np.abs(w) ** 2 / lam_safe)) / self.n2**2

    # ---------------- bilinear F2 ----------------
    def _real(self, a_hat: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(a_hat.reshape(self.N, self.N))

    def F2_pair(self, a_hat: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
        """F2(a (x) b) = -J(psi_a, omega_b) with J = psi_x omega_y - psi_y omega_x."""
        a = a_hat.reshape(self.N, self.N)
        b = b_hat.reshape(self.N, self.N)
        px = np.fft.ifft2(self.kxpsi * a)
        py = np.fft.ifft2(self.kypsi * a)
        ox = np.fft.ifft2(1j * self.sx * b)
        oy = np.fft.ifft2(1j * self.sy * b)
        J = px * oy - py * ox
        return (-np.fft.fft2(J)).ravel()

    # F2 = F2x + F2y with
    #   F2x(M)_i = -[ (Kx M Dy^T) ]_{ii}   (psi_x * omega_y part)
    #   F2y(M)_i = +[ (Ky M Dx^T) ]_{ii}   (psi_y * omega_x part)
    # each is "diagonal extraction of (multiplier on a) (x) (multiplier on b)".
    def _F2_part(self, M: np.ndarray, ma: np.ndarray, mb: np.ndarray, sign: float) -> np.ndarray:
        N = self.N
        M4 = M.reshape(N, N, N, N)
        A = np.fft.ifft2(ma[:, :, None, None] * M4, axes=(0, 1))
        B = np.fft.ifft2(mb[None, None, :, :] * A, axes=(2, 3))
        J = np.einsum("ijij->ij", B)
        return (sign * np.fft.fft2(J)).ravel()

    def _F2_part_adj(self, v_hat: np.ndarray, ma: np.ndarray, mb: np.ndarray, sign: float) -> np.ndarray:
        N = self.N
        v = np.fft.ifft2(v_hat.reshape(N, N)) * (N * N)  # adjoint of fft2 is N^2 * ifft2
        D = np.zeros((N, N, N, N), dtype=complex)
        idx = np.arange(N)
        D[idx[:, None], idx[None, :], idx[:, None], idx[None, :]] = sign * v
        t = np.fft.fft2(D, axes=(2, 3)) * np.conj(mb)[None, None, :, :]
        t = np.fft.fft2(t, axes=(0, 1)) * np.conj(ma)[:, :, None, None]
        return (t / (self.n2 * self.n2)).reshape(self.n2 * self.n2)

    def F2x(self, M):      return self._F2_part(M, self.kxpsi, 1j * self.sy, -1.0)
    def F2y(self, M):      return self._F2_part(M, self.kypsi, 1j * self.sx, +1.0)
    def F2x_adj(self, v):  return self._F2_part_adj(v, self.kxpsi, 1j * self.sy, -1.0)
    def F2y_adj(self, v):  return self._F2_part_adj(v, self.kypsi, 1j * self.sx, +1.0)

    def F2_mat(self, M: np.ndarray) -> np.ndarray:
        """F2 applied to a general element M of V (x) V (shape (N^2, N^2), Fourier in
        both indices):  out_i = sum_{a,b} coef[i;a,b] M_{ab}."""
        return self.F2x(M) + self.F2y(M)

    def F2_adj(self, v_hat: np.ndarray) -> np.ndarray:
        """Adjoint V -> V (x) V of F2_mat."""
        return self.F2x_adj(v_hat) + self.F2y_adj(v_hat)

    # ---------------- Carleman operator pieces on the Carleman state ----------------
    def split(self, x: np.ndarray):
        return x[: self.n2], x[self.n2 :]

    def C_apply(self, x: np.ndarray) -> np.ndarray:
        """C x for the Carleman state x = (omega, omega(x)omega)."""
        w, M = self.split(x)
        out = np.empty_like(x)
        out[: self.n2] = self.f1.ravel() * w
        if self.NC == 2:
            out[: self.n2] += self.F2_mat(M.reshape(self.n2, self.n2))
            f1 = self.f1.ravel()
            out[self.n2 :] = ((f1[:, None] + f1[None, :]) * M.reshape(self.n2, self.n2)).ravel()
        return out

    def C_adj(self, x: np.ndarray) -> np.ndarray:
        w, M = self.split(x)
        out = np.empty_like(x)
        f1 = self.f1.ravel()
        out[: self.n2] = np.conj(f1) * w
        if self.NC == 2:
            out[self.n2 :] = (np.conj(f1[:, None] + f1[None, :]) * M.reshape(self.n2, self.n2)).ravel()
            out[self.n2 :] += self.F2_adj(w)
        return out

    # ---------------- theta-method step operators ----------------
    def Aplus(self, x):  return x - self.theta * self.dt * self.C_apply(x)
    def Aplus_adj(self, x):  return x - self.theta * self.dt * self.C_adj(x)
    def Aminus(self, x):  return x + (1 - self.theta) * self.dt * self.C_apply(x)
    def Aminus_adj(self, x):  return x + (1 - self.theta) * self.dt * self.C_adj(x)

    def Aplus_solve(self, b: np.ndarray) -> np.ndarray:
        """A_plus^{-1} b by block back-substitution (upper block-triangular)."""
        th = self.theta * self.dt
        f1 = self.f1.ravel()
        b1, b2 = self.split(b)
        out = np.empty_like(b)
        if self.NC == 2:
            y2 = (b2.reshape(self.n2, self.n2) / (1 - th * (f1[:, None] + f1[None, :]))).ravel()
            out[self.n2 :] = y2
            rhs = b1 + th * self.F2_mat(y2.reshape(self.n2, self.n2))
        else:
            rhs = b1
        out[: self.n2] = rhs / (1 - th * f1)
        return out

    def Aplus_solve_adj(self, b: np.ndarray) -> np.ndarray:
        """A_plus^{-dagger} b (lower block-triangular)."""
        th = self.theta * self.dt
        f1 = self.f1.ravel()
        b1, b2 = self.split(b)
        out = np.empty_like(b)
        y1 = b1 / np.conj(1 - th * f1)
        out[: self.n2] = y1
        if self.NC == 2:
            rhs2 = b2 + th * self.F2_adj(y1)
            out[self.n2 :] = (rhs2.reshape(self.n2, self.n2) / np.conj(1 - th * (f1[:, None] + f1[None, :]))).ravel()
        return out

    # ---------------- global space-time system ----------------
    def slices(self, x: np.ndarray):
        return x.reshape(self.Nt + 1, self.dim_c)

    def L_apply(self, x: np.ndarray) -> np.ndarray:
        X = self.slices(x)
        Y = np.empty_like(X)
        Y[0] = X[0]
        for k in range(1, self.Nt + 1):
            Y[k] = self.Aplus(X[k]) - self.Aminus(X[k - 1])
        return Y.ravel()

    def L_adj(self, x: np.ndarray) -> np.ndarray:
        X = self.slices(x)
        Y = np.empty_like(X)
        for k in range(self.Nt + 1):
            Y[k] = self.Aplus_adj(X[k]) if k > 0 else X[k].copy()
            if k < self.Nt:
                Y[k] -= self.Aminus_adj(X[k + 1])
        return Y.ravel()

    def L_solve(self, b: np.ndarray) -> np.ndarray:
        B = self.slices(b)
        X = np.empty_like(B)
        X[0] = B[0]
        for k in range(1, self.Nt + 1):
            X[k] = self.Aplus_solve(B[k] + self.Aminus(X[k - 1]))
        return X.ravel()

    def L_solve_adj(self, b: np.ndarray) -> np.ndarray:
        B = self.slices(b)
        X = np.empty_like(B)
        X[self.Nt] = self.Aplus_solve_adj(B[self.Nt])
        for k in range(self.Nt - 1, -1, -1):
            r = B[k] + self.Aminus_adj(X[k + 1])
            X[k] = self.Aplus_solve_adj(r) if k > 0 else r
        return X.ravel()

    def as_linop(self, apply, adj, dim=None) -> LinearOperator:
        dim = dim or self.dim
        return LinearOperator((dim, dim), matvec=apply, rmatvec=adj, dtype=complex)

    # ---------------- initial data and classical emulation ----------------
    def initial_state(self, perturb: float = 0.0) -> np.ndarray:
        """Carleman initial vector b: (omega_0, omega_0 (x) omega_0), with an optional
        perturbation by a (2,1) mode of relative amplitude `perturb` (activates the
        nonlinearity, which vanishes identically on the pure TGV)."""
        w0 = self.omega_exact_hat(0.0)
        if perturb:
            x = (np.arange(self.N) + 0.5) * self.h
            X, Y = np.meshgrid(x, x, indexing="ij")
            w0 = w0 + perturb * np.fft.fft2(2 * self.V0 * np.sin(2 * X) * np.sin(Y)).ravel()
        x0 = np.zeros(self.dim_c, dtype=complex)
        x0[: self.n2] = w0
        if self.NC == 2:
            x0[self.n2 :] = np.outer(w0, w0).ravel()
        return x0

    def march(self, x0: np.ndarray, store: bool = False):
        """Classical theta-method time marching; returns final Carleman state
        (and the full history if store)."""
        x = x0.copy()
        hist = [x.copy()] if store else None
        for _ in range(self.Nt):
            x = self.Aplus_solve(self.Aminus(x))
            if store:
                hist.append(x.copy())
        return (x, hist) if store else x

    def rhs_global(self, x0: np.ndarray) -> np.ndarray:
        b = np.zeros(self.dim, dtype=complex)
        b[: self.dim_c] = x0
        return b

    # ---------------- spectral norms ----------------
    def opnorm(self, apply, adj, dim=None, tol=1e-3, maxiter=200) -> float:
        op = self.as_linop(apply, adj, dim)
        try:
            s = svds(op, k=1, which="LM", tol=tol, maxiter=maxiter, return_singular_vectors=False)
            return float(s[0])
        except Exception:
            # fallback: power iteration on A^dagger A
            rng = np.random.default_rng(0)
            v = rng.standard_normal(op.shape[1]) + 1j * rng.standard_normal(op.shape[1])
            v /= np.linalg.norm(v)
            val = 0.0
            for _ in range(maxiter):
                w = adj(apply(v))
                nv = np.linalg.norm(w)
                if abs(nv - val) < tol * max(nv, 1e-30):
                    val = nv
                    break
                val = nv
                v = w / nv
            return math.sqrt(val)

    def condition_number(self, tol=1e-3) -> tuple[float, float, float]:
        nL = self.opnorm(self.L_apply, self.L_adj, tol=tol)
        nLinv = self.opnorm(self.L_solve, self.L_solve_adj, tol=tol)
        return nL * nLinv, nL, nLinv


# --------------------------------------------------------------------------
# self-tests
# --------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(1)
    for NC in (1, 2):
        g = TGV(N=4, Re=10.0, NC=NC)
        # adjoint consistency of every operator pair
        for name, (A, At) in {
            "C": (g.C_apply, g.C_adj), "L": (g.L_apply, g.L_adj),
            "Linv": (g.L_solve, g.L_solve_adj)}.items():
            d = g.dim if name != "C" else g.dim_c
            x = rng.standard_normal(d) + 1j * rng.standard_normal(d)
            y = rng.standard_normal(d) + 1j * rng.standard_normal(d)
            lhs = np.vdot(y, A(x)); rhs = np.vdot(At(y), x)
            assert abs(lhs - rhs) < 1e-8 * (1 + abs(lhs)), (NC, name, lhs, rhs)
        # inverse consistency
        x = rng.standard_normal(g.dim) + 1j * rng.standard_normal(g.dim)
        assert np.linalg.norm(g.L_apply(g.L_solve(x)) - x) < 1e-9 * np.linalg.norm(x)
        # F2 on a product equals the pair formula
        if NC == 2:
            a = rng.standard_normal(g.n2) + 1j * rng.standard_normal(g.n2)
            b = rng.standard_normal(g.n2) + 1j * rng.standard_normal(g.n2)
            assert np.allclose(g.F2_mat(np.outer(a, b)), g.F2_pair(a, b), atol=1e-10)
            # nonlinearity vanishes identically on the exact TGV mode
            w0 = g.omega_exact_hat(0.0)
            assert np.linalg.norm(g.F2_pair(w0, w0)) < 1e-9 * np.linalg.norm(w0) ** 2
    # decay rate: KE(T)/KE(0) -> exp(-4 nu T) as N grows, dt -> 0 (Crank-Nicolson)
    errs = []
    for N in (8, 16, 32):
        g = TGV(N=N, Re=100.0, NC=1, cfl=0.25)
        x = g.march(g.initial_state())
        ratio = g.kinetic_energy(x[: g.n2]) / g.kinetic_energy(g.initial_state()[: g.n2])
        errs.append(abs(ratio - math.exp(-4 * g.nu * g.T)))
    assert errs[0] > errs[1] > errs[2], errs
    print("tgv_operator self-test OK; KE-decay errors N=8,16,32:", ["%.2e" % e for e in errs])


if __name__ == "__main__":
    _selftest()
