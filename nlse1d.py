"""1D defocusing nonlinear Schroedinger equation (Fig. 3d).

    i dpsi/dtau = [ -1/2 d^2/dx^2 + |psi|^2 ] psi

in the units of the paper (x in xi, time tau in z_NL, speeds in c_s for rho = 1).
Solved with a split-step Fourier method (Strang splitting).
"""

from __future__ import annotations

import numpy as np


def gaussian_dip_density(x, sigma_outer=80.0, depth=0.6, sigma_inner=10.0):
    """Gaussian background with a Gaussian hole at the centre (density)."""
    return np.exp(-(x**2) / (2 * sigma_outer**2)) - depth * np.exp(
        -(x**2) / (2 * sigma_inner**2)
    )


def gaussian_velocity(x, peak=1.8, sigma=10.0, x0=0.0):
    """Gaussian velocity pulse (c_s units)."""
    return peak * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


# Presets used for the numerical panels: a strong initial flow that develops a
# dispersive shock wave with a vacuum point, and a weak one that does not.
PRESETS = {
    "dsw": dict(depth=0.6, peak=1.8),
    "no_dsw": dict(depth=0.1, peak=0.9),
}


def simulate(
    x: np.ndarray,
    rho0: np.ndarray,
    v0: np.ndarray,
    t_snapshots,
    dt: float = 0.01,
):
    """Evolve psi_0 = sqrt(rho0) exp(i phi0) with d(phi0)/dx = v0.

    Args:
        x: uniform grid (xi).
        rho0, v0: initial density and velocity on the grid.
        t_snapshots: sorted times at which the state is stored.
        dt: time step.

    Returns:
        dict {t: {"rho", "v", "u"}} with v the phase gradient and
        u = sqrt(rho) v the density-weighted velocity (normalised like the data:
        sqrt(rho) is divided by sqrt(rho_max) of the initial state).
    """
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
    phi0 = np.cumsum(v0) * dx
    psi = np.sqrt(rho0) * np.exp(1j * phi0)
    norm = np.sqrt(np.max(rho0))

    out = {}
    t, targets = 0.0, list(t_snapshots)

    def record(tt):
        rho = np.abs(psi) ** 2
        v = np.gradient(np.unwrap(np.angle(psi)), dx)
        out[tt] = {"rho": rho, "v": v, "u": np.sqrt(rho) / norm * v}

    if targets and targets[0] == 0:
        record(targets.pop(0))
    while targets:
        n = max(1, int(np.ceil((targets[0] - t) / dt - 1e-9)))
        h = (targets[0] - t) / n
        half = np.exp(-0.25j * k**2 * h)
        for _ in range(n):
            psi = np.fft.ifft(half * np.fft.fft(psi))
            psi = psi * np.exp(-1j * np.abs(psi) ** 2 * h)
            psi = np.fft.ifft(half * np.fft.fft(psi))
        t = targets[0]
        record(targets.pop(0))
    return out
