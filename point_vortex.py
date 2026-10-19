"""Point-vortex model with an outward background flow (Eq. 3 of the paper).

Units: lengths in xi, speeds in c_s, time in z_NL (the circulation 2*pi is absorbed
in the time unit, i.e. the speed induced by a vortex of charge l at distance r is
l / r).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

import config


def initial_configuration(
    d: float = config.PV_PAIR_DISTANCE, big_d: float = config.PV_PAIR_SEPARATION
):
    """Two vortex-antivortex pairs on the y axis.

    Order: (+, y=D/2+d/2), (+, y=D/2-d/2), (-, y=-D/2-d/2), (-, y=-D/2+d/2).
    """
    positions = np.array(
        [
            [0.0, big_d / 2 + d / 2],
            [0.0, big_d / 2 - d / 2],
            [0.0, -big_d / 2 - d / 2],
            [0.0, -big_d / 2 + d / 2],
        ]
    )
    charges = np.array([1, 1, -1, -1])
    return positions, charges


def radial_background(
    v_max: float = config.PV_BACKGROUND_VMAX,
    r_max: float = config.PV_BACKGROUND_RMAX,
) -> Callable[[np.ndarray], np.ndarray]:
    """Outward flow profile v(r) = 2 v_max (r/r_max) / (1 + r/r_max), 0 beyond r_max.

    It reproduces the measured background flow of Fig. 5 (linear growth at small r,
    saturating at ~0.35 c_s). Any callable r -> v(r) can be used instead, e.g. an
    interpolation of the measured profile (see `radial_background_from_data`).
    """

    def v(r):
        rho = np.asarray(r, dtype=float) / r_max
        out = v_max * (rho / (1 + rho)) / 0.5
        return np.where(rho <= 1.0, out, 0.0)

    return v


def radial_background_from_data(r: np.ndarray, speed: np.ndarray, r_max: float):
    """Background flow interpolated from a measured profile (r in xi, speed in c_s)."""

    def v(rr):
        out = np.interp(rr, r, speed)
        return np.where(np.asarray(rr) <= r_max, out, 0.0)

    return v


def velocity(positions, charges, background=None):
    """Velocity of every vortex: Biot-Savart of the others + outward flow."""
    diff = positions[:, None, :] - positions[None, :, :]  # diff[i, j] = r_i - r_j
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, np.inf)  # no self interaction
    w = charges[None, :] / r2
    v = np.stack(
        [-np.sum(w * diff[..., 1], axis=1), np.sum(w * diff[..., 0], axis=1)], axis=1
    )
    if background is not None:
        r = np.linalg.norm(positions, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            radial = np.where(r[:, None] > 0, positions / r[:, None], 0.0)
        v = v + background(r)[:, None] * radial
    return v


def simulate(
    positions: Optional[np.ndarray] = None,
    charges: Optional[np.ndarray] = None,
    dt: float = config.PV_DT,
    t_max: float = config.PV_TMAX,
    background: Optional[Callable] = None,
    method: str = "euler",
):
    """Integrate the vortex trajectories.

    Args:
        method: "euler" (explicit Euler, the scheme used to produce the figure of
            the paper) or "rk4" (accurate; explicit Euler slowly drifts the
            orbits outwards).

    Returns:
        history: array (n_steps + 1, n_vortices, 2) of positions, history[0] being
        the initial condition. times: (n_steps + 1,).
    """
    if positions is None or charges is None:
        positions, charges = initial_configuration()
    if background is None:
        background = radial_background()
    pos = np.array(positions, dtype=float)
    n_steps = int(round(t_max / dt))
    history = np.empty((n_steps + 1,) + pos.shape)
    history[0] = pos

    def f(p):
        return velocity(p, charges, background)

    for n in range(n_steps):
        if method == "euler":
            pos = pos + dt * f(pos)
        elif method == "rk4":
            k1 = f(pos)
            k2 = f(pos + 0.5 * dt * k1)
            k3 = f(pos + 0.5 * dt * k2)
            k4 = f(pos + dt * k3)
            pos = pos + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError("method must be 'euler' or 'rk4'")
        history[n + 1] = pos
    return history, np.arange(n_steps + 1) * dt
