"""Velocity field, Helmholtz decomposition and detection of critical points.

Conventions (identical to the original code):
  * arrays are indexed [row, column] = [y, x]; vectors are stored as (v_x, v_y);
  * the velocity is the gradient of the phase, v = grad(phi), expressed in c_s when
    the pixel size `dx` is given in units of the healing length xi;
  * the "total velocity" of the paper is the density weighted velocity
    u = sqrt(rho) v, with sqrt(rho) the field amplitude normalised to its maximum.

Pure numpy/scipy implementation (no numba / pyfftw required).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# Velocity and Helmholtz decomposition
# --------------------------------------------------------------------------
def phase_velocity(phase: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Gradient of the phase, shape (2, Ny, Nx) = (v_x, v_y).

    The phase is unwrapped independently along each axis before differentiation,
    so that 2*pi jumps do not produce spikes (this is what makes it robust in the
    presence of vortices, where a 2D unwrapping is ill defined).
    """
    ux = np.unwrap(phase, axis=1)
    uy = np.unwrap(phase, axis=0)
    return np.stack([np.gradient(ux, dx, axis=1), np.gradient(uy, dx, axis=0)])


def helmholtz_decomposition(field: np.ndarray, dx: float = 1.0):
    """Split the density-weighted velocity into incompressible/compressible parts.

    Args:
        field: complex 2D field (amplitude normalised to its maximum by the caller).
        dx: pixel size, in units of xi to obtain velocities in units of c_s.

    Returns:
        (u_tot, u_inc, u_comp), each of shape (2, Ny, Nx), with u_tot = u_inc + u_comp
        and u_comp the curl-free (compressible) part.
    """
    sy, sx = field.shape
    kx = 2 * np.pi * np.fft.rfftfreq(sx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(sy, d=dx)
    K = np.array(np.meshgrid(kx, ky))  # (2, sy, sx//2+1) = (kx, ky)
    u_tot = np.abs(field) * phase_velocity(np.angle(field), dx)
    U = np.fft.rfft2(u_tot)
    # projection on k: the compressible part is k (k . U) / k^2
    proj = np.sum(U * K, axis=0) / (np.sum(K**2, axis=0) + 1e-15)
    u_comp = np.fft.irfft2(K * proj, s=(sy, sx))
    return u_tot, u_tot - u_comp, u_comp


def background_speed(field_ref: np.ndarray, dx: float) -> np.ndarray:
    """|u| of the vortex-free fluid (the outward flow of Fig. 5)."""
    u_tot, _, _ = helmholtz_decomposition(field_ref, dx)
    return np.hypot(u_tot[0], u_tot[1])


# --------------------------------------------------------------------------
# Plaquette sums (vectorised with cumulative sums, O(N) for any plaquette size)
# --------------------------------------------------------------------------
def _window_sum(a: np.ndarray, n: int, axis: int) -> np.ndarray:
    """Sum of n consecutive elements along `axis` (output is n-1 shorter)."""
    cs = np.cumsum(a, axis=axis, dtype=np.float64)
    zero_shape = list(a.shape)
    zero_shape[axis] = 1
    cs = np.concatenate([np.zeros(zero_shape), cs], axis=axis)
    hi = np.take(cs, np.arange(n, cs.shape[axis]), axis=axis)
    lo = np.take(cs, np.arange(0, cs.shape[axis] - n), axis=axis)
    return hi - lo


def _place_centered(values: np.ndarray, shape, r: int) -> np.ndarray:
    """Embed (N-r, M-r) plaquette values at the plaquette centre of an (N, M) map."""
    out = np.zeros(shape, dtype=np.float64)
    h = r // 2
    out[h : h + values.shape[0], h : h + values.shape[1]] = values
    return out


def winding(velo: np.ndarray, r: int) -> np.ndarray:
    """Circulation of the phase gradient around square plaquettes of side r.

    The value (in radians) sits at the plaquette centre; plaquettes crossing the
    image border are not evaluated (value 0).
    """
    n, m = velo.shape[1:]
    along_x = _window_sum(velo[0], r + 1, axis=1)  # (N, M-r)
    along_y = _window_sum(velo[1], r + 1, axis=0)  # (N-r, M)
    circ = (
        along_x[: n - r, :]
        - along_x[r:, :]
        + along_y[:, r:]
        - along_y[:, : m - r]
    )
    return _place_centered(circ, (n, m), r)


def poincare(angle: np.ndarray, r: int) -> np.ndarray:
    """Winding of a vector-field angle around square plaquettes of side r (index)."""
    n, m = angle.shape

    def wrap(d):
        return (d + np.pi) % (2 * np.pi) - np.pi

    dxa = wrap(np.diff(angle, axis=1))  # (N, M-1)
    dya = wrap(np.diff(angle, axis=0))  # (N-1, M)
    sx = _window_sum(dxa, r, axis=1)  # (N, M-r)
    sy = _window_sum(dya, r, axis=0)  # (N-r, M)
    idx = (sx[: n - r, :] - sx[r:, :] + sy[:, r:] - sy[:, : m - r]) / (2 * np.pi)
    return _place_centered(idx, (n, m), r)


def _merge_pixels(mask: np.ndarray):
    """Centroids (x, y) of the connected groups of True pixels."""
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros((0, 2))
    com = ndimage.center_of_mass(mask, labels, index=np.arange(1, n + 1))
    return np.array([[c[1], c[0]] for c in com])


def _collect(cond_plus, cond_minus, merge: bool):
    pts = []
    for cond, sign in ((cond_plus, 1.0), (cond_minus, -1.0)):
        if merge:
            xy = _merge_pixels(cond)
        else:
            yy, xx = np.nonzero(cond)
            xy = np.stack([xx, yy], axis=1).astype(float)
        pts.append(np.column_stack([xy, np.full(len(xy), sign)]))
    return np.concatenate(pts, axis=0)


# --------------------------------------------------------------------------
# Detection of vortices, saddles and nodes
# --------------------------------------------------------------------------
def detect_vortices(phase: np.ndarray, r: int = 2, merge: bool = True) -> np.ndarray:
    """Vortices from the quantised circulation of the phase.

    A pixel is a vortex of charge +/-1 if the circulation is +/-2*pi on all the
    nested plaquettes of side 1..r centred on it (r > 1 rejects noise).

    Returns:
        array (N, 3) of rows (x, y, charge) in pixels, positive charges first,
        each group sorted by increasing y then x.
    """
    velo = phase_velocity(phase)
    wind = np.stack([winding(velo, k) for k in range(1, r + 1)])
    plus = np.all(wind >= 2 * np.pi - 1e-6, axis=0)
    minus = np.all(wind <= -2 * np.pi + 1e-6, axis=0)
    return _collect(plus, minus, merge)


def detect_saddles_nodes(
    phase: np.ndarray,
    r: int = 2,
    vortices: np.ndarray | None = None,
    exclude_radius: float = 3.0,
    merge: bool = True,
):
    """Zeros of the velocity field with non-zero Poincare index.

    The angle of the velocity is atan2(v_x, v_y); with this convention a saddle
    (Poincare index -1) gives +1 and a node (index +1) gives -1. Vortex cores also
    have index +1, so detections closer than `exclude_radius` px to a vortex are
    discarded when `vortices` is given.

    Returns:
        (saddles, nodes): arrays (N, 2) of (x, y) in pixels.
    """
    velo = phase_velocity(phase)
    angle = np.arctan2(velo[0], velo[1])
    idx = np.mean([poincare(angle, k) for k in range(1, r + 1)], axis=0)
    pts = _collect(idx >= 1, idx <= -1, merge)
    saddles, nodes = pts[pts[:, 2] > 0, :2], pts[pts[:, 2] < 0, :2]
    if vortices is not None and len(vortices) and len(nodes):
        d = np.hypot(
            nodes[:, None, 0] - vortices[None, :, 0],
            nodes[:, None, 1] - vortices[None, :, 1],
        )
        nodes = nodes[np.all(d > exclude_radius, axis=1)]
    return saddles, nodes
