"""Tracking of the four vortices of the leapfrogging configuration (Fig. 1c)."""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import config
import velocity as vel
from io_utils import (
    Scan,
    fluid_radius,
    gaussian_filter_complex,
)

# Initial positions (x, y) in xi, same order as the tracked vortices:
# (+, inner), (+, outer), (-, outer), (-, inner)
INITIAL_POSITIONS = np.array(
    [[0.0, 2.5 - 0.75], [0.0, 2.5 + 0.75], [0.0, -2.5 - 0.75], [0.0, -2.5 + 0.75]]
)
CHARGES = np.array([1, 1, -1, -1])


def _detect_in_frame(scan: Scan, i: int, j: int, sigma: float, r: int):
    """Vortices of frame (i, j): array (N, 3) of (x, y, charge), x and y in units of
    xi and relative to the image centre (y positive = larger row index)."""
    field = np.asarray(scan.field[i, j], dtype=np.complex128)
    ref = np.asarray(scan.field_ref[i, j], dtype=np.complex128)
    field = gaussian_filter_complex(field * np.exp(-1j * np.angle(ref)), sigma)
    ny, nx = field.shape
    amp_ref = np.abs(ref) / np.abs(ref).max()

    window = int(fluid_radius(ref) / 1.5)  # keep vortices well inside the fluid
    vort = vel.detect_vortices(np.angle(field), r=r)
    vort[:, 0] -= nx // 2
    vort[:, 1] -= ny // 2
    vort = vort[vort[:, 0] ** 2 + vort[:, 1] ** 2 <= window**2]
    if len(vort) == 0:
        return vort
    # local healing length: xi grows as the density decreases (xi ~ 1/sqrt(rho))
    cx = int(np.clip(round(np.mean(vort[:, 0])) + nx // 2, 0, nx - 1))
    cy = int(np.clip(round(np.mean(vort[:, 1])) + ny // 2, 0, ny - 1))
    xi_local = scan.xis[i] / amp_ref[cy, cx]
    vort[:, :2] *= config.PIXEL_SIZE / xi_local
    return vort


def _canonical_order(v: np.ndarray) -> np.ndarray:
    """+ charges first, then by increasing y (matches INITIAL_POSITIONS)."""
    return v[np.lexsort((v[:, 1], -v[:, 2]))]


def _match(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Reorder `cur` so that row k is the continuation of `prev[k]`.

    The cost is the distance, plus a large penalty for mismatching charges.
    """
    cost = np.linalg.norm(cur[None, :, :2] - prev[:, None, :2], axis=-1)
    cost = cost + 1e3 * (cur[None, :, 2] != prev[:, None, 2])
    _, col = linear_sum_assignment(cost)
    return cur[col]


def track_vortices(
    scan: Scan,
    sigma: float = 2.0,
    r: int = 2,
    drop_last: int = 3,
    manual_swaps: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[str, np.ndarray]:
    """Track the 4 vortices of every frame, averaging over the N_avg repetitions.

    For each tau index, every repetition j in which exactly four vortices are found
    is used; the positions are averaged over those repetitions. Vortices are
    followed from frame to frame with the Hungarian algorithm.

    Args:
        scan: dataset.
        sigma: smoothing (px) applied to the field before detection.
        r: plaquette size for the vortex detection.
        drop_last: number of last frames ignored (vortices have left the fluid).
        manual_swaps: {frame_index: (a, b)} swaps the identity of tracks a and b
            at that frame, to fix a wrong automatic assignment (the original
            analysis needed this at frame 5 of the feedback-loop scan).

    Returns:
        dict with
          "trajectories": (4, N_frames, 2) x, y in xi (NaN where tracking failed),
          "charges": (4,),
          "taus": (N_frames,),
          "n_found": (N_frames, N_avg) number of vortices found in each repetition.
    """
    manual_swaps = manual_swaps or {}
    n_frames = scan.n_tau - drop_last
    traj = np.full((4, n_frames, 2), np.nan)
    n_found = np.zeros((n_frames, scan.n_avg), dtype=int)
    prev = None
    for i in range(n_frames):
        good = []
        for j in range(scan.n_avg):
            v = _detect_in_frame(scan, i, j, sigma, r)
            n_found[i, j] = len(v)
            if len(v) != 4 or sorted(v[:, 2]) != [-1, -1, 1, 1]:
                continue
            v = _canonical_order(v) if prev is None else _match(prev, v)
            if i in manual_swaps:
                a, b = manual_swaps[i]
                order = np.arange(4)
                order[[a, b]] = order[[b, a]]
                v = v[order]
            good.append(v)
        if not good:
            warnings.warn(f"Frame {i} (tau={scan.taus[i]:.0f}): no repetition with 4 vortices.")
            continue
        cur = np.mean(good, axis=0)
        traj[:, i, :] = cur[:, :2]
        prev = cur
    return {
        "trajectories": traj,
        "charges": CHARGES.copy(),
        "taus": scan.taus[:n_frames].copy(),
        "n_found": n_found,
    }


def remove_jumps(
    trajectories: np.ndarray, jump_indices: Sequence[int]
) -> np.ndarray:
    """Remove the position discontinuity at the given frame indices.

    At a feedback-loop transition the wavefunction is rebuilt analytically from the
    detected vortex positions, which introduces a small jump. For every index k in
    `jump_indices`, the step trajectory[k] - trajectory[k-1] is subtracted from all
    frames >= k. The steps are measured once, on the raw data, so that they do not
    depend on the order of the corrections (the original code recomputed them while
    modifying the array, which only corrected frame k itself).
    """
    out = np.array(trajectories, dtype=float, copy=True)
    steps = {k: trajectories[:, k, :] - trajectories[:, k - 1, :] for k in jump_indices}
    for k, step in steps.items():
        out[:, k:, :] -= step[:, None, :]
    return out
