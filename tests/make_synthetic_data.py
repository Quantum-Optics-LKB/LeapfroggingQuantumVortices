"""Synthetic dataset with the same layout as the Zenodo data (for tests and demos).

Four vortices (two pairs) are imprinted on a Gaussian beam and moved with the
point-vortex model; every frame has a different "tau". Usage:

    python tests/make_synthetic_data.py OUTPUT_FOLDER
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402
import point_vortex as pv  # noqa: E402

XI_PX = 8.0  # healing length in pixels (typical value of the experiment)
N = 384


def vortex_field(x, y, x0, y0, charge, xi):
    dx, dy = x - x0, y - y0
    r = np.hypot(dx, dy)
    return r / np.sqrt(r**2 + (xi / 0.83) ** 2) * np.exp(1j * charge * np.arctan2(dy, dx))


def make_scan(folder, n_tau=12, n_avg=2, seed=0, config_name="leapfrog"):
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    yy, xx = np.mgrid[:N, :N].astype(float)
    xc, yc = N / 2, N / 2
    beam = np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * 110.0**2))
    ref_phase = 2e-5 * ((xx - xc) ** 2 + (yy - yc) ** 2)  # defocusing curvature

    history, _ = pv.simulate(t_max=24.0)
    steps = np.linspace(0, len(history) - 1, n_tau).astype(int)
    taus = np.linspace(30, 140, n_tau)

    field = np.zeros((n_tau, n_avg, N, N), np.complex64)
    field_ref = np.zeros_like(field)
    for i, s in enumerate(steps):
        pos, ch = history[s], pv.initial_configuration()[1]
        psi = beam.astype(complex)
        for (px, py), q in zip(pos, ch):
            psi = psi * vortex_field(xx, yy, xc + px * XI_PX, yc + py * XI_PX, q, XI_PX)
        for j in range(n_avg):
            g = np.exp(1j * rng.uniform(0, 2 * np.pi))  # random global phase
            noise = 1 + 0.01 * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
            field[i, j] = psi * np.exp(1j * ref_phase) * g * noise
            field_ref[i, j] = beam * np.exp(1j * ref_phase) * g
    np.save(os.path.join(folder, "field.npy"), field)
    np.save(os.path.join(folder, "field_ref.npy"), field_ref)
    np.save(os.path.join(folder, "taus.npy"), taus)
    np.save(os.path.join(folder, "xis.npy"), np.full(n_tau, XI_PX * config.PIXEL_SIZE))
    return folder


if __name__ == "__main__":
    make_scan(sys.argv[1])
