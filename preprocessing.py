"""From raw interferograms (.tiff) to complex fields (field.npy ...).

Only needed if you start from the raw images of the dataset; the figures themselves
only need the .npy files. Expected files in the scan folder, for every time index i
and repetition j:

    dn_{i}_{j}.tiff          fluid with vortices
    dn_ref_{i}_{j}.tiff      reference fluid (no vortex)
    dn_vortex_{i}_{j}.tiff   fluid with one vortex (healing length measurement)

Output: field.npy, field_ref.npy, field_vortex.npy (N_tau, N_avg, Ny/2, Nx/2,
complex64), taus.npy and taus_err.npy.
"""

from __future__ import annotations

import os

import numpy as np

import config


def demodulate(im: np.ndarray, center=None, radius=None) -> np.ndarray:
    """Off-axis holography: complex field from an interferogram.

    The interferogram carries fringes at (row, col) = `center` in the half-plane
    Fourier transform. The sideband is cropped around it, low-pass filtered with a
    disk of `radius` px and transformed back. The result has half the number of
    pixels of `im` along each axis. Its global phase is fixed by the central pixel.
    """
    ny, nx = im.shape
    center = (ny // 2, nx // 4) if center is None else center
    radius = ny // 8 if radius is None else radius
    spec = np.fft.rfft2(im.astype(np.float32))
    spec = np.roll(
        spec,
        (-center[0] + spec.shape[0] // 2, -center[1] + spec.shape[1] // 2),
        axis=(0, 1),
    )
    spec = spec[spec.shape[0] // 4 : -(spec.shape[0] // 4), : spec.shape[1] - 1]
    yy, xx = np.ogrid[: spec.shape[0], : spec.shape[1]]
    disk = (yy - spec.shape[0] // 2) ** 2 + (xx - spec.shape[1] // 2) ** 2 < radius**2
    field = np.fft.ifft2(np.fft.fftshift(spec * disk))
    centre_phase = np.angle(field[field.shape[0] // 2, field.shape[1] // 2])
    return (field * np.exp(-1j * centre_phase)).astype(np.complex64)


def nonlinear_phase(field_ref: np.ndarray, threshold: float = 2e-2) -> float:
    """tau = peak-to-peak phase of the reference beam (nonlinear phase shift)."""
    from skimage.restoration import unwrap_phase  # optional dependency

    amp2 = np.abs(field_ref) ** 2
    mask = amp2 < threshold * amp2.max()
    phi = unwrap_phase(np.ma.array(np.angle(field_ref), mask=mask), wrap_around=(True, True))
    return float(np.abs(np.nanmax(phi) - np.nanmin(phi)))


def _read(path: str) -> np.ndarray:
    from PIL import Image  # optional dependency

    return np.array(Image.open(path), dtype=np.float32)


def process_scan(scan_dir: str, n_tau: int, n_avg: int, with_vortex: bool = True):
    """Demodulate all interferograms of a time scan and write the .npy files."""
    from tqdm import tqdm  # optional dependency

    first = _read(os.path.join(scan_dir, "dn_0_0.tiff"))
    shape = (n_tau, n_avg, first.shape[0] // 2, first.shape[1] // 2)
    field = np.zeros(shape, dtype=np.complex64)
    field_ref = np.zeros(shape, dtype=np.complex64)
    field_vortex = np.zeros(shape, dtype=np.complex64) if with_vortex else None
    taus = np.zeros(n_tau)
    taus_err = np.zeros(n_tau)

    for i in tqdm(range(n_tau), desc="Demodulating"):
        tau_j = np.zeros(n_avg)
        for j in range(n_avg):
            field[i, j] = demodulate(_read(os.path.join(scan_dir, f"dn_{i}_{j}.tiff")))
            field_ref[i, j] = demodulate(
                _read(os.path.join(scan_dir, f"dn_ref_{i}_{j}.tiff"))
            )
            if with_vortex:
                field_vortex[i, j] = demodulate(
                    _read(os.path.join(scan_dir, f"dn_vortex_{i}_{j}.tiff"))
                )
            tau_j[j] = nonlinear_phase(field_ref[i, j])
        taus[i], taus_err[i] = tau_j.mean(), tau_j.std()

    np.save(os.path.join(scan_dir, "field.npy"), field)
    np.save(os.path.join(scan_dir, "field_ref.npy"), field_ref)
    if with_vortex:
        np.save(os.path.join(scan_dir, "field_vortex.npy"), field_vortex)
    np.save(os.path.join(scan_dir, "taus.npy"), taus)
    np.save(os.path.join(scan_dir, "taus_err.npy"), taus_err)
    return taus


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("scan_dir")
    p.add_argument("--n-tau", type=int, required=True)
    p.add_argument("--n-avg", type=int, required=True)
    p.add_argument("--no-vortex", action="store_true")
    a = p.parse_args()
    process_scan(a.scan_dir, a.n_tau, a.n_avg, with_vortex=not a.no_vortex)
