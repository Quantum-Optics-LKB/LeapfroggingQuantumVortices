"""Loading of the datasets and small array helpers shared by all analyses."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

import config


@dataclass
class Scan:
    """One dataset folder.

    Attributes:
        path: folder containing the .npy files.
        field: complex array (N_tau, N_avg, Ny, Nx) of the fluid with vortices.
        field_ref: same shape, reference fluid (Gaussian beam without vortices).
        taus: (N_tau,) effective interaction times tau = delta_phi.
        xis: (N_tau,) healing length in metres for each tau.
        field_vortex: optional, fluid with a single vortex (used for xi(tau)).
    """

    path: str
    field: np.ndarray
    field_ref: np.ndarray
    taus: np.ndarray
    xis: np.ndarray
    field_vortex: Optional[np.ndarray] = None

    @property
    def n_tau(self) -> int:
        return self.field.shape[0]

    @property
    def n_avg(self) -> int:
        return self.field.shape[1]

    def index_of_tau(self, tau: float, tol: float = 3.0) -> int:
        """Index of the frame whose tau is closest to `tau`."""
        idx = int(np.argmin(np.abs(self.taus - tau)))
        if abs(self.taus[idx] - tau) > tol:
            warnings.warn(
                f"Requested tau={tau} but the closest frame has tau="
                f"{self.taus[idx]:.1f} (index {idx})."
            )
        return idx


def healing_length_from_tau(taus: np.ndarray) -> np.ndarray:
    """xi = sqrt(L / (k0 tau)), healing length deduced from the nonlinear phase."""
    return np.sqrt(config.CELL_LENGTH / (config.K0 * np.asarray(taus, dtype=float)))


def load_scan(path: str, with_vortex: bool = False, mmap: bool = True) -> Scan:
    """Load a dataset folder.

    The big arrays are memory mapped so that loading is instantaneous and only the
    frames that are actually used are read from disk. They are never modified in
    place by the analysis code.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Dataset folder not found: {path!r}. Download the data from Zenodo and "
            "pass the folder with --data (see README)."
        )
    mode = "r" if mmap else None

    def _load(name, mmap_mode=None, required=True):
        f = os.path.join(path, name)
        if not os.path.exists(f):
            if required:
                raise FileNotFoundError(f"Missing file {f}")
            return None
        return np.load(f, mmap_mode=mmap_mode)

    field = _load("field.npy", mode)
    field_ref = _load("field_ref.npy", mode)
    if field.ndim != 4 or field.shape != field_ref.shape:
        raise ValueError(
            "field.npy and field_ref.npy must have the same shape "
            f"(N_tau, N_avg, Ny, Nx); got {field.shape} and {field_ref.shape}."
        )
    taus = np.asarray(_load("taus.npy"), dtype=float)
    xis = _load("xis.npy", required=False)
    if xis is None:
        warnings.warn("xis.npy not found: using xi = sqrt(L/(k0*tau)).")
        xis = healing_length_from_tau(taus)
    field_vortex = _load("field_vortex.npy", mode, required=with_vortex)
    return Scan(path, field, field_ref, taus, np.asarray(xis, dtype=float), field_vortex)


def gaussian_filter_complex(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian filter of a complex array (real and imaginary parts separately)."""
    if sigma <= 0:
        return np.asarray(field, dtype=np.complex128)
    out = np.empty(field.shape, dtype=np.complex128)
    out.real = ndimage.gaussian_filter(np.real(field), sigma)
    out.imag = ndimage.gaussian_filter(np.imag(field), sigma)
    return out


def referenced_field(
    scan: Scan, i: int, j: int = 0, sigma: float = config.GAUSS_SIGMA
) -> np.ndarray:
    """Fluid field of frame (i, j) with the reference phase removed, then smoothed.

    The result is a new array; the dataset is left untouched.
    """
    f = np.asarray(scan.field[i, j], dtype=np.complex128)
    ref = np.asarray(scan.field_ref[i, j], dtype=np.complex128)
    f = f * np.exp(-1j * np.angle(ref))
    return gaussian_filter_complex(f, sigma)


def normalized_amplitude(field: np.ndarray) -> np.ndarray:
    amp = np.abs(field)
    return amp / np.max(amp)


def crop(img: np.ndarray, cy: int, cx: int, half_y: int, half_x: Optional[int] = None):
    """Crop a window centred on pixel (cy, cx) with half sizes (half_y, half_x).

    Raises if the window leaves the image: a plain numpy slice with a negative start
    would silently return a wrong (wrapped or empty) region.
    """
    half_x = half_y if half_x is None else half_x
    y0, y1 = int(cy - half_y), int(cy + half_y)
    x0, x1 = int(cx - half_x), int(cx + half_x)
    if y0 < 0 or x0 < 0 or y1 > img.shape[-2] or x1 > img.shape[-1]:
        raise ValueError(
            f"Crop window rows[{y0}:{y1}] cols[{x0}:{x1}] exceeds image "
            f"{img.shape[-2:]}; reduce the ROI or the offsets."
        )
    return img[..., y0:y1, x0:x1]


def fluid_radius(field_ref_frame: np.ndarray, threshold: float = config.FLUID_THRESHOLD):
    """Equivalent radius (px) of the region where |field_ref| > threshold * max."""
    amp = np.abs(field_ref_frame)
    n_fluid = np.count_nonzero(amp >= threshold * amp.max())
    return float(np.sqrt(n_fluid / np.pi))


def extent_in_xi(shape, xi: float):
    """imshow extent (x/xi, y/xi) of an image centred on its middle pixel."""
    ny, nx = shape[-2:]
    s = config.PIXEL_SIZE / xi
    return [-nx * s / 2, nx * s / 2, -ny * s / 2, ny * s / 2]
