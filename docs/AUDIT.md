# Audit of the original code

The original files (`leapfrog_processing.py`, `velocity.py`, `contrast.py`, `nlse.py`,
`nlse_1d.py`, ~10 000 lines) are no longer in the tree (they remain in the git history, commit `de9d212`, and locally in the git-ignored `old/` folder). This document
lists what the clean version keeps, what it drops, and the bugs that were fixed.
Only `__main__` of `leapfrog_processing.py` was ever run, with every call commented except one, and
none of the functions saved a figure except through `plt.show()`.

## 1. Function by function (`leapfrog_processing.py`)

| Original function | Verdict | Where it went |
|---|---|---|
| `compute_phase_time` | kept (optional) | `preprocessing.process_scan` |
| `compute_phase_full_time`, `compute_phase_dist`, `compute_phase_time_single` | dropped | variants for other acquisitions / full-resolution fields |
| `compute_n2_Isat` | kept | `figures.supp_n2_isat` |
| `compute_final_xi` | kept | `figures.supp_healing_length` |
| `vortex_tracking_corrected` | kept | `tracking.track_vortices`, `tracking.remove_jumps` |
| `vortex_tracking`, `plot_traj` | dropped | earlier versions of the above |
| `point_vortex_BEC` | kept | `point_vortex.simulate` |
| `point_vortex` | dropped | prototype without outward flow |
| `velocity_decomp` | kept (as template) | `figures._frame_analysis`, `fig2_phase_slips`, `supp_flow_patterns` |
| `phase_x` | kept | unwrapped phase profile in `_frame_analysis` |
| `field_velocity_x` | kept | velocity profile along x (`_frame_analysis`, `_row_profiles`) |
| `velocity_bgd` | kept | `figures.background_profiles`, `fig5_background_flow` |
| `plot_fields`, `plot_energy`, `energy_time`, `make_mp4_and_gif` | dropped | movies and energy diagnostics, not in the article |
| `velocity_fields`, `velocity_x`, `one_jrs`, `single_shot` | dropped | early prototypes of the Fig. 2 maps |
| `rho_x` | dropped | prototype with a bug (see 3) |
| `field_velocity_x_fit` | dropped | Lorentzian fits of the pair velocity, no figure of the article |
| `velocity_bgd_test` | dropped | alternative estimate of the background |
| `vortex_fit` | dropped | duplicate of the vortex-core fit in `compute_final_xi` |
| `equiphase` | dropped | exploratory (stream-function reconstruction); it was the only active call of `__main__` |
| `vortex_jrs_analytic` (defined twice), `JRS_padé`, `Soliton1D`, `Vortex`, `Vortex2D` | dropped (except the core profile used for the xi fit) | SLM/analytic initial states, not needed to analyse the data |
| `load_cached_array`, `mask_vortices`, `reinstate_vortices`, `power_law`, `natural_sort` | dropped | see 3 |

`velocity.py`: kept `velocity`, `helmholtz_decomp`, `vortex_detection`, `sell_detection` (rewritten
in numpy). Dropped: energy spectra, clustering (`cluster_*`, `ck`), correlations, Bessel reductions,
drag force, `velocity_fft`, `vortex_detection_schiebler`, `helmholtz_decomp_phase`.
`contrast.py`: only `im_osc_fast_t` / `exp_angle_fast` were needed (now `preprocessing.demodulate`
and numpy). `nlse.py` (GPU classes, 2400 lines) is not used by any figure; `nlse_1d.py` is the
1D simulation, reduced to `nlse1d.py`.

## 2. Heavy dependencies removed

`numba`, `pyfftw` (and its `fft.wisdom` file written in the working directory at every call),
`cupy`/GPU, `cv2`, `regex`, `cmasher`, `polarTransform`, `networkx`, `faulthandler`.

## 3. Bugs and hazards fixed

1. **Hard-coded paths**: `path_cache = "C:/Users/.../data_cache"` made every function fail on
   another machine; `load_cached_array` was also keyed on the file name only, so data of a previous
   scan could be reloaded silently for 10 hours. Replaced by `np.load(..., mmap_mode="r")`.
2. **In-place modification of the dataset**: `contrast.exp_angle_fast(field, field_ref)` modified
   `fields[i, j]` (a view of the loaded array). Re-running or averaging after a `continue` used
   already corrected data. The new code always works on copies.
3. **Loop-transition correction (Fig. 1c)**: `trajectory[j] -= trajectory[30] - trajectory[29]` was applied while
   `trajectory[30]` itself was being modified, so after `j = 30` the step was zero: only frame 30 was
   corrected. `tracking.remove_jumps` measures the step once and removes it from all later frames,
   which is what the code intended. The saved `.npy` was also written *before* the correction.
   **This may slightly change the last part of the experimental trajectories of Fig. 1c.**
4. **Wrong index order** in the local healing length of the tracking
   (`xi[mean_y + Nx//2, mean_x + Ny//2]`): only correct for square images.
5. **Silent wrap-around in crops**: `a[c - roi : c + roi]` with a negative start returns an empty or
   wrong region. `io_utils.crop` raises an explicit error.
6. **`plt.show()` ignoring `plot=False`** (40 pop-up windows in several functions), no `plt.close`.
7. **Ragged arrays** (`np.array(all_vortices)`) in the tracking, unused but a crash on recent numpy;
   a frame without four vortices crashed `np.nanmean` of an empty list.
8. **Vortex detection** used numba functions with a fixed `float32` signature and wrapped around the
   image border (spurious detections); the vectorised version ignores plaquettes crossing the border
   and merges adjacent flagged pixels into a single point.
9. **Saddle/node detection printed 3 lines per call**, and nodes included vortex cores (index +1);
   they are now excluded within 3 px of a detected vortex.
10. **`rho_x`** multiplied the 2D phase by the 1D averaged amplitude (wrong broadcast) and
    `delta_Rs[i]` overran its array. Dropped.
11. **`complex` Gaussian filtering**: `scipy.ndimage.gaussian_filter` on complex arrays is not
    supported on old scipy; real and imaginary parts are filtered separately.
12. **First-frame ordering** of the tracked vortices depended on the detection order; they are now
    ordered by charge then y, and the matching penalises charge changes.
13. **Euler time stepping** of the point-vortex model is not symplectic. With `dt = 0.1` the final
    positions differ from an RK4 integration by about 1 xi (compare `method="euler"` and
    `"rk4"`). Euler is kept as the default so that the figure of the article is reproduced;
    showing the RK4 curves as a check is worthwhile.
14. `velocity_decomp` called the Helmholtz decomposition with `dx` in metres instead of units of
    `xi` (velocity not in `c_s`); the new code always uses `PIXEL_SIZE / xi`.
15. **Writing into the data folder** (`field_processed_*.npy`, `v_x_t.npy`, ...): outputs now go to
    `--out`.

## 4. Inconsistencies noticed in the article (to check)

* Fig. 2 caption: times `89, 120, 125, 144`; the text and the panels say `123`.
* Fig. 2 caption: initial positions `(0, ±1.75 xi)` and `(0, ±3.25 xi)` (the configuration of
  Fig. 1), whereas the text says `(0, ±3 xi)` and `(5 xi, ±3 xi)` (`Delta R = 6`, `Delta r = 5`).
* Suppl. Fig. 2 caption: `n2 I/(1 + I/I_sat)` versus `n2 I/(1 + I/I_sat)^2 + b` in the code.
* Fig. 1c: the same vortex had different colours in the tracking and in the model plots.
  Colours now follow the vortex identity.
* Appendix B: `Lambda = 0.82` in the text; the SLM code of the original repository used `xi/0.83`.

## 5. Fig. 3 / 1D NLSE with measured profiles

The article states that the 1D simulations start from the measured input profiles. The shipped
presets (`nlse1d.PRESETS`) are Gaussian approximations. To use the measured ones:

```python
from figures import _row_profiles
from io_utils import load_scan
import nlse1d, numpy as np

scan = load_scan("zenodo/04151542_sheep_big")
p = _row_profiles(scan, 0, roi=60, x0_px=0, y0_px=0, sigma=3.0)   # first frame = input
x = p["x"]
rho0 = p["amp"] ** 2
v0 = p["u_inc"] / np.maximum(p["amp"], 0.05)                       # u = sqrt(rho) v
out = nlse1d.simulate(x, rho0, v0, [140.0])[140.0]
```

(the grid must be uniform and large enough that the density is flat at the boundaries; pad with
the edge values otherwise).

## 6. Not reproduced

The dispersive-shock-wave theory of the Supplementary Material (Gurevich-Pitaevskii solution,
Figs 6-7, `Gamma(u_L - u_R)`) is not in the original repository and is not part of this version.
