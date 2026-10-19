# Vortex leapfrogging and superfluid dissipation mechanisms in a fluid of light

Code to process the data and reproduce the figures of

> M. Baker-Rasooli, N. du Toit, N. Pavloff, Q. Glorieux,
> *Vortex leapfrogging and superfluid dissipation mechanisms in a fluid of light*,
> arXiv:submit/7060665 [cond-mat.quant-gas] (2025).

**Data:** Zenodo, DOI/link: **TODO: add the Zenodo link here** (`https://doi.org/10.5281/zenodo.XXXXXXX`).

## 1. Quick start

```bash
pip install -r requirements.txt
# 1. download the dataset from Zenodo (link at the top of this page) and unzip it, e.g. in ./zenodo
#    it must contain the three folders listed in section 2
# 2. reproduce a figure
python run_figures.py --data ./zenodo --out results fig2
# or all of them
python run_figures.py --data ./zenodo --out results all
```

Figures are written as `png` and `pdf` in `results/`, together with an `.npz` containing the
plotted numbers.

To check your installation without the dataset, run the self-tests, which build a small
synthetic dataset with the same layout:

```bash
python tests/test_core.py   # unit tests
python tests/run_demo.py    # synthetic data -> figures in results/demo/ (git-ignored)
```

## 2. Dataset layout

Each experiment is a folder of the Zenodo archive:

| Folder | Used for |
|---|---|
| `04302233_sheep_feedbackloop` | Fig. 1 (leapfrogging, 4 feedback loops), Fig. 5 (background flow) |
| `03201801_sheep_jrs` | Fig. 2 (phase slips), Suppl. Figs 2, 3, 5 |
| `04151542_sheep_big` | Fig. 3 (multicharged vortices, dispersive shock wave) |

Files inside a folder (`N_tau` effective times, `N_avg` repetitions, `Ny x Nx` pixels):

| File | Content |
|---|---|
| `field.npy` | complex array `(N_tau, N_avg, Ny, Nx)`, fluid with vortices |
| `field_ref.npy` | same shape, reference fluid without vortices (Gaussian beam) |
| `field_vortex.npy` | same shape, fluid with a single vortex (only for Suppl. Fig. 3) |
| `taus.npy` | `(N_tau,)` effective time `tau = k0 L delta_n` (nonlinear phase shift) |
| `xis.npy` | `(N_tau,)` healing length in metres (if missing: `xi = sqrt(L / (k0 tau))`) |
| `laser_powers.npy` | optional, `(N_tau,)` laser power in W (Suppl. Fig. 2) |

The pixel size of the fields is `PIXEL_SIZE = 6.9 um` (`config.py`).

If you start from the raw interferograms (`dn_{i}_{j}.tiff`, `dn_ref_{i}_{j}.tiff`,
`dn_vortex_{i}_{j}.tiff`), `preprocessing.py` rebuilds the `.npy` files:

```bash
python preprocessing.py /path/to/scan --n-tau 40 --n-avg 5
```

## 3. Which function makes which figure

| Figure | Function (`figures.py`) | What it does |
|---|---|---|
| Fig. 1b | `fig1_snapshots` | amplitude snapshots of the 4 loops |
| Fig. 1c | `fig1_trajectories` | vortex tracking (`tracking.py`), point-vortex model (`point_vortex.py`), barycentre lines |
| Fig. 2 | `fig2_phase_slips` | amplitude, phase + streamlines + vortices/saddles/nodes, unwrapped phase and total velocity along x |
| Fig. 3 | `fig3_dispersive_shock` | input `u_inc`, output amplitude and `u_tot` along x, 1D NLSE (`nlse1d.py`) |
| Fig. 4 | (schematic, no code) | |
| Fig. 5 | `fig5_background_flow` | azimuthally averaged outward flow of the vortex-free beam |
| Suppl. 2 | `supp_n2_isat` | fit of `delta_n(I)`, gives `n2` and `I_sat` |
| Suppl. 3 | `supp_healing_length` | healing length from `tau` and from the vortex core |
| Suppl. 5 | `supp_flow_patterns` | flow patterns during the bifurcations |

Every function has keyword arguments for the ROI, offsets, tau values..., see the docstrings and
the dictionaries `FIG1_*`, `FIG2`, `FIG3` at the top of each section of `figures.py`.

## 4. Method (short)

* **Field.** `referenced_field` divides the field by the phase of `field_ref` (removes the beam
  curvature, `phi_0` in Fig. 2) and smooths it with a Gaussian filter (real and imaginary parts).
* **Velocity.** `velocity.phase_velocity`: gradient of the phase, unwrapped independently along x and
  y (2D unwrapping fails at vortices). With the pixel size in units of `xi` the velocity is in `c_s`.
  The "total velocity" is `u_tot = sqrt(rho) grad(phi)` with `sqrt(rho)` normalised to its maximum.
* **Helmholtz decomposition.** `helmholtz_decomposition` splits `u_tot` into compressible and
  incompressible parts in Fourier space (Appendix A).
* **Critical points.** `detect_vortices`: quantised circulation `±2 pi` on nested plaquettes
  (`r` = largest plaquette). `detect_saddles_nodes`: Poincaré index of the velocity direction
  (Appendix C); detections on top of a vortex core are removed.
* **Tracking.** `tracking.track_vortices`: four vortices per frame, averaged over the repetitions
  that contain exactly four, matched frame to frame with the Hungarian algorithm (with charge
  penalty). Positions are in units of the *local* healing length `xi / sqrt(rho_ref/rho_ref_max)`.
* **Point-vortex model.** `point_vortex.simulate`: Eq. (3), speed `l / r` induced by a vortex of
  charge `l` (units `c_s`, `xi`), plus the outward flow
  `v(r) = 0.35 c_s * 2 (r/20)/(1 + r/20)` for `r < 20 xi`. The default integrator is explicit Euler
  with `dt = 0.1` (the scheme used for the article); `method="rk4"` is available and more accurate.

## 5. Things to know / assumptions

These choices come from the original analysis and could not be checked against the data when this
version was written. They are all parameters, not hidden constants:

1. **Feedback-loop frames (Fig. 1).** The scan is assumed to hold 4 loops of 10 frames
   (`FIG1_LOOP_LENGTH`); the barycentre lines are drawn at frames 10, 20, 30. The re-injection at
   the last transition leaves a jump that was removed by hand (`FIG1_JUMPS = (30,)`), and the
   vortex assignment at frame 5 had to be swapped by hand (`FIG1_SWAPS`). Set `jumps=()` and
   `manual_swaps={}` to see the raw tracking.
2. **Fig. 2 coordinates.** The maps are centred `x0_px = 30`, `y0_px = 4` pixels away from the image
   centre (`FIG2`), the requested times are `config.FIG2_TAUS` and the closest available `tau` is
   used (a warning is issued if it is more than 3 away).
3. **Fig. 3 input.** The "input" profile is taken from the first frame of the scan (lowest power,
   hence almost no nonlinear propagation). The 1D simulations use Gaussian initial states
   (`nlse1d.PRESETS`) with the peak velocity and depth chosen to reproduce the two regimes.
   Section "Fig. 3 / 1D NLSE" of `docs/AUDIT.md` explains how to use measured profiles instead.
4. **Fig. 5 units.** The velocity is in units of `c_s` computed with `xi = sqrt(L/(k0 tau))`, while
   the ROI and the radial axis use the measured `xis.npy`, as in the original code.
5. **Suppl. Fig. 2.** The model is `delta_n = n2 I / (1 + I/I_sat)^2 + b` (what the original code
   fitted). The caption of the supplementary figure quotes `n2 I / (1 + I/I_sat)`: one of the two
   should be corrected.

## 6. Files

```
config.py          constants and default parameters
io_utils.py        dataset loading, reference-phase removal, safe cropping
velocity.py        phase velocity, Helmholtz decomposition, vortex / saddle / node detection
tracking.py        vortex tracking, removal of loop-transition jumps
point_vortex.py    point-vortex model with outward flow
nlse1d.py          1D defocusing NLSE (split-step)
preprocessing.py   raw interferograms -> complex fields (optional)
figures.py         one function per figure
run_figures.py     command line interface
tests/             synthetic dataset + self-tests
docs/AUDIT.md      what was removed from the original code and which bugs were fixed
```

## 7. Citation

If you use this code or the data, please cite the article above and the Zenodo record
(**TODO: add DOI**).

## 8. History

The original research code (one year of accumulated functions) is not part of the current tree.
It is kept in the git history (commit `de9d212` and earlier); `docs/AUDIT.md` lists what was
kept, removed and fixed.

## 9. Use of AI assistance

Claude (Anthropic) was used only for the cleaning of the analysis code, the writing of the tests
and the documentation (commit "Clean analysis code, tests and documentation"), and for nothing
else. The experiments, the data, the analysis methods, the scientific results and the article
are the work of the authors.
