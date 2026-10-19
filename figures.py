"""Figure generation: one function per figure (or panel group) of the article.

Every function takes the root folder of the dataset (`data`) and an output folder
(`out`), saves the figure (png + pdf) and the plotted numbers (.npz), and returns
the dictionary of computed quantities.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import ndimage, optimize

import config
import nlse1d
import point_vortex as pv
import tracking
import velocity as vel
from io_utils import (
    Scan,
    crop,
    extent_in_xi,
    fluid_radius,
    gaussian_filter_complex,
    load_scan,
    normalized_amplitude,
    referenced_field,
)

# colours shared by all figures: (+ inner, + outer, - outer, - inner)
VORTEX_COLORS = ["mediumaquamarine", "tab:blue", "tab:red", "darksalmon"]


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _save(fig, out: str, name: str):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")


def _scan_path(data: str, name: str) -> str:
    return os.path.join(data, name)


# ==========================================================================
# Figure 1 : leapfrogging trajectories, experiment vs point-vortex model
# ==========================================================================
# frames shown in Fig. 1b (4 loops x 4 snapshots, 10 frames per loop)
FIG1_FRAMES = (0, 1, 3, 9, 10, 11, 13, 19, 20, 21, 23, 29, 30, 31, 33, 39)
FIG1_LOOP_LENGTH = 10  # frames per feedback loop
FIG1_JUMPS = (30,)  # frame indices where the loop re-injection leaves a jump
FIG1_SWAPS = {5: (0, 1)}  # manual fix of the vortex assignment at frame 5


def fig1_trajectories(
    data: str,
    out: str,
    scan_name: str = config.SCAN_FIG1,
    jumps: Sequence[int] = FIG1_JUMPS,
    manual_swaps: Optional[dict] = None,
    pv_method: str = "euler",
):
    """Fig. 1c: trajectories of the upper (positive) vortices.

    Top: tracked experimental trajectories. Bottom: point-vortex model with the
    outward flow. Dashed lines: x position of the pair barycentre at every loop
    transition.
    """
    plt = _plt()
    scan = load_scan(_scan_path(data, scan_name))
    res = tracking.track_vortices(
        scan, manual_swaps=FIG1_SWAPS if manual_swaps is None else manual_swaps
    )
    raw = res["trajectories"]
    jumps = [k for k in jumps if 0 < k < raw.shape[1]]  # ignore indices not in this scan
    traj = tracking.remove_jumps(raw, jumps) if jumps else raw

    history, _ = pv.simulate(method=pv_method)
    # model order is (+outer, +inner, -outer, -inner); tracking order is
    # (+inner, +outer, -outer, -inner)
    model = history[:, [1, 0, 2, 3], :]

    n_loops = max(1, traj.shape[1] // FIG1_LOOP_LENGTH)
    transitions = [k * FIG1_LOOP_LENGTH for k in range(1, n_loops)]
    bary_x = [np.nanmean(traj[:2, k, 0]) for k in transitions if k < traj.shape[1]]

    fig, axes = plt.subplots(2, 1, figsize=(4, 6), sharex=True)
    for ax, title in zip(axes, ("Experiment", "Point-vortex model")):
        for k in range(2):  # upper (positive) vortices only
            c = VORTEX_COLORS[k]
            if title == "Experiment":
                ax.plot(traj[k, :, 0], traj[k, :, 1], "-o", color=c, mfc="none", ms=3)
            else:
                ax.plot(model[:, k, 0], model[:, k, 1], "-", color=c)
            ax.plot(*tracking.INITIAL_POSITIONS[k], "o", color=c, zorder=3)
        for xb in bary_x:
            ax.axvline(xb, color="k", ls="--", lw=0.8)
        ax.set_ylabel(r"$y/\xi$")
        ax.set_ylim(0, 10)
        ax.set_title(title, fontsize=9)
    axes[-1].set_xlim(-2.5, 15)
    axes[-1].set_xlabel(r"$x/\xi$")
    fig.tight_layout()
    _save(fig, out, "fig1c_trajectories")
    np.savez(
        os.path.join(out, "fig1c_trajectories.npz"),
        trajectories_raw=raw,
        trajectories=traj,
        model=model,
        taus=res["taus"],
        n_found=res["n_found"],
    )
    return {"trajectories": traj, "raw": raw, "model": model}


def fig1_snapshots(
    data: str,
    out: str,
    scan_name: str = config.SCAN_FIG1,
    frames: Sequence[int] = FIG1_FRAMES,
    n_cols: int = 4,
):
    """Fig. 1b: amplitude snapshots sqrt(rho) at the given frame indices."""
    plt = _plt()
    scan = load_scan(_scan_path(data, scan_name))
    frames = [i for i in frames if i < scan.n_tau]  # ignore frames absent from this scan
    n_rows = int(np.ceil(len(frames) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * n_rows))
    for ax, i in zip(np.ravel(axes), frames):
        f = referenced_field(scan, i, 0, sigma=1.0)
        xi = scan.xis[i]
        half = int(7.5 * xi / config.PIXEL_SIZE)
        cy, cx = f.shape[0] // 2, f.shape[1] // 2
        view = crop(np.abs(f), cy, cx, half)
        ax.imshow(view / view.max(), cmap="gray", vmin=0, vmax=1, origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$\tau$={scan.taus[i]:.0f}", fontsize=6)
    fig.tight_layout()
    _save(fig, out, "fig1b_snapshots")
    return {}


# ==========================================================================
# Figure 2 : phase slip nucleation (maps, unwrapped phase, velocity)
# ==========================================================================
FIG2 = dict(roi=7.5, x0_px=30, y0_px=4, sigma=3.0, r_vortex=10, r_critical=2)


def _frame_analysis(scan: Scan, i: int, roi: float, x0_px: int, y0_px: int,
                    sigma: float, r_vortex: int, r_critical: int):
    """Everything needed to draw one frame: cropped maps and critical points."""
    xi = scan.xis[i]
    dx = config.PIXEL_SIZE / xi  # pixel size in xi
    f = referenced_field(scan, i, 0, sigma=sigma)
    amp = normalized_amplitude(f)
    phase = np.angle(f)
    u_tot, u_inc, u_comp = vel.helmholtz_decomposition(amp * np.exp(1j * phase), dx)
    vortices = vel.detect_vortices(phase, r=r_vortex)
    saddles, nodes = vel.detect_saddles_nodes(phase, r=r_critical, vortices=vortices)

    half = int(roi / dx)
    cy, cx = f.shape[0] // 2 - y0_px, f.shape[1] // 2 + x0_px

    def to_xi(p):  # pixel (x, y) -> xi coordinates relative to the crop centre
        return (np.asarray(p, float)[:, :2] - [cx, cy]) * dx if len(p) else np.zeros((0, 2))

    raw = np.asarray(scan.field[i, 0], dtype=np.complex128)
    raw = raw * np.exp(-1j * np.angle(np.asarray(scan.field_ref[i, 0])))
    return dict(
        xi=xi,
        dx=dx,
        extent=[-half * dx, half * dx, -half * dx, half * dx],
        amp_raw=normalized_amplitude(crop(raw, cy, cx, half)),
        phase_raw=np.angle(crop(raw, cy, cx, half)),
        u_tot=crop(u_tot, cy, cx, half),
        u_inc=crop(u_inc, cy, cx, half),
        # profiles along x through the crop centre
        x=(np.arange(2 * half) - half + 0.5) * dx,
        amp_x=crop(amp, cy, cx, half)[half, :] / crop(amp, cy, cx, half).max(),
        phase_x=np.unwrap(crop(phase, cy, cx, half)[half, :]),
        u_x=crop(u_tot, cy, cx, half)[0][half, :],
        vortices=np.column_stack([to_xi(vortices), vortices[:, 2]]) if len(vortices) else np.zeros((0, 3)),
        saddles=to_xi(saddles),
        nodes=to_xi(nodes),
    )


def _in_window(p, half):
    return p[(np.abs(p[:, 0]) <= half) & (np.abs(p[:, 1]) <= half)]


def fig2_phase_slips(
    data: str,
    out: str,
    scan_name: str = config.SCAN_FIG2,
    taus: Sequence[float] = config.FIG2_TAUS,
    params: Optional[dict] = None,
):
    """Fig. 2: amplitude, phase + streamlines + critical points, unwrapped phase and
    total velocity along x, at the requested effective times."""
    plt = _plt()
    p = {**FIG2, **(params or {})}
    scan = load_scan(_scan_path(data, scan_name))
    idx = [scan.index_of_tau(t) for t in taus]
    frames = [_frame_analysis(scan, i, p["roi"], p["x0_px"], p["y0_px"], p["sigma"],
                              p["r_vortex"], p["r_critical"]) for i in idx]

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(2, len(taus) + 1, width_ratios=[1] * len(taus) + [1.4])
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(taus)))
    half = p["roi"]
    for c, (fr, i) in enumerate(zip(frames, idx)):
        ax_a = fig.add_subplot(gs[0, c])
        ax_a.imshow(fr["amp_raw"], cmap="gray", vmin=0, vmax=1, origin="lower", extent=fr["extent"])
        ax_a.plot(fr["x"], fr["amp_x"] * half * 0.8 - half, color="w", lw=1)
        ax_a.axhline(0, color="k", lw=0.5)
        ax_a.set_title(rf"$\tau$={scan.taus[i]:.0f}", fontsize=8)

        ax_p = fig.add_subplot(gs[1, c])
        ax_p.imshow(fr["phase_raw"], cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi,
                    origin="lower", extent=fr["extent"])
        n = fr["u_tot"].shape[-1]
        xs = np.linspace(fr["extent"][0], fr["extent"][1], n)
        ys = np.linspace(fr["extent"][2], fr["extent"][3], n)
        ax_p.streamplot(xs, ys, fr["u_tot"][0], fr["u_tot"][1], color="w", linewidth=0.5, density=1)
        v = _in_window(fr["vortices"], half)
        for q, col in ((-1, "tab:red"), (1, "tab:blue")):
            m = v[:, 2] == q
            ax_p.plot(v[m, 0], v[m, 1], "o", color=col, mec="k", ms=4)
        for pts, col, mk in ((fr["saddles"], "orange", "D"), (fr["nodes"], "green", "D")):
            pts = _in_window(pts, half)
            ax_p.plot(pts[:, 0], pts[:, 1], mk, color=col, mec="k", ms=4)
        ax_p.set_xlim(-half, half)
        ax_p.set_ylim(-half, half)
        for ax in (ax_a, ax_p):
            ax.set_xlabel(r"$x/\xi$")
        ax_a.set_ylabel(r"$y/\xi$")
        ax_p.set_ylabel(r"$y/\xi$")
    ax_b = fig.add_subplot(gs[0, -1])
    ax_c = fig.add_subplot(gs[1, -1])
    for fr, i, col in zip(frames, idx, colors):
        ax_b.plot(fr["x"], (fr["phase_x"] - fr["phase_x"][len(fr["x"]) // 2]) / np.pi, color=col,
                  label=rf"$\tau$={scan.taus[i]:.0f}")
        ax_c.plot(fr["x"], fr["u_x"], color=col)
    ax_b.set_ylabel(r"$(\phi-\phi_0)/\pi$")
    ax_c.set_ylabel(r"$u_x^{tot}/c_s$")
    ax_b.legend(fontsize=6)
    for ax in (ax_b, ax_c):
        ax.set_xlabel(r"$x/\xi$")
        ax.set_xlim(-half, half)
    fig.tight_layout()
    _save(fig, out, "fig2_phase_slips")
    np.savez(
        os.path.join(out, "fig2_profiles.npz"),
        taus=scan.taus[idx],
        x=np.array([fr["x"] for fr in frames], dtype=object),
        phase_x=np.array([fr["phase_x"] for fr in frames], dtype=object),
        u_x=np.array([fr["u_x"] for fr in frames], dtype=object),
        allow_pickle=True,
    )
    return {"frames": frames, "indices": idx}


# ==========================================================================
# Figure 3 : multicharged vortices, dispersive shock wave
# ==========================================================================
FIG3 = dict(roi=60.0, x0_px=0, y0_px=0, sigma=3.0)


def _row_profiles(scan: Scan, i: int, roi: float, x0_px: int, y0_px: int, sigma: float):
    xi = scan.xis[i]
    dx = config.PIXEL_SIZE / xi
    f = referenced_field(scan, i, 0, sigma=sigma)
    amp = normalized_amplitude(f)
    u_tot, u_inc, _ = vel.helmholtz_decomposition(amp * np.exp(1j * np.angle(f)), dx)
    half = int(roi / dx)
    cy, cx = f.shape[0] // 2 - y0_px, f.shape[1] // 2 + x0_px
    x = (np.arange(2 * half) - half + 0.5) * dx
    return dict(
        x=x,
        amp=crop(amp, cy, cx, half)[half, :],
        u_tot=crop(u_tot, cy, cx, half)[0][half, :],
        u_inc=crop(u_inc, cy, cx, half)[0][half, :],
        amp_map=crop(amp, cy, cx, half),
        extent=[-half * dx, half * dx, -half * dx, half * dx],
    )


def fig3_dispersive_shock(
    data: str,
    out: str,
    scan_names: Sequence[str] = (config.SCAN_FIG3,),
    tau_out: float = config.FIG3_TAU,
    params: Optional[dict] = None,
    nlse_presets: Sequence[str] = ("no_dsw", "dsw"),
    tmax_nlse: float = config.FIG3_TAU,
):
    """Fig. 3: input incompressible velocity, output amplitude and total velocity
    along x for each dataset, and the 1D NLSE simulation.

    The "input" is the first frame of the scan (lowest power, hence almost no
    nonlinear evolution); the "output" is the frame closest to `tau_out`.
    """
    plt = _plt()
    p = {**FIG3, **(params or {})}
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    saved = {}
    for n, name in enumerate(scan_names):
        scan = load_scan(_scan_path(data, name))
        i_in, i_out = 0, scan.index_of_tau(tau_out)
        pin = _row_profiles(scan, i_in, p["roi"], p["x0_px"], p["y0_px"], p["sigma"])
        pout = _row_profiles(scan, i_out, p["roi"], p["x0_px"], p["y0_px"], p["sigma"])
        ax[0, 0].plot(pin["x"], pin["u_inc"], label=name)
        ax[0, 1].plot(pout["x"], pout["amp"], label=name)
        ax[1, 0].plot(pout["x"], pout["u_tot"], label=name)
        saved[name] = dict(input=pin, output=pout)
    ax[0, 0].set(xlabel=r"$x/\xi$", ylabel=r"$u_x^{inc}/c_s$", title=r"input ($\tau=0$)")
    ax[0, 1].set(xlabel=r"$x/\xi$", ylabel=r"$\sqrt{\rho}$", title=rf"output ($\tau={tau_out:.0f}$)")
    ax[1, 0].set(xlabel=r"$x/\xi$", ylabel=r"$u_x^{tot}/c_s$")
    ax[1, 0].axhline(0, color="k", lw=0.5, ls="--")
    ax[0, 0].legend(fontsize=6)

    # 1D simulation
    x = np.linspace(-500, 500, 2**13, endpoint=False)
    for name in nlse_presets:
        pr = nlse1d.PRESETS[name]
        rho0 = nlse1d.gaussian_dip_density(x, depth=pr["depth"])
        v0 = nlse1d.gaussian_velocity(x, peak=pr["peak"])
        res = nlse1d.simulate(x, rho0, v0, [tmax_nlse])[tmax_nlse]
        ax[1, 1].plot(x, np.sqrt(res["rho"]), label=name)
        saved[f"nlse_{name}"] = res
    ax[1, 1].set(xlabel=r"$x/\xi$", ylabel=r"$\sqrt{\rho}$", title="1D NLSE", xlim=(-100, 100))
    ax[1, 1].legend(fontsize=6)
    fig.tight_layout()
    _save(fig, out, "fig3_dispersive_shock")
    return saved


# ==========================================================================
# Figure 5 : background (outward) flow
# ==========================================================================
def background_profiles(scan: Scan, frames: Sequence[int], roi: float = 20.0,
                        sigma: float = 2.0, x0_px: int = -10):
    """Azimuthally averaged |u| of the vortex-free fluid versus r/xi."""
    profiles, radii = [], []
    for i in frames:
        xi = scan.xis[i]
        xi_cs = np.sqrt(config.CELL_LENGTH / (config.K0 * scan.taus[i]))
        f = gaussian_filter_complex(np.asarray(scan.field_ref[i, 0], dtype=np.complex128), sigma)
        amp = normalized_amplitude(f)
        # velocity in c_s: pixel size expressed in the healing length of the nonlinear phase
        speed = vel.background_speed(amp * np.exp(1j * np.angle(f)), config.PIXEL_SIZE / xi_cs)
        half = int(roi * xi / config.PIXEL_SIZE)
        shift = 2 * x0_px if i == 0 else x0_px  # frame 0 is centred twice as far from the centre
        view = crop(speed, f.shape[0] // 2, f.shape[1] // 2 + shift, half)
        n_r = half
        r_px = np.linspace(0, half, n_r, endpoint=False)
        theta = np.linspace(0, 2 * np.pi, 2 * half, endpoint=False)
        yy = half + r_px[:, None] * np.sin(theta)[None, :]
        xx = half + r_px[:, None] * np.cos(theta)[None, :]
        polar = ndimage.map_coordinates(view, [yy, xx], order=1)
        profiles.append(polar.mean(axis=1))
        radii.append(r_px * config.PIXEL_SIZE / xi)
    return radii, profiles


def fig5_background_flow(
    data: str,
    out: str,
    scan_name: str = config.SCAN_FIG1,
    taus: Sequence[float] = (12, 40, 62, 78, 91, 104, 112, 119, 126, 133),
):
    """Fig. 5: outward flow u^bgd/c_s versus r/xi for several tau (of one loop)."""
    plt = _plt()
    scan = load_scan(_scan_path(data, scan_name))
    idx = [scan.index_of_tau(t) for t in taus]
    r, prof = background_profiles(scan, idx)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for i, rr, pp in zip(idx, r, prof):
        ax.plot(rr, pp, label=f"{scan.taus[i]:.0f}")
    ax.axvline(11, color="k", ls="--", lw=0.8)
    ax.set(xlabel=r"$r/\xi$", ylabel=r"$u_x^{bgd}/c_s$", xlim=(0, 20))
    ax.legend(title=r"Effective time $\tau$", fontsize=6, ncol=2)
    fig.tight_layout()
    _save(fig, out, "fig5_background_flow")
    np.savez(os.path.join(out, "fig5_background_flow.npz"),
             taus=scan.taus[idx], r=np.array(r, dtype=object), u=np.array(prof, dtype=object),
             allow_pickle=True)
    return {"r": r, "u": prof}


# ==========================================================================
# Supplementary figures
# ==========================================================================
def _delta_n_model(intensity, i_sat, n2, offset):
    return n2 * intensity / (1 + intensity / i_sat) ** 2 + offset


def supp_n2_isat(data: str, out: str, scan_name: str = config.SCAN_FIG2,
                 powers: Optional[np.ndarray] = None):
    """Supplementary Fig. 2: nonlinear index change versus laser intensity.

    tau = delta_phi = k0 L delta_n. The laser powers are read from
    `laser_powers.npy` in the scan folder if it exists, otherwise a linear ramp from
    0.05 W to 2.56 W is assumed (the settings of the original acquisition).
    """
    plt = _plt()
    folder = _scan_path(data, scan_name)
    taus = np.load(os.path.join(folder, "taus.npy"))
    p_file = os.path.join(folder, "laser_powers.npy")
    if powers is None:
        powers = np.load(p_file) if os.path.exists(p_file) else np.linspace(0.05, 2.56, len(taus))
    intensity = 2 * powers / (np.pi * config.BEAM_WAIST**2)
    delta_n = taus / (config.K0 * config.CELL_LENGTH)
    popt, _ = optimize.curve_fit(_delta_n_model, intensity, delta_n, p0=[6e5, 1e-10, 1.0],
                                 maxfev=20000)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(intensity, delta_n, "o", mfc="none")
    ax.plot(intensity, _delta_n_model(intensity, *popt),
            label=rf"$I_{{sat}}$={popt[0]:.2e} W/m$^2$, $n_2$={popt[1]:.2e} m$^2$/W")
    ax.set(xlabel="Laser intensity (W/m$^2$)", ylabel=r"$\Delta n$")
    ax.legend(fontsize=6)
    fig.tight_layout()
    _save(fig, out, "supp2_n2_isat")
    return {"I_sat": popt[0], "n2": popt[1], "offset": popt[2]}


def _vortex_profile(coords, y0, x0, xi):
    y, x = coords
    r = np.hypot(x - x0, y - y0)
    return (r / np.sqrt(r**2 + (2 * xi) ** 2)).ravel()


def supp_healing_length(data: str, out: str, scan_name: str = config.SCAN_FIG2,
                        half_window: int = 10, detect_r: int = 50):
    """Supplementary Fig. 3: healing length versus tau.

    Compares the value deduced from tau with the radius of a single vortex
    measured on `field_vortex.npy` (fit of the core profile r/sqrt(r^2+(2 xi)^2)).
    """
    plt = _plt()
    scan = load_scan(_scan_path(data, scan_name), with_vortex=True)
    size = np.full(scan.n_tau, np.nan)
    size_err = np.full(scan.n_tau, np.nan)
    for i in range(scan.n_tau):
        radii = []
        for j in range(scan.n_avg):
            fv = gaussian_filter_complex(np.asarray(scan.field_vortex[i, j], dtype=np.complex128), 2)
            ref = gaussian_filter_complex(np.asarray(scan.field_ref[i, j], dtype=np.complex128), 2)
            fv = fv * np.exp(-1j * np.angle(ref))
            amp = np.abs(fv) / np.abs(fv).max()
            ny, nx = amp.shape
            v = vel.detect_vortices(np.angle(fv), r=detect_r)
            window = fluid_radius(ref) / 1.5
            v = v[np.hypot(v[:, 0] - nx // 2, v[:, 1] - ny // 2) <= window]
            if len(v) == 0:
                continue
            cx, cy = int(round(v[0, 0])), int(round(v[0, 1]))
            try:
                patch = crop(amp, cy, cx, half_window)
            except ValueError:
                continue
            yy, xx = np.mgrid[: 2 * half_window, : 2 * half_window]
            try:
                popt, _ = optimize.curve_fit(_vortex_profile, (yy, xx), patch.ravel(),
                                             p0=(half_window, half_window, 10))
            except RuntimeError:
                continue
            r_m = popt[2] * config.PIXEL_SIZE
            if r_m < 250e-6:
                radii.append(r_m)
        if radii:
            size[i], size_err[i] = np.mean(radii), np.std(radii)
    xi_tau = np.sqrt(config.CELL_LENGTH / (config.K0 * scan.taus))
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.errorbar(scan.taus, size * 1e6, yerr=size_err * 1e6, fmt="o-", label="Vortex radius")
    ax.plot(scan.taus, xi_tau * 1e6, "s-", label="Average xi")
    ax.set(xlabel=r"$\tau$", ylabel=r"$\xi$ ($\mu$m)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, "supp3_healing_length")
    np.savez(os.path.join(out, "supp3_healing_length.npz"), taus=scan.taus, xi_vortex=size,
             xi_vortex_err=size_err, xi_tau=xi_tau)
    return {"xi_vortex": size, "xi_tau": xi_tau}


def supp_flow_patterns(data: str, out: str, scan_name: str = config.SCAN_FIG2,
                       taus: Sequence[float] = (120.1, 123.2, 129.2, 139.7, 143.5, 144.3),
                       params: Optional[dict] = None):
    """Supplementary Fig. 5: streamlines + critical points for a sequence of frames
    (the experimental data are smoothed over 3 px)."""
    plt = _plt()
    p = {**FIG2, **(params or {})}
    scan = load_scan(_scan_path(data, scan_name))
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for ax, t in zip(np.ravel(axes), taus):
        i = scan.index_of_tau(t, tol=1.5)
        fr = _frame_analysis(scan, i, p["roi"], p["x0_px"], p["y0_px"], p["sigma"],
                             p["r_vortex"], p["r_critical"])
        ax.imshow(fr["amp_raw"], cmap="Blues_r", origin="lower", extent=fr["extent"])
        n = fr["u_tot"].shape[-1]
        xs = np.linspace(fr["extent"][0], fr["extent"][1], n)
        ys = np.linspace(fr["extent"][2], fr["extent"][3], n)
        ax.streamplot(xs, ys, fr["u_tot"][0], fr["u_tot"][1], color="k", linewidth=0.5, density=1.5)
        h = p["roi"]
        v = _in_window(fr["vortices"], h)
        for q, col in ((-1, "tab:red"), (1, "tab:blue")):
            m = v[:, 2] == q
            ax.plot(v[m, 0], v[m, 1], "o", color=col, mec="k", ms=4)
        ax.plot(*_in_window(fr["saddles"], h).T, "D", color="gold", mec="k", ms=4)
        ax.plot(*_in_window(fr["nodes"], h).T, "o", color="limegreen", mec="k", ms=4)
        ax.set_xlim(-h, h)
        ax.set_ylim(-h, h)
        ax.set_title(rf"$\tau$={scan.taus[i]:.1f}", fontsize=8)
    fig.tight_layout()
    _save(fig, out, "supp5_flow_patterns")
    return {}
