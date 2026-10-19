"""Self-tests on synthetic data. Run with:  python tests/test_core.py  (or pytest)."""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import config  # noqa: E402
import nlse1d  # noqa: E402
import point_vortex as pv  # noqa: E402
import tracking  # noqa: E402
import velocity as vel  # noqa: E402
from io_utils import load_scan  # noqa: E402
from make_synthetic_data import make_scan, vortex_field  # noqa: E402


def test_vortex_detection_and_charge():
    yy, xx = np.mgrid[:128, :128].astype(float)
    psi = vortex_field(xx, yy, 40.3, 60.2, +1, 6) * vortex_field(xx, yy, 90.1, 70.4, -1, 6)
    v = vel.detect_vortices(np.angle(psi), r=2)
    assert len(v) == 2, v
    plus = v[v[:, 2] > 0][0]
    minus = v[v[:, 2] < 0][0]
    assert np.hypot(plus[0] - 40.3, plus[1] - 60.2) < 1.5
    assert np.hypot(minus[0] - 90.1, minus[1] - 70.4) < 1.5


def test_helmholtz_is_a_decomposition():
    yy, xx = np.mgrid[:128, :128].astype(float)
    psi = vortex_field(xx, yy, 60, 64, +1, 6) * np.exp(1j * 0.001 * ((xx - 64) ** 2 + (yy - 64) ** 2))
    u, ui, uc = vel.helmholtz_decomposition(psi, 1.0)
    assert np.allclose(u, ui + uc)
    # the incompressible part is divergence free (spectrally)
    k = 2 * np.pi * np.fft.fftfreq(128)
    div = 1j * k[None, :] * np.fft.fft2(ui[0]) + 1j * k[:, None] * np.fft.fft2(ui[1])
    div[64, :] = 0  # Nyquist modes cannot be projected exactly on a real grid
    div[:, 64] = 0
    assert np.abs(div).max() < 1e-8 * np.abs(np.fft.fft2(u)).max()


def test_saddle_detected():
    # phase of two same-sign vortices has a saddle in between
    yy, xx = np.mgrid[:128, :128].astype(float)
    psi = vortex_field(xx, yy, 50, 64, 1, 5) * vortex_field(xx, yy, 78, 64, 1, 5)
    phase = np.angle(psi)
    v = vel.detect_vortices(phase, r=2)
    saddles, nodes = vel.detect_saddles_nodes(phase, r=2, vortices=v)
    assert len(v) == 2
    assert any(abs(s[0] - 64) < 3 and abs(s[1] - 64) < 3 for s in saddles), (saddles, nodes)


def test_point_vortex_pair_translates_and_euler_vs_rk4():
    pos = np.array([[0.0, 0.75], [0.0, -0.75]])
    ch = np.array([1, -1])
    h, _ = pv.simulate(pos, ch, dt=0.01, t_max=5, background=lambda r: 0 * r, method="rk4")
    # a dipole of separation d moves at speed 1/d, perpendicular to the separation
    speed = np.linalg.norm(h[-1, 0] - h[0, 0]) / 5
    assert abs(speed - 1 / 1.5) < 1e-6
    assert np.allclose(h[:, 0, 1] - h[:, 1, 1], 1.5)


def test_nlse_conserves_norm():
    x = np.linspace(-200, 200, 4096, endpoint=False)
    rho0 = nlse1d.gaussian_dip_density(x)
    v0 = nlse1d.gaussian_velocity(x)
    res = nlse1d.simulate(x, rho0, v0, [0, 20.0])
    n0, n1 = res[0]["rho"].sum(), res[20.0]["rho"].sum()
    assert abs(n1 - n0) / n0 < 1e-6


def test_tracking_and_figures():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        make_scan(os.path.join(d, config.SCAN_FIG1))
        scan = load_scan(os.path.join(d, config.SCAN_FIG1))
        res = tracking.track_vortices(scan, drop_last=0)
        traj = res["trajectories"]
        assert not np.isnan(traj).any(), res["n_found"]
        # tracked vortices start at the imposed positions (within 0.3 xi)
        assert np.abs(traj[:, 0, :] - tracking.INITIAL_POSITIONS).max() < 0.3, traj[:, 0, :]
        # and follow the model
        hist, _ = pv.simulate(t_max=24.0)
        model = hist[-1][[1, 0, 2, 3]]
        # positions are expressed in the local healing length xi/sqrt(rho_ref/rho_max)
        # (see tracking._detect_in_frame): undo it with the known synthetic beam
        r_bary = 8.0 * np.hypot(*model.mean(axis=0))
        amp_ref = np.exp(-(r_bary**2) / (2 * 110.0**2))
        assert np.abs(traj[:, -1, :] / amp_ref - model).max() < 0.6, (traj[:, -1, :], model)
        j = tracking.remove_jumps(traj, [5])
        assert np.allclose(j[:, 5, :], traj[:, 4, :])
        assert np.allclose(j[:, :5], traj[:, :5])

        import matplotlib

        matplotlib.use("Agg")
        import figures

        out = os.path.join(d, "out")
        figures.fig1_trajectories(d, out, jumps=(), manual_swaps={})
        figures.fig1_snapshots(d, out, frames=range(8))
        figures.fig5_background_flow(d, out, taus=(30, 60, 90, 120))
        # Fig 2 / Supp 5 / Fig 3 on the same synthetic scan
        make_scan(os.path.join(d, config.SCAN_FIG2))
        figures.fig2_phase_slips(d, out, taus=(30, 60, 100, 140), params=dict(x0_px=0, y0_px=0))
        figures.supp_flow_patterns(d, out, taus=(30, 40, 50, 60, 70, 80), params=dict(x0_px=0, y0_px=0))
        np.save(os.path.join(d, config.SCAN_FIG2, "laser_powers.npy"), np.linspace(0.05, 2.56, 12))
        figures.supp_n2_isat(d, out)
        make_scan(os.path.join(d, config.SCAN_FIG3))
        figures.fig3_dispersive_shock(d, out, params=dict(roi=20.0))
        assert os.path.exists(os.path.join(out, "fig2_phase_slips.png"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name, "...", end=" ", flush=True)
            fn()
            print("ok")
