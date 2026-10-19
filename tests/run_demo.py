"""Build a synthetic dataset and produce test figures, without the Zenodo data.

    python tests/run_demo.py            # data in demo_data/, figures in results/demo/

Both folders are ignored by git. The synthetic data only mimic the layout of the real
ones (four vortices moved by the point-vortex model on a Gaussian beam), so the figures
are meant to check the installation, not to reproduce the article.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import config  # noqa: E402
import figures  # noqa: E402
from make_synthetic_data import make_scan  # noqa: E402

DATA = os.path.join(ROOT, "demo_data")
OUT = os.path.join(ROOT, "results", "demo")

for name in (config.SCAN_FIG1, config.SCAN_FIG2, config.SCAN_FIG3):
    if not os.path.exists(os.path.join(DATA, name, "field.npy")):
        print("creating synthetic scan", name)
        make_scan(os.path.join(DATA, name))

# demo images are small (384 px) and have 12 frames: smaller ROIs / other tau values
small = dict(x0_px=0, y0_px=0)
print("fig1");  figures.fig1_trajectories(DATA, OUT); figures.fig1_snapshots(DATA, OUT, frames=range(8))
print("fig2");  figures.fig2_phase_slips(DATA, OUT, taus=(30, 60, 100, 140), params=small)
print("fig3");  figures.fig3_dispersive_shock(DATA, OUT, params=dict(roi=20.0))
print("fig5");  figures.fig5_background_flow(DATA, OUT, taus=(30, 60, 90, 120))
print("supp5"); figures.supp_flow_patterns(DATA, OUT, taus=(30, 40, 50, 60, 70, 80), params=small)
print("supp2")
import numpy as np  # noqa: E402

np.save(os.path.join(DATA, config.SCAN_FIG2, "laser_powers.npy"), np.linspace(0.05, 2.56, 12))
print(figures.supp_n2_isat(DATA, OUT))
print("Figures in", OUT)
