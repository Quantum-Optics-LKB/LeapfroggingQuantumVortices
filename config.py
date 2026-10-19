"""Physical constants and default analysis parameters.

Everything that was hard-coded inside the original analysis functions and that
a user may legitimately want to change lives here.
"""

import numpy as np

# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
WAVELENGTH = 780e-9  # m
K0 = 2 * np.pi / WAVELENGTH  # vacuum wavevector (1/m)
CELL_LENGTH = 20e-2  # m, length of the Rb vapour cell
BEAM_WAIST = 1.7e-3  # m, beam waist at the cell entrance
CAMERA_PIXEL = 3.45e-6  # m, physical camera pixel
# Fields are reconstructed with a 2x lower resolution than the raw interferogram
# (see preprocessing.demodulate), hence the effective pixel of the field arrays:
PIXEL_SIZE = 2 * CAMERA_PIXEL  # m  (= 6.9 um)

# --------------------------------------------------------------------------
# Image processing defaults
# --------------------------------------------------------------------------
GAUSS_SIGMA = 3.0  # px, smoothing applied before computing velocities
FLUID_THRESHOLD = 5e-2  # fraction of max(|field_ref|) defining "inside the fluid"

# --------------------------------------------------------------------------
# Point-vortex model (Fig. 1c), lengths in units of xi, speeds in units of c_s
# --------------------------------------------------------------------------
PV_PAIR_DISTANCE = 1.5  # Delta r : distance between same-sign vortices
PV_PAIR_SEPARATION = 5.0  # Delta R : mean distance between + and - vortices
PV_BACKGROUND_VMAX = 0.35  # outward flow speed at r = PV_BACKGROUND_RMAX (c_s)
PV_BACKGROUND_RMAX = 20.0  # xi, outward flow is switched off beyond this radius
PV_DT = 0.1
PV_TMAX = 57.0

# --------------------------------------------------------------------------
# Datasets (folder names inside the Zenodo archive)
# --------------------------------------------------------------------------
SCAN_FIG1 = "04302233_sheep_feedbackloop"
SCAN_FIG2 = "03201801_sheep_jrs"
SCAN_FIG3 = "04151542_sheep_big"

# Effective times shown in Fig. 2 (the closest available tau is used)
FIG2_TAUS = (89, 120, 123, 144)
# Effective time of the output snapshot in Fig. 3
FIG3_TAU = 140
