"""
config.py - Single source of truth for every constant used in the pipeline.

Purpose
-------
All physical properties, file names and analysis settings live here so that
no "magic numbers" are scattered through the other scripts. If you change the
fluid properties or the averaging window, change it once here and every
result and figure updates consistently.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Folders
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"     # raw Fluent / Workbench exports
FIGURES_DIR = PROJECT_ROOT / "figures"     # all generated figures go here
FIG_DPI = 300                              # publication-quality resolution

# --------------------------------------------------------------------------
# Fluid properties (air, Fluent defaults) and duct geometry
# --------------------------------------------------------------------------
RHO = 1.225          # density [kg/m^3]
MU = 1.7894e-5       # dynamic viscosity [Pa s]
D_H = 4.0            # hydraulic diameter of the 4 x 4 m square tunnel [m]
                     # (for a square duct D_h = 4A/P = 4*16/16 = 4 m)

# --------------------------------------------------------------------------
# 1) Mesh-independence study
# --------------------------------------------------------------------------
# (label, file name, cell count)
MESHES = [
    ("Coarse", "mesh1_coarse.out", 69_202),
    ("Medium", "mesh2_medium.out", 197_202),
    ("Fine",   "mesh3_fine.out",   466_684),
]
# The fine mesh run was stopped at iteration 559, so every mesh is averaged
# over the SAME window (300-559) to keep the comparison fair.
AVG_ITER_START = 300
AVG_ITER_END = 559

# --------------------------------------------------------------------------
# 2) Validation case: straight 4 x 4 m duct at 3 m/s
# --------------------------------------------------------------------------
VAL_UPSTREAM_FILE = "validation_p60.out"     # plane at x = 60 m
VAL_DOWNSTREAM_FILE = "validation_p100.out"  # plane at x = 100 m
VAL_LENGTH = 40.0                            # distance between planes [m]
VAL_VELOCITY = 3.0                           # bulk velocity [m/s]

# --------------------------------------------------------------------------
# 3) Parametric sweep (ANSYS Workbench design-point export)
# --------------------------------------------------------------------------
SWEEP_FILE = "sweep_transient.csv"
SWEEP_EXCLUDE = ["DP 0"]   # DP 0 came from a different run -> excluded
# Workbench names columns P1, P2, P3; map them to readable names.
SWEEP_COLUMNS = {"P1": "radius", "P2": "velocity", "P3": "dp"}

# --------------------------------------------------------------------------
# 4) ML surrogate settings
# --------------------------------------------------------------------------
SEED = 42                # makes training reproducible
HIDDEN_UNITS = 8         # one small hidden layer: 2 -> 8 -> 1 (33 weights)
N_ENSEMBLE = 5           # average 5 nets with different seeds (reduces
                         # sensitivity to random initialisation)
EPOCHS = 3000
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-3      # L2 penalty: keeps the fit smooth with 8 points
CFD_RUNTIME_HOURS = 2.5  # wall-clock time of one transient CFD design point
