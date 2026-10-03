"""
sweep.py - Load the parametric design-point sweep and add derived quantities.

Purpose
-------
ANSYS Workbench varied two inputs of the bent-tunnel model:
    radius   - bend centre-line radius [m]
    velocity - inlet velocity [m/s]
and recorded one output:
    dp       - time-averaged pressure drop of the tunnel (dp-avg-op) [Pa]

We convert these into the two dimensionless numbers engineers use for
duct fittings:

    Loss coefficient   K  = dp / (1/2 * rho * v^2)
        -> pressure loss expressed in "velocity heads". K is roughly
           constant at high Re, which is exactly why it is a better ML
           target than dp itself (dp grows ~ v^2, K barely changes).

    Reynolds number    Re = rho * v * D_h / mu

DP 0 is excluded: it was produced by a different run and is not consistent
with the rest of the sweep.
"""

from . import config
from .fluent_io import read_workbench_csv
from .validation import reynolds


def dynamic_pressure(velocity, rho=config.RHO):
    """Velocity head 1/2 rho v^2 [Pa]."""
    return 0.5 * rho * velocity ** 2


def load_sweep():
    df = read_workbench_csv(config.RESULTS_DIR / config.SWEEP_FILE,
                            rename=config.SWEEP_COLUMNS,
                            exclude=config.SWEEP_EXCLUDE)
    df["K"] = df["dp"] / dynamic_pressure(df["velocity"])
    df["Re"] = reynolds(df["velocity"])
    return df.sort_values(["radius", "velocity"]).reset_index(drop=True)
