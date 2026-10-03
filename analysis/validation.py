"""
validation.py - Verify the CFD set-up against a textbook result.

Purpose
-------
Before trusting the bend simulations we check that the same solver settings
reproduce a case with a known answer: fully developed turbulent flow in a
straight 4 x 4 m duct at 3 m/s.

CFD side
    Static pressure is monitored on two planes, at x = 60 m and x = 100 m
    (far enough downstream for the flow to be fully developed). The pressure
    drop over those 40 m is  dp_CFD = p(60) - p(100), taken at the final
    iteration.

Theory side (Darcy-Weisbach)
    dp = f * (L / D) * (1/2) * rho * v^2

    with the Petukhov friction factor for smooth pipes
    f = (0.790 * ln(Re) - 1.64)^(-2),   valid for 3e3 < Re < 5e6

    and Re = rho * v * D / mu. For a square duct the hydraulic diameter
    D_h = 4 m is used in place of D.

The % error between the two tells us how accurate the wall treatment and
turbulence model (k-omega SST) are for this class of flow.
"""

import numpy as np

from . import config
from .fluent_io import read_fluent_out


def reynolds(velocity, rho=config.RHO, mu=config.MU, d=config.D_H):
    """Reynolds number based on hydraulic diameter."""
    return rho * velocity * d / mu


def petukhov_friction_factor(re):
    """Darcy friction factor for smooth pipes (Petukhov, 1970)."""
    return (0.790 * np.log(re) - 1.64) ** -2


def darcy_weisbach_dp(velocity, length, rho=config.RHO, d=config.D_H):
    """Frictional pressure drop over a straight length of duct [Pa]."""
    f = petukhov_friction_factor(reynolds(velocity))
    return f * (length / d) * 0.5 * rho * velocity ** 2


def run_validation() -> dict:
    p_up = read_fluent_out(config.RESULTS_DIR / config.VAL_UPSTREAM_FILE).iloc[:, 0]
    p_dn = read_fluent_out(config.RESULTS_DIR / config.VAL_DOWNSTREAM_FILE).iloc[:, 0]

    dp_history = p_up - p_dn          # pressure drop at every iteration
    dp_cfd = float(dp_history.iloc[-1])  # converged (final-iteration) value

    v, L = config.VAL_VELOCITY, config.VAL_LENGTH
    re = reynolds(v)
    f = petukhov_friction_factor(re)
    dp_theory = darcy_weisbach_dp(v, L)

    return {
        "p_up": p_up,
        "p_dn": p_dn,
        "dp_history": dp_history,
        "dp_cfd": dp_cfd,
        "dp_theory": float(dp_theory),
        "re": float(re),
        "f": float(f),
        "pct_error": float((dp_cfd - dp_theory) / dp_theory * 100.0),
        "final_iter": int(dp_history.index.max()),
    }
