"""
mesh_study.py - Mesh-independence check.

Purpose
-------
A CFD result is only trustworthy if refining the mesh no longer changes it
meaningfully. Three meshes of the bent tunnel were run (coarse / medium /
fine). For each one we:

  1. read the inlet-pressure history (inlet pressure ~ total pressure drop of
     the tunnel, since the outlet is at 0 Pa gauge),
  2. average it over iterations 300-559 - late enough that the start-up
     transient has died out, and limited to 559 because the fine run was
     stopped there (all meshes must use the same window to be comparable),
  3. compute the % change from one mesh to the next.

A small change from medium -> fine means the medium mesh is "good enough"
and can be used for the parametric sweep at much lower cost.
"""

import pandas as pd

from . import config
from .fluent_io import read_fluent_out


def run_mesh_study() -> dict:
    """Return per-mesh histories and a summary table with averages and % changes."""
    histories = {}
    rows = []
    for label, fname, cells in config.MESHES:
        df = read_fluent_out(config.RESULTS_DIR / fname)
        series = df.iloc[:, 0]                       # the inlet-pressure column
        window = series.loc[config.AVG_ITER_START:config.AVG_ITER_END]
        histories[label] = series
        rows.append({
            "mesh": label,
            "cells": cells,
            "last_iter": int(series.index.max()),
            "p_avg": window.mean(),
            # Standard deviation inside the window shows how steady the
            # signal is (large value -> still oscillating / drifting).
            "p_std": window.std(),
        })

    table = pd.DataFrame(rows)
    # % change relative to the previous (coarser) mesh
    table["pct_change"] = table["p_avg"].pct_change() * 100.0
    return {"histories": histories, "table": table}
