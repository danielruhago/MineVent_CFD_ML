"""
figures.py - All plotting, kept separate from the calculations.

Purpose
-------
Each function takes the result dictionary of one analysis step and saves a
300-dpi PNG into figures/. Keeping plots apart from the maths means the
numbers can be re-used (e.g. in a report table) without re-plotting, and the
figure style is defined in one place.

Figures produced
----------------
fig1_mesh_study.png      convergence histories + averaged inlet pressure vs cells
fig2_validation.png      plane pressures + CFD dp vs Darcy-Weisbach
fig3_design_curves.png   dp vs velocity per bend radius (surrogate + CFD points)
fig4_pred_vs_actual.png  leave-one-out predicted vs CFD dp, with y = x line
"""

import matplotlib

matplotlib.use("Agg")      # render to file, no window needed
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .surrogate import predict_dp

# --------------------------------------------------------------------------
# Shared style: colour-blind-safe categorical colours (validated palette,
# first three slots), dark text, light recessive grid.
# --------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "savefig.dpi": config.FIG_DPI,
    "savefig.bbox": "tight",
})


def _save(fig, name):
    config.FIGURES_DIR.mkdir(exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def plot_mesh_study(mesh):
    table = mesh["table"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: convergence histories, with the averaging window shaded
    ax1.axvspan(config.AVG_ITER_START, config.AVG_ITER_END, color=GRID, alpha=0.6,
                lw=0, label=f"Averaging window ({config.AVG_ITER_START}-"
                            f"{config.AVG_ITER_END})")
    for color, (label, series) in zip(SERIES, mesh["histories"].items()):
        ax1.plot(series.index, series.values, color=color, lw=1.5, label=label)
    ax1.set(xlabel="Iteration", ylabel="Inlet pressure [Pa]",
            title="Convergence history")
    ax1.legend(loc="lower right")

    # Right: window-averaged inlet pressure vs cell count
    ax2.plot(table["cells"], table["p_avg"], color=INK_2, lw=1.5, zorder=1)
    ax2.scatter(table["cells"], table["p_avg"], c=SERIES, s=70, zorder=2,
                edgecolor="white", linewidth=2)
    for _, row in table.iterrows():
        txt = f"{row['mesh']}\n{row['p_avg']:.3f} Pa"
        if not np.isnan(row["pct_change"]):
            txt += f"\n({row['pct_change']:+.2f}%)"
        ax2.annotate(txt, (row["cells"], row["p_avg"]), textcoords="offset points",
                     xytext=(0, 12), ha="center", va="bottom", fontsize=9, color=INK)
    ax2.set(xlabel="Number of cells", ylabel="Mean inlet pressure [Pa]",
            title="Mesh independence")
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x / 1e3:.0f}k"))
    lo, hi = table["p_avg"].min(), table["p_avg"].max()
    pad = (hi - lo) * 0.6 + 0.05
    ax2.set_ylim(lo - pad * 0.4, hi + pad)
    ax2.set_xlim(0, table["cells"].max() * 1.12)
    fig.tight_layout()
    return _save(fig, "fig1_mesh_study.png")


# --------------------------------------------------------------------------
def plot_validation(val):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(val["p_up"].index, val["p_up"].values, color=BLUE, label="Plane x = 60 m")
    ax1.plot(val["p_dn"].index, val["p_dn"].values, color=ORANGE, label="Plane x = 100 m")
    ax1.set(xlabel="Iteration", ylabel="Static pressure [Pa]",
            title="Monitored plane pressures")
    ax1.legend()

    h = val["dp_history"]
    ax2.plot(h.index, h.values, color=BLUE, label="CFD  p(60) - p(100)")
    ax2.axhline(val["dp_theory"], color=INK_2, ls="--", lw=1.5,
                label="Darcy-Weisbach (Petukhov f)")
    ax2.annotate(f"CFD: {val['dp_cfd']:.4f} Pa\n"
                 f"Theory: {val['dp_theory']:.4f} Pa\n"
                 f"Error: {val['pct_error']:+.2f}%",
                 xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
                 fontsize=9, color=INK,
                 bbox=dict(boxstyle="round", fc="white", ec=GRID))
    ax2.set(xlabel="Iteration", ylabel="Pressure drop over 40 m [Pa]",
            title=f"Straight duct, v = {config.VAL_VELOCITY:g} m/s, "
                  f"Re = {val['re']:.2e}")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, "fig2_validation.png")


# --------------------------------------------------------------------------
def plot_design_curves(sweep, sur):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    v_fine = np.linspace(sweep["velocity"].min(), sweep["velocity"].max(), 100)

    for color, (radius, grp) in zip(SERIES, sweep.groupby("radius")):
        ax.plot(v_fine, predict_dp(sur["mlp"], radius, v_fine), color=color,
                label=f"R = {radius:g} m  (MLP surrogate)")
        ax.plot(v_fine, predict_dp(sur["linear"], radius, v_fine), color=color,
                lw=1, ls="--", alpha=0.8)
        ax.plot(v_fine, predict_dp(sur["const_k"], radius, v_fine), color=color,
                lw=1.2, ls=":", alpha=0.9)
        ax.scatter(grp["velocity"], grp["dp"], color=color, s=60, zorder=3,
                   edgecolor="white", linewidth=2)

    # Legend entries explaining the line/marker styles
    ax.plot([], [], color=INK_2, lw=1, ls="--", label="Linear baseline")
    ax.plot([], [], color=INK_2, lw=1.2, ls=":", label="Constant K per radius")
    ax.scatter([], [], color=INK_2, s=60, edgecolor="white", label="CFD design points")
    ax.set(xlabel="Inlet velocity [m/s]", ylabel="Pressure drop [Pa]",
           title="Design curves: bend pressure drop")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig3_design_curves.png")


# --------------------------------------------------------------------------
def plot_pred_vs_actual(sweep, sur):
    dp = sweep["dp"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lim = [0, dp.max() * 1.15]

    ax.plot(lim, lim, color=INK_2, lw=1.2, label="y = x (perfect)")
    ax.fill_between(lim, [l * 0.9 for l in lim], [l * 1.1 for l in lim],
                    color=GRID, alpha=0.6, lw=0, label="±10% band")
    ax.scatter(dp, sur["dp_loo_mlp"], color=BLUE, s=60, zorder=3,
               edgecolor="white", linewidth=2,
               label=f"MLP  (MAPE {sur['mape_mlp']:.1f}%)")
    ax.scatter(dp, sur["dp_loo_lin"], color=ORANGE, marker="s", s=50, zorder=3,
               edgecolor="white", linewidth=2,
               label=f"Linear  (MAPE {sur['mape_lin']:.1f}%)")
    ax.scatter(dp, sur["dp_loo_ck"], color=AQUA, marker="^", s=60, zorder=3,
               edgecolor="white", linewidth=2,
               label=f"Constant K/radius  (MAPE {sur['mape_ck']:.1f}%)")
    ax.set(xlim=lim, ylim=lim, aspect="equal",
           xlabel="CFD pressure drop [Pa]", ylabel="Predicted pressure drop [Pa]",
           title="Leave-one-out predictions")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig4_pred_vs_actual.png")
