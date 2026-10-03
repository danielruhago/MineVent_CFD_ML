"""
run_all.py - Run the complete post-processing pipeline end to end.

Usage
-----
    python run_all.py

What it does (in order)
-----------------------
1. Mesh study       - is the medium mesh fine enough?          (mesh_study.py)
2. Validation       - does CFD match Darcy-Weisbach?           (validation.py)
3. Parametric sweep - load design points, compute K and Re     (sweep.py)
4. ML surrogate     - MLP vs linear vs constant-K, LOO-CV, timing (surrogate.py)
5. Figures          - saved to figures/ at 300 dpi             (figures.py)
6. Summary          - printed to the console and saved to results_summary.txt
"""

from analysis import config, figures
from analysis.mesh_study import run_mesh_study
from analysis.surrogate import run_surrogate
from analysis.sweep import load_sweep
from analysis.validation import run_validation

LINE = "=" * 72


def format_summary(mesh, val, sweep, sur, fig_paths) -> str:
    out = []
    add = out.append

    add(LINE)
    add(" MINE VENTILATION TUNNEL (90-deg bend) - CFD + ML RESULTS SUMMARY")
    add(LINE)

    # 1. Mesh study --------------------------------------------------------
    add(f"\n1) MESH STUDY  (inlet pressure averaged over iterations "
        f"{config.AVG_ITER_START}-{config.AVG_ITER_END})")
    add(f"   {'Mesh':<8}{'Cells':>10}{'Last it.':>10}{'Mean p [Pa]':>14}"
        f"{'Std [Pa]':>11}{'Change':>10}")
    for _, r in mesh["table"].iterrows():
        chg = "-" if r["pct_change"] != r["pct_change"] else f"{r['pct_change']:+.2f}%"
        add(f"   {r['mesh']:<8}{r['cells']:>10,}{r['last_iter']:>10}"
            f"{r['p_avg']:>14.4f}{r['p_std']:>11.4f}{chg:>10}")

    # 2. Validation --------------------------------------------------------
    add(f"\n2) VALIDATION  (straight 4x4 m duct, v = {config.VAL_VELOCITY:g} m/s, "
        f"L = {config.VAL_LENGTH:g} m)")
    add(f"   Reynolds number            Re = {val['re']:,.0f}")
    add(f"   Petukhov friction factor    f = {val['f']:.5f}")
    add(f"   CFD dp (iteration {val['final_iter']})     = {val['dp_cfd']:.4f} Pa")
    add(f"   Darcy-Weisbach dp          = {val['dp_theory']:.4f} Pa")
    add(f"   Error                      = {val['pct_error']:+.2f} %")

    # 3. Sweep -------------------------------------------------------------
    add(f"\n3) PARAMETRIC SWEEP  ({len(sweep)} design points, "
        f"excluded: {', '.join(config.SWEEP_EXCLUDE)})")
    add(f"   {'DP':<7}{'R [m]':>7}{'v [m/s]':>9}{'dp [Pa]':>10}{'K [-]':>8}{'Re':>12}")
    for _, r in sweep.iterrows():
        add(f"   {r['Name']:<7}{r['radius']:>7.2f}{r['velocity']:>9.1f}"
            f"{r['dp']:>10.4f}{r['K']:>8.3f}{r['Re']:>12.3e}")

    # 4. Surrogate ---------------------------------------------------------
    add("\n4) ML SURROGATE  (leave-one-out cross-validation on dp)")
    add(f"   MLP (2-{config.HIDDEN_UNITS}-1 tanh, ensemble of {config.N_ENSEMBLE}) "
        f"MAPE = {sur['mape_mlp']:.2f} %")
    add(f"   Linear regression baseline           MAPE = {sur['mape_lin']:.2f} %")
    add(f"   Constant K per radius (physics)      MAPE = {sur['mape_ck']:.2f} %")
    add(f"\n   {'DP':<7}{'CFD dp':>9}{'MLP':>9}{'err':>8}{'Linear':>9}{'err':>8}"
        f"{'Const K':>9}{'err':>8}")
    for i, r in sweep.iterrows():
        row = f"   {r['Name']:<7}{r['dp']:>9.3f}"
        for key in ("dp_loo_mlp", "dp_loo_lin", "dp_loo_ck"):
            p = sur[key][i]
            row += f"{p:>9.3f}{(p - r['dp']) / r['dp'] * 100:>+7.1f}%"
        add(row)
    add(f"\n   Single surrogate prediction : {sur['t_pred'] * 1e3:.3f} ms")
    add(f"   Single CFD design point     : {config.CFD_RUNTIME_HOURS:g} h "
        f"({sur['t_cfd']:,.0f} s)")
    add(f"   Speed-up                    : ~{sur['speedup']:.1e} x")

    # 5. Figures -----------------------------------------------------------
    add("\n5) FIGURES")
    for p in fig_paths:
        add(f"   {p.relative_to(config.PROJECT_ROOT)}")
    add(LINE)
    return "\n".join(out)


def main():
    print("Running mesh study...")
    mesh = run_mesh_study()
    print("Running validation...")
    val = run_validation()
    print("Loading parametric sweep...")
    sweep = load_sweep()
    print("Training surrogate (leave-one-out CV)...")
    sur = run_surrogate(sweep)
    print("Saving figures...")
    fig_paths = [
        figures.plot_mesh_study(mesh),
        figures.plot_validation(val),
        figures.plot_design_curves(sweep, sur),
        figures.plot_pred_vs_actual(sweep, sur),
    ]

    summary = format_summary(mesh, val, sweep, sur, fig_paths)
    print("\n" + summary)
    (config.PROJECT_ROOT / "results_summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
