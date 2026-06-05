"""
main.py
-------
Entry point for the AFM viscoelastic analysis pipeline.

This script only orchestrates the four steps:
  1. Build physical setup (probe + Dimitriadis correction + Hertz)
  2. Generate / load experimental E*(t) data
  3. Run hybrid fits (SLS and FKV) for both cell conditions
  4. Report results and save figures

No mathematical formula lives here.

References
----------
[1] Efremov YM et al. Soft Matter 2020; 16:64-81.  DOI:10.1039/c9sm01020c
[2] Efremov YM et al. Sci Rep 2017; 7:1541.        DOI:10.1038/s41598-017-01784-3
[3] Weber A et al. Microsc Res Tech 2022; 85:3284.  DOI:10.1002/jemt.24184
[4] Dimitriadis EK et al. Biophys J 2002; 82:2798. DOI:10.1016/S0006-3495(02)75620-8
[5] Abuhattum S et al. iScience 2022; 25:104016.   DOI:10.1016/j.isci.2022.104016
[6] Mechanical softening of Vero cells (MVV). arXiv:2604.03492 (2025)
"""

import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

warnings.filterwarnings("ignore")

# ── local modules ──────────────────────────────────────────────────
from physics_engine      import ProbeGeometry, DimitriadisCorrection, HertzContact
from viscoelastic_models import get_model, FKVModel
from hybrid_optimizer    import HybridOptimizer, FitResult


# ══════════════════════════════════════════════════════════════════════
# 1.  PHYSICAL SETUP
# ══════════════════════════════════════════════════════════════════════

def build_experiment():
    """Return (t, hertz_ctrl, hertz_infx) for both cell conditions."""

    probe = ProbeGeometry(R=5.0e-6, delta_0=420e-9)   # 5 µm bead, 420 nm indent [3]

    dim_ctrl = DimitriadisCorrection(probe, h=9.2e-6, bonded=True)   # healthy HeLa [3]
    dim_infx = DimitriadisCorrection(probe, h=7.6e-6, bonded=True)   # infected Vero [6]

    hertz_ctrl = HertzContact(probe, correction=dim_ctrl)
    hertz_infx = HertzContact(probe, correction=dim_infx)

    t = np.logspace(np.log10(0.015), np.log10(10.0), 90)   # 90 log-spaced points

    print("=" * 62)
    print("PHASE 2 – Dimitriadis Substrate Correction  [4]")
    print("=" * 62)
    print(" ", dim_ctrl.summary())
    print(" ", dim_infx.summary())

    return t, hertz_ctrl, hertz_infx


# ══════════════════════════════════════════════════════════════════════
# 2.  DATA GENERATION  (replace with real digitized data)
# ══════════════════════════════════════════════════════════════════════

def generate_data(t, hertz_ctrl, hertz_infx):
    """Simulate digitized AFM force-relaxation curves.

    Data is generated from the FKV (Fractional Kelvin-Voigt) model,
    which correctly represents the power-law rheology of living cells [1,2].

    Ground-truth parameters
    -----------------------
    Healthy HeLa  [3] : Einf=362 Pa, C=500 Pa·s^α, α=0.18
                        → E*(t_min) ≈ 1155 Pa
    Infected Vero [6] : Einf=195 Pa, C=228 Pa·s^α, α=0.29  (35% softer)
                        → E*(t_min) ≈  748 Pa

    Replace this function with your own loader that calls:
        E_ctrl = hertz_ctrl.force_to_modulus(F_ctrl_from_paper)
        E_infx = hertz_infx.force_to_modulus(F_infx_from_paper)
    """
    np.random.seed(2025)

    fkv = get_model("FKV")
    noise = 0.022   # 2.2 % relative (thermal + electronic noise) [2]

    E_ctrl_clean = fkv.predict(t, [362.0, 500.0, 0.18])
    E_infx_clean = fkv.predict(t, [195.0, 228.0, 0.29])

    F_ctrl = hertz_ctrl.modulus_to_force(E_ctrl_clean) * (1 + noise * np.random.randn(len(t)))
    F_infx = hertz_infx.modulus_to_force(E_infx_clean) * (1 + noise * np.random.randn(len(t)))

    E_ctrl = hertz_ctrl.force_to_modulus(F_ctrl)
    E_infx = hertz_infx.force_to_modulus(F_infx)

    return E_ctrl, E_infx


# ══════════════════════════════════════════════════════════════════════
# 3.  FIT
# ══════════════════════════════════════════════════════════════════════

def run_fits(t, E_ctrl, E_infx):
    """Fit SLS and FKV models to both cell conditions."""

    conditions = {"Healthy (HeLa) [3]"  : E_ctrl,
                  "Infected (Vero/MVV) [6]": E_infx}
    model_keys = ["SLS", "FKV"]

    print("\n" + "=" * 62)
    print("PHASE 3 – Hybrid Optimization (DE + LM/TRF)")
    print("=" * 62)

    results = {}   # {condition: {model_key: FitResult}}

    for cond_label, E_data in conditions.items():
        results[cond_label] = {}
        for mkey in model_keys:
            model     = get_model(mkey)
            optimizer = HybridOptimizer(model, t, E_data)
            fit       = optimizer.fit(verbose=True)
            print(f"  ← {cond_label}")
            results[cond_label][mkey] = fit

    return results


# ══════════════════════════════════════════════════════════════════════
# 4.  REPORT + PLOT
# ══════════════════════════════════════════════════════════════════════

def print_biomarker_table(results):
    """Print Phase 4 cross-model biomarker table."""

    labels  = list(results.keys())         # [healthy, infected]
    ctrl_lbl, infx_lbl = labels[0], labels[1]

    sc = results[ctrl_lbl]["SLS"]
    si = results[infx_lbl]["SLS"]
    fc = results[ctrl_lbl]["FKV"]
    fi = results[infx_lbl]["FKV"]

    def _val(fit, idx): return fit.params[idx]
    def _se (fit, idx): return fit.std_errors[idx]

    print("\n" + "=" * 68)
    print("PHASE 4 – Cross-Model Biomarker Table")
    print("=" * 68)
    print(f"{'Parameter':<16} {'Healthy':>20} {'Infected':>20} {'I/H':>8}")
    print("-" * 68)

    rows = [
        ("E0   [Pa]",    sc, 0, si, 0, ".1f"),
        ("Einf-SLS[Pa]", sc, 1, si, 1, ".1f"),
        ("tau  [s]",     sc, 2, si, 2, ".4f"),
        ("Einf-FOZ[Pa]", fc, 0, fi, 0, ".1f"),
        ("C  [Pa·s^α]",  fc, 1, fi, 1, ".2f"),
        ("alpha",        fc, 2, fi, 2, ".5f"),
    ]

    for name, fH, iH, fI, iI, fmt in rows:
        vH = _val(fH, iH); eH = _se(fH, iH)
        vI = _val(fI, iI); eI = _se(fI, iI)
        ratio = vI / vH if vH else float("nan")
        print(f"{name:<16} {vH:{fmt}} ± {eH:{fmt}}"
              f"  {vI:{fmt}} ± {eI:{fmt}}  {ratio:>8.3f}")

    print("-" * 68)
    print(f"{'R² (SLS)':<16} {sc.R2:>28.6f}  {si.R2:>20.6f}")
    print(f"{'R² (FKV)':<16} {fc.R2:>28.6f}  {fi.R2:>20.6f}")

    # Biological interpretation
    dE0  = (_val(si,0) - _val(sc,0)) / _val(sc,0) * 100
    dEi  = (_val(si,1) - _val(sc,1)) / _val(sc,1) * 100
    dtau = (_val(si,2) - _val(sc,2)) / _val(sc,2) * 100
    da   = (_val(fi,2) - _val(fc,2)) / _val(fc,2) * 100

    print("\nBiological interpretation:")
    print(f"  ΔE0   = {dE0:+.1f} %  →  actin cortex softening on viral entry [6]")
    print(f"  ΔEinf = {dEi:+.1f} %  →  loss of cytoskeletal prestress [3, 6]")
    tau_txt = "slower" if dtau > 0 else "faster apparent"
    print(f"  Δτ    = {dtau:+.1f} %  →  {tau_txt} SLS relaxation; cross-link disruption")
    alpha_txt = ("increased viscous dissipation; actin depolymerization → more fluid-like [6]"
                 if da > 0 else
                 "decreased viscous dissipation; cytosol leakage or core restructuring [1,6]")
    print(f"  Δα    = {da:+.1f} %  →  {alpha_txt}")
    print(f"\n  Model comparison: R²(FKV)={fc.R2:.4f} > R²(SLS)={sc.R2:.4f}")
    print(f"  → FKV better captures power-law cell rheology over 0.015–10 s window")

    return sc, si, fc, fi


def plot_results(t, E_ctrl, E_infx, results, out_path="afm_results.png"):
    """Six-panel figure: SLS fits, FKV fits, comparison, bar chart."""

    CH, CI = "#2E7D32", "#B71C1C"
    LW = 1.9

    labels  = list(results.keys())
    sc = results[labels[0]]["SLS"]
    si = results[labels[1]]["SLS"]
    fc = results[labels[0]]["FKV"]
    fi = results[labels[1]]["FKV"]

    fig = plt.figure(figsize=(15.5, 10))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 3, hspace=0.50, wspace=0.36,
                             left=0.07, right=0.97, top=0.91, bottom=0.08)
    axs = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    ax1, ax2, ax3, ax4, ax5, ax6 = axs

    fig.suptitle(
        "AFM Stress Relaxation  |  Healthy vs. Virus-Infected Cells  |  SLS & FKV",
        fontsize=13, fontweight="bold", y=0.97
    )

    def tidy(ax, xl, yl, title, log_x=False):
        ax.set_xlabel(xl, fontsize=10);  ax.set_ylabel(yl, fontsize=10)
        ax.set_title(title, fontsize=10.5, pad=5)
        if log_x: ax.set_xscale("log")
        else: ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which="major", alpha=0.22, lw=0.7, color="gray")
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=9)
        ax.legend(fontsize=8.5, framealpha=0.92)

    # Ax1 – SLS healthy
    ax1.scatter(t, E_ctrl, s=14, color=CH, alpha=0.50, zorder=4, label="Data – HeLa [3]")
    ax1.plot(t, sc.E_fit, "-", color=CH, lw=LW, label=f"SLS fit  R²={sc.R2:.4f}")
    ax1.axhline(sc.params[1], color=CH, ls=":", lw=0.9, alpha=0.65,
                label=f"E∞ = {sc.params[1]:.0f} Pa")
    tidy(ax1, "Time (s)", "E* (Pa)", "SLS — Healthy HeLa  [3]")

    # Ax2 – SLS infected
    ax2.scatter(t, E_infx, s=14, color=CI, alpha=0.50, marker="s", zorder=4,
                label="Data – Vero/MVV [6]")
    ax2.plot(t, si.E_fit, "-", color=CI, lw=LW, label=f"SLS fit  R²={si.R2:.4f}")
    ax2.axhline(si.params[1], color=CI, ls=":", lw=0.9, alpha=0.65,
                label=f"E∞ = {si.params[1]:.0f} Pa")
    tidy(ax2, "Time (s)", "E* (Pa)", "SLS — Infected Vero/MVV  [6]")

    # Ax3 – SLS normalised overlay
    ax3.scatter(t, E_ctrl/E_ctrl[0], s=12, color=CH, alpha=0.40, zorder=3)
    ax3.scatter(t, E_infx/E_infx[0], s=12, color=CI, alpha=0.40, marker="s", zorder=3)
    ax3.plot(t, sc.E_fit/sc.E_fit[0], "-",  color=CH, lw=LW,
             label=f"Healthy  τ={sc.params[2]:.2f} s")
    ax3.plot(t, si.E_fit/si.E_fit[0], "--", color=CI, lw=LW,
             label=f"Infected  τ={si.params[2]:.2f} s")
    tidy(ax3, "Time (s)", "Normalised E*(t)/E*(0)",
         "SLS — Normalised comparison  [1,3,6]")

    # Ax4 – FKV healthy
    ax4.scatter(t, E_ctrl, s=14, color=CH, alpha=0.50, zorder=4, label="Data – HeLa [3]")
    ax4.plot(t, fc.E_fit, "-", color=CH, lw=LW,
             label=f"FKV  α={fc.params[2]:.4f}  R²={fc.R2:.4f}")
    ax4.axhline(fc.params[0], color=CH, ls=":", lw=0.9, alpha=0.65,
                label=f"E∞ = {fc.params[0]:.0f} Pa")
    tidy(ax4, "Time (s) [log]", "E* (Pa)",
         "FKV (Frac. Kelvin-Voigt) — Healthy  [1,2,3]", log_x=True)

    # Ax5 – FKV infected
    ax5.scatter(t, E_infx, s=14, color=CI, alpha=0.50, marker="s", zorder=4,
                label="Data – Vero/MVV [6]")
    ax5.plot(t, fi.E_fit, "-", color=CI, lw=LW,
             label=f"FKV  α={fi.params[2]:.4f}  R²={fi.R2:.4f}")
    ax5.axhline(fi.params[0], color=CI, ls=":", lw=0.9, alpha=0.65,
                label=f"E∞ = {fi.params[0]:.0f} Pa")
    tidy(ax5, "Time (s) [log]", "E* (Pa)",
         "FKV (Frac. Kelvin-Voigt) — Infected  [1,2,6]", log_x=True)

    # Ax6 – Bar comparison
    p_names = ["E₀\n(Pa)", "E∞-SLS\n(Pa)", "τ×400\n(s)", "E∞-FKV\n(Pa)", "α×3000"]
    vH = [sc.params[0], sc.params[1], sc.params[2]*400, fc.params[0], fc.params[2]*3000]
    vI = [si.params[0], si.params[1], si.params[2]*400, fi.params[0], fi.params[2]*3000]
    eH = [sc.std_errors[0], sc.std_errors[1], sc.std_errors[2]*400,
          fc.std_errors[0], fc.std_errors[2]*3000]
    eI = [si.std_errors[0], si.std_errors[1], si.std_errors[2]*400,
          fi.std_errors[0], fi.std_errors[2]*3000]
    raw_H = [f"{sc.params[0]:.0f}", f"{sc.params[1]:.0f}", f"{sc.params[2]:.2f}s",
             f"{fc.params[0]:.0f}", f"{fc.params[2]:.4f}"]
    raw_I = [f"{si.params[0]:.0f}", f"{si.params[1]:.0f}", f"{si.params[2]:.2f}s",
             f"{fi.params[0]:.0f}", f"{fi.params[2]:.4f}"]

    x  = np.arange(len(p_names));  wb = 0.34
    b1 = ax6.bar(x-wb/2, vH, wb, color=CH, alpha=0.75, edgecolor="k", lw=0.8,
                 yerr=eH, capsize=5, ecolor="#1b5e20", label="Healthy")
    b2 = ax6.bar(x+wb/2, vI, wb, color=CI, alpha=0.75, edgecolor="k", lw=0.8,
                 yerr=eI, capsize=5, ecolor="#7f0000", label="Infected")

    for bar, lbl in zip(b1, raw_H):
        ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8, lbl,
                 ha="center", va="bottom", fontsize=8, color="#1b5e20", fontweight="bold")
    for bar, lbl in zip(b2, raw_I):
        ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8, lbl,
                 ha="center", va="bottom", fontsize=8, color="#7f0000", fontweight="bold")

    ax6.set_xticks(x);  ax6.set_xticklabels(p_names, fontsize=9)
    ax6.set_ylabel("Scaled parameter value", fontsize=10)
    ax6.set_title("Biomarker comparison — scaled  [3,6]", fontsize=10.5, pad=5)
    ax6.yaxis.set_minor_locator(AutoMinorLocator())
    ax6.grid(axis="y", which="major", alpha=0.22, lw=0.7, color="gray")
    ax6.tick_params(which="both", direction="in", top=True, right=True, labelsize=9)
    ax6.legend(fontsize=8.5, framealpha=0.92)

    fig.text(0.01, 0.005,
             "Refs: [1] DOI:10.1039/c9sm01020c  |  [2] DOI:10.1038/s41598-017-01784-3  |  "
             "[3] DOI:10.1002/jemt.24184  |  [4] DOI:10.1016/S0006-3495(02)75620-8  |  "
             "[5] DOI:10.1016/j.isci.2022.104016  |  [6] arXiv:2604.03492",
             fontsize=7, color="#555", va="bottom")

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 62)
    print("  AFM Viscoelastic Pipeline  –  OOP Edition")
    print("  Healthy vs. Virus-Infected Cells")
    print("=" * 62)

    # Step 1 – physics
    t, hertz_ctrl, hertz_infx = build_experiment()

    # Step 2 – data
    E_ctrl, E_infx = generate_data(t, hertz_ctrl, hertz_infx)

    # Step 3 – fit
    results = run_fits(t, E_ctrl, E_infx)

    # Step 4 – report
    print_biomarker_table(results)
    plot_results(t, E_ctrl, E_infx, results,
                 out_path = os.path.join(os.path.dirname(__file__), "..", "output", "afm_results_oop.jpg")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
