"""
scripts/plot_survival.py
=========================
Two output figures per survival trait:

Figure 1 — Kaplan-Meier curves (one panel per ancestry component)
  - Three curves: Low / Mid / High tertile of ancestry proportion
  - 95% confidence intervals (shaded)
  - Censored observations as tick marks
  - Log-rank p-value + median survival per group in annotation box
  - At-risk table below each panel

Figure 2 — Cox regression forest plot
  - One row per ancestry component
  - HR + 95% CI (per SD change in ancestry proportion)
  - Vertical line at HR=1
  - p-value annotated
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

_log = open(str(snakemake.log[0]), "w", buffering=1)
sys.stdout = _log
sys.stderr = _log

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

merged = pd.read_csv(snakemake.input.merged, sep="\t")

with open(snakemake.input.meta) as fh:
    meta = json.load(fh)

stats_df      = pd.read_csv(snakemake.input.stats, sep="\t")
ancestry_cols = snakemake.params.ancestry_cols
label         = meta["label"]

# ---------------------------------------------------------------------------
# Dark theme constants
# ---------------------------------------------------------------------------

# NEW — light theme
BG_COLOR    = "white"
PANEL_COLOR = "#f8f8f8"
GRID_COLOR  = "#dddddd"
TEXT_COLOR  = "#111111"
ANNOT_BG    = "#ffffff"
ANNOT_EDGE  = "#bbbbbb"

PALETTE = {"EF": "#e8604c", "Steppe": "#4caf7d", "HG": "#5b9bd5"}
DEFAULT_COLOR = "#aaaaaa"

TERTILE_COLORS = {
    "Low":  "#5b9bd5",   # blue  = low ancestry
    "Mid":  "#888888",   # grey  = mid
    "High": "#e8604c",   # red   = high ancestry
}

def pval_str(p):
    if pd.isna(p):
        return "p = n.a."
    return f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"

def style_ax(ax):
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

def annot_box(ax, text, x=0.03, y=0.05, ha="left", va="bottom"):
    ax.text(x, y, text, transform=ax.transAxes,
            ha=ha, va=va, fontsize=7.5, color=TEXT_COLOR,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=ANNOT_BG,
                      edgecolor=ANNOT_EDGE, alpha=0.92))

# ===========================================================================
# Figure 1: Kaplan-Meier curves
# ===========================================================================

plt.style.use("default")

N_PANELS = len(ancestry_cols)
fig_km, axes_km = plt.subplots(
    1, N_PANELS,
    figsize=(5.5 * N_PANELS, 5.5),
    facecolor=BG_COLOR,
)
if N_PANELS == 1:
    axes_km = [axes_km]
fig_km.patch.set_facecolor(BG_COLOR)
fig_km.suptitle(
    f"Kaplan-Meier — {label}\n"
    f"n={meta['n_total']:,}  ·  events={meta['n_events']:,}  ·  "
    f"median age={meta['median_time']:.1f}y",
    fontsize=11, fontweight="bold", color=TEXT_COLOR, y=1.02,
)

for ax, anc in zip(axes_km, ancestry_cols):
    anc_color = PALETTE.get(anc, DEFAULT_COLOR)
    df = merged[[anc, "surv_time", "surv_event"]].dropna().copy()
    df["tertile"] = pd.qcut(df[anc], q=3, labels=["Low", "Mid", "High"])

    # Tertile boundary values for annotation
    cuts     = pd.qcut(df[anc], q=3, retbins=True)[1]
    t_lo_hi  = f"[{cuts[0]:.3f}–{cuts[1]:.3f}]"
    t_mid_hi = f"[{cuts[1]:.3f}–{cuts[2]:.3f}]"
    t_hi_hi  = f"[{cuts[2]:.3f}–{cuts[3]:.3f}]"
    tert_ranges = {"Low": t_lo_hi, "Mid": t_mid_hi, "High": t_hi_hi}

    kmf  = KaplanMeierFitter()
    lines = []
    annot_lines = []

    for tert in ["Low", "Mid", "High"]:
        mask   = df["tertile"] == tert
        color  = TERTILE_COLORS[tert]
        t_data = df.loc[mask, "surv_time"]
        e_data = df.loc[mask, "surv_event"]
        n_grp  = mask.sum()
        med    = float(t_data.median())

        kmf.fit(t_data, e_data, label=tert)

        # Survival curve
        kmf.plot_survival_function(
            ax=ax, ci_show=True, ci_alpha=0.12,
            color=color, linewidth=2.0,
            at_risk_counts=False,
            show_censors=True,
            censor_styles={"ms": 4, "marker": "|"},
        )
        annot_lines.append(
            f"{tert} {tert_ranges[tert]}  n={n_grp:,}  median={med:.1f}y"
        )

    # Log-rank p-value from stats
    lr_rows = stats_df[
        (stats_df["test"] == "log_rank_tertile_stats") &
        (stats_df["ancestry"] == anc)
    ]
    logrank_p = lr_rows["logrank_p_3way"].iloc[0] if len(lr_rows) else np.nan
    lh_p      = lr_rows["logrank_p_low_vs_high"].iloc[0] if len(lr_rows) else np.nan

    annot_text = (
        f"Log-rank (3-way): {pval_str(logrank_p)}\n"
        f"Low vs High:      {pval_str(lh_p)}\n"
        + "\n".join(annot_lines)
    )
    annot_box(ax, annot_text, x=0.03, y=0.05)

    ax.set_title(anc, fontsize=13, fontweight="bold", color=anc_color, pad=8)
    ax.set_xlabel("Parental age (years)", fontsize=10)
    ax.set_ylabel("Survival probability", fontsize=10)
    ax.set_xlim(40, 115)
    ax.set_ylim(-0.02, 1.05)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color=TERTILE_COLORS[t], lw=2, label=t)
        for t in ["Low", "Mid", "High"]
    ]
    ax.legend(handles=legend_elements, fontsize=8,
              facecolor=ANNOT_BG, edgecolor=ANNOT_EDGE,
              labelcolor=TEXT_COLOR, loc="upper right")
    style_ax(ax)

plt.tight_layout(pad=1.5)
plt.savefig(snakemake.output.km_plot, dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print(f"[plot_survival] KM plot saved: {snakemake.output.km_plot}", flush=True)

# ===========================================================================
# Figure 2: Cox forest plot
# ===========================================================================

cox_rows = stats_df[stats_df["test"] == "cox_regression"].copy()

fig_fp, ax_fp = plt.subplots(figsize=(7, 3.5), facecolor=BG_COLOR)
ax_fp.set_facecolor(PANEL_COLOR)
fig_fp.patch.set_facecolor(BG_COLOR)

y_positions = {anc: i for i, anc in enumerate(reversed(ancestry_cols))}
y_labels    = []

for anc in ancestry_cols:
    row = cox_rows[cox_rows["ancestry"] == anc]
    if len(row) == 0:
        continue
    row = row.iloc[0]
    ypos  = y_positions[anc]
    color = PALETTE.get(anc, DEFAULT_COLOR)
    hr    = row["HR_per_SD"]
    lo    = row["HR_CI_low"]
    hi    = row["HR_CI_high"]
    pval  = row["p_value"]

    # CI line
    ax_fp.plot([lo, hi], [ypos, ypos], color=color, lw=2.5, solid_capstyle="round")
    # Point estimate
    ax_fp.scatter([hr], [ypos], color=color, s=80, zorder=5)

    # Annotation
    ax_fp.text(
        max(hi, 1.0) + 0.005, ypos,
        f"HR={hr:.3f} [{lo:.3f}–{hi:.3f}]  {pval_str(pval)}",
        va="center", ha="left", fontsize=8,
        color=TEXT_COLOR, family="monospace",
    )

    y_labels.append((ypos, anc))

# Reference line
ax_fp.axvline(1.0, color="#ffffff", lw=1.0, linestyle="--", alpha=0.5)

ax_fp.set_yticks([y for y, _ in y_labels])
ax_fp.set_yticklabels([lbl for _, lbl in y_labels], fontsize=10,
                       fontweight="bold", color=TEXT_COLOR)
ax_fp.tick_params(colors=TEXT_COLOR, labelsize=9)
ax_fp.set_xlabel("Hazard Ratio (per SD of ancestry proportion)\n"
                 "HR < 1 = higher ancestry → lower mortality risk",
                 color=TEXT_COLOR, fontsize=9)
ax_fp.set_title(
    f"Cox Proportional Hazards — {label}\n"
    f"Adjusted for: age, sex, birth year",
    color=TEXT_COLOR, fontsize=11, fontweight="bold",
)
for spine in ax_fp.spines.values():
    spine.set_edgecolor(GRID_COLOR)
ax_fp.grid(True, color=GRID_COLOR, linewidth=0.4, linestyle="--",
           alpha=0.5, axis="x")
ax_fp.set_axisbelow(True)

# Expand x axis to fit annotations
x_max = cox_rows["HR_CI_high"].max() if len(cox_rows) else 1.5
ax_fp.set_xlim(cox_rows["HR_CI_low"].min() * 0.95 if len(cox_rows) else 0.7,
               x_max * 1.35)
ax_fp.set_ylim(-0.8, len(ancestry_cols) - 0.2)

plt.tight_layout(pad=1.5)
plt.savefig(snakemake.output.cox_plot, dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print(f"[plot_survival] Cox forest plot saved: {snakemake.output.cox_plot}", flush=True)