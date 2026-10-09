"""
scripts/plot_category_combined.py
==================================
Combined forest plot showing unadjusted and SNP-conditioned results
side by side for each trait row.

Layout: [p_EF] [forest_EF] [labels] [forest_HG] [p_HG]

Each trait row has two points vertically offset:
  - Circle   (y + 0.18): unadjusted regression
  - Diamond  (y - 0.18): SNP-conditioned regression

P-values shown for both, stacked matching the vertical offset.
"""

import sys
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

_log = open(str(snakemake.log[0]), "w", buffering=1)
sys.stdout = _log
sys.stderr = _log

LOG_PREFIX = "[plot_category_combined]"
def log(msg): print(f"{LOG_PREFIX} {msg}", flush=True)

# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

category_id     = snakemake.params.category_id
category_label  = snakemake.params.category_label
trait_list      = snakemake.params.trait_list
focus_covset    = snakemake.params.focus_covset
focus_ancestry  = snakemake.params.focus_ancestry
results_dir     = snakemake.params.results_dir
snp_cond_covset = snakemake.params.snp_cond_covset

stats_files          = snakemake.input.stats_files
stats_files_snp_cond = snakemake.input.stats_files_snp_cond
merged_files         = snakemake.input.merged_files

log(f"Category: {category_label}  ({len(trait_list)} traits)")

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

BG_COLOR    = "white"
PANEL_COLOR = "#f8f8f8"
GRID_COLOR  = "#dddddd"
TEXT_COLOR  = "#111111"
ANNOT_BG    = "#ffffff"
ANNOT_EDGE  = "#cccccc"

ANC_COLORS = {
    "EF":     "#A03020",
    "Steppe": "#2471a3",
    "HG":     "#0D3D66",
}
SIG_ALPHA    = 0.95
NONSIG_ALPHA = 0.38
DOT_SIZE     = 38
DOT_SIZE_SML = 32   # diamond slightly smaller

VOFFSET = 0.18   # vertical offset between circle and diamond

plt.style.use("default")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_axis_label(trait_label, field_id, n_cases=None, n_controls=None):
    fid_str = str(field_id)
    if fid_str.isdigit():
        base = f"{trait_label} ({fid_str})"
    elif trait_label == fid_str:
        base = trait_label.replace("_", " ").title()
    else:
        base = trait_label
    if n_cases is not None and n_controls is not None:
        base = f"{base}\n{n_cases:,} cases / {n_controls:,} controls"
    return base

def fmt_pval(p):
    if pd.isna(p) or p is None:
        return "n.s."
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"

def adaptive_fontsize(labels, col_width_in, fig, min_fs=5.0, max_fs=7.0):
    n = len(labels)
    fs_count_cap = 5.0 + 0.30 * n
    renderer = fig.canvas.get_renderer()
    col_px   = col_width_in * fig.dpi
    fs = min(max_fs, fs_count_cap)
    while fs >= min_fs:
        fits = True
        for lbl in labels:
            for line in lbl.split("\n"):
                t  = fig.text(0, 0, line, fontsize=fs)
                bb = t.get_window_extent(renderer=renderer)
                t.remove()
                if bb.width > col_px * 0.88:
                    fits = False
                    break
            if not fits:
                break
        if fits:
            break
        fs -= 0.25
    return float(np.clip(fs, min_fs, max_fs))

def label_n_lines(label):
    return label.count("\n") + 1

def get_stats_row(stats_df, test_name, anc):
    rows = stats_df[
        (stats_df["test"] == test_name) &
        (stats_df["ancestry"] == anc)
    ]
    return rows.iloc[0] if len(rows) > 0 else None

# ---------------------------------------------------------------------------
# Load both standard and SNP-conditioned stats per trait
# ---------------------------------------------------------------------------

records = []   # list of dicts, one per (trait, ancestry, model)

for trait_info, stats_f, stats_snp_f, merged_f in zip(
        trait_list, stats_files, stats_files_snp_cond, merged_files):

    fid         = trait_info["field_id"]
    trait_label = trait_info.get("label", str(fid))

    log(f"  Loading: {trait_label}  fid={fid}")

    # Load standard stats
    try:
        std_df = pd.read_csv(stats_f, sep="\t")
    except Exception as e:
        log(f"  WARNING: could not read standard stats for {fid}: {e}")
        std_df = pd.DataFrame()

    # Load SNP-conditioned stats
    try:
        snp_df = pd.read_csv(stats_snp_f, sep="\t")
    except Exception as e:
        log(f"  WARNING: could not read SNP-cond stats for {fid}: {e}")
        snp_df = pd.DataFrame()

    if std_df.empty and snp_df.empty:
        log(f"  WARNING: both stats empty for {fid}, skipping")
        continue

    # Detect analysis type
    all_tests = pd.concat([std_df, snp_df])["test"].values if (not std_df.empty and not snp_df.empty) else \
                (std_df["test"].values if not std_df.empty else snp_df["test"].values)
    is_cc = any("Logistic" in t for t in all_tests)

    # n_cases / n_controls for label
    n_cases = n_controls = None
    if is_cc:
        for df in [std_df, snp_df]:
            if df.empty:
                continue
            mw = df[df["test"] == "Mann-Whitney U"]
            if len(mw) > 0:
                n_cases    = int(mw.iloc[0]["n_cases"])
                n_controls = int(mw.iloc[0]["n_controls"])
                break

    axis_label = make_axis_label(trait_label, fid, n_cases, n_controls)

    # SD for standardisation
    trait_sd = None
    if not is_cc:
        try:
            mdf      = pd.read_csv(merged_f, sep="\t", usecols=["trait_value"])
            trait_sd = float(mdf["trait_value"].dropna().std())
        except Exception as e:
            log(f"  WARNING: could not compute SD for {fid}: {e}")

    def extract_record(df, model_tag, anc):
        if df.empty:
            return None
        if is_cc:
            # find logistic test — could be adj. or adj.+SNP
            test_name = next((t for t in df["test"].values
                              if "Logistic" in t), None)
            if test_name is None:
                return None
            row = get_stats_row(df, test_name, anc)
            if row is None:
                return None
            OR   = float(row["OR"])
            lo   = float(row["OR_CI_low"])
            hi   = float(row["OR_CI_high"])
            pval = float(row["p_value"])
            return {
                "axis_label":    axis_label,
                "ancestry":      anc,
                "analysis_type": "case_control",
                "model":         model_tag,
                "effect":        np.log(OR),
                "ci_low":        np.log(lo),
                "ci_high":       np.log(hi),
                "p_value":       pval,
                "significant":   pval < 0.05,
            }
        else:
            test_name = next((t for t in df["test"].values
                              if "Linear" in t), None)
            if test_name is None:
                return None
            row = get_stats_row(df, test_name, anc)
            if row is None:
                return None
            beta = float(row["beta"])
            lo   = float(row["CI_low"])
            hi   = float(row["CI_high"])
            pval = float(row["p_value"])
            if trait_sd and trait_sd > 0:
                beta /= trait_sd
                lo   /= trait_sd
                hi   /= trait_sd
            return {
                "axis_label":    axis_label,
                "ancestry":      anc,
                "analysis_type": "continuous",
                "model":         model_tag,
                "effect":        beta,
                "ci_low":        lo,
                "ci_high":       hi,
                "p_value":       pval,
                "significant":   pval < 0.05,
            }

    for anc in focus_ancestry:
        r_std = extract_record(std_df, "standard", anc)
        r_snp = extract_record(snp_df, "snp_cond", anc)
        if r_std:
            records.append(r_std)
        if r_snp:
            records.append(r_snp)

if not records:
    log("ERROR: no records — writing blank figure")
    fig, ax = plt.subplots(figsize=(6, 3), facecolor="white")
    ax.text(0.5, 0.5, f"No data for category: {category_label}",
            ha="center", va="center", fontsize=12, color="#666666",
            transform=ax.transAxes)
    ax.axis("off")
    for out in [snakemake.output.plot_png, snakemake.output.plot_pdf,
                snakemake.output.plot_svg]:
        plt.savefig(out, bbox_inches="tight")
    plt.close()
    sys.exit(0)

df = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Sort by EF standard effect size descending
# ---------------------------------------------------------------------------

ef_std = (
    df[(df["ancestry"] == focus_ancestry[0]) & (df["model"] == "standard")]
    .groupby("axis_label")["effect"].mean()
    .sort_values(ascending=False)
)
trait_order = list(ef_std.index)
extra = [t for t in df["axis_label"].unique() if t not in trait_order]
trait_order = trait_order + extra

n_traits     = len(trait_order)
primary_type = df["analysis_type"].mode().iloc[0]

# ---------------------------------------------------------------------------
# Y positions — two sub-rows per trait
# ---------------------------------------------------------------------------

LINE_H   = 0.28
ROW_PAD  = 0.30   # extra padding for two-point rows
row_heights = [LINE_H * label_n_lines(lbl) + ROW_PAD for lbl in trait_order]
content_h   = sum(row_heights)
fig_h       = max(3.0, content_h + 1.0)

cumulative = 0.0
y_pos_center = {}
for lbl in reversed(trait_order):
    h = LINE_H * label_n_lines(lbl) + ROW_PAD
    y_pos_center[lbl] = cumulative + h / 2
    cumulative += h

scale = (n_traits - 1) / max(cumulative, 1)
y_pos_center = {lbl: v * scale for lbl, v in y_pos_center.items()}
y_max = cumulative * scale

# Actual y positions for each model
def yp(label, model):
    c = y_pos_center[label]
    return c + VOFFSET if model == "standard" else c - VOFFSET

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

n_anc      = len(focus_ancestry)
forest_w   = 2.8
pval_w     = 0.65
label_w    = 2.8

label_rel  = label_w / forest_w
pval_rel   = pval_w  / forest_w

if n_anc == 2:
    width_ratios = [pval_rel, 1.0, label_rel, 1.0, pval_rel]
    col_pval_left  = 0
    col_forest     = {focus_ancestry[0]: 1, focus_ancestry[1]: 3}
    col_labels     = 2
    col_pval_right = 4
    n_cols         = 5
else:
    width_ratios   = [pval_rel] + [1.0] * n_anc + [label_rel, pval_rel]
    col_pval_left  = 0
    col_forest     = {anc: 1 + i for i, anc in enumerate(focus_ancestry)}
    col_labels     = 1 + n_anc
    col_pval_right = 2 + n_anc
    n_cols         = 3 + n_anc

total_w = pval_w * 2 + forest_w * n_anc + label_w
fig = plt.figure(figsize=(total_w, fig_h), facecolor=BG_COLOR)
fig.patch.set_facecolor(BG_COLOR)

gs = gridspec.GridSpec(1, n_cols, figure=fig,
                       width_ratios=width_ratios, wspace=0.04)

ax_pval_left  = fig.add_subplot(gs[0, col_pval_left])
ax_labels     = fig.add_subplot(gs[0, col_labels])
ax_forests    = {anc: fig.add_subplot(gs[0, col_forest[anc]]) for anc in focus_ancestry}
ax_pval_right = fig.add_subplot(gs[0, col_pval_right])

# Adaptive font size
fig.canvas.draw()
renderer        = fig.canvas.get_renderer()
ax_label_bbox   = ax_labels.get_window_extent(renderer=renderer)
actual_label_w  = ax_label_bbox.width / fig.dpi
label_fs        = adaptive_fontsize(trait_order, actual_label_w, fig)
log(f"Label font size: {label_fs:.1f}pt")

# ---------------------------------------------------------------------------
# Draw forest panels
# ---------------------------------------------------------------------------

def draw_forest(ax, anc, df, primary_type, n_traits):
    color    = ANC_COLORS.get(anc, "#555555")
    anc_data = df[df["ancestry"] == anc]

    for _, row in anc_data.iterrows():
        lbl   = row["axis_label"]
        model = row["model"]
        y     = yp(lbl, model)
        eff   = row["effect"]
        lo    = row["ci_low"]
        hi    = row["ci_high"]
        sig   = row["significant"]
        alpha = SIG_ALPHA if sig else NONSIG_ALPHA

        # CI line
        ax.plot([lo, hi], [y, y], color=color, lw=1.4, alpha=alpha,
                solid_capstyle="round", zorder=3)

        if model == "standard":
            # Circle
            if sig:
                ax.scatter([eff], [y], color=color, s=DOT_SIZE,
                           marker="o", zorder=5, alpha=SIG_ALPHA)
            else:
                ax.scatter([eff], [y], color=color, s=DOT_SIZE,
                           marker="o", zorder=5, alpha=NONSIG_ALPHA,
                           facecolors="none", linewidths=1.4)
        else:
            # Diamond
            if sig:
                ax.scatter([eff], [y], color=color, s=DOT_SIZE_SML,
                           marker="D", zorder=5, alpha=SIG_ALPHA)
            else:
                ax.scatter([eff], [y], color=color, s=DOT_SIZE_SML,
                           marker="D", zorder=5, alpha=NONSIG_ALPHA,
                           facecolors="none", linewidths=1.4)

    ax.axvline(0, color="#aaaaaa", lw=0.9, linestyle="--", zorder=2)
    ax.set_ylim(-0.6, y_max + 0.3)
    ax.set_yticks([])
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.xaxis.label.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4,
            linestyle="--", alpha=0.8, axis="x")
    ax.set_axisbelow(True)
    ax.set_title(anc, fontsize=9, fontweight="bold",
                 color=color, pad=4)

    if primary_type == "continuous":
        ax.set_xlabel("Std. β  (SD of trait / unit ancestry)",
                      fontsize=6.5, color=TEXT_COLOR)
    else:
        ax.set_xlabel("Odds Ratio  (log scale)", fontsize=6.5, color=TEXT_COLOR)
        xticks = ax.get_xticks()
        ax.set_xticklabels([f"{np.exp(x):.2f}" for x in xticks], fontsize=6.5)


# ---------------------------------------------------------------------------
# Draw p-value columns — two stacked per trait
# ---------------------------------------------------------------------------

def draw_pvals(ax, anc, df, ha="left"):
    color    = ANC_COLORS.get(anc, "#555555")
    anc_data = df[df["ancestry"] == anc]

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, y_max + 0.3)
    ax.axis("off")

    xpos = 0.08 if ha == "left" else 0.92

    for lbl in trait_order:
        for model in ["standard", "snp_cond"]:
            row_match = anc_data[
                (anc_data["axis_label"] == lbl) &
                (anc_data["model"] == model)
            ]
            if len(row_match) == 0:
                continue
            row  = row_match.iloc[0]
            y    = yp(lbl, model)
            sig  = row["significant"]
            txt  = fmt_pval(row["p_value"])
            fc   = color if sig else "#bbbbbb"
            # SNP-conditioned in italic to distinguish
            style = "italic" if model == "snp_cond" else "normal"
            ax.text(xpos, y, txt, va="center", ha=ha,
                    fontsize=5.8, color=fc,
                    fontfamily="monospace", style=style)

    ax.text(0.5, y_max + 0.05, f"p ({anc})",
            va="bottom", ha="center",
            fontsize=6.5, color=color, fontweight="bold")


# ---------------------------------------------------------------------------
# Draw label column
# ---------------------------------------------------------------------------

def draw_labels(ax, trait_order, fontsize):
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, y_max + 0.3)
    ax.axis("off")
    for lbl in trait_order:
        yc = y_pos_center[lbl]
        ax.text(0.5, yc, lbl, va="center", ha="center",
                fontsize=fontsize, color=TEXT_COLOR, linespacing=1.3)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

draw_pvals(ax_pval_left,  focus_ancestry[0],  df, ha="left")
draw_labels(ax_labels, trait_order, fontsize=label_fs)
for anc in focus_ancestry:
    draw_forest(ax_forests[anc], anc, df, primary_type, n_traits)
draw_pvals(ax_pval_right, focus_ancestry[-1], df, ha="right")

# ---------------------------------------------------------------------------
# Title + legend
# ---------------------------------------------------------------------------

fig.subplots_adjust(top=0.90)
fig.suptitle(category_label, fontsize=9, fontweight="bold",
             color=TEXT_COLOR, y=0.98)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
           markersize=6, label="Unadjusted for SNPs"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
           markeredgecolor="#555555", markeredgewidth=1.3,
           markersize=6, label="p ≥ 0.05"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#555555",
           markersize=5, label="SNP-conditioned"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="none",
           markeredgecolor="#555555", markeredgewidth=1.3,
           markersize=5, label="p ≥ 0.05 (SNP-cond.)"),
]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=4, fontsize=7, frameon=True,
           facecolor=ANNOT_BG, edgecolor=ANNOT_EDGE,
           bbox_to_anchor=(0.5, -0.06))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

save_kw = dict(bbox_inches="tight", facecolor=BG_COLOR)
plt.savefig(snakemake.output.plot_png, dpi=150, **save_kw)
plt.savefig(snakemake.output.plot_pdf, **save_kw)
plt.savefig(snakemake.output.plot_svg, **save_kw)
plt.close()
log("Saved: PNG / PDF / SVG")