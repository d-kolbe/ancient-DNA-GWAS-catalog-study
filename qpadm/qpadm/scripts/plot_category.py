"""
scripts/plot_category.py
=========================
Category-level forest plot.

Layout (left to right):
  [ EF p-values ] [ EF forest ] [ trait labels ] [ HG forest ] [ HG p-values ]

Outputs: PNG + PDF + SVG (all written to snakemake.output.plot_png / .plot_pdf / .plot_svg)
"""

import sys
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

_log = open(str(snakemake.log[0]), "w", buffering=1)
sys.stdout = _log
sys.stderr = _log

LOG_PREFIX = "[plot_category]"

def log(msg):
    print(f"{LOG_PREFIX} {msg}", flush=True)

# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

category_id     = snakemake.params.category_id
category_label  = snakemake.params.category_label
trait_list      = snakemake.params.trait_list
focus_covset    = snakemake.params.focus_covset
focus_ancestry  = snakemake.params.focus_ancestry
results_dir     = snakemake.params.results_dir

stats_files  = snakemake.input.stats_files
merged_files = snakemake.input.merged_files

log(f"Category: {category_label}  ({len(trait_list)} traits)")
log(f"Covariate set: {focus_covset}")
log(f"Ancestry focus: {focus_ancestry}")

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

BG_COLOR    = "white"
PANEL_COLOR = "#f8f8f8"
GRID_COLOR  = "#dddddd"
TEXT_COLOR  = "#111111"

ANC_COLORS = {
    "EF":     "#A03020",
    "Steppe": "#2471a3",
    "HG":     "#0D3D66",
}
SIG_ALPHA    = 0.95
NONSIG_ALPHA = 0.40
DOT_SIZE     = 40

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
    """
    Find the largest font size at which every label fits within col_width_in
    inches, measured using the actual Matplotlib renderer — no heuristics.
    Also caps by n_traits so sparse categories don't get oversized text.
    """
    n = len(labels)
    # Trait-count cap: fewer rows = more vertical space = risk of huge text
    fs_count_cap = 5.0 + 0.30 * n   # ~5.9 for 3 traits, ~8.0 for 10+

    renderer  = fig.canvas.get_renderer()
    col_px    = col_width_in * fig.dpi   # column width in pixels

    fs = min(max_fs, fs_count_cap)
    while fs >= min_fs:
        fits = True
        for lbl in labels:
            # Measure each line separately; take the widest
            for line in lbl.split("\n"):
                t = fig.text(0, 0, line, fontsize=fs)
                bb = t.get_window_extent(renderer=renderer)
                t.remove()
                if bb.width > col_px * 0.88:   # 12% margin
                    fits = False
                    break
            if not fits:
                break
        if fits:
            break
        fs -= 0.25   # step down by 0.25pt until everything fits

    return float(np.clip(fs, min_fs, max_fs))

def label_n_lines(label):
    return label.count("\n") + 1

# ---------------------------------------------------------------------------
# Load and standardise per trait
# ---------------------------------------------------------------------------

records = []

for trait_info, stats_f, merged_f in zip(trait_list, stats_files, merged_files):
    fid         = trait_info["field_id"]
    trait_label = trait_info.get("label", str(fid))

    log(f"  Loading: {trait_label}  fid={fid}")

    try:
        stats_df = pd.read_csv(stats_f, sep="\t")
    except Exception as e:
        log(f"  WARNING: could not read stats for {fid}: {e}")
        continue

    if stats_df.empty or "test" not in stats_df.columns:
        log(f"  WARNING: empty stats for {fid}, skipping")
        continue

    is_case_control = stats_df["test"].str.contains("Logistic").any()
    # Support both standard and SNP-conditioned test names
    logistic_test = "Logistic regression (adj.+SNP)" if "Logistic regression (adj.+SNP)" in stats_df["test"].values else "Logistic regression (adj.)"
    linear_test   = "Linear regression (adj.+SNP)"   if "Linear regression (adj.+SNP)"   in stats_df["test"].values else "Linear regression (adj.)"

    n_cases = n_controls = None
    if is_case_control:
        mw_rows = stats_df[stats_df["test"] == "Mann-Whitney U"]
        if len(mw_rows) > 0:
            n_cases    = int(mw_rows.iloc[0]["n_cases"])
            n_controls = int(mw_rows.iloc[0]["n_controls"])

    axis_label = make_axis_label(trait_label, fid, n_cases, n_controls)

    trait_sd = None
    if not is_case_control:
        try:
            merged_df = pd.read_csv(merged_f, sep="\t", usecols=["trait_value"])
            trait_sd  = float(merged_df["trait_value"].dropna().std())
            log(f"    SD(trait) = {trait_sd:.4f}")
        except Exception as e:
            log(f"  WARNING: could not compute SD for {fid}: {e}")

    for anc in focus_ancestry:
        if is_case_control:
            row = stats_df[
                (stats_df["test"] == logistic_test) &
                (stats_df["ancestry"] == anc)
            ]
            if len(row) == 0:
                continue
            row  = row.iloc[0]
            OR   = float(row["OR"])
            lo   = float(row["OR_CI_low"])
            hi   = float(row["OR_CI_high"])
            pval = float(row["p_value"])
            records.append({
                "axis_label":    axis_label,
                "field_id":      str(fid),
                "ancestry":      anc,
                "analysis_type": "case_control",
                "effect":        np.log(OR),
                "ci_low":        np.log(lo),
                "ci_high":       np.log(hi),
                "p_value":       pval,
                "significant":   pval < 0.05,
            })
        else:
            row = stats_df[
                (stats_df["test"] == linear_test) &
                (stats_df["ancestry"] == anc)
            ]
            if len(row) == 0:
                continue
            row  = row.iloc[0]
            beta = float(row["beta"])
            lo   = float(row["CI_low"])
            hi   = float(row["CI_high"])
            pval = float(row["p_value"])

            if trait_sd and trait_sd > 0:
                beta /= trait_sd
                lo   /= trait_sd
                hi   /= trait_sd

            records.append({
                "axis_label":    axis_label,
                "field_id":      str(fid),
                "ancestry":      anc,
                "analysis_type": "continuous",
                "effect":        beta,
                "ci_low":        lo,
                "ci_high":       hi,
                "p_value":       pval,
                "significant":   pval < 0.05,
            })

if not records:
    log("ERROR: no records to plot — all traits missing or failed")
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
# Sort traits by first focal ancestry effect size (descending)
# ---------------------------------------------------------------------------

ef_effects = (
    df[df["ancestry"] == focus_ancestry[0]]
    .groupby("axis_label")["effect"]
    .mean()
    .sort_values(ascending=False)
)
trait_order = list(ef_effects.index)
extra       = [t for t in df["axis_label"].unique() if t not in trait_order]
trait_order = trait_order + extra

n_traits     = len(trait_order)
y_pos        = {lbl: n_traits - 1 - i for i, lbl in enumerate(trait_order)}
primary_type = df["analysis_type"].mode().iloc[0]

log(f"Trait order: {trait_order}")

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
# Row height depends on max number of text lines across all labels in that row.
# Case/control labels have 2 lines; continuous have 1. We compute per-row line
# counts so the y-axis spacing never clips multi-line labels.

LINE_H      = 0.28   # inches per text line
ROW_PAD     = 0.18   # extra padding per row beyond text
row_heights = [LINE_H * label_n_lines(lbl) + ROW_PAD for lbl in trait_order]
content_h   = sum(row_heights)
fig_h       = max(2.6, content_h + 1.0)   # +1.0 for title + x-axis labels

# Evenly-spaced y positions scaled to actual row heights so taller rows
# get proportionally more space.
cumulative = 0.0
y_pos_scaled = {}
for lbl in reversed(trait_order):   # bottom to top
    h = LINE_H * label_n_lines(lbl) + ROW_PAD
    y_pos_scaled[lbl] = cumulative + h / 2
    cumulative += h

# Rescale to integer-like range for simpler axis limits
scale = (n_traits - 1) / max(cumulative, 1)
y_pos = {lbl: v * scale for lbl, v in y_pos_scaled.items()}
y_max = cumulative * scale

n_anc      = len(focus_ancestry)
forest_w   = 2.6
pval_w     = 0.60
label_w    = 2.8


label_rel  = label_w  / forest_w
pval_rel   = pval_w   / forest_w

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

# Compute label font size using actual rendered axes width (more accurate than nominal)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
ax_label_bbox = ax_labels.get_window_extent(renderer=renderer)
actual_label_w_in = ax_label_bbox.width / fig.dpi
log(f"Actual label column width: {actual_label_w_in:.3f}in (nominal: {label_w}in)")
label_fs = adaptive_fontsize(trait_order, actual_label_w_in, fig)
log(f"Label font size: {label_fs:.1f}pt")

# ---------------------------------------------------------------------------
# Draw helpers
# ---------------------------------------------------------------------------

def draw_forest(ax, anc, df, y_pos, y_max, primary_type, n_traits):
    color    = ANC_COLORS.get(anc, "#555555")
    anc_data = df[df["ancestry"] == anc]

    for _, row in anc_data.iterrows():
        yp   = y_pos.get(row["axis_label"])
        if yp is None:
            continue
        eff   = row["effect"]
        lo    = row["ci_low"]
        hi    = row["ci_high"]
        sig   = row["significant"]
        alpha = SIG_ALPHA if sig else NONSIG_ALPHA

        ax.plot([lo, hi], [yp, yp], color=color, lw=1.5, alpha=alpha,
                solid_capstyle="round", zorder=3)
        if sig:
            ax.scatter([eff], [yp], color=color, s=DOT_SIZE,
                       zorder=5, alpha=SIG_ALPHA)
        else:
            ax.scatter([eff], [yp], color=color, s=DOT_SIZE,
                       zorder=5, alpha=NONSIG_ALPHA,
                       facecolors="none", linewidths=1.4)

    ax.axvline(0, color="#aaaaaa", lw=0.9, linestyle="--", zorder=2)
    ax.set_ylim(-0.3, y_max + 0.3)
    ax.set_yticks([])
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.xaxis.label.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4,
            linestyle="--", alpha=0.8, axis="x")
    ax.set_axisbelow(True)
    ax.set_title(anc, fontsize=9, fontweight="bold", color=color, pad=4)

    if primary_type == "continuous":
        ax.set_xlabel("Std. β  (SD of trait / unit ancestry)",
                      fontsize=6.5, color=TEXT_COLOR)
    else:
        ax.set_xlabel("Odds Ratio  (log scale)", fontsize=6.5, color=TEXT_COLOR)
        xticks = ax.get_xticks()
        ax.set_xticklabels([f"{np.exp(x):.2f}" for x in xticks], fontsize=6.5)


def draw_pvals(ax, anc, df, y_pos, y_max, ha="left"):
    color    = ANC_COLORS.get(anc, "#555555")
    anc_data = df[df["ancestry"] == anc]

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, y_max + 0.3)
    ax.axis("off")

    xpos = 0.08 if ha == "left" else 0.92
    for _, row in anc_data.iterrows():
        yp   = y_pos.get(row["axis_label"])
        if yp is None:
            continue
        sig  = row["significant"]
        txt  = fmt_pval(row["p_value"])
        fc   = color if sig else "#bbbbbb"
        ax.text(xpos, yp, txt, va="center", ha=ha,
                fontsize=6.0, color=fc, fontfamily="monospace")

    # Header inside the data range (near top but not above)
    ax.text(0.5, y_max + 0.05, f"p ({anc})",
            va="bottom", ha="center",
            fontsize=6.5, color=color, fontweight="bold")


def draw_labels(ax, trait_order, y_pos, y_max, fontsize):
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, y_max + 0.3)
    ax.axis("off")
    for lbl in trait_order:
        yp = y_pos[lbl]
        ax.text(0.5, yp, lbl, va="center", ha="center",
                fontsize=fontsize, color=TEXT_COLOR, linespacing=1.3)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

draw_pvals(ax_pval_left,  focus_ancestry[0],  df, y_pos, y_max, ha="left")
draw_labels(ax_labels, trait_order, y_pos, y_max, fontsize=label_fs)
for anc in focus_ancestry:
    draw_forest(ax_forests[anc], anc, df, y_pos, y_max, primary_type, n_traits)
draw_pvals(ax_pval_right, focus_ancestry[-1], df, y_pos, y_max, ha="right")

# ---------------------------------------------------------------------------
# Title — smaller, with top margin so it never overlaps p-value headers
# ---------------------------------------------------------------------------

fig.subplots_adjust(top=0.90)
fig.suptitle(category_label, fontsize=9, fontweight="bold",
             color=TEXT_COLOR, y=0.98)

# ---------------------------------------------------------------------------
# Save PNG + PDF + SVG
# ---------------------------------------------------------------------------

save_kw = dict(bbox_inches="tight", facecolor=BG_COLOR)
plt.savefig(snakemake.output.plot_png, dpi=150, **save_kw)
plt.savefig(snakemake.output.plot_pdf, **save_kw)
plt.savefig(snakemake.output.plot_svg, **save_kw)
plt.close()
log(f"Saved: PNG / PDF / SVG")