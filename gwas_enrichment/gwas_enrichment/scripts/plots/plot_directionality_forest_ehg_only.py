"""
plot_directionality_forest_ehg_only.py
=======================================
2 x 3 panel forest plot of trait-level PGS directionality: EF vs EHG
only (no WHG series -- see plot_directionality_forest_ehg.py for the
dodged WHG+EHG overlay version).

Whiskers-only style, no central dot -- CI ranges only, matching the
bootstrap CI figure in the main pipeline's plot_directionality_forest.py
(fig3). Reads bootstrap_ci_lo/bootstrap_ci_hi directly from
bootstrap_ci_ehg.txt (produced by calculate_directionality_bootstrap_ehg.py,
Option A: 1000x resampling of individuals with replacement) rather than
computing an analytic CI -- this matches how Figure 3's whiskers panel is
actually computed, not the analytic-CI main dot panel.

Single whisker per trait row:
    - EF higher   -> color_ef_higher   (same as the main WHG/EF figure)
    - EHG higher  -> color_ehg_higher  (new, distinct from WHG's color)
    - not significant -> color_neutral

Sign/significance for coloring and row order both come from the original
point estimate (cohens_d, ttest_pval), not the bootstrap-reconstructed
values -- matching how the main pipeline's bootstrap figure does it.
"""

import sys, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# 1. PALETTE
# =============================================================================

p = snakemake.params

COLOR_EF  = str(getattr(p, 'color_ef_higher',  '#D6604D'))
COLOR_EHG = str(getattr(p, 'color_ehg_higher', '#E8A33D'))
COLOR_NS  = str(getattr(p, 'color_neutral',    '#999999'))

BG        = 'white'
MUTED     = '#888888'
GRID_CLR  = '#ebebeb'
SPINE_CLR = '#d0d0d0'


# =============================================================================
# 2. SNAKEMAKE I/O & PARAMS
# =============================================================================

directionality_ehg_file = snakemake.input['bootstrap_ci_ehg']
output_file              = snakemake.output['plot']

COL_CAT   = str(p.col_category)
COL_TRAIT = str(p.col_trait)
COL_D     = str(p.col_cohens_d)
COL_P     = str(p.col_pvalue)
COL_N     = str(getattr(p, 'col_n_extracted', 'n_snps_extracted'))

SIG_THRESH  = float(p.sig_threshold)
N_BOOTSTRAP = int(getattr(p, 'n_bootstrap', 1000))
MAX_LABEL   = int(getattr(p, 'max_label_length', 50))
N_TRAITS    = int(getattr(p, 'n_panel_traits', 10))
FIG_W       = float(getattr(p, 'panel_figure_width', 8.0))
FIG_H       = float(getattr(p, 'panel_figure_height', 18.0))
PANEL_ORDER = list(getattr(p, 'panel_order', []) or [])

os.makedirs(os.path.dirname(output_file), exist_ok=True)


# =============================================================================
# 3. HELPERS
# =============================================================================

def _stars(pval):
    if pval < 0.001: return '***'
    elif pval < 0.01: return '**'
    elif pval < 0.05: return '*'
    return ''

def _safe_int(row, col):
    try:
        v = row[col] if col and col in row.index else None
        return int(float(v)) if v is not None and pd.notna(v) else None
    except (TypeError, ValueError):
        return None

def _norm_str(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


# =============================================================================
# 4. LOAD DATA
# =============================================================================

df = pd.read_csv(directionality_ehg_file, sep='\t', comment='#')
print(f"[forest_ehg_only] {len(df)} trait rows (EF/EHG bootstrap). Cols: {list(df.columns)}", file=sys.stderr)

if df.empty:
    raise ValueError(f"[forest_ehg_only] {directionality_ehg_file} is empty -- nothing to plot")

df = df.copy()
df[COL_D] = pd.to_numeric(df[COL_D], errors='coerce')
df[COL_P] = pd.to_numeric(df[COL_P], errors='coerce')
df['bootstrap_ci_lo'] = pd.to_numeric(df.get('bootstrap_ci_lo'), errors='coerce')
df['bootstrap_ci_hi'] = pd.to_numeric(df.get('bootstrap_ci_hi'), errors='coerce')
n_before = len(df)
df = df.dropna(subset=[COL_D, COL_P, 'bootstrap_ci_lo', 'bootstrap_ci_hi'])
if len(df) < n_before:
    print(f"[forest_ehg_only] Dropped {n_before - len(df)} rows with missing "
          f"point estimate or bootstrap CI (skipped in the bootstrap script)",
          file=sys.stderr)

cats = list(df[COL_CAT].unique())
if PANEL_ORDER:
    order_idx = {_norm_str(n): i for i, n in enumerate(PANEL_ORDER)}
    for c in cats:
        if _norm_str(c) not in order_idx:
            print(f"[forest_ehg_only] WARNING: category not in panel_order (check spelling): {c}",
                  file=sys.stderr)
    cats = sorted(cats, key=lambda c: order_idx.get(_norm_str(c), 999))
else:
    cats = sorted(cats, key=lambda c: -len(df[df[COL_CAT] == c]))


# =============================================================================
# 5. FIRST PASS -- build panel data + global x range
#    CI comes directly from the bootstrap file (already computed), no
#    analytic approximation here.
# =============================================================================

panels   = []   # list of (cat_name, cat_df, ci_data)
global_x = []

for cat in cats:
    cat_df = (df[df[COL_CAT] == cat]
                .sort_values(COL_D, ascending=True)
                .reset_index(drop=True))
    if cat_df.empty:
        panels.append((cat, cat_df, []))
        continue

    ci_data = []
    for _, row in cat_df.iterrows():
        lo = float(row['bootstrap_ci_lo'])
        hi = float(row['bootstrap_ci_hi'])
        ci_data.append((lo, hi))
        global_x += [lo, hi]

    panels.append((cat, cat_df, ci_data))

if not global_x:
    raise ValueError("[forest_ehg_only] No CI data found across any panel.")

g_lo, g_hi = min(global_x), max(global_x)
g_span     = max(g_hi - g_lo, 0.10)
g_pad      = g_span * 0.18 + 0.04
X_LO, X_HI = g_lo - g_pad, g_hi + g_pad


# =============================================================================
# 6. DRAW ONE PANEL
# =============================================================================

def draw_panel(ax, cat_name, cat_df, ci_data, n_rows, show_xlabels):
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(-0.75, n_rows - 0.25)

    for i, ((_, row), (lo, hi)) in enumerate(zip(cat_df.iterrows(), ci_data)):
        d         = float(row[COL_D])
        pval      = float(row[COL_P])
        sig       = pval < SIG_THRESH
        dot_color = (COLOR_EF if d > 0 else COLOR_EHG) if sig else COLOR_NS

        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f4f4f4', linewidth=0, zorder=0)

        nblk  = _safe_int(row, COL_N)
        stars = _stars(pval)
        name  = str(row[COL_TRAIT]).replace('_', ' ')
        name  = (name[:MAX_LABEL - 1] + '...') if len(name) > MAX_LABEL else name
        n_str = f' (n={nblk})' if nblk is not None else ''
        label = f'{name}{n_str}  {stars}'.rstrip() if stars else f'{name}{n_str}'
        ax.text(X_LO + (X_HI - X_LO) * 0.018, i + 0.30,
                label, ha='left', va='bottom',
                fontsize=6.5, color='#363636', alpha=0.88,
                zorder=5, clip_on=True)

        serif = 0.10
        ax.plot([lo, hi], [i, i],
                color=dot_color, lw=1.8, solid_capstyle='butt', alpha=0.85, zorder=3)
        for xe in (lo, hi):
            ax.plot([xe, xe], [i - serif, i + serif],
                    color=dot_color, lw=1.8, alpha=0.85, zorder=3)

    ax.axvline(0, color='#c8c8c8', lw=0.9, linestyle='--', zorder=1)
    ax.set_yticks([])

    title_str = str(cat_name).replace('_', ' ')
    if len(title_str) > 52:
        title_str = title_str[:51] + '...'
    ax.set_title(title_str, color='#1a1a1a', fontsize=7.5,
                 fontweight='normal', pad=4, loc='left')

    ax.tick_params(axis='x', labelsize=7.5 if show_xlabels else 0,
                   colors=MUTED, length=3, width=0.7)
    if show_xlabels:
        ax.set_xlabel("Cohen's d", fontsize=8.5, color=MUTED, labelpad=4)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(SPINE_CLR)
        sp.set_linewidth(0.8)

    ax.grid(axis='x', linestyle=':', linewidth=0.4, color=GRID_CLR, zorder=0)
    ax.set_facecolor(BG)


# =============================================================================
# 7. BUILD FIGURE
# =============================================================================

N_ROWS, N_COLS = 3, 2
panels = panels[:N_ROWS * N_COLS]

row_max_n = []
for r in range(N_ROWS):
    rp = panels[r * N_COLS:(r + 1) * N_COLS]
    row_max_n.append(max((len(pp[1]) for pp in rp if not pp[1].empty), default=1))

fig_h = FIG_H * sum(row_max_n) / (N_ROWS * N_TRAITS)

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FIG_W, fig_h),
                          gridspec_kw={'height_ratios': row_max_n},
                          facecolor=BG)
fig.patch.set_facecolor(BG)

for idx, (cat, cat_df, ci_data) in enumerate(panels):
    row_idx = idx // N_COLS
    col_idx = idx % N_COLS
    ax = axes[row_idx, col_idx]

    if cat_df.empty:
        ax.set_visible(False)
        continue

    draw_panel(ax, cat, cat_df, ci_data, row_max_n[row_idx],
               show_xlabels=(row_idx == N_ROWS - 1))

for idx in range(len(panels), N_ROWS * N_COLS):
    axes.flatten()[idx].set_visible(False)


# =============================================================================
# 8. LEGEND
# =============================================================================

_mk = dict(marker='|', linestyle='none', markersize=14, markeredgewidth=2.0)

n_ehg_display = int(df['n_EHG_samples'].iloc[0]) if 'n_EHG_samples' in df.columns and len(df) else None
n_note = f'EHG n={n_ehg_display} -- interpret CIs with caution' if n_ehg_display else ''

legend_handles = [
    Line2D([0], [0], markeredgecolor=COLOR_EF,  color=COLOR_EF,
           label=f'EF > EHG  (bootstrap 95% CI, {N_BOOTSTRAP}x resample)', **_mk),
    Line2D([0], [0], markeredgecolor=COLOR_EHG, color=COLOR_EHG,
           label=f'EHG > EF  (bootstrap 95% CI, {N_BOOTSTRAP}x resample)', **_mk),
    Line2D([0], [0], markeredgecolor=COLOR_NS,  color=COLOR_NS,
           label=f'p >= {SIG_THRESH}  (ns)', **_mk),
    Line2D([0], [0], color='none', label=''),
    Line2D([0], [0], color='none', label='* p < 0.05   ** p < 0.01   *** p < 0.001'),
    Line2D([0], [0], color='none', label=n_note),
]

fig.legend(handles=legend_handles, loc='lower center', ncol=3,
           fontsize=8.0, framealpha=0.0, labelcolor=MUTED,
           bbox_to_anchor=(0.5, 0.0),
           handletextpad=0.4, columnspacing=1.0)

fig.tight_layout(rect=[0, 0.08, 1, 1.0], h_pad=2.0, w_pad=2.0)


# =============================================================================
# 9. SAVE
# =============================================================================

for ext in ('.pdf', '.svg'):
    path = output_file if ext == '.pdf' else output_file.replace('.pdf', '.svg')
    fig.savefig(path, bbox_inches='tight', dpi=180, facecolor=BG)
    print(f"[forest_ehg_only] Saved: {path}", file=sys.stderr)

plt.close(fig)