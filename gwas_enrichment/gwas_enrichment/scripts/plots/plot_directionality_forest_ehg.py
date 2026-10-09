"""
plot_directionality_forest_ehg.py
==================================
2 x 3 panel forest plot of trait-level PGS directionality, EHG addition.

Same panel layout, trait selection, and trait set as
plot_directionality_forest.py's main figure. Each trait row shows TWO
dodged series instead of one:

    - filled circle, upper offset : WHG vs EF   (from directionality_file)
    - open diamond,  lower offset : EHG vs EF   (from directionality_ehg_file)

Both series share the same colour convention and the same Cohen's d
x-axis (positive = EF higher), so the two dots on a row directly show
whether EHG tracks WHG or diverges from it for that trait.

Does not touch the original plot_directionality_forest.py or its
outputs. Reads its own inputs, writes its own output file.
"""

import sys, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy.stats import norm as _sp_norm
    def _z(ci): return _sp_norm.ppf(1 - (1 - ci) / 2)
except ImportError:
    def _z(ci): return 1.96


# =============================================================================
# 1. PALETTE (sourced from config where available, same defaults as the
#    main forest plot so the two figures read consistently together)
# =============================================================================

p = snakemake.params

COLOR_EF = str(getattr(p, 'color_ef_higher', '#C94F3A'))
COLOR_HG = str(getattr(p, 'color_hg_higher', '#1E5E99'))
COLOR_NS = str(getattr(p, 'color_neutral',   '#BBBBBB'))

BG        = 'white'
TEXT      = '#1a1a1a'
MUTED     = '#888888'
GRID_CLR  = '#ebebeb'
SPINE_CLR = '#d0d0d0'


# =============================================================================
# 2. SNAKEMAKE I/O & PARAMS
# =============================================================================

enrichment_file         = snakemake.input['enrichment']
directionality_file     = snakemake.input['directionality']
directionality_ehg_file = snakemake.input['directionality_ehg']
output_file              = snakemake.output['plot']

COL_CAT   = str(p.col_category)
COL_TRAIT = str(p.col_trait)
COL_D     = str(p.col_cohens_d)
COL_P_CFG = str(p.col_pvalue)
COL_NHG   = str(p.col_n_hg)
COL_NEF   = str(p.col_n_ef)
COL_N     = str(getattr(p, 'col_n_extracted', 'n_snps_extracted'))

SIG_THRESH = float(p.sig_threshold)
CI_LEVEL   = float(p.ci_level)
MIN_SNPS_EXTR = int(getattr(p, 'min_snps_extracted', 20))
EXCLUDE       = list(getattr(p, 'exclude_traits', []) or [])
MAX_LABEL     = int(getattr(p, 'max_label_length', 50))
N_CATS        = int(getattr(p, 'n_panel_categories', 6))
N_TRAITS      = int(getattr(p, 'n_panel_traits', 10))
FIG_W         = float(getattr(p, 'panel_figure_width', 8.0))
FIG_H         = float(getattr(p, 'panel_figure_height', 18.0))
PANEL_ORDER   = list(getattr(p, 'panel_order', []) or [])

# Vertical dodge between the two series (same convention as sensitivity_y_offset)
Y_OFFSET = float(getattr(p, 'ehg_y_offset', 0.18))

Z_CRIT = _z(CI_LEVEL)

os.makedirs(os.path.dirname(output_file), exist_ok=True)

ENR_P_CANDS = ['mannwhitneyu_pval', 'mw_p_value', 'p_value', 'pvalue',
               'mannwhitneyu_pval_twosided', 'kolmogorov_smirnov_pval', 'ttest_pval']
DIR_P_CANDS = ['ttest_pval', 'pgs_pvalue', 'mannwhitneyu_pval', 'p_value', 'pvalue']


# =============================================================================
# 3. HELPERS
# =============================================================================

def _resolve_col(col, cands, df, label, required=True):
    if col and col in df.columns:
        return col
    auto = next((c for c in cands if c in df.columns), None)
    if auto:
        print(f"[forest_ehg] '{col}' -> '{auto}' for {label}", file=sys.stderr)
        return auto
    if required:
        raise ValueError(f"Column '{label}' not found. Tried: {[col] + cands}")
    return None

def _stars(pval):
    if pval < 0.001: return '***'
    elif pval < 0.01: return '**'
    elif pval < 0.05: return '*'
    return ''

def _d_se(d, n1, n2):
    n1 = max(float(n1), 2.0)
    n2 = max(float(n2), 2.0)
    return np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2.0 * (n1 + n2 - 2.0)))

def _safe_float(row, col, default=30.0):
    try:
        v = row[col] if col and col in row.index else default
        return float(v) if pd.notna(v) else default
    except (TypeError, ValueError):
        return default

def _safe_int(row, col):
    try:
        v = row[col] if col and col in row.index else None
        return int(float(v)) if v is not None and pd.notna(v) else None
    except (TypeError, ValueError):
        return None

def _norm_str(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


# =============================================================================
# 4. LOAD & SELECT DATA -- identical selection logic to
#    plot_directionality_forest.py's main figure
# =============================================================================

enr = pd.read_csv(enrichment_file, sep='\t', comment='#')
enr_cat = next((c for c in ['category', 'Category', 'trait_category']
                if c in enr.columns), None)
if enr_cat is None:
    raise ValueError(f"No category column in {enrichment_file}")
enr_p = next((c for c in ENR_P_CANDS if c in enr.columns), None)
if enr_p is None:
    raise ValueError(f"No p-value column in {enrichment_file}")

top_cats_raw = (enr.dropna(subset=[enr_p])
                   .sort_values(enr_p)
                   .head(N_CATS)[enr_cat]
                   .tolist())
print(f"[forest_ehg] Enrichment top-{N_CATS}: {top_cats_raw}", file=sys.stderr)

df = pd.read_csv(directionality_file, sep='\t', comment='#')
print(f"[forest_ehg] {len(df)} trait rows (WHG/EF). Cols: {list(df.columns)}", file=sys.stderr)

COL_P = _resolve_col(COL_P_CFG, DIR_P_CANDS, df, 'directionality p-value')

if COL_N and COL_N in df.columns and MIN_SNPS_EXTR > 0:
    before = len(df)
    df = df[pd.to_numeric(df[COL_N], errors='coerce').fillna(0) >= MIN_SNPS_EXTR]
    print(f"[forest_ehg] min_snps_extracted ({COL_N}>={MIN_SNPS_EXTR}): {before}->{len(df)}",
          file=sys.stderr)

if EXCLUDE:
    df = df[~df[COL_TRAIT].isin(EXCLUDE)]

ehg_df = pd.read_csv(directionality_ehg_file, sep='\t', comment='#')
print(f"[forest_ehg] {len(ehg_df)} trait rows (EHG/EF)", file=sys.stderr)

ehg_lookup = {}
for _, r in ehg_df.iterrows():
    ehg_lookup[(str(r.get(COL_TRAIT, '')), str(r.get(COL_CAT, '')))] = r

actual_cats    = df[COL_CAT].unique()
norm_to_actual = {_norm_str(c): c for c in actual_cats}

def _resolve_cat(name):
    if name in actual_cats:
        return name
    return norm_to_actual.get(_norm_str(name))

top_cats = [r for raw in top_cats_raw
            for r in [_resolve_cat(raw)] if r is not None]
if not top_cats:
    raise ValueError("[forest_ehg] No valid categories after normalisation.")


# =============================================================================
# 5. FIRST PASS -- collect panel data + compute global x range
#    (includes EHG CI edges, since n=EHG is small and may widen the range)
# =============================================================================

panels   = []   # list of (cat_name, cat_df, ci_data, ehg_ci_data)
global_x = []
n_ehg_missing_total = 0

for cat in top_cats[:N_CATS]:
    cat_df = (
        df[df[COL_CAT] == cat]
          .copy()
          .assign(**{COL_P: lambda x: pd.to_numeric(x[COL_P], errors='coerce')})
          .dropna(subset=[COL_P])
          .nsmallest(N_TRAITS, COL_P)
          .sort_values(COL_D, ascending=True)
          .reset_index(drop=True)
    )
    if cat_df.empty:
        print(f"[forest_ehg] No data for '{cat}' -- panel hidden", file=sys.stderr)
        panels.append((cat, cat_df, [], []))
        continue

    ci_data = []
    for _, row in cat_df.iterrows():
        d  = float(row[COL_D])
        n1 = _safe_float(row, COL_NHG)
        n2 = _safe_float(row, COL_NEF)
        se = _d_se(d, n1, n2)
        ci = Z_CRIT * se
        ci_data.append((d, ci))
        global_x += [d - ci, d + ci]

    ehg_ci_data = []
    for _, row in cat_df.iterrows():
        er = ehg_lookup.get((str(row[COL_TRAIT]), cat))
        if er is None:
            ehg_ci_data.append(None)
            n_ehg_missing_total += 1
            continue
        try:
            ed  = float(er['cohens_d'])
            en1 = _safe_float(er, 'n_EF_samples')
            en2 = _safe_float(er, 'n_EHG_samples')
            ese = _d_se(ed, en1, en2)
            eci = Z_CRIT * ese
            ep  = float(er.get('ttest_pval', 1.0))
            ehg_ci_data.append((ed, eci, ep))
            global_x += [ed - eci, ed + eci]
        except (KeyError, ValueError, TypeError):
            ehg_ci_data.append(None)
            n_ehg_missing_total += 1

    panels.append((cat, cat_df, ci_data, ehg_ci_data))

if n_ehg_missing_total:
    print(f"[forest_ehg] {n_ehg_missing_total} trait rows have no matching EHG result "
          f"(will show WHG dot only)", file=sys.stderr)

if not global_x:
    raise ValueError("[forest_ehg] No CI data found across any panel.")

g_lo, g_hi = min(global_x), max(global_x)
g_span     = max(g_hi - g_lo, 0.10)
g_pad      = g_span * 0.18 + 0.04
X_LO, X_HI = g_lo - g_pad, g_hi + g_pad


# =============================================================================
# 6. DRAW ONE PANEL
# =============================================================================

def draw_panel(ax, cat_name, cat_df, ci_data, ehg_ci_data, show_xlabels):
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(-0.75, N_TRAITS - 0.25)

    for i, ((_, row), (d, ci)) in enumerate(zip(cat_df.iterrows(), ci_data)):
        pval      = float(row[COL_P])
        sig       = pval < SIG_THRESH
        bar_color = (COLOR_EF if d > 0 else COLOR_HG) if sig else COLOR_NS

        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f4f4f4', linewidth=0, zorder=0)

        nblk  = _safe_int(row, COL_N)
        stars = _stars(pval)
        name  = str(row[COL_TRAIT]).replace('_', ' ')
        name  = (name[:MAX_LABEL - 1] + '...') if len(name) > MAX_LABEL else name
        n_str = f' (n={nblk})' if nblk is not None else ''
        label = f'{name}{n_str}  {stars}'.rstrip() if stars else f'{name}{n_str}'
        if i < len(ehg_ci_data) and ehg_ci_data[i] is None:
            label += '  [no EHG data]'
        ax.text(X_LO + (X_HI - X_LO) * 0.018, i + 0.42,
                label, ha='left', va='bottom',
                fontsize=6.5, color='#363636', alpha=0.88,
                zorder=5, clip_on=True)

        # -- WHG vs EF: filled circle, upper offset --
        y_whg = i + Y_OFFSET
        ax.plot([d - ci, d + ci], [y_whg, y_whg],
                color=bar_color, lw=2.0, solid_capstyle='round', alpha=0.88, zorder=3)
        ax.scatter([d], [y_whg], marker='o', s=62,
                   color=bar_color, edgecolors='white', linewidths=1.0, zorder=4)

        # -- EHG vs EF: open diamond, lower offset --
        if i < len(ehg_ci_data) and ehg_ci_data[i] is not None:
            ed, eci, ep = ehg_ci_data[i]
            esig  = ep < SIG_THRESH
            ecolr = (COLOR_EF if ed > 0 else COLOR_HG) if esig else COLOR_NS
            y_ehg = i - Y_OFFSET
            ax.plot([ed - eci, ed + eci], [y_ehg, y_ehg],
                    color=ecolr, lw=1.6, solid_capstyle='round', alpha=0.65, zorder=3)
            ax.scatter([ed], [y_ehg], marker='D', s=42,
                       color='white', edgecolors=ecolr, linewidths=1.4, zorder=4)

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
        ax.set_xlabel("Cohen's d  (positive = EF higher)", fontsize=8.5, color=MUTED, labelpad=4)

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

if PANEL_ORDER:
    order_idx = {_norm_str(n): i for i, n in enumerate(PANEL_ORDER)}
    for pname, *_ in panels:
        if _norm_str(pname) not in order_idx:
            print(f"[forest_ehg] WARNING: category not in panel_order (check spelling): {pname}",
                  file=sys.stderr)
    panels = sorted(panels, key=lambda x: order_idx.get(_norm_str(x[0]), 999))
else:
    panels = sorted(panels, key=lambda x: (-len(x[1]), x[0]))

row_max_n = []
for r in range(N_ROWS):
    rp = panels[r * N_COLS:(r + 1) * N_COLS]
    row_max_n.append(max((len(pp[1]) for pp in rp if not pp[1].empty), default=1))

fig_h = FIG_H * sum(row_max_n) / (N_ROWS * N_TRAITS)

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FIG_W, fig_h),
                          gridspec_kw={'height_ratios': row_max_n},
                          facecolor=BG)
fig.patch.set_facecolor(BG)

for idx, (cat, cat_df, ci_data, ehg_ci_data) in enumerate(panels[:N_ROWS * N_COLS]):
    row_idx = idx // N_COLS
    col_idx = idx % N_COLS
    ax = axes[row_idx, col_idx]

    if cat_df.empty:
        ax.set_visible(False)
        continue

    draw_panel(ax, cat, cat_df, ci_data, ehg_ci_data,
               show_xlabels=(row_idx == N_ROWS - 1))
    ax.set_ylim(-0.75, row_max_n[row_idx] - 0.25)

for idx in range(len(panels), N_ROWS * N_COLS):
    axes.flatten()[idx].set_visible(False)


# =============================================================================
# 8. SHARED LEGEND
# =============================================================================

_mk_main = dict(marker='o', linestyle='none', markersize=7,
                markeredgecolor='white', markeredgewidth=0.8)
_mk_ehg  = dict(marker='D', linestyle='none', markerfacecolor='white', markeredgewidth=1.4)

legend_handles = [
    Line2D([0], [0], markerfacecolor=COLOR_HG, label='WHG > EF  (d < 0)', **_mk_main),
    Line2D([0], [0], markerfacecolor=COLOR_EF, label='EF > WHG  (d > 0)', **_mk_main),
    Line2D([0], [0], markerfacecolor=COLOR_NS, label=f'p >= {SIG_THRESH}  (ns)', **_mk_main),
    Line2D([0], [0], color='none', label=''),
    Line2D([0], [0], markeredgecolor=COLOR_HG, label='EHG > EF  (diamond, d < 0)', **_mk_ehg),
    Line2D([0], [0], markeredgecolor=COLOR_EF, label='EF > EHG  (diamond, d > 0)', **_mk_ehg),
    Line2D([0], [0], color='none', label=''),
    Line2D([0], [0], color='none', label='* p < 0.05   ** p < 0.01   *** p < 0.001'),
    Line2D([0], [0], color='none', label=f'EHG n={int(next(iter(ehg_lookup.values()))["n_EHG_samples"]) if ehg_lookup else "?"} -- interpret CIs with caution'),
]

fig.legend(handles=legend_handles, loc='lower center', ncol=3,
           fontsize=8.0, framealpha=0.0, labelcolor=MUTED,
           bbox_to_anchor=(0.5, 0.0),
           handletextpad=0.4, columnspacing=1.0)

fig.tight_layout(rect=[0, 0.09, 1, 1.0], h_pad=2.0, w_pad=2.0)


# =============================================================================
# 9. SAVE
# =============================================================================

for ext in ('.pdf', '.svg'):
    path = output_file if ext == '.pdf' else output_file.replace('.pdf', '.svg')
    fig.savefig(path, bbox_inches='tight', dpi=180, facecolor=BG)
    print(f"[forest_ehg] Saved: {path}", file=sys.stderr)

plt.close(fig)