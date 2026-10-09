"""
plot_directionality_forest.py
=============================
2 × 3 panel forest plot of trait-level PGS directionality.

• Shared x-axis across all panels (globally computed CI range).
• Per row: CI bar (thin, round caps) + centre dot above a strip of
  SNP contribution dots sized by |contribution| and labelled for the
  top 2 per trait.
• Light theme only.
"""

import sys, os, re, glob
import zlib
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1 · SEMANTIC COLOUR MAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SEMANTIC = {
    'neuropsychiatric': '#7B68C8', 'neurological': '#7B68C8',
    'psychiatric':      '#9480D4', 'cognitive':    '#8B78D0',
    'brain':            '#7B68C8', 'sleep':        '#9B88D8',
    'cardiovascular':   '#C44444', 'cardiac':      '#C44444',
    'heart':            '#C44444', 'hematological':'#A02828',
    'haematological':   '#A02828', 'blood':        '#A02828',
    'coagulation':      '#B43030',
    'lipids':           '#C8A820', 'metabolic':    '#C8A820',
    'diabetes':         '#E08820', 'glycemic':     '#E08820',
    'thyroid':          '#B89820', 'endocrine':    '#B89820',
    'gastrointestinal': '#DD7030', 'gut':          '#DD7030',
    'liver':            '#B85C20', 'hepatic':      '#B85C20',
    'immune':           '#D06020', 'autoimmune':   '#D06020',
    'inflammatory':     '#C05A18', 'allergy':      '#D07828',
    'anthropometry':    '#4C72B0', 'body':         '#4C72B0',
    'behavioral':       '#8172B2', 'behaviour':    '#8172B2',
    'substance':        '#9482C0',
    'musculoskeletal':  '#C05858', 'bone':         '#C05858',
    'joint':            '#C05858', 'connective':   '#C05858',
    'sensory':          '#2A9EA8', 'vision':       '#2A9EA8',
    'hearing':          '#2A9EA8', 'ocular':       '#3AAEBC',
    'lifestyle':        '#55A868', 'diet':         '#55A868',
    'physical':         '#55A868',
    'aging':            '#888888', 'ageing':       '#888888',
    'longevity':        '#888888',
    'reproductive':     '#C87AAC', 'fertility':    '#C87AAC',
    'pregnancy':        '#C87AAC', 'maternal':     '#C87AAC',
    'cancer':           '#505050', 'oncological':  '#505050',
    'respiratory':      '#5898C8', 'pulmonary':    '#5898C8',
    'renal':            '#5060A8', 'kidney':       '#5060A8',
    'dental':           '#60B4A8', 'oral':         '#60B4A8',
    'skin':             '#B89050', 'dermatological':'#B89050',
}
_FALLBACK = ['#4C72B0', '#DD8452', '#55A868', '#8172B2', '#937860', '#CCB974']

def _cat_color(cat_name):
    name  = str(cat_name).replace('_', ' ')
    group = name.split(' - ')[0].strip() if ' - ' in name else name.split()[0]
    key   = group.lower().strip()
    return (_SEMANTIC.get(key)
            or _SEMANTIC.get(key.split()[0])
            or _FALLBACK[zlib.crc32(key.encode('utf-8')) % len(_FALLBACK)])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2 · PALETTE CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CI bar / centre dot  — saturated directional colours
COLOR_EF   = '#C94F3A'
COLOR_HG   = '#1E5E99'
COLOR_NS   = '#BBBBBB'

# SNP dots — lighter, softer versions so they don't compete with the bar
SNP_CLR_EF = '#E8957E'
SNP_CLR_HG = '#7BADD4'
SNP_CLR_NS = '#CCCCCC'

BG         = 'white'
TEXT       = '#1a1a1a'
MUTED      = '#888888'
GRID_CLR   = '#ebebeb'
SPINE_CLR  = '#d0d0d0'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3 · SNAKEMAKE I/O & PARAMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

enrichment_file     = snakemake.input['enrichment']
directionality_file = snakemake.input['directionality']
contrib_dir         = str(snakemake.input.get('snp_contributions_dir', ''))
output_file       = snakemake.output['plot']
output_file_sens  = snakemake.output.get('plot_sensitivity', '')
output_file_boot  = snakemake.output.get('plot_bootstrap', '')

p = snakemake.params

COL_CAT        = str(p.col_category)
COL_TRAIT      = str(p.col_trait)
COL_D          = str(p.col_cohens_d)
COL_P_CFG      = str(p.col_pvalue)
COL_NHG        = str(p.col_n_hg)
COL_NEF        = str(p.col_n_ef)
COL_N          = str(getattr(p, 'col_n_extracted', 'n_snps_extracted'))

SIG_THRESH     = float(p.sig_threshold)
CI_LEVEL       = float(p.ci_level)
MIN_SNPS_EXTR  = int(getattr(p, 'min_snps_extracted',      10))
EXCLUDE        = list(getattr(p, 'exclude_traits',          []) or [])
MAX_LABEL      = int(getattr(p, 'max_label_length',         22))
N_CATS         = int(getattr(p, 'n_panel_categories',        6))
N_TRAITS       = int(getattr(p, 'n_panel_traits',            5))
TOP_SNPS       = int(getattr(p, 'top_n_snp_overlay',       400))
# SNP x-scaling: max |contrib| maps to this fraction of the global x half-span
SNP_SCALE      = float(getattr(p, 'snp_contribution_scale', 0.45))
# How many SNPs to label per trait row (by |contribution|)
N_SNP_LABELS   = int(getattr(p, 'n_snp_labels',              2))
FIG_W          = float(getattr(p, 'panel_figure_width',     13.0))
FIG_H          = float(getattr(p, 'panel_figure_height',    10.0))

# Sensitivity (optional -- leave path blank to skip)
SENS_FILE  = str(getattr(p, 'sensitivity_directionality_file', ''))
Y_OFFSET   = float(getattr(p, 'sensitivity_y_offset', 0.18))

# Bootstrap CI plot (optional -- leave path blank to skip)
BOOT_FILE  = str(getattr(p, 'bootstrap_ci_file', ''))
BOOT_OFFSET = float(getattr(p, 'bootstrap_y_offset', 0.18))
LOO_FILE   = str(getattr(p, 'loo_file', ''))

# Bootstrap colour variants (lighter/darker of direction colours)
BOOT_EF_LIGHT  = '#E8957E'   # bootstrap sample CI — EF>HG
BOOT_EF_DARK   = '#A03020'   # SNP-dropout CI     — EF>HG
BOOT_HG_LIGHT  = '#6B9EC8'   # bootstrap sample CI — HG>EF
BOOT_HG_DARK   = '#0D3D66'   # SNP-dropout CI     — HG>EF
BOOT_NS_LIGHT  = '#CCCCCC'
BOOT_NS_DARK   = '#999999'

Z_CRIT = _z(CI_LEVEL)

os.makedirs(os.path.dirname(output_file), exist_ok=True)

ENR_P_CANDS = ['mannwhitneyu_pval', 'mw_p_value', 'p_value', 'pvalue',
               'mannwhitneyu_pval_twosided', 'kolmogorov_smirnov_pval', 'ttest_pval']
DIR_P_CANDS = ['ttest_pval', 'pgs_pvalue', 'mannwhitneyu_pval', 'p_value', 'pvalue']


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4 · HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _resolve_col(col, cands, df, label, required=True):
    if col and col in df.columns: return col
    auto = next((c for c in cands if c in df.columns), None)
    if auto:
        print(f"[forest] '{col}' → '{auto}' for {label}", file=sys.stderr)
        return auto
    if required: raise ValueError(f"Column '{label}' not found. Tried: {[col]+cands}")
    return None

def _stars(pval):
    if   pval < 0.001: return '***'
    elif pval < 0.01:  return '**'
    elif pval < 0.05:  return '*'
    return ''

def _d_se(d, n1, n2):
    n1 = max(float(n1), 2.0)
    n2 = max(float(n2), 2.0)
    return np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2.0 * (n1 + n2 - 2.0)))

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

def _find_cat_dir(base, category):
    if not base or not os.path.isdir(base): return None
    for cand in [category, re.sub(r'[^\w\-]', '_', str(category)).strip('_')]:
        path = os.path.join(base, cand)
        if os.path.isdir(path): return path
    nc = _norm_str(category)
    for d in os.listdir(base):
        if _norm_str(d) == nc: return os.path.join(base, d)
    return None

def _load_snps(category, trait, top_n):
    cat_dir = _find_cat_dir(contrib_dir, category)
    if cat_dir is None: return None
    nt    = _norm_str(trait)
    fpath = None
    for f in glob.glob(os.path.join(cat_dir, '*_snp_contributions.txt')):
        stem = os.path.basename(f).replace('_snp_contributions.txt', '')
        if _norm_str(stem.replace('_', ' ')) == nt or _norm_str(stem) == nt:
            fpath = f; break
    if fpath is None: return None
    try:
        sdf = pd.read_csv(fpath, sep='\t', low_memory=False)
        col = next((c for c in ['pgs_contribution', 'contribution']
                    if c in sdf.columns), None)
        if col is None: return None
        sdf[col] = pd.to_numeric(sdf[col], errors='coerce')
        sdf = sdf.dropna(subset=[col]).rename(columns={col: 'pgs_contribution'})
        idx = sdf['pgs_contribution'].abs().nlargest(top_n).index
        return sdf.loc[idx].reset_index(drop=True)
    except Exception as exc:
        print(f"[forest] SNPs not loaded for '{trait}': {exc}", file=sys.stderr)
        return None

def _snp_label(snp_row):
    """gene-rsID label, mirroring plot_snp_contributions.py column names."""
    # Gene: MAPPED_GENE (GWAS catalog export); take first if comma-separated
    gene = next((str(snp_row[c]) for c in ['MAPPED_GENE', 'gene', 'nearest_gene', 'gene_name']
                 if c in snp_row.index and pd.notna(snp_row.get(c))
                 and str(snp_row.get(c)).strip() not in ('', 'nan', 'NaN')), None)
    if gene:
        gene = gene.split(',')[0].strip()

    # rsID: SNPS column (GWAS catalog), then fallbacks
    rsid = next((str(snp_row[c]) for c in ['SNPS', 'snp_id', 'rsid', 'rsID', 'variant_id']
                 if c in snp_row.index and pd.notna(snp_row.get(c))
                 and str(snp_row.get(c)).strip() not in ('', 'nan', 'NaN')), None)

    if gene and rsid:
        lbl = f'{gene}-{rsid}'
    else:
        lbl = gene or rsid or ''
    # Discard bare genomic coordinates (chr:pos or chr:pos:ref:alt)
    import re as _re
    if _re.match(r'^\d+:\d+(:\w+:\w+)?$', lbl.strip()):
        return ''
    return lbl


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5 · LOAD & SELECT DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

enr = pd.read_csv(enrichment_file, sep='\t', comment='#')
enr_cat = next((c for c in ['category', 'Category', 'trait_category']
                if c in enr.columns), None)
if enr_cat is None: raise ValueError(f"No category column in {enrichment_file}")
enr_p = next((c for c in ENR_P_CANDS if c in enr.columns), None)
if enr_p is None: raise ValueError(f"No p-value column in {enrichment_file}")

top_cats_raw = (enr.dropna(subset=[enr_p])
                   .sort_values(enr_p)
                   .head(N_CATS)[enr_cat]
                   .tolist())
print(f"[forest] Enrichment top-{N_CATS}: {top_cats_raw}", file=sys.stderr)

df = pd.read_csv(directionality_file, sep='\t', comment='#')
print(f"[forest] {len(df)} trait rows. Cols: {list(df.columns)}", file=sys.stderr)

COL_P = _resolve_col(COL_P_CFG, DIR_P_CANDS, df, 'directionality p-value')

if COL_N and COL_N in df.columns and MIN_SNPS_EXTR > 0:
    before = len(df)
    df = df[pd.to_numeric(df[COL_N], errors='coerce').fillna(0) >= MIN_SNPS_EXTR]
    print(f"[forest] min_snps_extracted ({COL_N}>={MIN_SNPS_EXTR}): {before}→{len(df)}",
          file=sys.stderr)
elif COL_N and COL_N not in df.columns:
    print(f"[forest] Warning: col_n_extracted '{COL_N}' not found — no SNP filter",
          file=sys.stderr)

if EXCLUDE:
    df = df[~df[COL_TRAIT].isin(EXCLUDE)]

sens_df = None
if SENS_FILE and os.path.isfile(SENS_FILE):
    try:
        sens_df = pd.read_csv(SENS_FILE, sep='\t', comment='#')
        print(f'[forest] Sensitivity: {len(sens_df)} rows', file=sys.stderr)
    except Exception as e:
        print(f'[forest] Warning: {e}', file=sys.stderr)
elif SENS_FILE:
    print(f'[forest] Warning: sensitivity file not found: {SENS_FILE}', file=sys.stderr)

# Pre-index for O(1) lookup in loops
sens_lookup = {}
if sens_df is not None:
    for _, _sr in sens_df.iterrows():
        sens_lookup[(str(_sr.get(COL_TRAIT,'')), str(_sr.get(COL_CAT,'')))] = _sr

boot_df = None
if BOOT_FILE and os.path.isfile(BOOT_FILE):
    try:
        boot_df = pd.read_csv(BOOT_FILE, sep='\t', comment='#')
        print(f'[forest] Bootstrap CI: {len(boot_df)} rows', file=sys.stderr)
    except Exception as e:
        print(f'[forest] Warning: bootstrap load failed: {e}', file=sys.stderr)
elif BOOT_FILE:
    print(f'[forest] Warning: bootstrap file not found: {BOOT_FILE}', file=sys.stderr)

boot_lookup = {}
if boot_df is not None:
    for _, _br in boot_df.iterrows():
        boot_lookup[(str(_br.get(COL_TRAIT,'')), str(_br.get(COL_CAT,'')))] = _br

loo_lookup = {}
if LOO_FILE and os.path.isfile(LOO_FILE):
    try:
        loo_df = pd.read_csv(LOO_FILE, sep='\t', comment='#')
        print(f'[forest] LOO file: {len(loo_df)} rows', file=sys.stderr)
        for _, _lr in loo_df.iterrows():
            _lk = (str(_lr.get(COL_TRAIT,'')), str(_lr.get(COL_CAT,'')))
            loo_lookup.setdefault(_lk, []).append(_lr)
    except Exception as e:
        print(f'[forest] Warning: LOO load failed: {e}', file=sys.stderr)
elif LOO_FILE:
    print(f'[forest] Warning: LOO file not found: {LOO_FILE}', file=sys.stderr)

actual_cats    = df[COL_CAT].unique()
norm_to_actual = {_norm_str(c): c for c in actual_cats}

def _resolve_cat(name):
    if name in actual_cats: return name
    return norm_to_actual.get(_norm_str(name))

top_cats = [r for raw in top_cats_raw
            for r in [_resolve_cat(raw)] if r is not None]
if not top_cats:
    raise ValueError("[forest] No valid categories after normalisation.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6 · FIRST PASS — collect panel data + compute global x range
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

panels     = []   # list of (cat_name, cat_df, ci_data)
global_x   = []

for cat in top_cats[:6]:
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
        print(f"[forest] No data for '{cat}' — panel hidden", file=sys.stderr)
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

    sens_ci = []
    if sens_df is not None:
        for _, row in cat_df.iterrows():
            sr = sens_lookup.get((str(row[COL_TRAIT]), cat))
            if sr is None: sens_ci.append(None)
            else:
                try:
                    sd  = float(sr[COL_D])
                    sn1 = _safe_float(sr, COL_NHG)
                    sn2 = _safe_float(sr, COL_NEF)
                    sse = _d_se(sd, sn1, sn2)
                    sci = Z_CRIT * sse
                    sp  = float(sr.get(COL_P, 1.0))
                    sens_ci.append((sd, sd - sci, sd + sci, sp))
                except: sens_ci.append(None)
    panels.append((cat, cat_df, ci_data, sens_ci))

if not global_x:
    raise ValueError("[forest] No CI data found across any panel.")

g_lo, g_hi = min(global_x), max(global_x)
g_span     = max(g_hi - g_lo, 0.10)
g_pad      = g_span * 0.18 + 0.04
X_LO, X_HI = g_lo - g_pad, g_hi + g_pad
X_HALF     = (X_HI - X_LO) / 2.0   # used for SNP scaling


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7 · DRAW ONE PANEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_panel(ax, cat_name, cat_df, ci_data, sens_ci, show_xlabels):
    n         = len(cat_df)
    cat_color = _cat_color(cat_name)

    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(-0.75, N_TRAITS - 0.25)

    for i, ((_, row), (d, ci)) in enumerate(zip(cat_df.iterrows(), ci_data)):
        pval      = float(row[COL_P])
        sig       = pval < SIG_THRESH
        bar_color = (COLOR_EF if d > 0 else COLOR_HG) if sig else COLOR_NS

        # ── Alternating row shading ────────────────────────────────────
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f4f4f4', linewidth=0, zorder=0)

        # ── Inside trait label (top of row) ───────────────────────────
        nblk  = _safe_int(row, COL_N)
        stars = _stars(pval)
        name  = str(row[COL_TRAIT]).replace('_', ' ')
        name  = (name[:MAX_LABEL - 1] + '…') if len(name) > MAX_LABEL else name
        n_str = f' (n={nblk})' if nblk is not None else ''
        label = f'{name}{n_str}  {stars}'.rstrip() if stars else f'{name}{n_str}'
        ax.text(X_LO + (X_HI - X_LO) * 0.018, i + 0.30,
                label, ha='left', va='bottom',
                fontsize=6.5, color='#363636', alpha=0.88,
                zorder=5, clip_on=True)

        # ── CI bar ─────────────────────────────────────────────────────
        # Main CI bar + dot at y=i (unchanged)
        ax.plot([d - ci, d + ci], [i, i],
                color=bar_color, lw=2.2,
                solid_capstyle='round', alpha=0.88, zorder=3)
        ax.scatter([d], [i], marker='o', s=72,
                   color=bar_color, edgecolors='white', linewidths=1.1, zorder=4)

    # ── Zero reference ────────────────────────────────────────────────────────
    ax.axvline(0, color='#c8c8c8', lw=0.9, linestyle='--', zorder=1)

    ax.set_yticks([])

    # ── Panel title ───────────────────────────────────────────────────────────
    title_str = str(cat_name).replace('_', ' ')
    if len(title_str) > 52: title_str = title_str[:51] + '…'
    ax.set_title(title_str, color='#1a1a1a', fontsize=7.5,
                 fontweight='normal', pad=4, loc='left')

    # ── x-axis: ticks on all panels, labels only on bottom row ───────────────
    ax.tick_params(axis='x', labelsize=7.5 if show_xlabels else 0,
                   colors=MUTED, length=3, width=0.7)
    if show_xlabels:
        ax.set_xlabel("Cohen's d", fontsize=8.5, color=MUTED, labelpad=4)

    # ── Spines ───────────────────────────────────────────────────────────────
    # Full box frame
    for _sp in ax.spines.values():
        _sp.set_visible(True)
        _sp.set_color(SPINE_CLR)
        _sp.set_linewidth(0.8)

    ax.grid(axis='x', linestyle=':', linewidth=0.4, color=GRID_CLR, zorder=0)
    ax.set_facecolor(BG)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8 · BUILD FIGURE  (second pass)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

N_ROWS, N_COLS = 3, 2

# Sort panels: explicit order from config if given, else by trait count desc
PANEL_ORDER = list(getattr(p, 'panel_order', []) or [])
if PANEL_ORDER:
    _order_idx = {_norm_str(n): i for i, n in enumerate(PANEL_ORDER)}
    for _pname, _pdf, *_ in panels:
        if _norm_str(_pname) not in _order_idx:
            _msg = '[' + 'forest' + '] WARNING: category not in panel_order (check spelling): ' + str(_pname)
            print(_msg, file=sys.stderr)
    panels = sorted(panels,
                    key=lambda x: _order_idx.get(_norm_str(x[0]), 999))
else:
    panels = sorted(panels, key=lambda x: (-len(x[1]), x[0]))

# Height ratio per row = max trait count in that row
row_max_n = []
for _r in range(N_ROWS):
    _rp = panels[_r*N_COLS:(_r+1)*N_COLS]
    row_max_n.append(max((len(p[1]) for p in _rp if not p[1].empty), default=1))

# Shrink figure height proportionally to actual trait rows
_fig_h = FIG_H * sum(row_max_n) / (N_ROWS * N_TRAITS)

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FIG_W, _fig_h),
                          gridspec_kw={'height_ratios': row_max_n},
                          facecolor=BG)
fig.patch.set_facecolor(BG)

for idx, (cat, cat_df, ci_data, sens_ci) in enumerate(panels[:N_ROWS * N_COLS]):
    row_idx = idx // N_COLS
    col_idx = idx % N_COLS
    ax      = axes[row_idx, col_idx]

    if cat_df.empty:
        ax.set_visible(False)
        continue

    draw_panel(ax, cat, cat_df, ci_data, sens_ci,
               show_xlabels=(row_idx == N_ROWS - 1))
    ax.set_ylim(-0.75, row_max_n[row_idx] - 0.25)

for idx in range(len(panels), N_ROWS * N_COLS):
    axes.flatten()[idx].set_visible(False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9 · SHARED LEGEND
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_mk = dict(marker='o', linestyle='none', markersize=7,
           markeredgecolor='white', markeredgewidth=0.8)
legend_handles = [
    Line2D([0], [0], markerfacecolor=COLOR_HG, label='HG > EF  (d < 0)', **_mk),
    Line2D([0], [0], markerfacecolor=COLOR_EF, label='EF > HG  (d > 0)', **_mk),
    Line2D([0], [0], markerfacecolor=COLOR_NS, label=f'p ≥ {SIG_THRESH}  (ns)', **_mk),
    Line2D([0], [0], color='none', label=''),
    Line2D([0], [0], color='none',
           label='* p < 0.05   ** p < 0.01   *** p < 0.001'),
]
if sens_df is not None:
    _mk_s = dict(marker='D', linestyle='none', markerfacecolor='white', markeredgewidth=1.4)
    legend_handles += [
        Line2D([0],[0], color='none', label=''),
        Line2D([0],[0], markeredgecolor=COLOR_HG,
               label='HG > EF (sensitivity, excl. top SNPs)', **_mk_s),
        Line2D([0],[0], markeredgecolor=COLOR_EF,
               label='EF > HG (sensitivity, excl. top SNPs)', **_mk_s),
    ]

fig.legend(handles=legend_handles, loc='lower center', ncol=5,
           fontsize=8.5, framealpha=0.0, labelcolor=MUTED,
           bbox_to_anchor=(0.5, 0.0),
           handletextpad=0.4, columnspacing=1.0)

fig.tight_layout(rect=[0, 0.06, 1, 1.0], h_pad=2.0, w_pad=2.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10 · SAVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

for ext in ('.pdf', '.svg'):
    path = output_file if ext == '.pdf' else output_file.replace('.pdf', '.svg')
    fig.savefig(path, bbox_inches='tight', dpi=180, facecolor=BG)
    print(f"[forest] Saved: {path}", file=sys.stderr)

plt.close(fig)

import time as _t; _t0=_t.time(); print(f'[forest] Main figure done in {_t.time()-_t0:.1f}s', file=sys.stderr); _t0=_t.time()
# =========================================================================
# SENSITIVITY FIGURE — same layout, paired dots per row
# =========================================================================
if sens_df is not None and output_file_sens:

    def draw_panel_sens(ax, cat_name, cat_df, ci_data, sens_ci, show_xlabels):
        n         = len(cat_df)
        cat_color = _cat_color(cat_name)
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
            ax.text(X_LO + (X_HI - X_LO) * 0.018, i + 0.38,
                    label, ha='left', va='bottom',
                    fontsize=6.5, color='#363636', alpha=0.88,
                    zorder=5, clip_on=True)
            # Main: shifted up
            y_main = i + Y_OFFSET
            ax.plot([d - ci, d + ci], [y_main, y_main],
                    color=bar_color, lw=2.2,
                    solid_capstyle='round', alpha=0.88, zorder=3)
            ax.scatter([d], [y_main], marker='o', s=72,
                       color=bar_color, edgecolors='white', linewidths=1.1, zorder=4)
            # Sensitivity: shifted down
            if sens_ci and i < len(sens_ci) and sens_ci[i] is not None:
                sd, slo, shi, sp = sens_ci[i]
                s_sig   = sp < SIG_THRESH
                s_color = (COLOR_EF if sd > 0 else COLOR_HG) if s_sig else COLOR_NS
                y_sens  = i - Y_OFFSET
                ax.plot([slo, shi], [y_sens, y_sens],
                        color=s_color, lw=1.6, solid_capstyle='round', alpha=0.62, zorder=3)
                ax.scatter([sd], [y_sens], marker='D', s=40,
                           color='white', edgecolors=s_color, linewidths=1.4, zorder=4)
        ax.axvline(0, color='#c8c8c8', lw=0.9, linestyle='--', zorder=1)
        ax.set_yticks([])
        title_str = str(cat_name).replace('_', ' ')
        if len(title_str) > 52: title_str = title_str[:51] + '...'
        ax.set_title(title_str, color='#1a1a1a', fontsize=7.5,
                     fontweight='normal', pad=4, loc='left')
        ax.tick_params(axis='x', labelsize=7.5 if show_xlabels else 0,
                       colors=MUTED, length=3, width=0.7)
        if show_xlabels:
            ax.set_xlabel("Cohen's d", fontsize=8.5, color=MUTED, labelpad=4)
        for _sp in ax.spines.values():
            _sp.set_visible(True)
            _sp.set_color(SPINE_CLR)
            _sp.set_linewidth(0.8)
        ax.grid(axis='x', linestyle=':', linewidth=0.4, color=GRID_CLR, zorder=0)
        ax.set_facecolor(BG)

    fig2, axes2 = plt.subplots(N_ROWS, N_COLS, figsize=(FIG_W, _fig_h),
                                gridspec_kw={'height_ratios': row_max_n},
                                facecolor=BG)
    fig2.patch.set_facecolor(BG)
    for idx2, (cat, cat_df, ci_data, sens_ci) in enumerate(panels[:N_ROWS * N_COLS]):
        row_idx2 = idx2 // N_COLS
        col_idx2 = idx2 % N_COLS
        ax2 = axes2[row_idx2, col_idx2]
        if cat_df.empty:
            ax2.set_visible(False); continue
        draw_panel_sens(ax2, cat, cat_df, ci_data, sens_ci,
                        show_xlabels=(row_idx2 == N_ROWS - 1))
        ax2.set_ylim(-0.75, row_max_n[row_idx2] - 0.25)
    for idx2 in range(len(panels), N_ROWS * N_COLS):
        axes2.flatten()[idx2].set_visible(False)

    _mk_s = dict(marker='D', linestyle='none', markerfacecolor='white',
                 markeredgewidth=1.4)
    leg2_handles = legend_handles + [
        Line2D([0],[0], color='none', label=''),
        Line2D([0],[0], markeredgecolor=COLOR_HG,
               label='HG > EF (sensitivity, excl. top SNPs)', **_mk_s),
        Line2D([0],[0], markeredgecolor=COLOR_EF,
               label='EF > HG (sensitivity, excl. top SNPs)', **_mk_s),
    ]
    fig2.legend(handles=leg2_handles, loc='lower center',
                ncol=4, fontsize=8.5, framealpha=0.0, labelcolor=MUTED,
                bbox_to_anchor=(0.5, 0.0),
                handletextpad=0.4, columnspacing=1.0)
    fig2.tight_layout(rect=[0, 0.06, 1, 1.0], h_pad=2.0, w_pad=2.0)
    for ext in ('.pdf', '.svg'):
        path2 = (output_file_sens if ext == '.pdf'
                 else output_file_sens.replace('.pdf', '.svg'))
        fig2.savefig(path2, bbox_inches='tight', dpi=180, facecolor=BG)
        print(f'[forest] Saved sensitivity: {path2}', file=sys.stderr)
    plt.close(fig2)

print(f'[forest] Sensitivity figure done in {_t.time()-_t0:.1f}s', file=sys.stderr); _t0=_t.time()
# =========================================================================
# BOOTSTRAP CI FIGURE — CI ranges only, no central dot
# =========================================================================
if boot_df is not None and output_file_boot:

    def _boot_row(ax, cat_df, i, row, serif=0.10):
        """Draw bootstrap + SNP-dropout CI ranges for one trait row."""
        d    = float(row[COL_D])
        pval = float(row[COL_P]) if COL_P in row.index else 1.0
        sig  = pval < SIG_THRESH
        c_lt = (BOOT_EF_LIGHT if d > 0 else BOOT_HG_LIGHT) if sig else BOOT_NS_LIGHT
        c_dk = (BOOT_EF_DARK  if d > 0 else BOOT_HG_DARK)  if sig else BOOT_NS_DARK

        br = boot_lookup.get((str(row[COL_TRAIT]), str(row[COL_CAT])))
        if br is None: return

        def _ci_line(ax, lo, hi, y, color, ls, lw=1.8):
            ax.plot([lo, hi], [y, y], color=color, lw=lw,
                    linestyle=ls, solid_capstyle='butt', alpha=0.85, zorder=3)
            for xe in [lo, hi]:  # serif end caps
                ax.plot([xe, xe], [y-serif, y+serif],
                        color=color, lw=lw, alpha=0.85, zorder=3)

        try:
            blo = float(br['bootstrap_ci_lo'])
            bhi = float(br['bootstrap_ci_hi'])
            _ci_line(ax, blo, bhi, i + BOOT_OFFSET, c_lt, '-')
        except (KeyError, ValueError): pass

        # Leave-one-out dots + most-impactful label
        _lk   = (str(row[COL_TRAIT]), str(row[COL_CAT]))
        _loos = loo_lookup.get(_lk, [])
        if _loos:
            _ys = i - BOOT_OFFSET
            _d0 = float(row[COL_D]) if COL_D in row.index else 0.0
            _xs = []
            for _lr in _loos:
                try: _xs.append(float(_lr['cohens_d_loo']))
                except: pass
            if _xs:
                ax.scatter(_xs, [_ys]*len(_xs), s=5, color=c_lt,
                           alpha=0.20, zorder=2, linewidths=0)
                # Most impactful: most opposite to d_original
                _key_fn = (lambda r: float(r.get('cohens_d_loo', _d0)))
                _extreme = min(_loos, key=_key_fn) if _d0 >= 0 \
                           else max(_loos, key=_key_fn)
                _ex_x = float(_extreme.get('cohens_d_loo', _d0))
                _gene = str(_extreme.get('MAPPED_GENE','')).split(',')[0].strip()
                _rsid = str(_extreme.get('SNPS','')).strip()
                _lbl  = f'{_gene}-{_rsid}' if (_gene and _rsid) \
                        else (_gene or _rsid or '')
                ax.scatter([_ex_x], [_ys], s=20, color=c_dk,
                           alpha=0.90, zorder=4, linewidths=0)
                if _lbl:
                    ax.text(_ex_x, _ys - 0.09, _lbl,
                            ha='center', va='top', fontsize=4.5,
                            color=c_dk, alpha=0.88, zorder=5, clip_on=True)

    # Compute global x range including bootstrap CIs
    _bx = []
    for _cat, _cdf, _cid, _sc in panels[:N_ROWS * N_COLS]:
        for _, _row in _cdf.iterrows():
            _br = boot_lookup.get((str(_row[COL_TRAIT]), _cat))
            if _br is not None:
                for _col in ['bootstrap_ci_lo','bootstrap_ci_hi',
                             'snp_dropout_ci_lo','snp_dropout_ci_hi']:
                    try: _bx.append(float(_br[_col]))
                    except: pass
    for _loos in loo_lookup.values():
        for _lr in _loos:
            try: _bx.append(float(_lr['cohens_d_loo']))
            except: pass
    _bx += global_x  # also include main CI data
    if _bx:
        _blo2, _bhi2 = min(_bx), max(_bx)
        _bspan = max(_bhi2 - _blo2, 0.10)
        _bpad  = _bspan * 0.18 + 0.04
        BX_LO, BX_HI = _blo2 - _bpad, _bhi2 + _bpad
    else:
        BX_LO, BX_HI = X_LO, X_HI

    fig3, axes3 = plt.subplots(N_ROWS, N_COLS, figsize=(FIG_W, _fig_h),
                                gridspec_kw={'height_ratios': row_max_n},
                                facecolor=BG)
    fig3.patch.set_facecolor(BG)

    for idx3, (cat, cat_df, ci_data, sens_ci) in enumerate(panels[:N_ROWS * N_COLS]):
        row_idx3 = idx3 // N_COLS
        col_idx3 = idx3 % N_COLS
        ax3 = axes3[row_idx3, col_idx3]
        if cat_df.empty: ax3.set_visible(False); continue

        n3        = len(cat_df)
        cat_color = _cat_color(cat)
        ax3.set_xlim(BX_LO, BX_HI)
        ax3.set_ylim(-0.75, row_max_n[row_idx3] - 0.25)

        for i3, (_, row) in enumerate(cat_df.iterrows()):
            if i3 % 2 == 0:
                ax3.axhspan(i3-0.5, i3+0.5, color='#f4f4f4', linewidth=0, zorder=0)
            # Inside label
            nblk  = _safe_int(row, COL_N)
            stars = _stars(float(row[COL_P]))
            name  = str(row[COL_TRAIT]).replace('_', ' ')
            name  = (name[:MAX_LABEL-1]+'...') if len(name)>MAX_LABEL else name
            n_str = f' (n={nblk})' if nblk is not None else ''
            lbl   = f'{name}{n_str}  {stars}'.rstrip() if stars else f'{name}{n_str}'
            ax3.text(BX_LO+(BX_HI-BX_LO)*0.018, i3+0.38,
                     lbl, ha='left', va='bottom',
                     fontsize=6.5, color='#363636', alpha=0.88,
                     zorder=5, clip_on=True)
            _boot_row(ax3, cat_df, i3, row)

        ax3.set_ylim(-0.75, row_max_n[row_idx3] - 0.25)
        ax3.axvline(0, color='#c8c8c8', lw=0.9, linestyle='--', zorder=1)
        ax3.set_yticks([])
        title3 = str(cat).replace('_',' ')
        if len(title3) > 52: title3 = title3[:51]+'...'
        ax3.set_title(title3, color='#1a1a1a', fontsize=7.5,
                      fontweight='normal', pad=4, loc='left')
        ax3.tick_params(axis='x', labelsize=7.5 if row_idx3==N_ROWS-1 else 0,
                        colors=MUTED, length=3, width=0.7)
        if row_idx3 == N_ROWS - 1:
            ax3.set_xlabel("Cohen's d", fontsize=8.5, color=MUTED, labelpad=4)
        for _sp3 in ax3.spines.values():
            _sp3.set_visible(True)
            _sp3.set_color(SPINE_CLR)
            _sp3.set_linewidth(0.8)
        ax3.grid(axis='x', linestyle=':', linewidth=0.4, color=GRID_CLR, zorder=0)
        ax3.set_facecolor(BG)

    for idx3 in range(len(panels), N_ROWS * N_COLS):
        axes3.flatten()[idx3].set_visible(False)

    _boot_leg = [
        Line2D([0],[0], color=BOOT_EF_LIGHT, lw=2.0, ls='-',
               label='Bootstrap CI — EF>HG (sample resampling, 1000x)'),
        Line2D([0],[0], color=BOOT_HG_LIGHT, lw=2.0, ls='-',
               label='Bootstrap CI — HG>EF (sample resampling, 1000x)'),
        Line2D([0],[0], color=BOOT_EF_DARK,  lw=1.8, ls='--',
               label='SNP-dropout CI — EF>HG (10% removed, 1000x)'),
        Line2D([0],[0], color=BOOT_HG_DARK,  lw=1.8, ls='--',
               label='ns (p >= 0.05)'),
    ]
    fig3.legend(handles=_boot_leg, loc='lower center', ncol=2,
                fontsize=8.0, framealpha=0.0, labelcolor=MUTED,
                bbox_to_anchor=(0.5, 0.0),
                handletextpad=0.6, columnspacing=1.2)
    fig3.tight_layout(rect=[0, 0.08, 1, 1.0], h_pad=2.0, w_pad=2.0)
    for ext in ('.pdf', '.svg'):
        path3 = (output_file_boot if ext == '.pdf'
                 else output_file_boot.replace('.pdf', '.svg'))
        fig3.savefig(path3, bbox_inches='tight', dpi=180, facecolor=BG)
        print(f'[forest] Saved bootstrap: {path3}', file=sys.stderr)
    plt.close(fig3)