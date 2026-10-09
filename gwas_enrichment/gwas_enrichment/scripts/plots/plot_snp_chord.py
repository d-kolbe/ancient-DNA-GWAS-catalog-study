"""
plot_snp_chord.py  --  LD-block Contribution Chord Diagram
==========================================================
2*3 panel figure matching the forest panel category selection.

Per panel (category):
  - Outer ring: top N traits as coloured arc-wedges (red EF>HG, blue HG>EF, grey ns).
  - One chord per trait pair (max C(N,2) = 10 for N=5).
  Chord weight = sum of mean(|contrib_i|, |contrib_j|) over shared LD blocks.
  Chord thickness proportional to weight (globally normalised).
  Chord colour = net direction of shared blocks (red EF>HG, blue HG>EF).
  "no shared LD blocks" message when no pairs overlap.

LD blocks matched by the block_id column in the SNP contribution files.
Light theme only.
"""

import sys, os, re, glob
import zlib
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D


# --------------------------------------------------------------------------
# 1  SEMANTIC COLOUR MAP
# --------------------------------------------------------------------------

_SEMANTIC = {
    'neuropsychiatric': '#7B68C8', 'neurological': '#7B68C8',
    'psychiatric':      '#9480D4', 'cognitive':    '#8B78D0',
    'brain':            '#7B68C8', 'sleep':        '#9B88D8',
    'cardiovascular':   '#C44444', 'cardiac':      '#C44444',
    'heart':            '#C44444', 'hematological':'#A02828',
    'haematological':   '#A02828', 'blood':        '#A02828',
    'lipids':           '#C8A820', 'metabolic':    '#C8A820',
    'diabetes':         '#E08820', 'thyroid':      '#B89820',
    'gastrointestinal': '#DD7030', 'gut':          '#DD7030',
    'immune':           '#D06020', 'autoimmune':   '#D06020',
    'inflammatory':     '#C05A18', 'allergy':      '#D07828',
    'anthropometry':    '#4C72B0', 'body':         '#4C72B0',
    'behavioral':       '#8172B2', 'behaviour':    '#8172B2',
    'musculoskeletal':  '#C05858', 'bone':         '#C05858',
    'sensory':          '#2A9EA8', 'vision':       '#2A9EA8',
    'hearing':          '#2A9EA8', 'ocular':       '#3AAEBC',
    'lifestyle':        '#55A868', 'diet':         '#55A868',
    'aging':            '#888888', 'ageing':       '#888888',
    'reproductive':     '#C87AAC', 'fertility':    '#C87AAC',
    'cancer':           '#505050', 'respiratory':  '#5898C8',
    'renal':            '#5060A8', 'dental':       '#60B4A8',
    'skin':             '#B89050',
}
_FALLBACK = ['#4C72B0', '#DD8452', '#55A868', '#8172B2', '#937860', '#CCB974']

def _cat_color(cat_name):
    name  = str(cat_name).replace('_', ' ')
    group = name.split(' - ')[0].strip() if ' - ' in name else name.split()[0]
    key   = group.lower().strip()
    return (_SEMANTIC.get(key) or _SEMANTIC.get(key.split()[0])
            or _FALLBACK[zlib.crc32(key.encode('utf-8')) % len(_FALLBACK)])


# --------------------------------------------------------------------------
# 2  PALETTE
# --------------------------------------------------------------------------

COLOR_EF   = '#C94F3A'
COLOR_HG   = '#1E5E99'
COLOR_NS   = '#BBBBBB'
CHORD_CLR  = '#888888'   # single neutral chord colour

# Distinct colours for same-block SNP grouping (colorblind-friendly)
# Block-match palette: original 8 + 4 extensions, no blue/coral.
_BLOCK_PAL = [
    '#9B2226', '#6A0572', '#CC79A7', '#8B4513',
    '#4A4E69', '#009E73', '#F4D03F', '#5D4037',
    # Extensions:
    '#76448A', '#D4AC0D', '#7B241C', '#1D6A5A',
]

BG    = 'white'
TEXT  = '#1a1a1a'
MUTED = '#999999'


# --------------------------------------------------------------------------
# 3  SNAKEMAKE I/O & PARAMS
# --------------------------------------------------------------------------

enrichment_file     = snakemake.input['enrichment']
directionality_file = snakemake.input['directionality']
contrib_dir         = str(snakemake.input.get('snp_contributions_dir', ''))
output_file         = snakemake.output['plot']

p = snakemake.params

COL_CAT   = str(p.col_category)
COL_TRAIT = str(p.col_trait)
COL_D     = str(p.col_cohens_d)
COL_P_CFG = str(p.col_pvalue)
COL_N     = str(getattr(p, 'col_n_extracted', 'n_snps_extracted'))

SIG_THRESH    = float(p.sig_threshold)
MIN_SNPS_EXTR = int(getattr(p, 'min_snps_extracted', 10))
EXCLUDE       = list(getattr(p, 'exclude_traits', []) or [])
N_CATS        = int(getattr(p, 'n_panel_categories', 6))
N_TRAITS      = int(getattr(p, 'n_panel_traits',     5))

cp = snakemake.config.get('snp_chord_plot', {})
TOP_SNPS_PER_TRAIT  = int(cp.get('top_snps_per_trait',   500))
MIN_SHARED_BLOCKS   = int(cp.get('min_shared_blocks',      1))
MAX_LABEL_CHARS     = int(cp.get('max_label_chars',        18))
N_TICKS     = int(cp.get('n_snp_ticks',      3))   # top SNPs per arc
MAX_BAR_LEN = float(cp.get('max_bar_length', 0.28)) # max radial bar len
MIN_BAR_LEN         = float(cp.get('min_bar_length',      0.04))
TITLE_FS            = float(cp.get('panel_title_fontsize',  7.0))
TITLE_X             = float(cp.get('panel_title_x',        -1.46))  # data coords
TITLE_Y             = float(cp.get('panel_title_y',         1.60))  # data coords; can exceed ylim since clip_on=False
HSPACE              = float(cp.get('panel_hspace',          0.07))  # vertical gap
WSPACE              = float(cp.get('panel_wspace',          0.05))  # horizontal gap
TRAIT_LBL_FS        = float(cp.get('trait_label_fontsize',  5.0))
TRAIT_WRAP_CHARS    = int(cp.get('trait_label_wrap_chars',  14))  # wrap to new line after this many chars
SNP_LBL_FS          = float(cp.get('snp_label_fontsize',   4.2))
FIG_W               = float(cp.get('figure_width',        13.0))
FIG_H               = float(cp.get('figure_height',       15.0))

os.makedirs(os.path.dirname(output_file), exist_ok=True)

ENR_P_CANDS = ['mannwhitneyu_pval', 'mw_p_value', 'p_value', 'pvalue',
               'mannwhitneyu_pval_twosided', 'ttest_pval']
DIR_P_CANDS = ['ttest_pval', 'pgs_pvalue', 'mannwhitneyu_pval', 'p_value', 'pvalue']


# --------------------------------------------------------------------------
# 4  HELPERS
# --------------------------------------------------------------------------

def _norm_str(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _find_cat_dir(base, category):
    if not base or not os.path.isdir(base): return None
    for cand in [category, re.sub(r'[^\w\-]', '_', str(category)).strip('_')]:
        if os.path.isdir(os.path.join(base, cand)):
            return os.path.join(base, cand)
    nc = _norm_str(category)
    for d in os.listdir(base):
        if _norm_str(d) == nc: return os.path.join(base, d)
    return None

def load_block_contribs(category, trait):
    """
    Load SNP contribution file and aggregate by block_id.
    Returns dict {block_id: summed_pgs_contribution} or {} on failure.
    """
    cat_dir = _find_cat_dir(contrib_dir, category)
    if cat_dir is None: return {}
    nt    = _norm_str(trait)
    fpath = None
    for f in glob.glob(os.path.join(cat_dir, '*_snp_contributions.txt')):
        stem = os.path.basename(f).replace('_snp_contributions.txt', '')
        if _norm_str(stem.replace('_', ' ')) == nt or _norm_str(stem) == nt:
            fpath = f; break
    if fpath is None: return {}
    try:
        sdf = pd.read_csv(fpath, sep='\t', low_memory=False)
        contrib_col = next((c for c in ['pgs_contribution', 'contribution']
                            if c in sdf.columns), None)
        if contrib_col is None or 'block_id' not in sdf.columns:
            if 'block_id' not in sdf.columns:
                print(f"[chord] No block_id column in {fpath}", file=sys.stderr)
            return {}
        sdf[contrib_col] = pd.to_numeric(sdf[contrib_col], errors='coerce')
        sdf = sdf.dropna(subset=[contrib_col])
        return sdf.groupby('block_id')[contrib_col].sum().to_dict()
    except Exception as e:
        print(f"[chord] Error loading '{trait}': {e}", file=sys.stderr)
        return {}


def pairwise_shared(blocks_i, blocks_j):
    """
    Shared genetic architecture as a proportion.

    proportion = sum(mean(|c_i|, |c_j|)) across shared blocks
                 divided by
                 0.5 * (sum(|c_i|) all blocks of i  +  sum(|c_j|) all blocks of j)

    = fraction of combined genetic architecture that is shared.
    Close to 1: same loci driving both traits. Close to 0: genetically distinct.
    """
    shared = set(blocks_i) & set(blocks_j)
    if not shared:
        return 0.0, 0, 0.0
    shared_contrib = sum((abs(blocks_i[b]) + abs(blocks_j[b])) / 2 for b in shared)
    net_dir        = sum(blocks_i[b] + blocks_j[b] for b in shared)
    total_i  = sum(abs(v) for v in blocks_i.values())
    total_j  = sum(abs(v) for v in blocks_j.values())
    combined = 0.5 * (total_i + total_j)
    proportion = shared_contrib / combined if combined > 0 else 0.0
    return proportion, len(shared), net_dir


def load_top_snps(category, trait, top_n):
    "Load top N SNPs as (gene-rsID label, contribution) pairs."
    cat_dir = _find_cat_dir(contrib_dir, category)
    if cat_dir is None: return []
    nt = _norm_str(trait)
    fpath = None
    for f in glob.glob(os.path.join(cat_dir, '*_snp_contributions.txt')):
        stem = os.path.basename(f).replace('_snp_contributions.txt', '')
        if _norm_str(stem.replace('_', ' ')) == nt or _norm_str(stem) == nt:
            fpath = f; break
    if fpath is None: return []
    try:
        sdf = pd.read_csv(fpath, sep='\t', low_memory=False)
        cc = next((x for x in ['pgs_contribution','contribution']
                   if x in sdf.columns), None)
        if cc is None: return []
        sdf[cc] = pd.to_numeric(sdf[cc], errors='coerce')
        sdf = sdf.dropna(subset=[cc])
        sdf = sdf.loc[sdf[cc].abs().nlargest(top_n * 5).index].copy()
        results = []
        for _, row in sdf.iterrows():
            gene = next((str(row[x]) for x in
                ['MAPPED_GENE','gene','nearest_gene']
                if x in row.index and pd.notna(row.get(x))
                and str(row.get(x)).strip() not in ('','nan','NaN')), None)
            if gene: gene = gene.split(',')[0].strip()
            rsid = next((str(row[x]) for x in
                ['SNPS','snp_id','rsid','rsID']
                if x in row.index and pd.notna(row.get(x))
                and str(row.get(x)).strip() not in ('','nan','NaN')), None)
            lbl = (f'{gene}-{rsid}' if (gene and rsid)
                   else (gene or rsid or ''))
            if not lbl: continue
            import re as _re2
            if _re2.match(r'^\d+:\d+(:\w+:\w+)?$', lbl.strip()): continue
            bid = str(row['block_id']) if 'block_id' in row.index and pd.notna(row.get('block_id')) else ''
            results.append((lbl, float(row[cc]), bid))
            if len(results) >= top_n: break
        return results
    except Exception as ex:
        print(f'[chord] SNP labels error for {trait}: {ex}', file=sys.stderr)
        return []


def _short(s, n):
    s = str(s).replace('_', ' ')
    return (s[:n - 1] + '...') if len(s) > n else s


# --------------------------------------------------------------------------
# 5  LOAD & SELECT DATA
# --------------------------------------------------------------------------

enr = pd.read_csv(enrichment_file, sep='\t', comment='#')
enr_cat = next((c for c in ['category', 'Category'] if c in enr.columns), None)
enr_p   = next((c for c in ENR_P_CANDS if c in enr.columns), None)
if not enr_cat or not enr_p:
    raise ValueError("Enrichment file missing category or p-value column")

top_cats_raw = (enr.dropna(subset=[enr_p]).sort_values(enr_p)
                   .head(N_CATS)[enr_cat].tolist())

df = pd.read_csv(directionality_file, sep='\t', comment='#')
COL_P = next((c for c in [COL_P_CFG] + DIR_P_CANDS if c and c in df.columns), None)
if COL_P is None: raise ValueError("No p-value column in directionality file")

if COL_N in df.columns and MIN_SNPS_EXTR > 0:
    df = df[pd.to_numeric(df[COL_N], errors='coerce').fillna(0) >= MIN_SNPS_EXTR]
if EXCLUDE:
    df = df[~df[COL_TRAIT].isin(EXCLUDE)]

actual_cats    = df[COL_CAT].unique()
norm_to_actual = {_norm_str(c): c for c in actual_cats}

top_cats = [r for raw in top_cats_raw
            for r in [norm_to_actual.get(_norm_str(raw), raw if raw in actual_cats else None)]
            if r is not None and r in actual_cats]

print(f"[chord] Categories: {top_cats}", file=sys.stderr)


# --------------------------------------------------------------------------
# 6  FIRST PASS -- load blocks, compute pairwise chords
# --------------------------------------------------------------------------

panels      = []   # (cat, cat_df, chords)
# chord = (i, j, weight, n_shared, net_dir)
all_weights = []

for cat in top_cats[:6]:
    cat_df = (
        df[df[COL_CAT] == cat].copy()
          .assign(**{COL_P: lambda x: pd.to_numeric(x[COL_P], errors='coerce')})
          .dropna(subset=[COL_P])
          .nsmallest(N_TRAITS, COL_P)
          .sort_values(COL_D, ascending=True)
          .reset_index(drop=True)
    )
    if cat_df.empty:
        panels.append((cat, cat_df, [], {})); continue

    traits = [str(r[COL_TRAIT]) for _, r in cat_df.iterrows()]

    # Load block contributions (chords) and top SNPs (tick labels)
    trait_blocks = {t: load_block_contribs(cat, t) for t in traits}
    trait_snps   = {t: load_top_snps(cat, t, N_TICKS) for t in traits}
    n_loaded = sum(1 for v in trait_blocks.values() if v)
    print(f"[chord] {cat}: {n_loaded}/{len(traits)} traits with block data", file=sys.stderr)

    # Compute all pairwise chords
    chords = []
    for i, j in combinations(range(len(traits)), 2):
        weight, n_shared, net_dir = pairwise_shared(
            trait_blocks[traits[i]], trait_blocks[traits[j]]
        )
        if n_shared >= MIN_SHARED_BLOCKS and weight > 0:
            chords.append((i, j, weight, n_shared, net_dir))
            all_weights.append(weight)

    chords.sort(key=lambda x: x[2])   # draw weakest first
    panels.append((cat, cat_df, chords, trait_snps))
    print(f"[chord]   -> {len(chords)} chords", file=sys.stderr)



# --------------------------------------------------------------------------
# 7  DRAWING HELPERS
# --------------------------------------------------------------------------

R_OUTER  = 1.00
RING_W   = 0.13
R_INNER  = R_OUTER - RING_W
R_CHORD  = R_INNER - 0.03
CTRL     = 0.28   # bezier control-point pull towards origin


def _bezier_chord(ax, deg1, deg2, color, lw, alpha):
    t1 = np.radians(deg1);  t2 = np.radians(deg2)
    x1, y1 = R_CHORD * np.cos(t1), R_CHORD * np.sin(t1)
    x2, y2 = R_CHORD * np.cos(t2), R_CHORD * np.sin(t2)
    path = Path(
        [(x1,y1), (x1*CTRL, y1*CTRL), (x2*CTRL, y2*CTRL), (x2,y2)],
        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    )
    ax.add_patch(PathPatch(path, fc='none', ec=color, lw=lw,
                            alpha=alpha, zorder=2, capstyle='round'))



# --------------------------------------------------------------------------
# 8  DRAW ONE PANEL
# --------------------------------------------------------------------------

def draw_panel(ax, cat_name, cat_df, chords, trait_snps):
    N         = len(cat_df)
    cat_color = _cat_color(cat_name)

    if N == 0:
        ax.axis('off'); return

    GAP      = 9.0
    STEP     = 360.0 / N
    ARC_SPAN = STEP - GAP
    ARC_HALF = ARC_SPAN / 2.0
    centers  = [90.0 - i * STEP for i in range(N)]

    # -- Trait arc wedges -----------------------------------------------------
    for i, (_, row) in enumerate(cat_df.iterrows()):
        pval = float(row[COL_P])
        d    = float(row[COL_D])
        sig  = pval < SIG_THRESH
        arc_c = (COLOR_EF if d > 0 else COLOR_HG) if sig else COLOR_NS

        ax.add_patch(Wedge(
            (0,0), R_OUTER, centers[i] - ARC_HALF, centers[i] + ARC_HALF,
            width=RING_W, facecolor=arc_c, edgecolor='white',
            linewidth=0.5, alpha=0.88, zorder=3
        ))

        # Trait label: counter-clockwise half of the arc, just outside ring
        lbl_ang  = centers[i] - ARC_HALF * 0.60   # CCW side
        theta_lbl = np.radians(lbl_ang)
        ang_deg_l = lbl_ang % 360
        if 90 < ang_deg_l <= 270:
            t_rot = ang_deg_l - 180
            t_ha  = 'right'
        else:
            t_rot = ang_deg_l
            t_ha  = 'left'
        r_lbl = R_OUTER + 0.06
        _lbl_str = _short(row[COL_TRAIT], MAX_LABEL_CHARS)
        # Wrap to next line every TRAIT_WRAP_CHARS characters,
        # breaking at the nearest space before the limit.
        def _wrap(s, w):
            words = s.split()
            lines, cur = [], ''
            for wd in words:
                test = (cur + ' ' + wd).strip()
                if len(test) <= w:
                    cur = test
                else:
                    if cur: lines.append(cur)
                    cur = wd
            if cur: lines.append(cur)
            return '\n'.join(lines)
        _lbl_wrapped = _wrap(_lbl_str, TRAIT_WRAP_CHARS)
        _n_lines = _lbl_wrapped.count('\n') + 1
        # Font shrinks slightly for multi-line to keep total height in check
        _fs = max(3.5, TRAIT_LBL_FS - (_n_lines - 1) * 0.4)
        ax.text(r_lbl * np.cos(theta_lbl), r_lbl * np.sin(theta_lbl),
                _lbl_wrapped,
                ha=t_ha, va='center',
                fontsize=_fs, fontweight='bold', color=TEXT,
                rotation=t_rot, rotation_mode='anchor',
                linespacing=1.25,
                zorder=5, clip_on=False)

    # -- Radial SNP tick marks ------------------------------------------------
    # Collect all block_ids across panel and assign consistent colours
    all_bids = [bid for snps in trait_snps.values()
                for _, _, bid in snps if bid]
    # Count occurrences; blocks appearing >1 trait get a palette colour
    from collections import Counter as _Counter
    bid_counts = _Counter(all_bids)
    shared_bids = sorted(b for b, n in bid_counts.items() if n > 1)
    block_clr = {bid: _BLOCK_PAL[i % len(_BLOCK_PAL)]
                 for i, bid in enumerate(shared_bids)}

    # Per-trait normalisation: top SNP in each trait gets MAX_BAR_LEN.
    # Gives full resolution on within-trait relative contributions.
    trait_max = {}
    for t, snps in trait_snps.items():
        vals = [abs(v) for _, v, _ in snps]
        trait_max[t] = max(vals) if vals else 1.0

    for i, (_, row) in enumerate(cat_df.iterrows()):
        trait   = str(row[COL_TRAIT])
        snps    = trait_snps.get(trait, [])
        if not snps: continue
        pval    = float(row[COL_P])
        d       = float(row[COL_D])
        sig     = pval < SIG_THRESH
        bar_col = (COLOR_EF if d > 0 else COLOR_HG) if sig else COLOR_NS
        n_s     = len(snps)
        # Bars on the clockwise half: from arc centre+margin to +ARC_HALF-margin
        margin  = ARC_HALF * 0.14
        if n_s == 1:
            tick_angles = [centers[i] + ARC_HALF * 0.50]
        else:
            t_start = centers[i] + margin
            t_end   = centers[i] + ARC_HALF - margin
            tick_angles = [
                t_start + k * (t_end - t_start) / (n_s - 1)
                for k in range(n_s)
            ]
        for k, (lbl, contrib, bid) in enumerate(snps):
            theta   = np.radians(tick_angles[k])
            t_max   = trait_max.get(trait, 1.0)
            frac    = abs(contrib) / t_max if t_max > 0 else 0.0
            bar_len = frac * MAX_BAR_LEN   # starts from 0, full range used
            x1 = R_OUTER * np.cos(theta)
            y1 = R_OUTER * np.sin(theta)
            x2 = (R_OUTER + bar_len) * np.cos(theta)
            y2 = (R_OUTER + bar_len) * np.sin(theta)
            # Fill: SNP's own direction (sign of contribution)
            fill_col = COLOR_EF if contrib > 0 else COLOR_HG
            # Border: block-match palette colour; none for unmatched
            border_col = block_clr.get(bid, None) if bid else None
            if border_col:
                ax.plot([x1, x2], [y1, y2],
                        color=border_col, lw=3.8, alpha=0.90, zorder=4,
                        solid_capstyle='round')
            ax.plot([x1, x2], [y1, y2],
                    color=fill_col, lw=2.0, alpha=0.88, zorder=5,
                    solid_capstyle='round')
            lbl_s   = lbl  # full gene-rsID label, no truncation
            ang_deg = tick_angles[k] % 360
            if 90 < ang_deg <= 270:
                rot = ang_deg - 180
                ha  = 'right'
            else:
                rot = ang_deg
                ha  = 'left'
            xl = (R_OUTER + bar_len + 0.03) * np.cos(theta)
            yl = (R_OUTER + bar_len + 0.03) * np.sin(theta)
            # Label: border colour if shared block, else black
            lbl_col = border_col if border_col else '#222222'
            ax.text(xl, yl, lbl_s,
                    ha=ha, va='center', fontsize=SNP_LBL_FS,
                    color=lbl_col,
                    rotation=rot, rotation_mode='anchor',
                    alpha=0.90, zorder=6, clip_on=False)

    # -- Chords ---------------------------------------------------------------
    for i, j, weight, n_shared, net_dir in chords:
        frac  = min(float(weight), 1.0)  # already a proportion [0,~1]
        lw    = 0.5 + frac * 5.0           # [0.5, 5.5]
        alpha = 0.22 + min(frac, 1) * 0.58 # [0.22, 0.80]

        chord_c = CHORD_CLR

        _bezier_chord(ax, centers[i], centers[j], chord_c, lw, alpha)

    # -- No-sharing message ----------------------------------------------------
    if not chords:
        ax.text(0, 0, 'no shared LD blocks\nbetween traits',
                ha='center', va='center', fontsize=7,
                color=MUTED, style='italic', zorder=4)

    # -- Panel title -----------------------------------------------------------
    ax.text(TITLE_X, TITLE_Y, str(cat_name).replace('_', ' '),
            color='#222222', fontsize=TITLE_FS, fontweight='normal',
            ha='left', va='top', zorder=10, clip_on=False)

    ax.set_xlim(-1.50, 1.50)
    ax.set_ylim(-1.50, 1.50)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_facecolor(BG)


# --------------------------------------------------------------------------
# 9  BUILD FIGURE
# --------------------------------------------------------------------------

N_ROWS, N_COLS = 3, 2

# Sort panels: explicit order from config if given, else by trait count desc
PANEL_ORDER = list(getattr(p, 'panel_order', []) or [])
if PANEL_ORDER:
    _order_idx = {_norm_str(n): i for i, n in enumerate(PANEL_ORDER)}
    for _pname, _pdf, *_ in panels:
        if _norm_str(_pname) not in _order_idx:
            _msg = '[' + 'chord' + '] WARNING: category not in panel_order (check spelling): ' + str(_pname)
            print(_msg, file=sys.stderr)
    panels = sorted(panels,
                    key=lambda x: _order_idx.get(_norm_str(x[0]), 999))
else:
    panels = sorted(panels, key=lambda x: (-len(x[1]), x[0]))

# Height ratio per row = max arc count in that row
_row_max_n = []
for _r in range(N_ROWS):
    _rp = panels[_r*N_COLS:(_r+1)*N_COLS]
    _row_max_n.append(max((len(p[1]) for p in _rp if not p[1].empty), default=1))

# Compute FIG_H: square cells, height scaled to actual trait rows
_L, _R, _T = 0.005, 0.995, 0.995
_ws, _hs   = WSPACE, HSPACE
_leg_in    = 1.1
_cell_w    = FIG_W * (_R - _L) / (N_COLS + _ws * (N_COLS - 1))
_fig_h_uni = (_cell_w * (N_ROWS + _hs * (N_ROWS - 1)) + _leg_in) / _T
_fig_h     = _fig_h_uni * sum(_row_max_n) / (N_ROWS * max(_row_max_n))
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FIG_W, _fig_h),
                          gridspec_kw={'height_ratios': _row_max_n},
                          facecolor=BG)
fig.patch.set_facecolor(BG)

for idx, (cat, cat_df, chords, trait_snps) in enumerate(panels[:N_ROWS * N_COLS]):
    ax = axes[idx // N_COLS, idx % N_COLS]
    if cat_df.empty: ax.axis('off'); continue
    draw_panel(ax, cat, cat_df, chords, trait_snps)

for idx in range(len(panels), N_ROWS * N_COLS):
    axes.flatten()[idx].axis('off')


# --------------------------------------------------------------------------
# 10  LEGEND
# --------------------------------------------------------------------------

_sq = dict(marker='s', linestyle='none', markersize=8,
           markeredgecolor='none')
legend_handles = [
    Line2D([0],[0], markerfacecolor=COLOR_HG, label='HG > EF  (arc)',  **_sq),
    Line2D([0],[0], markerfacecolor=COLOR_EF, label='EF > HG  (arc)',  **_sq),
    Line2D([0],[0], markerfacecolor=COLOR_NS, label='p >= 0.05  (arc)', **_sq),
    Line2D([0],[0], color=CHORD_CLR, lw=1.5,
           label='chord width ~ shared / combined contribution  (fraction of shared genetic architecture)'),
]

fig.legend(handles=legend_handles, loc='lower center', ncol=3,
           fontsize=7.5, framealpha=0.0, labelcolor=MUTED,
           bbox_to_anchor=(0.5, 0.0),
           handletextpad=0.4, columnspacing=1.2)

fig.subplots_adjust(left=_L, right=_R, top=_T,
                    bottom=_leg_in/_fig_h, hspace=_hs, wspace=_ws)


# --------------------------------------------------------------------------
# 11  SAVE
# --------------------------------------------------------------------------

import time as _time, os as _os

for ext in ('.pdf', '.svg'):
    path = output_file if ext == '.pdf' else output_file.replace('.pdf', '.svg')
    fig.savefig(path, bbox_inches='tight', dpi=180, facecolor=BG)
    # fsync to force Lustre/parallel filesystem to flush metadata
    try:
        with open(path, 'rb') as _fh:
            _os.fsync(_fh.fileno())
    except Exception:
        pass
    print(f"[chord] Saved: {path}", file=sys.stderr)

plt.close(fig)

# Small pause for parallel filesystem metadata propagation
_time.sleep(5)