"""
plot_category_enrichment.py — FST enrichment volcano, dark / light theme.

Produces two output files:
  • plot_volcano      — raw p-value version
  • plot_volcano_fdr  — BH-FDR version (with significance band)

Dots are coloured by category group (the prefix before " - ").
Only dots with -log10(y) >= label_min_neg_logq are labelled.
Label positions are de-overlapped with an iterative 1-D repulsion pass;
adjustText is used instead if available in the environment.
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── optional adjustText ────────────────────────────────────────────────────
try:
    from adjustText import adjust_text as _adjust_text
    _HAS_AT = True
except ImportError:
    _HAS_AT = False
    print("[volcano] adjustText not found — using built-in repulsion", file=sys.stderr)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1 · THEME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THEME = str(getattr(snakemake.params, 'theme', 'dark')).lower()

if THEME == 'dark':
    plt.style.use('dark_background')
    BG           = '#181818'
    PANEL        = '#1e1e1e'
    TEXT         = '#eeeeee'
    MUTED        = '#777777'
    GRID         = '#2e2e2e'
    SPINE        = '#4a4a4a'
    LEG_FACE     = '#252525'
    LEG_EDGE     = '#444444'
    THRESH_CLR   = '#E07070'
    THRESH_FILL  = '#5a1515'
    NS_ALPHA     = 0.20
    SIG_ALPHA    = 0.88
    EDGE_CLR     = '#eeeeee'
else:
    matplotlib.rcdefaults()
    BG           = '#f8f8f8'
    PANEL        = '#ffffff'
    TEXT         = '#1a1a1a'
    MUTED        = '#999999'
    GRID         = '#e8e8e8'
    SPINE        = '#bbbbbb'
    LEG_FACE     = '#ffffff'
    LEG_EDGE     = '#cccccc'
    THRESH_CLR   = '#C03030'
    THRESH_FILL  = '#ffe8e8'
    NS_ALPHA     = 0.18
    SIG_ALPHA    = 0.90
    EDGE_CLR     = '#222222'
    matplotlib.rcParams.update({
        'text.color':        TEXT,
        'axes.labelcolor':   TEXT,
        'xtick.color':       TEXT,
        'ytick.color':       TEXT,
        'figure.facecolor':  BG,
        'axes.facecolor':    PANEL,
    })

# ── Semantic colour map ────────────────────────────────────────────────────
# Biologically motivated: colour families track organ system / disease domain.
# Keys are lowercased first tokens; unknown groups fall back to _FALLBACK_PALETTE.
_SEMANTIC = {
    # Neurological / cognitive / psychiatric  → violet family
    'neuropsychiatric': '#7B68C8',
    'neurological':     '#7B68C8',
    'psychiatric':      '#9480D4',
    'cognitive':        '#8B78D0',
    'brain':            '#7B68C8',
    'sleep':            '#9B88D8',
    # Cardiovascular / haematological  → red family
    'cardiovascular':   '#C44444',
    'cardiac':          '#C44444',
    'heart':            '#C44444',
    'hematological':    '#A02828',
    'haematological':   '#A02828',
    'blood':            '#A02828',
    'coagulation':      '#B43030',
    # Metabolic / lipids / diabetes / endocrine  → gold / amber family
    'lipids':           '#C8A820',
    'metabolic':        '#C8A820',
    'diabetes':         '#E08820',
    'glycemic':         '#E08820',
    'thyroid':          '#B89820',
    'endocrine':        '#B89820',
    # Gastrointestinal / hepatic  → burnt orange
    'gastrointestinal': '#DD7030',
    'gut':              '#DD7030',
    'liver':            '#B85C20',
    'hepatic':          '#B85C20',
    # Immune / inflammatory / autoimmune  → orange-red
    'immune':           '#D06020',
    'autoimmune':       '#D06020',
    'inflammatory':     '#C05A18',
    'allergy':          '#D07828',
    # Anthropometry / body composition  → steel blue
    'anthropometry':    '#4C72B0',
    'body':             '#4C72B0',
    # Behavioral / substance use  → muted purple
    'behavioral':       '#8172B2',
    'behaviour':        '#8172B2',
    'substance':        '#9482C0',
    # Musculoskeletal / bone / connective tissue  → muted coral
    'musculoskeletal':  '#C05858',
    'bone':             '#C05858',
    'joint':            '#C05858',
    'connective':       '#C05858',
    # Sensory (hearing, vision, smell)  → teal
    'sensory':          '#2A9EA8',
    'vision':           '#2A9EA8',
    'hearing':          '#2A9EA8',
    'ocular':           '#3AAEBC',
    # Lifestyle / diet / physical activity  → green
    'lifestyle':        '#55A868',
    'diet':             '#55A868',
    'physical':         '#55A868',
    # Aging / longevity  → grey
    'aging':            '#888888',
    'ageing':           '#888888',
    'longevity':        '#888888',
    # Reproductive / fertility / pregnancy  → mauve / pink
    'reproductive':     '#C87AAC',
    'fertility':        '#C87AAC',
    'pregnancy':        '#C87AAC',
    'maternal':         '#C87AAC',
    # Cancer / oncology  → dark charcoal
    'cancer':           '#505050',
    'oncological':      '#505050',
    'oncology':         '#505050',
    # Respiratory / pulmonary  → sky blue
    'respiratory':      '#5898C8',
    'pulmonary':        '#5898C8',
    'asthma':           '#5898C8',
    # Renal / kidney  → indigo-blue
    'renal':            '#5060A8',
    'kidney':           '#5060A8',
    'urological':       '#5060A8',
    # Dental / oral  → soft teal-green
    'dental':           '#60B4A8',
    'oral':             '#60B4A8',
    # Skin / dermatological  → warm tan
    'skin':             '#B89050',
    'dermatological':   '#B89050',
}

# Fallback palette for any group not matched above
_FALLBACK_PALETTE = [
    '#4C72B0', '#DD8452', '#55A868', '#8172B2', '#937860',
    '#CCB974', '#64B5CD', '#264D8E', '#A85B1A', '#2D7A46',
]


def _group_color(group_name, fallback_idx=0):
    """Return a semantically meaningful colour for a category group name."""
    key   = group_name.lower().strip()
    first = key.split()[0]
    return (_SEMANTIC.get(key)
            or _SEMANTIC.get(first)
            or _FALLBACK_PALETTE[fallback_idx % len(_FALLBACK_PALETTE)])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2 · SNAKEMAKE I/O & PARAMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

enrichment_file = snakemake.input['enrichment']
out_p           = snakemake.output['plot_volcano']
out_fdr         = snakemake.output['plot_volcano_fdr']

p              = snakemake.params
COL_CAT        = p.col_category
COL_P_CFG      = p.col_pvalue
COL_FDR_CFG    = p.col_fdr
COL_NCAT       = p.col_n_category
COL_EFF_FB     = getattr(p, 'col_effect_size_fallback', 'fold_enrichment')
FDR_THRESH     = float(p.fdr_threshold)
MAX_LABEL      = int(p.max_label_length)

vp = snakemake.config.get('volcano_plot', {})
DOT_MIN          = float(vp.get('dot_size_min',        60))
DOT_MAX          = float(vp.get('dot_size_max',       520))
LABEL_MIN_Y      = float(vp.get('label_min_neg_logq',  1.0))
LABEL_FS         = float(vp.get('label_size_min',      10.5))

P_CANDS   = ['mannwhitneyu_pval', 'mw_p_value', 'pvalue', 'p_value',
             'mannwhitneyu_pval_twosided', 'kolmogorov_smirnov_pval', 'ttest_pval']
FDR_CANDS = ['mannwhitneyu_fdr', 'fdr_p_value', 'q_value', 'fdr',
             'mannwhitneyu_pval_bonf']
EFF_CANDS = ['fold_enrichment', 'cohens_d', 'mean_fst', 'median_fst']


def resolve(col, cands, df, label, required=True):
    if col and col in df.columns:
        return col
    auto = next((c for c in cands if c in df.columns), None)
    if auto:
        print(f"[volcano] '{col}' not found → using '{auto}' for {label}", file=sys.stderr)
        return auto
    if required:
        raise ValueError(f"Column '{label}' not found. Tried: {[col] + cands}")
    return None


os.makedirs(os.path.dirname(out_p), exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3 · LOAD & DERIVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

df = pd.read_csv(enrichment_file, sep='\t', comment='#')
print(f"[volcano] {len(df)} categories  |  columns: {list(df.columns)}", file=sys.stderr)

COL_P   = resolve(COL_P_CFG,   P_CANDS,   df, 'p-value',   required=True)
COL_FDR = resolve(COL_FDR_CFG, FDR_CANDS, df, 'FDR',       required=False)
eff_col = resolve(COL_EFF_FB,  EFF_CANDS, df, 'effect size', required=True)

df['x_val']        = df[eff_col].astype(float)
df['neg_log10_p']  = -np.log10(df[COL_P].clip(lower=1e-300))
df['neg_log10_fdr'] = (
    -np.log10(df[COL_FDR].clip(lower=1e-300))
    if (COL_FDR and COL_FDR in df.columns)
    else np.nan
)
df['is_sig'] = (df[COL_FDR] < FDR_THRESH) if (COL_FDR and COL_FDR in df.columns) else False

# ── category group → colour ───────────────────────────────────────────────
def _group(cat):
    name = str(cat).replace('_', ' ')
    return name.split(' - ')[0].strip() if ' - ' in name else name.split()[0]

df['group'] = df[COL_CAT].apply(_group)
sorted_groups = sorted(df['group'].unique())
# Semantic lookup first; fallback index only increments for truly unknown groups
_fb_idx = 0
g2c = {}
for _g in sorted_groups:
    _c = _group_color(_g, _fb_idx)
    if _c in _FALLBACK_PALETTE:
        _fb_idx += 1
    g2c[_g] = _c
df['color'] = df['group'].map(g2c)

# ── dot sizes ─────────────────────────────────────────────────────────────
if COL_NCAT and COL_NCAT in df.columns:
    sz_raw   = df[COL_NCAT].astype(float)
    sz_range = sz_raw.max() - sz_raw.min()
    df['dot_size'] = (
        DOT_MIN + (sz_raw - sz_raw.min()) / (sz_range + 1e-9) * (DOT_MAX - DOT_MIN)
        if sz_range > 0 else (DOT_MIN + DOT_MAX) / 2
    )
else:
    sz_raw = sz_range = None
    df['dot_size'] = (DOT_MIN + DOT_MAX) / 2

# ── label text ────────────────────────────────────────────────────────────
def _label(cat):
    n = str(cat).replace('_', ' ')
    return (n[:MAX_LABEL - 1] + '…') if len(n) > MAX_LABEL else n

df['label'] = df[COL_CAT].apply(_label)

eff_is_fold = 'fold' in eff_col.lower()
x_ref       = 1.0 if eff_is_fold else 0.0
x_label_str = r'F$_{\mathrm{ST}}$ fold enrichment over genome-wide background' if eff_is_fold else eff_col.replace('_', ' ').title()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4 · LABEL REPULSION  (fallback when adjustText is absent)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _repel_y(ys, y_data_range, fig_h_in=8.5, ax_h_frac=0.80, font_pt=13.0,
             n_iter=400):
    """
    Iterative pairwise repulsion in y (data units).

    min_gap is derived from font size so labels don't collide after
    rendering — no axes draw required.
    """
    pts_per_unit = (fig_h_in * 72.0 * ax_h_frac) / (y_data_range + 1e-9)
    min_gap      = font_pt / pts_per_unit        # data units

    ys = np.array(ys, dtype=float)
    if len(ys) <= 1:
        return ys

    for _ in range(n_iter):
        moved = False
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                diff = ys[i] - ys[j]
                gap  = min_gap - abs(diff)
                if gap > 1e-9:
                    push    = gap / 2.0 + 1e-7
                    sign_ij = 1.0 if diff >= 0 else -1.0
                    ys[i]  += push * sign_ij
                    ys[j]  -= push * sign_ij
                    moved   = True
        if not moved:
            break
    return ys


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5 · DRAW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_volcano(y_col, y_label_str, thresh_y, output_path):

    x  = df['x_val'].values.astype(float)
    y  = df[y_col].values.astype(float)
    y  = np.where(np.isfinite(y), y, 0.0)

    x_extent = x.max() - x_ref
    y_max    = max(y.max() * 1.12 + 0.25, 1.2)

    # right-margin for labels
    x_min = x_ref - x_extent * 0.04
    x_max = x.max() + x_extent * 0.30

    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    # ── set limits early so repulsion is accurate ──────────────────────
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.18, y_max)

    # ── FDR significance band ──────────────────────────────────────────
    if thresh_y is not None:
        # subtle fill above threshold
        ax.axhspan(
            thresh_y, y_max * 1.05,
            color=THRESH_FILL,
            alpha=0.12 if THEME == 'dark' else 0.22,
            zorder=0, linewidth=0
        )
        # clean solid line (not dashed)
        ax.axhline(thresh_y, color=THRESH_CLR, linewidth=1.1,
                   alpha=0.80, zorder=2, linestyle='-')
        # inline tag — right edge, just above the line
        ax.text(
            x_max - x_extent * 0.005, thresh_y,
            f' FDR = {FDR_THRESH} ',
            va='bottom', ha='right',
            fontsize=7.5, color=THRESH_CLR, alpha=0.85,
            style='italic'
        )

    # ── reference line at x_ref (no-enrichment) ───────────────────────
    ax.axvline(x_ref, color=SPINE, linewidth=0.7, alpha=0.55,
               zorder=1, linestyle=':')

    # ── non-significant dots (drawn first → background layer) ─────────
    ns  = ~df['is_sig'].values
    sig =  df['is_sig'].values
    ax.scatter(
        x[ns], y[ns],
        s=df['dot_size'].values[ns],
        c=df['color'].values[ns],
        alpha=NS_ALPHA, linewidths=0, zorder=3
    )

    # ── significant dots (outlined) ────────────────────────────────────
    ax.scatter(
        x[sig], y[sig],
        s=df['dot_size'].values[sig],
        c=df['color'].values[sig],
        alpha=SIG_ALPHA, linewidths=1.4,
        edgecolors=EDGE_CLR, zorder=4
    )

    # ── labels ─────────────────────────────────────────────────────────
    mask_label = y >= LABEL_MIN_Y
    ldf        = df[mask_label].copy()
    lx         = df.loc[mask_label, 'x_val'].values.astype(float)
    ly         = df.loc[mask_label, y_col].values.astype(float)

    if len(ldf):
        if _HAS_AT:
            # adjustText handles everything
            texts = []
            for i, (_, row) in enumerate(ldf.iterrows()):
                t = ax.text(
                    lx[i], ly[i], row['label'],
                    fontsize=LABEL_FS, color=TEXT,
                    alpha=0.95 if row['is_sig'] else 0.55,
                    va='center', ha='left',
                    zorder=6
                )
                texts.append(t)
            _adjust_text(
                texts, ax=ax,
                x=lx, y=ly,
                arrowprops=dict(
                    arrowstyle='-',
                    color=MUTED, lw=0.6, alpha=0.6
                ),
                expand_points=(1.4, 1.6),
                force_text=(0.1, 0.35),
                only_move={'points': 'y', 'text': 'xy'}
            )
        else:
            # Built-in fallback: repel y, draw annotate arrows
            x_offset   = x_extent * 0.024   # label x offset from dot
            y_data_rng = y_max - (-0.18)

            # Sort by y descending so repulsion starts from the top
            order  = np.argsort(ly)[::-1]
            lx_ord = lx[order]
            ly_ord = ly[order]
            ldf_ord = ldf.iloc[order]

            ly_adj = _repel_y(ly_ord, y_data_rng, fig_h_in=8.5, ax_h_frac=0.80,
                               font_pt=LABEL_FS * 1.4, n_iter=400)
            ly_adj = np.clip(ly_adj, 0.05, y_max * 0.97)

            for i, (_, row) in enumerate(ldf_ord.iterrows()):
                x_dot, y_dot = lx_ord[i], ly_ord[i]
                y_lbl        = ly_adj[i]
                col          = row['color']
                is_sig       = row['is_sig']

                # leader-line + label via annotate
                ax.annotate(
                    row['label'],
                    xy=(x_dot, y_dot),
                    xytext=(x_dot + x_offset, y_lbl),
                    fontsize=LABEL_FS,
                    color=TEXT,
                    alpha=0.95 if is_sig else 0.52,
                    va='center', ha='left',
                    annotation_clip=False,
                    zorder=6,
                    arrowprops=dict(
                        arrowstyle='-',
                        color=col if is_sig else MUTED,
                        lw=0.65, alpha=0.55,
                        connectionstyle='arc3,rad=0.0'
                    ),
                )

    # ── category-group colour legend (only if ≤ 14 groups) ────────────
    n_groups = len(sorted_groups)
    if n_groups <= 14:
        from matplotlib.lines import Line2D
        group_handles = [
            Line2D([0], [0], marker='o', color='none',
                   markerfacecolor=g2c[g], markersize=7,
                   alpha=0.85, label=g)
            for g in sorted_groups
        ]
        ax2 = ax.inset_axes([0.995, 0.0, 0.18, 1.0],
                             transform=ax.transAxes)
        ax2.axis('off')
        ax2.legend(
            handles=group_handles,
            title='Category group', title_fontsize=7.5,
            fontsize=7.5, framealpha=0.30,
            facecolor=LEG_FACE, edgecolor=LEG_EDGE,
            labelcolor=TEXT, loc='center left',
            handletextpad=0.4, borderpad=0.8,
            labelspacing=0.55
        ).get_title().set_color(TEXT)

    # ── axes styling ───────────────────────────────────────────────────
    ax.set_xlabel(x_label_str, fontsize=12, color=TEXT, labelpad=9)
    ax.set_ylabel(y_label_str, fontsize=12, color=TEXT, labelpad=9)



    ax.tick_params(colors=TEXT, labelsize=9.5, length=3.5, width=0.8)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='major', linestyle=':', linewidth=0.4,
            color=GRID, alpha=0.8, zorder=0)
    ax.grid(which='minor', linestyle=':', linewidth=0.2,
            color=GRID, alpha=0.4, zorder=0)

    # full box frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)

    fig.tight_layout(rect=[0, 0, 0.82 if n_groups <= 14 else 1.0, 1.0])
    for ext in ('.pdf', '.svg'):
        path = output_path if ext == '.pdf' else output_path.replace('.pdf', '.svg')
        fig.savefig(path, bbox_inches='tight', dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"[volcano] Saved: {output_path}", file=sys.stderr)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6 · RUN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

draw_volcano(
    y_col       = 'neg_log10_p',
    y_label_str = r'$-\log_{10}(p)$',
    thresh_y    = None,
    output_path = out_p,
)

if df['neg_log10_fdr'].notna().any():
    draw_volcano(
        y_col       = 'neg_log10_fdr',
        y_label_str = r'$-\log_{10}(q)$',
        thresh_y    = -np.log10(FDR_THRESH),
        output_path = out_fdr,
    )
else:
    print("[volcano] No FDR column — skipping FDR volcano", file=sys.stderr)