"""
plot_snp_contributions.py
=========================
Manhattan-style plot of per-SNP PGS contributions.
Dark theme.

Category & trait selection mirrors the forest plots exactly:
  - Categories: top N by enrichment p-value (config: top_n_categories),
                ranked independently from the directionality top_n.
  - Traits:     top N by ttest_pval, min_snps_for_plot, exclude_traits —
                same filters as the forest plot so both figures show
                the same set of traits.

If this category is not in the top-N, the output directory is created
empty and the script exits cleanly (Snakemake target is satisfied).

Outputs: one PDF + SVG per trait, written into output_dir.
"""

import sys, os, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use('dark_background')

# ---------------------------------------------------------------------------
# Theme setup — dark or light depending on snakemake.params.theme
# ---------------------------------------------------------------------------
THEME = str(getattr(snakemake.params, 'theme', 'dark')).lower()

if THEME == 'dark':
    plt.style.use('dark_background')
    BG          = '#1a1a1a'
    TEXT        = 'white'
    GRID_COLOR  = '#555555'
    SPINE_COLOR = '#555555'
    LEG_FACE    = '#1a1a1a'
    LEG_EDGE    = '#444444'
    BAND_A      = '#252525'
    BAND_B      = '#1a1a1a'
else:
    plt.style.use('default')
    BG          = 'white'
    TEXT        = 'black'
    GRID_COLOR  = '#cccccc'
    SPINE_COLOR = '#cccccc'
    LEG_FACE    = 'white'
    LEG_EDGE    = '#cccccc'
    BAND_A      = '#f5f5f5'
    BAND_B      = 'white'


# ---------------------------------------------------------------------------
# Snakemake I/O
# ---------------------------------------------------------------------------

contrib_dir       = str(snakemake.input['contributions_dir'])
enrichment_file   = str(snakemake.input['enrichment'])
directionality_file = str(snakemake.input['directionality'])
output_dir        = str(snakemake.output['plot_dir'])
p                 = snakemake.params

CATEGORY      = str(p.category)
TOP_N_CATS    = int(getattr(p, 'top_n_categories',  10))
RANK_BY       = str(getattr(p, 'rank_by',           'mannwhitneyu_pval'))
TOP_N_TRAITS  = int(getattr(p, 'top_n_traits',       20))
MIN_SNPS      = int(getattr(p, 'min_snps_for_plot',  15))
EXCLUDE       = list(getattr(p, 'exclude_traits',    []) or [])
TOP_N_LABELS  = int(getattr(p, 'top_n_labels',       10))
MIN_CONTRIB   = float(getattr(p, 'min_contribution',  0.0))
FIG_W         = float(getattr(p, 'figure_width',     16))
FIG_H         = float(getattr(p, 'figure_height',     6))

COL_CHR   = 'chr'
COL_POS   = 'pos'
COL_CONT  = 'pgs_contribution'
COL_ABS   = 'abs_pgs_contribution'
COL_GENE  = 'MAPPED_GENE'
COL_RSID  = 'SNPS'
COL_TRAIT = 'trait'
COL_CAT   = 'category'
COL_NBLKS = 'n_snps_in_blocks'
COL_PVAL  = 'ttest_pval'

P_CANDIDATES = ['ttest_pval', 'pgs_pvalue', 'mannwhitneyu_pval', 'p_value']

os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Determine top-N categories from enrichment file
# ---------------------------------------------------------------------------

import re as _re

def _norm(s):
    return _re.sub(r'[^a-z0-9]', '', str(s).lower())

enr = pd.read_csv(enrichment_file, sep='\t', comment='#')
cat_col = next((c for c in ['category','Category','trait_category'] if c in enr.columns), None)
if cat_col is None:
    print(f"Error: no category column in enrichment file. Columns: {list(enr.columns)}",
          file=sys.stderr)
    sys.exit(1)

rank_col = RANK_BY
if rank_col not in enr.columns:
    rank_col = next((c for c in P_CANDIDATES if c in enr.columns), None)
    if rank_col:
        print(f"Warning: '{RANK_BY}' not found, using '{rank_col}'", file=sys.stderr)
    else:
        print(f"Error: no p-value column found in enrichment file.", file=sys.stderr)
        sys.exit(1)

enr = enr.dropna(subset=[rank_col])
import re as _re2
ascending = bool(_re2.search(r'p.?val', rank_col, flags=_re2.IGNORECASE))
top_cats = (enr.sort_values(rank_col, ascending=ascending)
               .head(TOP_N_CATS)[cat_col]
               .tolist())

print(f"Top {TOP_N_CATS} categories for SNP contribution plots:", file=sys.stderr)
for i, c in enumerate(top_cats, 1):
    print(f"  {i:2d}. {c}", file=sys.stderr)

# Normalised lookup for matching
norm_to_cat = {_norm(c): c for c in top_cats}

# Check if this category is in the top-N
this_cat_norm = _norm(CATEGORY)
matched_cat = norm_to_cat.get(this_cat_norm)

if matched_cat is None:
    print(f"Category '{CATEGORY}' is not in top-{TOP_N_CATS} — skipping.",
          file=sys.stderr)
    # Write a marker file so Snakemake sees the directory as non-empty
    with open(os.path.join(output_dir, 'NOT_IN_TOP_N.txt'), 'w') as fh:
        fh.write(f"Category '{CATEGORY}' not in top-{TOP_N_CATS} by {rank_col}\n")
    sys.exit(0)

print(f"Category '{CATEGORY}' → matched as '{matched_cat}'", file=sys.stderr)

# ---------------------------------------------------------------------------
# Step 2: Determine which traits to plot (mirror forest plot filters)
# ---------------------------------------------------------------------------

dir_df = pd.read_csv(directionality_file, sep='\t', comment='#')

# Resolve p-value column
p_col = next((c for c in P_CANDIDATES if c in dir_df.columns), None)
if p_col is None:
    print(f"Error: no p-value column in directionality file.", file=sys.stderr)
    sys.exit(1)
if p_col != COL_PVAL:
    print(f"Warning: using '{p_col}' as p-value column", file=sys.stderr)

# Resolve category: normalised match
norm_to_actual = {_norm(c): c for c in dir_df[COL_CAT].unique()}
actual_cat = norm_to_actual.get(_norm(matched_cat), matched_cat)

cat_df = dir_df[dir_df[COL_CAT] == actual_cat].copy()
print(f"Traits in directionality file for this category: {len(cat_df)}", file=sys.stderr)

# Apply same filters as forest plot
if MIN_SNPS > 0 and COL_NBLKS in cat_df.columns:
    cat_df = cat_df[pd.to_numeric(cat_df[COL_NBLKS], errors='coerce').fillna(0) >= MIN_SNPS]

if EXCLUDE:
    cat_df = cat_df[~cat_df[COL_TRAIT].isin(EXCLUDE)]

cat_df = (cat_df.sort_values(p_col, ascending=True)
                .head(TOP_N_TRAITS)
                .reset_index(drop=True))

traits_to_plot = set(cat_df[COL_TRAIT].tolist())
print(f"Traits to plot after filters: {sorted(traits_to_plot)}", file=sys.stderr)

if not traits_to_plot:
    with open(os.path.join(output_dir, 'NO_TRAITS.txt'), 'w') as fh:
        fh.write(f"No traits passed filters for category '{matched_cat}'\n")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Step 3: Find contribution files for this category
# ---------------------------------------------------------------------------

def find_category_dir(base, category):
    direct = os.path.join(base, category)
    if os.path.isdir(direct): return direct
    safe = _re.sub(r'[^\w]', '_', category).strip('_')
    safe_path = os.path.join(base, safe)
    if os.path.isdir(safe_path): return safe_path
    for d in os.listdir(base):
        if _norm(d) == _norm(category):
            return os.path.join(base, d)
    return None

cat_dir = find_category_dir(contrib_dir, CATEGORY)
if cat_dir is None:
    cat_dir = find_category_dir(contrib_dir, matched_cat)
if cat_dir is None:
    print(f"Warning: no contribution directory found for '{CATEGORY}'", file=sys.stderr)
    with open(os.path.join(output_dir, 'NO_CONTRIB_DIR.txt'), 'w') as fh:
        fh.write(f"No contribution directory found for {CATEGORY}\n")
    sys.exit(0)

all_files = glob.glob(os.path.join(cat_dir, '*_snp_contributions.txt'))

# Match files to selected traits (normalised)
norm_trait = lambda s: _re.sub(r'[^a-z0-9]', '', str(s).lower())
trait_norm_map = {norm_trait(t): t for t in traits_to_plot}

files_to_plot = []
for fpath in sorted(all_files):
    fname = os.path.basename(fpath).replace('_snp_contributions.txt', '')
    fname_display = fname.replace('_', ' ')
    if norm_trait(fname_display) in trait_norm_map:
        files_to_plot.append((fpath, trait_norm_map[norm_trait(fname_display)]))
    elif norm_trait(fname) in trait_norm_map:
        files_to_plot.append((fpath, trait_norm_map[norm_trait(fname)]))

print(f"Matched {len(files_to_plot)}/{len(traits_to_plot)} traits to contribution files",
      file=sys.stderr)

# ---------------------------------------------------------------------------
# Chromosome layout
# ---------------------------------------------------------------------------

CHR_ORDER = list(range(1, 23)) + ['X', 'Y']
CHR_LENGTHS = {
    1:249, 2:243, 3:198, 4:191, 5:181, 6:171, 7:159, 8:146,
    9:141, 10:136, 11:135, 12:134, 13:115, 14:107, 15:103,
    16:90,  17:81,  18:78,  19:59,  20:63,  21:48,  22:51,
    'X':155,'Y':59
}

def build_chr_offsets(chrs_present):
    offsets = {}
    cumulative = 0
    gap = 5_000_000
    for c in CHR_ORDER:
        if c not in chrs_present: continue
        offsets[c] = cumulative
        cumulative += CHR_LENGTHS.get(c, 100) * 1_000_000 + gap
    return offsets

def safe_chr(val):
    try:
        v = int(val)
        return v if 1 <= v <= 22 else str(val)
    except (ValueError, TypeError):
        return str(val).strip()

def label_for_snp(row):
    CONSEQUENCE_SHORT = {
        'missense_variant':                   'missense',
        'synonymous_variant':                 'synonymous',
        'intron_variant':                     'intronic',
        'intergenic_variant':                 'intergenic',
        '3_prime_UTR_variant':                "3'UTR",
        '5_prime_UTR_variant':                "5'UTR",
        'stop_gained':                        'stop gained',
        'frameshift_variant':                 'frameshift',
        'splice_region_variant':              'splice region',
        'splice_donor_variant':               'splice donor',
        'splice_acceptor_variant':            'splice acceptor',
        'non_coding_transcript_exon_variant': 'nc exon',
        'upstream_gene_variant':              'upstream',
        'downstream_gene_variant':            'downstream',
        'regulatory_region_variant':          'regulatory',
    }

    # Gene — first gene if comma-separated
    gene = str(row.get(COL_GENE, '')).strip()
    gene = gene.split(',')[0].strip() if gene not in ('nan', 'NaN', '') else ''

    # rsID
    rsid = str(row.get(COL_RSID, '')).strip()
    if not rsid or rsid in ('nan', 'NaN', ''):
        rsid = str(row.get('snp_id', f"{row[COL_CHR]}:{row[COL_POS]}"))

    # Consequence
    ctx = str(row.get('CONTEXT', '')).strip()
    ctx_first = ctx.split(',')[0].split(';')[0].strip()
    ctx_short = CONSEQUENCE_SHORT.get(ctx_first, ctx_first)
    ctx_part  = f' ({ctx_short})' if ctx_short and ctx_short not in ('nan','NaN','') else ''

    if gene:
        return f"{gene}-{rsid}{ctx_part}"
    return f"{rsid}{ctx_part}"

# ---------------------------------------------------------------------------
# Step 4: Plot each selected trait
# ---------------------------------------------------------------------------

for fpath, trait_name in files_to_plot:
    print(f"  Plotting: {trait_name}", file=sys.stderr)

    try:
        df = pd.read_csv(fpath, sep='\t', low_memory=False)
    except Exception as e:
        print(f"  Error reading {fpath}: {e}", file=sys.stderr)
        continue

    if df.empty or COL_CONT not in df.columns:
        print(f"  Skipping — empty or missing {COL_CONT}", file=sys.stderr)
        continue

    df[COL_CHR]  = df[COL_CHR].apply(safe_chr)
    df[COL_POS]  = pd.to_numeric(df[COL_POS],  errors='coerce')
    df[COL_CONT] = pd.to_numeric(df[COL_CONT], errors='coerce')
    df = df.dropna(subset=[COL_CHR, COL_POS, COL_CONT]).copy()

    if MIN_CONTRIB > 0:
        df = df[df[COL_CONT].abs() >= MIN_CONTRIB].copy()

    if df.empty:
        print(f"  Skipping — no data after filters", file=sys.stderr)
        continue

    df[COL_ABS] = df[COL_CONT].abs()
    chrs_present = set(df[COL_CHR].unique())
    offsets = build_chr_offsets(chrs_present)
    df['x'] = df.apply(lambda r: offsets.get(r[COL_CHR], 0) + r[COL_POS], axis=1)
    df['chr_idx'] = df[COL_CHR].apply(
        lambda c: CHR_ORDER.index(c) if c in CHR_ORDER else 99)

    pos_mask = df[COL_CONT] >= 0
    neg_mask  = ~pos_mask
    top_idx   = df.nlargest(TOP_N_LABELS, COL_ABS).index

    # Figure
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax.set_facecolor(BG)

    # Chromosome bands
    for c in CHR_ORDER:
        if c not in offsets: continue
        x0 = offsets[c]
        x1 = x0 + CHR_LENGTHS.get(c, 100) * 1_000_000
        shade = BAND_A if CHR_ORDER.index(c) % 2 == 0 else BAND_B
        ax.axvspan(x0, x1, facecolor=shade, alpha=0.6, zorder=0, linewidth=0)

    ax.axhline(0, color=TEXT, linewidth=0.6, alpha=0.25, zorder=1)

    if pos_mask.any():
        ax.scatter(df.loc[pos_mask, 'x'], df.loc[pos_mask, COL_CONT],
                   s=18, c='#D6604D', alpha=0.78, linewidths=0, zorder=3,
                   label='EF > HG')
    if neg_mask.any():
        ax.scatter(df.loc[neg_mask, 'x'], df.loc[neg_mask, COL_CONT],
                   s=18, c='#2166AC', alpha=0.78, linewidths=0, zorder=3,
                   label='HG > EF')

    # Highlighted top-N dots
    top_colors = df.loc[top_idx, COL_CONT].apply(
        lambda v: '#FF6B4A' if v >= 0 else '#4A9FDD')
    ax.scatter(df.loc[top_idx, 'x'], df.loc[top_idx, COL_CONT],
               s=60, c=top_colors, edgecolors=TEXT, linewidths=0.7, zorder=4)

    # Labels — anchored just to the top-right (positive) or bottom-right (negative)
    # of each highlighted dot. x_offset is a tiny fraction of the x range so the
    # label starts right beside the dot. adjustText is used only for fine-tuning
    # in the rare case two top SNPs are close together.
    x_range = (df['x'].max() - df['x'].min()) if len(df) > 1 else 1
    x_off   = x_range * 0.005   # ~0.5% of genome width — just clears the dot edge
    y_range = df[COL_CONT].abs().max() * 2 if df[COL_CONT].abs().max() > 0 else 1
    y_off_small = y_range * 0.008   # tiny vertical nudge so label doesn't sit on dot

    try:
        from adjustText import adjust_text
        texts = []
        for idx in top_idx:
            row = df.loc[idx]
            lbl = label_for_snp(row)
            va  = 'bottom' if row[COL_CONT] >= 0 else 'top'
            y   = row[COL_CONT] + (y_off_small if row[COL_CONT] >= 0 else -y_off_small)
            texts.append(
                ax.text(row['x'] + x_off, y, lbl,
                        fontsize=7, color='white',
                        va=va, ha='left', fontweight='bold')
            )
        adjust_text(
            texts, ax=ax,
            only_move={'points': 'y', 'texts': 'y'},   # only nudge vertically
            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.5, alpha=0.7),
            expand_text=(1.0, 1.1),
            force_text=(0.0, 0.15)
        )
    except ImportError:
        for idx in top_idx:
            row = df.loc[idx]
            lbl = label_for_snp(row)
            va  = 'bottom' if row[COL_CONT] >= 0 else 'top'
            y   = row[COL_CONT] + (y_off_small if row[COL_CONT] >= 0 else -y_off_small)
            ax.text(row['x'] + x_off, y, lbl,
                    fontsize=7, color=TEXT,
                    va=va, ha='left', fontweight='bold')

    # X-axis chromosome ticks
    chr_ticks, chr_labels = [], []
    for c in CHR_ORDER:
        if c not in offsets: continue
        chr_ticks.append(offsets[c] + CHR_LENGTHS.get(c, 100) * 1_000_000 / 2)
        chr_labels.append(str(c))
    ax.set_xticks(chr_ticks)
    ax.set_xticklabels(chr_labels, fontsize=7, color=TEXT)
    ax.set_xlim(min(offsets.values()) - 5e6,
                max(offsets[c] + CHR_LENGTHS.get(c, 100) * 1e6 for c in offsets) + 5e6)

    ax.set_ylabel('PGS contribution  (EF − HG)', fontsize=10, color=TEXT)
    ax.set_xlabel('Chromosome', fontsize=10, color=TEXT)
    ax.tick_params(colors='white', labelsize=8)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(SPINE_COLOR)
    ax.spines['left'].set_color(SPINE_COLOR)
    ax.grid(axis='y', linestyle=':', linewidth=0.4, alpha=0.3)

    handles = [
        mpatches.Patch(color='#D6604D', label='EF > HG  (+)'),
        mpatches.Patch(color='#2166AC', label='HG > EF  (−)'),
    ]
    ax.legend(handles=handles, fontsize=8, framealpha=0.25,
              facecolor=BG, edgecolor='#555', loc='upper right')
    ax.set_title(trait_name, fontsize=11, color=TEXT, pad=8)

    safe_name = _re.sub(r'[^\w\-]', '_', trait_name).strip('_')
    out_base  = os.path.join(output_dir, safe_name)
    fig.savefig(out_base + '.pdf', bbox_inches='tight', dpi=180,
                facecolor=BG)
    fig.savefig(out_base + '.svg', bbox_inches='tight',
                facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out_base}.pdf/.svg", file=sys.stderr)

print(f"Done. {len(files_to_plot)} plots written to {output_dir}", file=sys.stderr)