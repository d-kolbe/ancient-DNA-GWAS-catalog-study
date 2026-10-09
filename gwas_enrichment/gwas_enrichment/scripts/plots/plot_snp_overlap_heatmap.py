"""
plot_snp_overlap_heatmap.py
===========================
Produces two Jaccard heatmaps comparing SNP overlap between traits,
and a companion summary table.

Trait selection:
  - Top-N categories (from directionality_categories.txt)
  - Within each category: top K traits by p-value (config: top_n_traits_per_category)

Two heatmaps:
  1. ALL SNPs  — Jaccard index using every SNP in each trait's contribution file
  2. TOP SNPs  — Jaccard index using only the "high-leverage" SNPs per trait,
                 defined as the minimal set whose cumulative |contribution|
                 reaches the configured threshold (e.g. 80% of total effect)

Summary table:
  - All pairwise combinations
  - Columns: trait_A, category_A, trait_B, category_B,
             n_snps_A, n_snps_B, n_shared_all, jaccard_all,
             n_top_A, n_top_B, n_shared_top, jaccard_top,
             shared_snp_ids_all, shared_snp_ids_top

Inputs:
    snakemake.input.directionality      — trait_directionality_summary.txt
    snakemake.input.dir_cats            — directionality_categories.txt
    snakemake.input.snp_contributions_dir — directory containing {category}/{trait}.txt

Outputs:
    snakemake.output.heatmap_all        — jaccard_all_snps.pdf
    snakemake.output.heatmap_top        — jaccard_top_snps.pdf
    snakemake.output.summary_table      — overlap_summary_table.tsv
"""

import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# ---------------------------------------------------------------------------
# Snakemake I/O
# ---------------------------------------------------------------------------

directionality_file   = snakemake.input['directionality']
dir_cats_file         = snakemake.input['dir_cats']
contrib_dir           = str(snakemake.input['snp_contributions_dir'])
out_all               = snakemake.output['heatmap_all']
out_top               = snakemake.output['heatmap_top']
out_table             = snakemake.output['summary_table']
p                     = snakemake.params

COL_SNP   = p.col_snp_id       # "snp_id"
COL_CONT  = p.col_contribution  # "pgs_contribution"
COL_TRAIT = 'trait'
COL_CAT   = 'category'
COL_P     = 'pgs_pvalue'

TOP_K_TRAITS   = int(p.top_n_traits_per_category)
CUM_THRESH     = float(p.cumulative_contribution_threshold)
MIN_SNPS       = int(p.min_snps_for_overlap)
FIG_W          = float(p.figure_width)
FIG_H          = float(p.figure_height)
CMAP           = str(p.cmap)
ANNOTATE       = bool(p.annotate_cells)
ANN_THRESH     = float(p.annotate_threshold)

for path in [out_all, out_top, out_table]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_filename(name):
    """Convert a trait/category name to a safe filename component."""
    return re.sub(r'[^\w\-]', '_', str(name)).strip('_')


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return np.nan
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def top_snps_by_cumulative(snps, contribs, threshold):
    """
    Return the set of SNP IDs whose cumulative |contribution|
    first reaches `threshold` fraction of the total absolute contribution.
    """
    abs_c  = np.abs(contribs)
    total  = abs_c.sum()
    if total == 0:
        return set(snps)
    order  = np.argsort(-abs_c)   # descending
    cumsum = np.cumsum(abs_c[order])
    cutoff = np.searchsorted(cumsum, threshold * total)
    top_idx = order[:cutoff + 1]
    return set(np.array(snps)[top_idx])


# ---------------------------------------------------------------------------
# Load directionality summary
# ---------------------------------------------------------------------------

df_dir = pd.read_csv(directionality_file, sep='\t', comment='#')
print(f'Loaded directionality: {len(df_dir)} traits', file=sys.stderr)
print(f'Directionality columns: {list(df_dir.columns)}', file=sys.stderr)

# Read top-N categories
with open(dir_cats_file) as fh:
    top_cats_raw = [
        line.strip() for line in fh
        if line.strip() and not line.startswith('#')
    ]

# ---------------------------------------------------------------------------
# Resolve category name format mismatch
# ---------------------------------------------------------------------------
# directionality_categories.txt may use filesystem-safe names (underscores)
# while trait_directionality_summary.txt uses original names (spaces/dashes).

def normalise(name):
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

actual_cats = df_dir[COL_CAT].unique()
norm_to_actual = {normalise(c): c for c in actual_cats}

top_cats = []
for raw in top_cats_raw:
    if raw in actual_cats:
        top_cats.append(raw)
    else:
        match = norm_to_actual.get(normalise(raw))
        if match:
            print(f'Category name resolved: "{raw}" -> "{match}"', file=sys.stderr)
            top_cats.append(match)
        else:
            print(f'Warning: category "{raw}" not found in directionality file '
                  f'(even after normalisation) — skipping', file=sys.stderr)

print(f'Resolved {len(top_cats)}/{len(top_cats_raw)} categories', file=sys.stderr)

# ---------------------------------------------------------------------------
# Select traits: top K per category by p-value
# ---------------------------------------------------------------------------

# Auto-detect p-value column
P_CANDIDATES = ['ttest_pval', 'pgs_pvalue', 'mannwhitneyu_pval', 'mw_p_value',
                'p_value', 'pvalue', 'kolmogorov_smirnov_pval']
if COL_P not in df_dir.columns:
    auto = next((c for c in P_CANDIDATES if c in df_dir.columns), None)
    if auto:
        print(f'Warning: p-value col "{COL_P}" not found, auto-using "{auto}"',
              file=sys.stderr)
        COL_P = auto
    else:
        raise ValueError(f'No p-value column found. Available: {list(df_dir.columns)}')

selected_traits = []
for cat in top_cats:
    cat_df = df_dir[df_dir[COL_CAT] == cat].copy()
    if cat_df.empty:
        print(f'Warning: no traits for resolved category "{cat}"', file=sys.stderr)
        continue
    top_k = cat_df.sort_values(COL_P, ascending=True).head(TOP_K_TRAITS)
    selected_traits.append(top_k)

if not selected_traits:
    print('Error: no traits selected after name resolution.', file=sys.stderr)
    print(f'  Categories in directionality file: {sorted(actual_cats)}', file=sys.stderr)
    print(f'  Categories requested (raw): {top_cats_raw}', file=sys.stderr)
    sys.exit(1)


traits_df = pd.concat(selected_traits, ignore_index=True)
print(f"Selected {len(traits_df)} traits across {len(top_cats)} categories",
      file=sys.stderr)

# ---------------------------------------------------------------------------
# Load SNP contribution files and build SNP sets
# ---------------------------------------------------------------------------
# Expected file layout: {contrib_dir}/{category_safe}/{trait_safe}.txt
# The script tries multiple name sanitisation patterns if the first fails.

_dir_listed = False

def find_contrib_file(base_dir, category, trait):
    """
    Try to locate the SNP contribution file for a given category/trait.
    On the first call, prints the top-2-level directory tree so path
    patterns can be diagnosed from the log.
    """
    global _dir_listed
    if not _dir_listed:
        _dir_listed = True
        print(f"\n--- SNP contributions directory structure (top 2 levels) ---",
              file=sys.stderr)
        if os.path.isdir(base_dir):
            for root, dirs, files in os.walk(base_dir):
                depth = root.replace(base_dir, '').count(os.sep)
                if depth > 1:
                    dirs[:] = []   # don't recurse deeper
                    continue
                indent = '  ' * depth
                print(f"{indent}{os.path.basename(root)}/", file=sys.stderr)
                if depth == 1:
                    for f in files[:5]:   # show first 5 files per subdir
                        print(f"{indent}  {f}", file=sys.stderr)
                    if len(files) > 5:
                        print(f"{indent}  ... ({len(files)} files total)", file=sys.stderr)
        else:
            print(f"  Directory does not exist: {base_dir}", file=sys.stderr)
        print(f"--- end directory listing ---\n", file=sys.stderr)

    cat_safe   = safe_filename(category)
    trait_safe = safe_filename(trait)

    candidates = [
        # Actual on-disk format: {cat_safe}/{trait_safe}_snp_contributions.txt
        os.path.join(base_dir, cat_safe,  trait_safe + '_snp_contributions.txt'),
        os.path.join(base_dir, category,  trait_safe + '_snp_contributions.txt'),
        # Fallback: bare .txt in case format changes
        os.path.join(base_dir, cat_safe,  trait_safe + '.txt'),
        os.path.join(base_dir, cat_safe,  trait_safe + '.tsv'),
        os.path.join(base_dir, category,  trait + '.txt'),
        # Flat layout (no subdir)
        os.path.join(base_dir, trait_safe + '_snp_contributions.txt'),
        os.path.join(base_dir, trait_safe + '.txt'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


trait_snp_sets_all = {}   # label → set of all SNP IDs
trait_snp_sets_top = {}   # label → set of top SNP IDs
trait_meta         = {}   # label → (category, trait, n_all, n_top)

missing = []
for _, row in traits_df.iterrows():
    cat   = row[COL_CAT]
    trait = row[COL_TRAIT]
    label = f'{cat} | {trait}'

    fpath = find_contrib_file(contrib_dir, cat, trait)
    if fpath is None:
        missing.append(label)
        print(f"  Missing contribution file for: {label}", file=sys.stderr)
        continue

    try:
        cdf = pd.read_csv(fpath, sep='\t', comment='#')
    except Exception as e:
        print(f"  Error reading {fpath}: {e}", file=sys.stderr)
        continue

    if COL_SNP not in cdf.columns or COL_CONT not in cdf.columns:
        # Auto-detect SNP id and contribution columns
        SNP_CANDIDATES  = ["snp_id", "rsid", "SNP", "variant_id", "chrpos",
                            "chr_pos", "ID"]
        CONT_CANDIDATES = ["pgs_contribution", "contribution", "snp_contribution",
                            "beta_x_delta_af", "effect"]
        resolved_snp  = next((c for c in SNP_CANDIDATES  if c in cdf.columns), None)
        resolved_cont = next((c for c in CONT_CANDIDATES if c in cdf.columns), None)
        if resolved_snp and resolved_cont:
            print(f"  Auto-resolved columns in {os.path.basename(fpath)}: "
                  f"snp={resolved_snp}, contribution={resolved_cont}", file=sys.stderr)
            COL_SNP  = resolved_snp
            COL_CONT = resolved_cont
        else:
            print(f"  Cannot resolve columns in {fpath}\n"
                  f"  Actual columns: {list(cdf.columns)}", file=sys.stderr)
            continue

    cdf = cdf.dropna(subset=[COL_SNP, COL_CONT])
    if len(cdf) < MIN_SNPS:
        print(f"  Too few SNPs ({len(cdf)}) for trait: {label}", file=sys.stderr)
        continue

    snps    = cdf[COL_SNP].astype(str).values
    contribs = cdf[COL_CONT].values.astype(float)

    all_set = set(snps)
    top_set = top_snps_by_cumulative(snps, contribs, CUM_THRESH)

    trait_snp_sets_all[label] = all_set
    trait_snp_sets_top[label] = top_set
    trait_meta[label] = (cat, trait, len(all_set), len(top_set))

if missing:
    print(f"\n{len(missing)} contribution files not found; these traits will be skipped.",
          file=sys.stderr)

labels = list(trait_snp_sets_all.keys())
print(f"\nBuilding heatmaps for {len(labels)} traits", file=sys.stderr)

if len(labels) < 2:
    print("Error: fewer than 2 traits with contribution data — cannot build heatmap",
          file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Build pairwise Jaccard matrices
# ---------------------------------------------------------------------------

n = len(labels)
mat_all = np.full((n, n), np.nan)
mat_top = np.full((n, n), np.nan)

for i in range(n):
    mat_all[i, i] = 1.0
    mat_top[i, i] = 1.0
    for j in range(i + 1, n):
        j_all = jaccard(trait_snp_sets_all[labels[i]], trait_snp_sets_all[labels[j]])
        j_top = jaccard(trait_snp_sets_top[labels[i]], trait_snp_sets_top[labels[j]])
        mat_all[i, j] = mat_all[j, i] = j_all
        mat_top[i, j] = mat_top[j, i] = j_top

df_all = pd.DataFrame(mat_all, index=labels, columns=labels)
df_top = pd.DataFrame(mat_top, index=labels, columns=labels)

# ---------------------------------------------------------------------------
# Heatmap drawing helper
# ---------------------------------------------------------------------------

def draw_heatmap(matrix_df, title, output_path, cmap, annotate, ann_thresh,
                 fig_w, fig_h):
    n = len(matrix_df)
    # Auto-scale figure so labels don't overlap
    dim = max(fig_w, n * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(dim, dim * fig_h / fig_w))

    # Annotation matrix: show value only if >= threshold
    annot_data = None
    if annotate:
        annot_data = matrix_df.applymap(
            lambda v: f'{v:.2f}' if (not np.isnan(v) and v >= ann_thresh) else ''
        )

    # Shorten labels for readability: show trait name only (drop category prefix)
    short_labels = [lbl.split(' | ')[-1] for lbl in matrix_df.index]
    # Append category as a short tag in brackets
    cat_tags = [lbl.split(' | ')[0].split(' - ')[-1][:15] for lbl in matrix_df.index]
    display_labels = [f'{t[:35]}  [{c}]' for t, c in zip(short_labels, cat_tags)]

    sns.heatmap(
        matrix_df,
        ax=ax,
        cmap=cmap,
        vmin=0, vmax=1,
        annot=annot_data,
        fmt='',
        linewidths=0.3,
        linecolor='white',
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar_kws={'label': 'Jaccard index', 'shrink': 0.6}
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_title(title, fontsize=11, pad=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=180)
    plt.close()
    print(f"Saved heatmap: {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Draw heatmaps
# ---------------------------------------------------------------------------

draw_heatmap(
    df_all,
    title=f'SNP Overlap (All SNPs) — Top {TOP_K_TRAITS} traits per category\nJaccard index',
    output_path=out_all,
    cmap=CMAP, annotate=ANNOTATE, ann_thresh=ANN_THRESH,
    fig_w=FIG_W, fig_h=FIG_H
)

draw_heatmap(
    df_top,
    title=(f'SNP Overlap (Top SNPs — {int(CUM_THRESH*100)}% cumulative |contribution|) '
           f'— Top {TOP_K_TRAITS} traits per category\nJaccard index'),
    output_path=out_top,
    cmap=CMAP, annotate=ANNOTATE, ann_thresh=ANN_THRESH,
    fig_w=FIG_W, fig_h=FIG_H
)

# ---------------------------------------------------------------------------
# Build summary table
# ---------------------------------------------------------------------------

rows = []
for la, lb in combinations(labels, 2):
    cat_a, trait_a, n_all_a, n_top_a = trait_meta[la]
    cat_b, trait_b, n_all_b, n_top_b = trait_meta[lb]

    set_all_a = trait_snp_sets_all[la]
    set_all_b = trait_snp_sets_all[lb]
    set_top_a = trait_snp_sets_top[la]
    set_top_b = trait_snp_sets_top[lb]

    shared_all = set_all_a & set_all_b
    shared_top = set_top_a & set_top_b

    rows.append({
        'trait_A':         trait_a,
        'category_A':      cat_a,
        'trait_B':         trait_b,
        'category_B':      cat_b,
        'n_snps_A':        n_all_a,
        'n_snps_B':        n_all_b,
        'n_shared_all':    len(shared_all),
        'jaccard_all':     round(jaccard(set_all_a, set_all_b), 4),
        'n_top_snps_A':    n_top_a,
        'n_top_snps_B':    n_top_b,
        'n_shared_top':    len(shared_top),
        'jaccard_top':     round(jaccard(set_top_a, set_top_b), 4),
        'shared_snp_ids_all': ';'.join(sorted(shared_all)) if shared_all else '',
        'shared_snp_ids_top': ';'.join(sorted(shared_top)) if shared_top else '',
    })

summary_df = (
    pd.DataFrame(rows)
    .sort_values('jaccard_top', ascending=False)
    .reset_index(drop=True)
)
summary_df.to_csv(out_table, sep='\t', index=False)
print(f"Saved summary table ({len(summary_df)} pairs): {out_table}", file=sys.stderr)