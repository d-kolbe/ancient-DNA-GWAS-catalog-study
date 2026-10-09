"""
identify_directionality_categories.py

Selects the top N categories for trait decomposition and PGS/directionality
analysis, ranked by p-value (or optionally by effect size / Mann-Whitney U
statistic). This is independent of FDR significance — all top-N categories
are processed regardless of whether they survive multiple-testing correction.

Inputs (from snakemake.input):
    enrichment_results : path to category_enrichment.txt

Params (from snakemake.params):
    top_n       : number of categories to select (default 10)
    rank_by     : column to sort by; 'p_value' (default) or 'effect_size'
                  (any numeric column present in the enrichment file works)

Output (snakemake.output):
    directionality_categories : one category name per line
"""

import pandas as pd
import sys

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

enrichment_file = snakemake.input['enrichment_results']
output_file     = snakemake.output['directionality_categories']
top_n           = int(snakemake.params.get('top_n', 10))
rank_by         = snakemake.params.get('rank_by', 'p_value')

# Common p-value column names used across pipeline versions — for auto-detection
P_VALUE_CANDIDATES = [
    'p_value', 'pvalue', 'mw_p_value', 'mannwhitneyu_pval',
    'mannwhitneyu_pval_twosided', 'kolmogorov_smirnov_pval',
]

# ---------------------------------------------------------------------------
# Load enrichment results
# ---------------------------------------------------------------------------

df = pd.read_csv(enrichment_file, sep='\t', comment='#')

print(f"Loaded {len(df)} categories from {enrichment_file}", file=sys.stderr)
print(f"Columns: {list(df.columns)}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Determine category name column
# ---------------------------------------------------------------------------

# Accept 'category' or 'Category' column
cat_col = None
for candidate in ['category', 'Category', 'trait_category', 'CATEGORY']:
    if candidate in df.columns:
        cat_col = candidate
        break

if cat_col is None:
    raise ValueError(
        f"Could not find a category column in enrichment file. "
        f"Available columns: {list(df.columns)}"
    )

# ---------------------------------------------------------------------------
# Validate ranking column
# ---------------------------------------------------------------------------

if rank_by not in df.columns:
    # Try to auto-detect a sensible p-value column before raising
    auto = next((c for c in P_VALUE_CANDIDATES if c in df.columns), None)
    if auto:
        print(
            f"Warning: configured rank_by='{rank_by}' not found. "
            f"Auto-detected '{auto}' — using that instead. "
            f"Set directionality_rank_by in config to silence this warning.",
            file=sys.stderr
        )
        rank_by = auto
    else:
        available = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        raise ValueError(
            f"rank_by column '{rank_by}' not found, and no known p-value column "
            f"could be auto-detected. Set directionality_rank_by in config to "
            f"one of: {available}"
        )

# ---------------------------------------------------------------------------
# Drop rows with missing rank column values
# ---------------------------------------------------------------------------

n_before = len(df)
df = df.dropna(subset=[rank_by])
n_dropped = n_before - len(df)
if n_dropped > 0:
    print(
        f"Warning: dropped {n_dropped} categories with missing '{rank_by}' values",
        file=sys.stderr
    )

# ---------------------------------------------------------------------------
# Sort and select top N
# ---------------------------------------------------------------------------

# For p-values: ascending (smallest p = most significant)
# For effect sizes: descending (largest effect first)
# Detect p-value columns by name pattern rather than exact string match
import re as _re
ascending = bool(_re.search(r'p.?val', rank_by, flags=_re.IGNORECASE))
df_sorted = df.sort_values(rank_by, ascending=ascending)

actual_n = min(top_n, len(df_sorted))
if actual_n < top_n:
    print(
        f"Warning: requested top {top_n} but only {actual_n} categories available",
        file=sys.stderr
    )

top_categories = df_sorted.head(actual_n)[cat_col].tolist()

# ---------------------------------------------------------------------------
# Report selection
# ---------------------------------------------------------------------------

print(f"\nTop {actual_n} categories selected for directionality analysis "
      f"(ranked by {rank_by}, {'ascending' if ascending else 'descending'}):",
      file=sys.stderr)
for i, cat in enumerate(top_categories, 1):
    rank_val = df_sorted.iloc[i - 1][rank_by]
    print(f"  {i:2d}. {cat}  ({rank_by}={rank_val:.4g})", file=sys.stderr)

# Check how many are also FDR-significant (informational only)
if 'fdr_p_value' in df.columns or 'q_value' in df.columns:
    fdr_col = 'fdr_p_value' if 'fdr_p_value' in df.columns else 'q_value'
    n_sig = df_sorted.head(actual_n)[fdr_col].lt(0.05).sum()
    print(f"\n  {n_sig}/{actual_n} of selected categories are also FDR-significant (q<0.05)",
          file=sys.stderr)

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

with open(output_file, 'w') as f:
    f.write(f"# Top {actual_n} categories ranked by {rank_by}\n")
    f.write(f"# Generated by identify_directionality_categories.py\n")
    for cat in top_categories:
        f.write(f"{cat}\n")

print(f"\nWrote {actual_n} categories to {output_file}", file=sys.stderr)