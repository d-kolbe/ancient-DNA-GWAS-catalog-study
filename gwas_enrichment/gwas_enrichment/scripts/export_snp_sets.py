#!/usr/bin/env python3
"""
Export SNP set files for downstream analysis.
One file per trait, written into the category's snp_contributions directory.
Format: chr  pos  effect_allele  beta
SNPs are exactly those in the SNP contributions files (LD-deduplicated).
Only traits with >= min_snps SNPs are exported.
"""

import os
import sys
import glob
import re
import pandas as pd

RUN_DIR           = snakemake.params['run_dir']
ENRICHMENT_FILE   = snakemake.input['enrichment']
CONTRIB_BASE_DIR  = snakemake.params['contrib_base_dir']
OUTPUT_DIR        = snakemake.params['output_dir']
MIN_SNPS          = int(snakemake.params['min_snps'])
FDR_COL           = 'mannwhitneyu_fdr'
FDR_THRESH        = 0.05

output_summary    = snakemake.output['summary']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: identify FDR-significant categories
# ---------------------------------------------------------------------------

enr = pd.read_csv(ENRICHMENT_FILE, sep='\t')
sig_cats = set(enr[enr[FDR_COL] < FDR_THRESH]['category'].tolist())

# normalise to filesystem-safe names
def to_safe(s):
    return re.sub(r'[^\w\-]', '_', str(s)).strip('_')

sig_cats_safe = {to_safe(c) for c in sig_cats}

print(f"FDR < {FDR_THRESH} categories: {len(sig_cats)}")
for c in sorted(sig_cats):
    print(f"  {c}")

# ---------------------------------------------------------------------------
# Step 2: for each significant category, find contribution files
# ---------------------------------------------------------------------------

exported = []

for cat_safe in sorted(sig_cats_safe):
    cat_contrib_dir = os.path.join(CONTRIB_BASE_DIR, cat_safe)
    cat_output_dir  = os.path.join(OUTPUT_DIR, cat_safe)

    if not os.path.isdir(cat_contrib_dir):
        print(f"  WARNING: contrib dir not found: {cat_contrib_dir}")
        continue

    contrib_files = glob.glob(os.path.join(cat_contrib_dir, '*_snp_contributions.txt'))
    print(f"\n{cat_safe}: {len(contrib_files)} contribution files")

    for fpath in sorted(contrib_files):
        try:
            df = pd.read_csv(fpath, sep='\t', low_memory=False)
        except Exception as e:
            print(f"  ERROR reading {fpath}: {e}")
            continue

        if df.empty:
            continue

        trait_name = df['trait'].iloc[0] if 'trait' in df.columns else \
                     os.path.basename(fpath).replace('_snp_contributions.txt', '').replace('_', ' ')

        # Apply min SNP filter
        if len(df) < MIN_SNPS:
            print(f"  SKIP {trait_name}: {len(df)} SNPs < {MIN_SNPS}")
            continue

        # Extract required columns
        out = pd.DataFrame({
            'chr':           df['chr'],
            'pos':           df['pos'],
            'effect_allele': df['risk_allele'],
            'beta':          df['beta'],
        })

        # Write output
        os.makedirs(cat_output_dir, exist_ok=True)
        trait_safe = os.path.basename(fpath).replace('_snp_contributions.txt', '')
        out_path   = os.path.join(cat_output_dir, f"{trait_safe}_snpset.txt")
        out.to_csv(out_path, sep='\t', index=False)

        print(f"  ✓ {trait_name}: {len(out)} SNPs → {out_path}")
        exported.append({
            'category':   cat_safe,
            'trait':      trait_name,
            'n_snps':     len(out),
            'file':       out_path,
        })

# ---------------------------------------------------------------------------
# Write summary
# ---------------------------------------------------------------------------

summary_df = pd.DataFrame(exported)
summary_df.to_csv(output_summary, sep='\t', index=False)

print(f"\n{'='*60}")
print(f"Exported {len(exported)} trait SNP sets")
print(f"Summary: {output_summary}")
print(f"{'='*60}")