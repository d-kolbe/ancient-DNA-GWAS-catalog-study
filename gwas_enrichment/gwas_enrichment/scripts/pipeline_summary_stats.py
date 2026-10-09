#!/usr/bin/env python3
"""
Pipeline Summary Statistics

Pulls together basic counting statistics across pipeline stages:
  - Unique SNPs and categories at the genome-wide LD block stage
  - Unique traits and SNPs after trait decomposition
  - Unique traits and SNPs that made it into the final PGS/directionality stage
  - Mean / median SNPs per category at each stage

Writes a single human-readable summary text file plus a machine-readable
TSV with one row per category for the per-category breakdown.
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Snakemake I/O
# ---------------------------------------------------------------------------

merged_members_file   = snakemake.input['merged_members']      # genome-wide LD blocks (all categories)
all_traits_file        = snakemake.input['all_traits_enrichment']
directionality_file    = snakemake.input['directionality_summary']
per_snp_file            = snakemake.input['directionality_per_snp']

output_summary_txt = snakemake.output['summary_txt']
output_per_cat_tsv = snakemake.output['per_category_tsv']

os.makedirs(os.path.dirname(output_summary_txt), exist_ok=True)

lines = []
def log(s=''):
    print(s)
    lines.append(s)

log("="*80)
log("PIPELINE SUMMARY STATISTICS")
log("="*80)

# ---------------------------------------------------------------------------
# Stage 1: Genome-wide LD blocks (all categories, all SNPs incl. proxies)
# ---------------------------------------------------------------------------

log("\n--- Stage 1: Genome-wide LD blocks (category-level, GWAS + proxy SNPs) ---")

gw = pd.read_csv(merged_members_file, sep='\t')

# Columns expected: block_id, snp_id, ceu_id, is_gwas_snp, category (if present)
has_category = 'category' in gw.columns

n_unique_snps_gw   = gw['snp_id'].nunique()
n_unique_blocks_gw = gw['block_id'].nunique() if not has_category else gw.groupby('category')['block_id'].nunique().sum()
n_gwas_snps_gw     = gw[gw['is_gwas_snp'] == True]['snp_id'].nunique() if 'is_gwas_snp' in gw.columns else np.nan

log(f"  Unique SNPs (genome-wide, all categories, incl. proxies): {n_unique_snps_gw:,}")
if not np.isnan(n_gwas_snps_gw):
    log(f"  Unique GWAS SNPs (genome-wide, excl. proxies):              {n_gwas_snps_gw:,}")

per_cat_gw = None
if has_category:
    n_categories_gw = gw['category'].nunique()
    per_cat_gw = gw.groupby('category').agg(
        n_snps_total=('snp_id', 'nunique'),
        n_blocks=('block_id', 'nunique')
    ).reset_index()
    log(f"  Categories represented:                                     {n_categories_gw}")
    log(f"  Mean SNPs per category:                                     {per_cat_gw['n_snps_total'].mean():.1f}")
    log(f"  Median SNPs per category:                                   {per_cat_gw['n_snps_total'].median():.1f}")
    log(f"  Mean LD blocks per category:                                {per_cat_gw['n_blocks'].mean():.1f}")
    log(f"  Median LD blocks per category:                              {per_cat_gw['n_blocks'].median():.1f}")
else:
    log("  (No 'category' column found — per-category breakdown skipped for this stage)")

# ---------------------------------------------------------------------------
# Stage 2: After trait decomposition (top-N categories only)
# ---------------------------------------------------------------------------

log("\n--- Stage 2: Trait decomposition (top-N categories, all traits) ---")

td = pd.read_csv(all_traits_file, sep='\t')

n_unique_traits_td = td['trait'].nunique()
n_categories_td     = td['category'].nunique()

log(f"  Unique traits:                                              {n_unique_traits_td:,}")
log(f"  Categories represented:                                     {n_categories_td}")
if 'n_snps' in td.columns:
    log(f"  Mean SNPs per trait:                                        {td['n_snps'].mean():.1f}")
    log(f"  Median SNPs per trait:                                      {td['n_snps'].median():.1f}")

per_cat_td = (td.groupby('category')
                .agg(n_traits=('trait', 'nunique'),
                     mean_snps_per_trait=('n_snps', 'mean') if 'n_snps' in td.columns else ('trait', 'size'))
                .reset_index())
log(f"  Mean traits per category:                                   {per_cat_td['n_traits'].mean():.1f}")
log(f"  Median traits per category:                                 {per_cat_td['n_traits'].median():.1f}")

# ---------------------------------------------------------------------------
# Stage 3: Final PGS / directionality analysis (traits that passed all filters)
# ---------------------------------------------------------------------------

log("\n--- Stage 3: Final PGS / directionality analysis ---")

dir_df = pd.read_csv(directionality_file, sep='\t')
n_unique_traits_dir = dir_df['trait'].nunique()
n_categories_dir     = dir_df['category'].nunique()

log(f"  Unique traits analyzed:                                     {n_unique_traits_dir:,}")
log(f"  Categories represented:                                     {n_categories_dir}")

if 'n_snps_extracted' in dir_df.columns:
    log(f"  Mean SNPs per trait (extracted/used in PGS):                {dir_df['n_snps_extracted'].mean():.1f}")
    log(f"  Median SNPs per trait (extracted/used in PGS):              {dir_df['n_snps_extracted'].median():.1f}")
    log(f"  Min / Max SNPs per trait:                                   {dir_df['n_snps_extracted'].min()} / {dir_df['n_snps_extracted'].max()}")

per_cat_dir = (dir_df.groupby('category')
                  .agg(n_traits=('trait', 'nunique'),
                       mean_snps_per_trait=('n_snps_extracted', 'mean') if 'n_snps_extracted' in dir_df.columns else ('trait', 'size'),
                       median_snps_per_trait=('n_snps_extracted', 'median') if 'n_snps_extracted' in dir_df.columns else ('trait', 'size'))
                  .reset_index())

# Unique SNPs actually used across all final PGS calculations (deduplicated genome-wide)
per_snp = pd.read_csv(per_snp_file, sep='\t')
n_unique_snps_used = per_snp['snp_id'].nunique() if 'snp_id' in per_snp.columns else np.nan
n_total_snp_trait_rows = len(per_snp)

log(f"\n  Unique SNPs used across ALL final PGS calculations (deduplicated genome-wide): {n_unique_snps_used:,}")
log(f"  Total SNP-trait pairs (same SNP can be reused across traits):                  {n_total_snp_trait_rows:,}")

# ---------------------------------------------------------------------------
# Merge per-category tables for output
# ---------------------------------------------------------------------------

per_cat_final = per_cat_dir.rename(columns={
    'n_traits':              'n_traits_final',
    'mean_snps_per_trait':   'mean_snps_per_trait_final',
    'median_snps_per_trait': 'median_snps_per_trait_final',
})

if per_cat_gw is not None:
    per_cat_final = per_cat_final.merge(
        per_cat_gw.rename(columns={
            'n_snps_total': 'n_snps_genome_wide',
            'n_blocks':     'n_blocks_genome_wide'
        }),
        on='category', how='left'
    )

per_cat_final = per_cat_final.sort_values('n_traits_final', ascending=False)

# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

with open(output_summary_txt, 'w') as f:
    f.write('\n'.join(lines) + '\n')

per_cat_final.to_csv(output_per_category_tsv := output_per_cat_tsv, sep='\t', index=False)

log(f"\n{'='*80}")
log(f"Saved: {output_summary_txt}")
log(f"Saved: {output_per_cat_tsv}")
log("="*80)

# Re-write summary file including the final log lines above
with open(output_summary_txt, 'w') as f:
    f.write('\n'.join(lines) + '\n')