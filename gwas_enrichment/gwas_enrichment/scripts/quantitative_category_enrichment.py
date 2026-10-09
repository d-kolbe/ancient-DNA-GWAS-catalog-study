#!/usr/bin/env python3
"""
Quantitative Category Enrichment Analysis
Tests whether GWAS trait categories show elevated Fst compared to genome-wide background
Uses Mann-Whitney U and Kolmogorov-Smirnov tests on LD block maximum Fst values
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

print("="*80)
print("QUANTITATIVE CATEGORY ENRICHMENT (LD BLOCK-BASED)")
print("="*80)

# Get inputs
fst_file = snakemake.input['fst_results']
blocks_file = snakemake.input['merged_blocks']
members_file = snakemake.input['merged_members']

# Get outputs
enrichment_out = snakemake.output['enrichment']
distributions_out = snakemake.output['distributions']
blocks_with_fst_out = snakemake.output['blocks_with_fst']

# Get parameters
min_blocks = snakemake.params.get('min_blocks', 5)
block_fst_metric = snakemake.params.get('block_fst_metric', 'max')
seed = int(snakemake.params.get('seed', 42))
rng = np.random.default_rng(seed)

print(f"\nInputs:")
print(f"  Fst results: {fst_file}")
print(f"  LD blocks: {blocks_file}")
print(f"  Block members: {members_file}")
print(f"\nParameters:")
print(f"  Minimum blocks per category: {min_blocks}")
print(f"  Block Fst metric: {block_fst_metric}")

# =========================================================================
# Load data
# =========================================================================

print(f"\n{'='*80}")
print("LOADING DATA")
print(f"{'='*80}")

# Load Fst data
print(f"\nLoading Fst results...")
fst_df = pd.read_csv(fst_file, sep='\t')
print(f"  Total SNPs: {len(fst_df):,}")

# Check column names
if 'CHROM' in fst_df.columns and 'POS' in fst_df.columns and 'WEIR_AND_COCKERHAM_FST' in fst_df.columns:
    # vcftools format
    fst_df['chr'] = fst_df['CHROM']
    fst_df['pos'] = fst_df['POS']
    fst_df['fst'] = fst_df['WEIR_AND_COCKERHAM_FST']
elif 'CHR' in fst_df.columns:
    # Alternative format
    fst_df['chr'] = fst_df['CHR']
    fst_df['pos'] = fst_df['POS']
    fst_df['fst'] = fst_df['FST']
else:
    print(f"  Columns: {fst_df.columns.tolist()}")
    raise ValueError("Unexpected Fst file format")

# Create chr:pos identifier
fst_df['snp_id'] = fst_df['chr'].astype(str) + ':' + fst_df['pos'].astype(int).astype(str)

# Handle missing and negative Fst values
n_missing = fst_df['fst'].isna().sum()
n_negative = (fst_df['fst'] < 0).sum()

print(f"  Missing Fst: {n_missing:,}")
print(f"  Negative Fst: {n_negative:,} (excluded)")

# Remove missing and negative Fst values
fst_df = fst_df[fst_df['fst'].notna()].copy()
fst_df = fst_df[fst_df['fst'].notna() & (fst_df['fst'] >= 0)].copy()

print(f"  Valid Fst values: {len(fst_df):,}")
print(f"  Mean Fst: {fst_df['fst'].mean():.6f}")
print(f"  Median Fst: {fst_df['fst'].median():.6f}")

# Load LD blocks
print(f"\nLoading LD blocks...")
blocks_df = pd.read_csv(blocks_file, sep='\t')
print(f"  Total blocks: {len(blocks_df):,}")
print(f"  Categories: {blocks_df['category'].nunique():,}")

# Load block members
print(f"\nLoading block members...")
members_df = pd.read_csv(members_file, sep='\t')
print(f"  Total member rows: {len(members_df):,}")

if 'is_gwas_snp' in members_df.columns:
    gwas_count = (members_df['is_gwas_snp'] == True).sum()
    proxy_count = (members_df['is_gwas_snp'] == False).sum()
    print(f"  GWAS SNPs: {gwas_count:,}")
    print(f"  Proxy SNPs: {proxy_count:,}")

# =========================================================================
# Assign Fst to block members
# =========================================================================

print(f"\n{'='*80}")
print("ASSIGNING FST TO BLOCKS")
print(f"{'='*80}")

# Merge Fst values with block members
members_with_fst = members_df.merge(
    fst_df[['snp_id', 'fst']],
    on='snp_id',
    how='left'
)

print(f"\nSNPs with Fst data: {members_with_fst['fst'].notna().sum():,} / {len(members_with_fst):,}")

# Calculate block Fst using specified metric (to avoid pseudo-replication from LD)
print(f"\nCalculating {block_fst_metric} Fst per block...")

if block_fst_metric == 'max':
    block_fst_summary = members_with_fst.groupby('block_id')['fst'].max().reset_index()
elif block_fst_metric == 'mean':
    block_fst_summary = members_with_fst.groupby('block_id')['fst'].mean().reset_index()
elif block_fst_metric == 'median':
    block_fst_summary = members_with_fst.groupby('block_id')['fst'].median().reset_index()
else:
    raise ValueError(f"Unknown block_fst_metric: {block_fst_metric}. Must be 'max', 'mean', or 'median'")

block_fst_summary.columns = ['block_id', 'block_fst']

# Merge back to get category info
blocks_with_fst = blocks_df.merge(block_fst_summary, on='block_id', how='left')

# Remove blocks with no Fst data
blocks_with_fst = blocks_with_fst[blocks_with_fst['block_fst'].notna()].copy()
print(f"\nBlocks with Fst data: {len(blocks_with_fst):,}")
print(f"  Using {block_fst_metric} Fst per block")

# =========================================================================
# Test enrichment per category
# =========================================================================

print(f"\n{'='*80}")
print("TESTING CATEGORY ENRICHMENT")
print(f"{'='*80}\n")

# Get genome-wide background (all blocks with Fst)
genome_wide_fst = blocks_with_fst['block_fst'].values
print(f"Genome-wide background ({block_fst_metric} Fst per block):")
print(f"  Blocks: {len(genome_wide_fst):,}")
print(f"  Mean Fst: {np.mean(genome_wide_fst):.6f}")
print(f"  Median Fst: {np.median(genome_wide_fst):.6f}")
print(f"  SD: {np.std(genome_wide_fst):.6f}")

# Test each category
enrichment_results = []
distribution_data = []

categories = blocks_with_fst['category'].unique()
print(f"\nTesting {len(categories)} categories...\n")

for category in sorted(categories):
    cat_blocks = blocks_with_fst[blocks_with_fst['category'] == category]
    cat_fst = cat_blocks['block_fst'].values
    
    n_blocks = len(cat_fst)
    
    # Skip if too few blocks
    if n_blocks < min_blocks:
        print(f"  SKIP: {category} (only {n_blocks} blocks)")
        continue
    
    # Calculate statistics
    cat_mean = np.mean(cat_fst)
    cat_median = np.median(cat_fst)
    cat_sd = np.std(cat_fst)
    
    # Mann-Whitney U test (one-sided: category > background)
    u_stat, mw_pval_two = stats.mannwhitneyu(cat_fst, genome_wide_fst, alternative='two-sided')
    _, mw_pval_greater = stats.mannwhitneyu(cat_fst, genome_wide_fst, alternative='greater')
    
    # Kolmogorov-Smirnov test (two-sided)
    ks_stat, ks_pval = stats.ks_2samp(cat_fst, genome_wide_fst)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(cat_fst) + np.var(genome_wide_fst)) / 2)
    cohens_d = (cat_mean - np.mean(genome_wide_fst)) / pooled_std if pooled_std > 0 else 0
    
    # Store results
    enrichment_results.append({
        'category': category,
        'n_blocks': n_blocks,
        'mean_fst': cat_mean,
        'median_fst': cat_median,
        'sd_fst': cat_sd,
        'genome_mean_fst': np.mean(genome_wide_fst),
        'genome_median_fst': np.median(genome_wide_fst),
        'fold_enrichment': cat_mean / np.mean(genome_wide_fst) if np.mean(genome_wide_fst) > 0 else np.nan,
        'cohens_d': cohens_d,
        'mannwhitneyu_pval': mw_pval_greater,
        'mannwhitneyu_pval_twosided': mw_pval_two,
        'kolmogorov_smirnov_stat': ks_stat,
        'kolmogorov_smirnov_pval': ks_pval
    })
    
    # Store distribution data for plotting
    for fst_val in cat_fst:
        distribution_data.append({
            'category': category,
            'fst': fst_val,
            'is_genome_wide': False
        })
    
    # Print summary
    sig_marker = "***" if mw_pval_greater < 0.001 else "**" if mw_pval_greater < 0.01 else "*" if mw_pval_greater < 0.05 else ""
    print(f"  {category}")
    print(f"    Blocks: {n_blocks:,}, Mean Fst: {cat_mean:.6f}, MW p-value: {mw_pval_greater:.3e} {sig_marker}")

# Add genome-wide background to distribution data
for fst_val in rng.choice(genome_wide_fst, min(10000, len(genome_wide_fst)), replace=False):
    distribution_data.append({
        'category': 'Genome_Wide_Background',
        'fst': fst_val,
        'is_genome_wide': True
    })

# =========================================================================
# Multiple testing correction
# =========================================================================

print(f"\n{'='*80}")
print("MULTIPLE TESTING CORRECTION")
print(f"{'='*80}")

enrichment_df = pd.DataFrame(enrichment_results)

if len(enrichment_df) > 0:
    # Bonferroni correction
    enrichment_df['mannwhitneyu_pval_bonf'] = enrichment_df['mannwhitneyu_pval'] * len(enrichment_df)
    enrichment_df['mannwhitneyu_pval_bonf'] = enrichment_df['mannwhitneyu_pval_bonf'].clip(upper=1.0)
    
    # Benjamini-Hochberg FDR
    from scipy.stats import false_discovery_control
    enrichment_df['mannwhitneyu_fdr'] = false_discovery_control(enrichment_df['mannwhitneyu_pval'].values)
    
    # Sort by p-value
    enrichment_df = enrichment_df.sort_values('mannwhitneyu_pval')
    
    # Count significant results
    n_sig_raw = (enrichment_df['mannwhitneyu_pval'] < 0.05).sum()
    n_sig_bonf = (enrichment_df['mannwhitneyu_pval_bonf'] < 0.05).sum()
    n_sig_fdr = (enrichment_df['mannwhitneyu_fdr'] < 0.05).sum()
    
    print(f"\nCategories tested: {len(enrichment_df)}")
    print(f"Significant (p < 0.05, uncorrected): {n_sig_raw}")
    print(f"Significant (Bonferroni p < 0.05): {n_sig_bonf}")
    print(f"Significant (FDR < 0.05): {n_sig_fdr}")

# =========================================================================
# Save results
# =========================================================================

print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

os.makedirs(os.path.dirname(enrichment_out), exist_ok=True)

enrichment_df.to_csv(enrichment_out, sep='\t', index=False)
print(f"\n✓ Enrichment results: {enrichment_out}")

distribution_df = pd.DataFrame(distribution_data)
distribution_df.to_csv(distributions_out, sep='\t', index=False)
print(f"✓ Fst distributions: {distributions_out}")

# Save blocks with Fst for downstream trait decomposition
blocks_with_fst.to_csv(blocks_with_fst_out, sep='\t', index=False)
print(f"✓ Blocks with Fst: {blocks_with_fst_out}")

# Print top results
print(f"\n{'='*80}")
print("TOP ENRICHED CATEGORIES (by p-value)")
print(f"{'='*80}\n")

if len(enrichment_df) > 0:
    top_cats = enrichment_df.head(10)
    for idx, row in top_cats.iterrows():
        print(f"{row['category']}")
        print(f"  Mean Fst: {row['mean_fst']:.6f} vs genome {row['genome_mean_fst']:.6f}")
        print(f"  Fold enrichment: {row['fold_enrichment']:.3f}x")
        print(f"  Mann-Whitney U p-value: {row['mannwhitneyu_pval']:.3e}")
        print(f"  FDR: {row['mannwhitneyu_fdr']:.3e}")
        print()

print(f"{'='*80}")
