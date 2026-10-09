#!/usr/bin/env python3
"""
Identify Significant Categories
Filters category enrichment results to find categories with FDR < threshold
"""

import pandas as pd
import os

print("="*80)
print("IDENTIFYING SIGNIFICANT CATEGORIES")
print("="*80)

# Get inputs from Snakemake
enrichment_file = snakemake.input['enrichment_results']
output_list = snakemake.output['significant_categories']

# Parameters
fdr_threshold = snakemake.params.get('fdr_threshold', 0.05)

print(f"\nInputs:")
print(f"  Enrichment results: {enrichment_file}")
print(f"  FDR threshold: {fdr_threshold}")

# Load enrichment results
enrichment_df = pd.read_csv(enrichment_file, sep='\t')
print(f"\nTotal categories tested: {len(enrichment_df)}")

# Filter to significant categories
sig_df = enrichment_df[enrichment_df['mannwhitneyu_fdr'] < fdr_threshold].copy()

print(f"Significant categories (FDR < {fdr_threshold}): {len(sig_df)}")

if len(sig_df) == 0:
    print("\nWARNING: No significant categories found!")
    print("No trait decomposition will be performed.")
    # Create empty output
    with open(output_list, 'w') as f:
        f.write("# No significant categories\n")
else:
    # Print significant categories
    print(f"\nSignificant categories:")
    for idx, row in sig_df.iterrows():
        print(f"  {row['category']}")
        print(f"    Mean Fst: {row['mean_fst']:.6f} vs {row['genome_mean_fst']:.6f}")
        print(f"    FDR: {row['mannwhitneyu_fdr']:.3e}")
        print(f"    Blocks: {row['n_blocks']:,}")
    
    # Save category list
    os.makedirs(os.path.dirname(output_list), exist_ok=True)
    with open(output_list, 'w') as f:
        for cat in sig_df['category'].tolist():
            f.write(f"{cat}\n")
    
    print(f"\n✓ Saved significant categories to: {output_list}")

print("="*80)