#!/usr/bin/env python3
"""
Merge Trait Enrichment Results
Combines trait-level block assignment summaries from all top-N categories.

Since decompose_category_traits.py no longer calculates FST or p-values,
this script simply concatenates the per-category trait_enrichment.txt files
and sorts by category then trait name.
"""

import pandas as pd
import os
import sys

print("="*80)
print("MERGING TRAIT ENRICHMENT RESULTS")
print("="*80)

trait_files    = snakemake.input['trait_files']
output_summary = snakemake.output['summary']

print(f"\nInput files: {len(trait_files)}")

all_traits = []
for trait_file in trait_files:
    if not os.path.exists(trait_file):
        print(f"  WARNING: File not found: {trait_file}")
        continue
    try:
        df = pd.read_csv(trait_file, sep='\t')
        if len(df) > 0:
            all_traits.append(df)
            print(f"  ✓ {os.path.basename(os.path.dirname(trait_file))}: {len(df)} traits")
    except pd.errors.EmptyDataError:
        print(f"  WARNING: Empty file: {trait_file}")
    except Exception as e:
        print(f"  ERROR reading {trait_file}: {str(e)}")

if not all_traits:
    print("\nWARNING: No trait data to merge!")
    pd.DataFrame().to_csv(output_summary, sep='\t', index=False)
    sys.exit(0)

merged_df = pd.concat(all_traits, ignore_index=True)

# Sort by category then trait name
merged_df = merged_df.sort_values(['category', 'trait']).reset_index(drop=True)

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"\nTotal traits: {len(merged_df)}")
print(f"Categories:   {merged_df['category'].nunique()}")

print(f"\nTraits per category:")
for cat, grp in merged_df.groupby('category'):
    print(f"  {cat}: {len(grp)} traits  "
          f"(median blocks={grp['n_blocks'].median():.0f}, "
          f"median GWAS SNPs={grp['n_gwas_snps'].median():.0f})")

os.makedirs(os.path.dirname(output_summary), exist_ok=True)
merged_df.to_csv(output_summary, sep='\t', index=False)
print(f"\n✓ Saved: {output_summary}")
print("="*80)