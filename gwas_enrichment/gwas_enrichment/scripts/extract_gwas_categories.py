#!/usr/bin/env python3
"""
Extract GWAS Categories - Exact Match Version
Matches GWAS catalog traits against exact trait names from categories.yaml
"""

import pandas as pd
import sys
import os

# Import category utilities
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from category_utils import load_categories, sanitize_category_name

print("="*80)
print("EXTRACTING GWAS CATEGORIES (EXACT MATCH)")
print("="*80)

# Read inputs from Snakemake
gwas_catalog_file = snakemake.input['gwas_catalog']
categories_yaml = snakemake.input['categories_yaml']
category_list_out = snakemake.output['category_list']
category_snps_out = snakemake.output['category_snps']

min_pvalue = float(snakemake.params['min_pvalue'])
ancestry_filter_str = snakemake.params.get('ancestry_filter', '')
ancestry_filter = [x.strip() for x in ancestry_filter_str.split(',') if x.strip()]
min_snps_per_category = int(snakemake.params.get('min_snps_per_category', 30))

print(f"\nInputs:")
print(f"  GWAS catalog: {gwas_catalog_file}")
print(f"  Categories YAML: {categories_yaml}")

# Load categories from YAML
print(f"\nLoading categories...")
categories = load_categories(categories_yaml)

print(f"✓ Loaded {len(categories)} categories from YAML")

# Load GWAS catalog
print(f"\nLoading GWAS catalog...")
gwas_df = pd.read_csv(gwas_catalog_file, sep='\t', low_memory=False)
print(f"  Total associations: {len(gwas_df):,}")

# Apply filters
if ancestry_filter:
    mask = gwas_df['INITIAL SAMPLE SIZE'].str.contains('|'.join(ancestry_filter), case=False, na=False)
    gwas_df = gwas_df[mask].copy()
    print(f"  After ancestry filter: {len(gwas_df):,}")

gwas_df['P-VALUE'] = pd.to_numeric(gwas_df['P-VALUE'], errors='coerce')
gwas_df = gwas_df[gwas_df['P-VALUE'].notna() & (gwas_df['P-VALUE'] < min_pvalue)].copy()
print(f"  After p-value filter (p<{min_pvalue}): {len(gwas_df):,}")

gwas_df = gwas_df[
    gwas_df['STRONGEST SNP-RISK ALLELE'].notna() & 
    gwas_df['STRONGEST SNP-RISK ALLELE'].str.contains('-', na=False)
].copy()
print(f"  After risk allele filter: {len(gwas_df):,}")

# Create chr:pos
gwas_df = gwas_df[gwas_df['CHR_ID'].notna() & gwas_df['CHR_POS'].notna()].copy()
gwas_df['chr_num'] = pd.to_numeric(gwas_df['CHR_ID'], errors='coerce')
gwas_df = gwas_df[gwas_df['chr_num'].notna()].copy()
gwas_df['CHR_POS'] = pd.to_numeric(gwas_df['CHR_POS'], errors='coerce')
gwas_df = gwas_df[gwas_df['CHR_POS'].notna()].copy()

print(f"  Final associations with positions: {len(gwas_df):,}")
print(f"  Unique traits: {gwas_df['DISEASE/TRAIT'].nunique():,}")

# Normalize GWAS traits for matching (case-insensitive, stripped)
gwas_df['trait_normalized'] = gwas_df['DISEASE/TRAIT'].str.lower().str.strip()

print(f"\n{'='*80}")
print(f"CATEGORIZING TRAITS (EXACT MATCH)")
print(f"{'='*80}\n")

# Process each category
category_summary = []

for category_name, trait_list in categories.items():
    # Normalize trait list for matching
    trait_list_normalized = {trait.lower().strip(): trait for trait in trait_list}
    
    # Find matching GWAS entries
    mask = gwas_df['trait_normalized'].isin(trait_list_normalized.keys())
    category_df = gwas_df[mask].copy()
    
    if len(category_df) == 0:
        continue
    
    n_associations = len(category_df)
    n_traits = category_df['DISEASE/TRAIT'].nunique()
    n_snps = category_df[['CHR_ID', 'CHR_POS']].drop_duplicates().shape[0]
    
    # Create safe filename using utility function
    safe_name = sanitize_category_name(category_name)
    
    category_summary.append({
        'category': category_name,
        'safe_name': safe_name,
        'n_associations': n_associations,
        'n_traits': n_traits,
        'n_snps': n_snps,
        'trait_list_size': len(trait_list)
    })
    
    print(f"{category_name}")
    print(f"  Traits defined: {len(trait_list)}, Traits matched: {n_traits}, Associations: {n_associations:,}, SNPs: {n_snps:,}")

# Filter categories by minimum SNP count
print(f"\n{'='*80}")
print(f"FILTERING CATEGORIES")
print(f"{'='*80}\n")

print(f"Minimum SNPs per category: {min_snps_per_category}")
print(f"Categories before filtering: {len(category_summary)}")

category_summary_filtered = [c for c in category_summary if c['n_snps'] >= min_snps_per_category]

print(f"Categories after filtering: {len(category_summary_filtered)}")
print(f"Removed {len(category_summary) - len(category_summary_filtered)} categories with < {min_snps_per_category} SNPs")

if len(category_summary_filtered) < len(category_summary):
    print(f"\nRemoved categories:")
    for c in category_summary:
        if c['n_snps'] < min_snps_per_category:
            print(f"  - {c['category']} ({c['n_snps']} SNPs)")

# Save category list
categories_with_data = [c['safe_name'] for c in category_summary_filtered]

os.makedirs(os.path.dirname(category_list_out), exist_ok=True)
with open(category_list_out, 'w') as f:
    for cat_safe in categories_with_data:
        f.write(f"{cat_safe}\n")

print(f"\n✓ Saved {len(categories_with_data)} categories to: {category_list_out}")

# Save summary
summary_df = pd.DataFrame(category_summary)
summary_df = summary_df.sort_values('n_snps', ascending=False)
summary_df.to_csv(category_snps_out, sep='\t', index=False)

print(f"✓ Saved summary to: {category_snps_out}")

print("\n" + "="*80)
print(f"EXTRACTION COMPLETE")
print(f"  Categories (passed filter): {len(categories_with_data)}")
print(f"  Total SNPs across categories: {sum(c['n_snps'] for c in category_summary_filtered):,}")
print("="*80)