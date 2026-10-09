#!/usr/bin/env python3
"""
Decompose Category into Trait-Level Block Assignments

Simplified version: reads pre-computed category LD blocks (block_members.txt)
and writes trait_blocks.txt mapping each trait's GWAS SNPs to their block IDs.

No PLINK calls. No FST calculation.
Filtering by min SNPs / min blocks happens downstream in
calculate_trait_directionality.py.

Outputs:
  trait_blocks.txt    — one row per SNP, with trait, block_id, snp_id, is_gwas_snp
  trait_enrichment.txt — one row per trait with basic SNP counts (for merge step)
"""

import pandas as pd
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from category_utils import load_categories, sanitize_category_name

print("="*80)
category_safe = snakemake.params['category']
print(f"TRAIT DECOMPOSITION FOR CATEGORY: {category_safe}")
print("="*80)

# ---------------------------------------------------------------------------
# Inputs / outputs / params
# ---------------------------------------------------------------------------

gwas_catalog_file  = snakemake.input['gwas_catalog']
categories_yaml    = snakemake.input['categories_yaml']
block_members_file = snakemake.input['block_members']

output_enrichment  = snakemake.output['trait_enrichment']
output_blocks      = snakemake.output['trait_blocks']

min_pvalue          = float(snakemake.params['min_pvalue'])
ancestry_filter_str = snakemake.params.get('ancestry_filter', '')
ancestry_filter     = [x.strip() for x in ancestry_filter_str.split(',') if x.strip()]

os.makedirs(os.path.dirname(output_enrichment), exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 0: Load category trait list from YAML
# ---------------------------------------------------------------------------

print(f"\n[0/3] Loading category from YAML...")

categories = load_categories(categories_yaml)

category_name = None
trait_list    = None
for cat_name, traits in categories.items():
    if sanitize_category_name(cat_name) == category_safe:
        category_name = cat_name
        trait_list    = traits
        break

if trait_list is None:
    print(f"ERROR: Category '{category_safe}' not found in YAML!")
    sys.exit(1)

print(f"  ✓ Category: {category_name}")
print(f"  ✓ Traits defined: {len(trait_list)}")

# ---------------------------------------------------------------------------
# STEP 1: Load pre-computed category LD blocks
# ---------------------------------------------------------------------------

print(f"\n[1/3] Loading pre-computed LD blocks from {block_members_file}...")

block_members = pd.read_csv(block_members_file, sep='\t')
print(f"  ✓ Loaded {len(block_members):,} block member rows")
print(f"  ✓ Unique blocks: {block_members['block_id'].nunique():,}")
print(f"  ✓ GWAS SNPs: {block_members['is_gwas_snp'].sum():,}")

# Build lookup: chrpos -> block_id
chrpos_to_block = dict(zip(block_members['snp_id'], block_members['block_id']))

# ---------------------------------------------------------------------------
# STEP 2: Load GWAS catalog and filter to this category's traits
# ---------------------------------------------------------------------------

print(f"\n[2/3] Loading GWAS catalog...")

gwas_df = pd.read_csv(gwas_catalog_file, sep='\t', low_memory=False)

if ancestry_filter:
    mask    = gwas_df['INITIAL SAMPLE SIZE'].str.contains(
                  '|'.join(ancestry_filter), case=False, na=False)
    gwas_df = gwas_df[mask].copy()

gwas_df['P-VALUE'] = pd.to_numeric(gwas_df['P-VALUE'], errors='coerce')
gwas_df = gwas_df[gwas_df['P-VALUE'].notna() & (gwas_df['P-VALUE'] < min_pvalue)].copy()

gwas_df = gwas_df[
    gwas_df['STRONGEST SNP-RISK ALLELE'].notna() &
    gwas_df['STRONGEST SNP-RISK ALLELE'].str.contains('-', na=False)
].copy()

gwas_df = gwas_df[gwas_df['CHR_ID'].notna() & gwas_df['CHR_POS'].notna()].copy()
gwas_df['chr_num'] = pd.to_numeric(gwas_df['CHR_ID'], errors='coerce')
gwas_df = gwas_df[gwas_df['chr_num'].notna()].copy()
gwas_df['CHR_POS'] = pd.to_numeric(gwas_df['CHR_POS'], errors='coerce')
gwas_df = gwas_df[gwas_df['CHR_POS'].notna()].copy()
gwas_df['chrpos'] = (gwas_df['chr_num'].astype(int).astype(str) + ':' +
                     gwas_df['CHR_POS'].astype(int).astype(str))

gwas_df['trait_normalized'] = gwas_df['DISEASE/TRAIT'].str.lower().str.strip()

print(f"  ✓ GWAS catalog loaded: {len(gwas_df):,} associations after filters")

# ---------------------------------------------------------------------------
# STEP 3: For each trait, map GWAS SNPs to pre-computed blocks
# ---------------------------------------------------------------------------

print(f"\n[3/3] Mapping traits to pre-computed blocks...")

trait_blocks_rows = []
trait_enrichment_rows = []

for trait_name in trait_list:
    trait_norm = trait_name.lower().strip()
    trait_gwas = gwas_df[gwas_df['trait_normalized'] == trait_norm].copy()

    if trait_gwas.empty:
        continue

    gwas_snps = set(trait_gwas['chrpos'].unique())

    # Find which GWAS SNPs have a pre-computed block assignment
    matched = {cp: chrpos_to_block[cp] for cp in gwas_snps if cp in chrpos_to_block}

    if not matched:
        print(f"  {trait_name}: no GWAS SNPs matched to blocks — skipping")
        continue

    # Get all block IDs that contain at least one GWAS SNP for this trait
    trait_block_ids = set(matched.values())

    # Pull all SNPs (GWAS + proxies) from those blocks
    trait_block_members = block_members[block_members['block_id'].isin(trait_block_ids)].copy()

    n_gwas   = len(matched)
    n_blocks = len(trait_block_ids)
    n_snps   = len(trait_block_members)

    print(f"  {trait_name}: {n_gwas} GWAS SNPs → {n_blocks} blocks → {n_snps} total SNPs")

    # Write block membership rows for this trait
    for _, row in trait_block_members.iterrows():
        trait_blocks_rows.append({
            'trait':       trait_name,
            'trait_safe':  sanitize_category_name(trait_name),
            'block_id':    row['block_id'],
            'snp_id':      row['snp_id'],
            'ceu_id':      row['ceu_id'],
            'is_gwas_snp': row['snp_id'] in gwas_snps
        })

    # Basic summary row for the merge step
    trait_enrichment_rows.append({
        'category':    category_name,
        'trait':       trait_name,
        'trait_safe':  sanitize_category_name(trait_name),
        'n_gwas_snps': n_gwas,
        'n_blocks':    n_blocks,
        'n_snps':      n_snps,
    })

# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

blocks_df = pd.DataFrame(trait_blocks_rows)
blocks_df.to_csv(output_blocks, sep='\t', index=False)
print(f"\n✓ trait_blocks.txt: {len(blocks_df):,} rows, "
      f"{blocks_df['trait'].nunique()} traits")

enrichment_df = pd.DataFrame(trait_enrichment_rows)
enrichment_df.to_csv(output_enrichment, sep='\t', index=False)
print(f"✓ trait_enrichment.txt: {len(enrichment_df)} traits")

print("="*80)