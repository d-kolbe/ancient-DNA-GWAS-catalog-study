#!/usr/bin/env python3
"""
Per-Category LD Block Generation - Exact Match Version
Finds GWAS SNPs + LD proxies, creates independent LD blocks
Uses exact trait matching instead of regex

Fix: added --ld-window 999999 to both PLINK LD calls so that the SNP-count
window (default=10) does not silently prevent R² computation between variants
that are within the physical --ld-window-kb limit but more than 10 SNPs apart
in the extracted dataset. Both filters apply simultaneously in PLINK; omitting
--ld-window caused high-LD SNPs in dense regions (e.g. FADS2) to be assigned
to separate blocks.
"""

import pandas as pd
import numpy as np
import os
import sys
import subprocess
import tempfile
import networkx as nx

# Import category utilities
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from category_utils import load_categories, sanitize_category_name

print("="*80)
category_safe = snakemake.params['category']
print(f"CATEGORY: {category_safe}")
print("="*80)

# Get inputs
vcf_file        = snakemake.input['vcf']
gwas_catalog_file = snakemake.input['gwas_catalog']
categories_yaml = snakemake.input['categories_yaml']

# Parameters
r2_threshold       = snakemake.params['r2_threshold']
window_kb          = snakemake.params['window_kb']
ref_panel          = snakemake.params['ref_panel']
min_pvalue         = float(snakemake.params['min_pvalue'])
ancestry_filter_str = snakemake.params.get('ancestry_filter', '')
ancestry_filter    = [x.strip() for x in ancestry_filter_str.split(',') if x.strip()]

# Outputs
output_blocks  = snakemake.output['ld_blocks']
output_members = snakemake.output['block_members']
output_stats   = snakemake.output['stats']

print(f"Parameters:")
print(f"  R² threshold: {r2_threshold}")
print(f"  Window: {window_kb} kb")
print(f"  Reference: {ref_panel}")

# =========================================================================
# STEP 0: Load category from YAML
# =========================================================================

print(f"[0/6] Loading category from YAML...")

categories = load_categories(categories_yaml)

category_name = None
trait_list    = None
for cat_name, traits in categories.items():
    if sanitize_category_name(cat_name) == category_safe:
        category_name = cat_name
        trait_list    = traits
        break

if trait_list is None:
    print(f"ERROR: Category '{category_safe}' not found!")
    sys.exit(1)

print(f"  ✓ Category: {category_name}")
print(f"  ✓ Traits: {len(trait_list)}")

# =========================================================================
# STEP 1: Load and filter GWAS catalog for this category
# =========================================================================

print("[1/6] Loading GWAS catalog and filtering for category...")

gwas_df = pd.read_csv(gwas_catalog_file, sep='\t', low_memory=False)

if ancestry_filter:
    mask   = gwas_df['INITIAL SAMPLE SIZE'].str.contains('|'.join(ancestry_filter), case=False, na=False)
    gwas_df = gwas_df[mask].copy()

gwas_df['P-VALUE'] = pd.to_numeric(gwas_df['P-VALUE'], errors='coerce')
gwas_df = gwas_df[gwas_df['P-VALUE'].notna() & (gwas_df['P-VALUE'] < min_pvalue)].copy()
gwas_df = gwas_df[
    gwas_df['STRONGEST SNP-RISK ALLELE'].notna() &
    gwas_df['STRONGEST SNP-RISK ALLELE'].str.contains('-', na=False)
].copy()

gwas_df['trait_normalized']  = gwas_df['DISEASE/TRAIT'].str.lower().str.strip()
trait_list_normalized        = [t.lower().strip() for t in trait_list]
category_gwas                = gwas_df[gwas_df['trait_normalized'].isin(trait_list_normalized)].copy()

print(f"  ✓ Category associations: {len(category_gwas):,}")

if len(category_gwas) == 0:
    print("ERROR: No GWAS associations for this category")
    sys.exit(1)

category_gwas = category_gwas[category_gwas['CHR_ID'].notna() & category_gwas['CHR_POS'].notna()].copy()
category_gwas['chr_num'] = pd.to_numeric(category_gwas['CHR_ID'], errors='coerce')
category_gwas = category_gwas[category_gwas['chr_num'].notna()].copy()
category_gwas['CHR_POS'] = pd.to_numeric(category_gwas['CHR_POS'], errors='coerce')
category_gwas = category_gwas[category_gwas['CHR_POS'].notna()].copy()
category_gwas['chrpos'] = (category_gwas['chr_num'].astype(int).astype(str) + ':' +
                            category_gwas['CHR_POS'].astype(int).astype(str))

gwas_variants = set(category_gwas['chrpos'].unique())
print(f"  ✓ Unique GWAS SNPs: {len(gwas_variants):,}")

# =========================================================================
# STEP 2: Load ancient VCF positions
# =========================================================================

print("[2/6] Loading ancient VCF positions...")

cmd    = ['bcftools', 'query', '-f', '%CHROM:%POS\n', vcf_file]
result = subprocess.run(cmd, capture_output=True, text=True, check=True)

ancient_variants  = set(result.stdout.strip().split('\n'))
print(f"  ✓ Ancient variants: {len(ancient_variants):,}")

gwas_in_ancient = gwas_variants & ancient_variants
print(f"  ✓ GWAS SNPs in ancient: {len(gwas_in_ancient):,}")

if len(gwas_in_ancient) == 0:
    print("ERROR: No GWAS SNPs in ancient VCF")
    sys.exit(1)

# =========================================================================
# STEP 3: Load CEU reference panel
# =========================================================================

print("[3/6] Loading CEU reference panel...")

ceu_bim = pd.read_csv(f"{ref_panel}.bim", sep='\t', header=None,
                      names=['chr', 'snp_id', 'cm', 'pos', 'a1', 'a2'])

print(f"  ✓ CEU SNPs: {len(ceu_bim):,}")

def extract_chrpos(snp_id):
    parts = str(snp_id).split(':')
    return f"{parts[0]}:{parts[1]}" if len(parts) >= 2 else None

ceu_bim['chrpos'] = ceu_bim['snp_id'].apply(extract_chrpos)
ceu_bim = ceu_bim[ceu_bim['chrpos'].notna()].copy()

gwas_in_ceu = gwas_in_ancient & set(ceu_bim['chrpos'].unique())
print(f"  ✓ GWAS SNPs in CEU: {len(gwas_in_ceu):,}")

if len(gwas_in_ceu) == 0:
    print("ERROR: No category SNPs in CEU reference")
    sys.exit(1)

# =========================================================================
# STEP 4: LD expansion — find proxy SNPs for all GWAS SNPs
# =========================================================================
# FIX: --ld-window 999999 added so PLINK does not silently skip pairs that
# are within --ld-window-kb but more than 10 SNP positions apart (the
# undocumented default). Both limits apply simultaneously; omitting
# --ld-window caused spurious block splits in dense regions like FADS2.

print(f"[4/6] Calculating LD to find proxy SNPs...")

# Use stable_dir for temp files so they're accessible across nodes
temp_base = os.path.join(snakemake.params['stable_dir'], 'tmp_ld_blocks')
os.makedirs(temp_base, exist_ok=True)
temp_dir = tempfile.mkdtemp(prefix=f'cat_{category_safe}_', dir=temp_base)

gwas_ceu_ids   = ceu_bim[ceu_bim['chrpos'].isin(gwas_in_ceu)]['snp_id'].tolist()
gwas_list_file = os.path.join(temp_dir, 'gwas_snps.txt')
with open(gwas_list_file, 'w') as f:
    f.write('\n'.join(gwas_ceu_ids) + '\n')

gwas_prefix = os.path.join(temp_dir, 'gwas_snps')
subprocess.run(
    ['plink', '--bfile', ref_panel, '--extract', gwas_list_file,
     '--make-bed', '--out', gwas_prefix, '--allow-extra-chr', '--silent'],
    capture_output=True, check=True
)

ld_output = os.path.join(temp_dir, 'ld_expansion')
result = subprocess.run(
    ['plink', '--bfile', ref_panel,
     '--r2',
     '--ld-snp-list', gwas_list_file,
     '--ld-window-kb', str(window_kb),
     '--ld-window',    '999999',        # ← FIX: remove SNP-count cap
     '--ld-window-r2', str(r2_threshold),
     '--out', ld_output, '--allow-extra-chr', '--silent'],
    capture_output=True
)

ld_file = f"{ld_output}.ld" if result.returncode == 0 else None

# =========================================================================
# STEP 5: Collect all SNPs (GWAS + LD proxies) in ancient+CEU
# =========================================================================

print("[5/6] Collecting GWAS SNPs + LD proxies...")

all_snps_in_category = set(gwas_ceu_ids)

if ld_file and os.path.exists(ld_file):
    ld_df  = pd.read_csv(ld_file, sep=r'\s+')
    print(f"  LD pairs found: {len(ld_df):,}")
    proxy_snps = set(ld_df['SNP_B'].unique())
    all_snps_in_category.update(proxy_snps)
    print(f"  Proxy SNPs: {len(proxy_snps):,}")

print(f"  Total SNPs (GWAS + proxies): {len(all_snps_in_category):,}")

all_snps_chrpos            = {extract_chrpos(s) for s in all_snps_in_category}
all_snps_chrpos.discard(None)
all_snps_in_ancient_and_ceu = all_snps_chrpos & ancient_variants
print(f"  SNPs in ancient+CEU: {len(all_snps_in_ancient_and_ceu):,}")

# =========================================================================
# STEP 6: Create LD blocks from all category SNPs (GWAS + proxies)
# =========================================================================
# FIX: --ld-window 999999 added here too — this is the critical call that
# determines block membership. Without it, high-LD SNP pairs in dense
# regions are missed and incorrectly assigned to separate blocks.

print(f"[6/6] Creating LD blocks...")

all_cat_ceu_ids = ceu_bim[ceu_bim['chrpos'].isin(all_snps_in_ancient_and_ceu)]['snp_id'].tolist()

all_list_file = os.path.join(temp_dir, 'all_category_snps.txt')
with open(all_list_file, 'w') as f:
    f.write('\n'.join(all_cat_ceu_ids) + '\n')

all_prefix = os.path.join(temp_dir, 'all_cat_snps')
subprocess.run(
    ['plink', '--bfile', ref_panel, '--extract', all_list_file,
     '--make-bed', '--out', all_prefix, '--allow-extra-chr', '--silent'],
    capture_output=True, check=True
)

ld_all_output = os.path.join(temp_dir, 'ld_all')
result = subprocess.run(
    ['plink', '--bfile', all_prefix,
     '--r2',
     '--ld-window-kb', str(window_kb),
     '--ld-window',    '999999',        # ← FIX: remove SNP-count cap
     '--ld-window-r2', str(r2_threshold),
     '--out', ld_all_output, '--allow-extra-chr', '--silent'],
    capture_output=True
)

if result.returncode != 0 or not os.path.exists(f"{ld_all_output}.ld"):
    print("  WARNING: LD calculation failed — treating each SNP as its own block")
    blocks_df = pd.DataFrame({
        'block_id': range(len(all_cat_ceu_ids)),
        'snps':     [[s] for s in all_cat_ceu_ids]
    })
else:
    ld_pairs = pd.read_csv(f"{ld_all_output}.ld", sep=r'\s+')

    G = nx.Graph()
    G.add_nodes_from(all_cat_ceu_ids)
    for _, row in ld_pairs.iterrows():
        G.add_edge(row['SNP_A'], row['SNP_B'])

    components = list(nx.connected_components(G))
    blocks_df  = pd.DataFrame({
        'block_id': range(len(components)),
        'snps':     [list(comp) for comp in components]
    })

print(f"  ✓ Created {len(blocks_df)} LD blocks")

# =========================================================================
# Write outputs
# =========================================================================

os.makedirs(os.path.dirname(output_blocks), exist_ok=True)

blocks_df['n_snps']   = blocks_df['snps'].apply(len)
blocks_df['snp_ids']  = blocks_df['snps'].apply(','.join)
blocks_df[['block_id', 'snp_ids', 'n_snps']].to_csv(output_blocks, sep='\t', index=False)

members = []
for _, row in blocks_df.iterrows():
    for snp_id in row['snps']:
        chrpos = extract_chrpos(snp_id)
        members.append({
            'block_id':    row['block_id'],
            'snp_id':      chrpos,
            'ceu_id':      snp_id,
            'is_gwas_snp': chrpos in gwas_in_ceu
        })

members_df = pd.DataFrame(members)
members_df.to_csv(output_members, sep='\t', index=False)

with open(output_stats, 'w') as f:
    f.write(f"category\t{category_name}\n")
    f.write(f"n_blocks\t{len(blocks_df)}\n")
    f.write(f"n_snps\t{len(members_df)}\n")
    f.write(f"n_gwas_snps\t{members_df['is_gwas_snp'].sum()}\n")
    f.write(f"n_proxy_snps\t{(~members_df['is_gwas_snp']).sum()}\n")

print(f"✓ Outputs written:")
print(f"  {output_blocks}")
print(f"  {output_members}")
print(f"  {output_stats}")

# ---------------------------------------------------------------------------
# Temp files: removed unless ld.keep_temp is set in the config
# ---------------------------------------------------------------------------
if snakemake.params.get('keep_temp', False):
    print(f"\n  Temp files kept at: {temp_dir}")
else:
    import shutil
    shutil.rmtree(temp_dir)

print("="*80)
