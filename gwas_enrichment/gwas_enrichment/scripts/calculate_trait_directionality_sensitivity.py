#!/usr/bin/env python3
"""
==============================================================================
TRAIT DIRECTIONALITY SENSITIVITY ANALYSIS — TOP-N SNP EXCLUSION
==============================================================================

Identical to calculate_trait_directionality.py with one addition:
before extracting genotypes for each trait, the top N SNPs by
|pgs_contribution| are read from the existing SNP contributions file
and excluded from the PGS calculation.

Purpose: verify that the directionality signal is not driven by a small
number of high-leverage SNPs.

Output format is identical to trait_directionality_summary.txt with two
extra columns: n_snps_excluded and exclude_top_n.
==============================================================================
"""

import os
import pandas as pd
import numpy as np
import sys
import subprocess
import functools
from scipy import stats

print = functools.partial(print, flush=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
try:
    from category_utils import sanitize_category_name
except ImportError:
    def sanitize_category_name(name):
        name = (name.replace(' - ', '_-_').replace(' & ', '_and_')
                    .replace(' ', '_').replace('/', '-')
                    .replace('(', '').replace(')', ''))
        for apostrophe in ["'", "\u2019", "\u2018", "`", "\u00b4", "\u2032"]:
            name = name.replace(apostrophe, '')
        for char in ['"', '\u201c', '\u201d', '$', '!', ';', '|', '<', '>', '&', '*', '?', '[', ']', '{', '}']:
            name = name.replace(char, '')
        return name

print("="*80)
print("TRAIT DIRECTIONALITY SENSITIVITY ANALYSIS (TOP-N SNP EXCLUSION)")
print("="*80)

# ---------------------------------------------------------------------------
# Snakemake I/O
# ---------------------------------------------------------------------------

trait_enrichment_file = snakemake.input['trait_enrichment']
gwas_catalog_file     = snakemake.input['gwas_catalog']
vcf_file              = snakemake.input['vcf']
hg_samples_file       = snakemake.input['hg_samples']
ef_samples_file       = snakemake.input['ef_samples']

trait_decomp_dir      = snakemake.params['trait_decomp_dir']
contrib_base_dir      = snakemake.params['contrib_base_dir']
output_dir            = snakemake.params['output_dir']
EXCLUDE_TOP_N         = int(snakemake.params['exclude_top_n'])
MIN_SNPS              = int(snakemake.params['min_snps_after_exclusion'])
min_pvalue            = float(snakemake.params['min_pvalue'])
ancestry_filter_str   = snakemake.params.get('ancestry_filter', '')
ancestry_filter       = [x.strip() for x in ancestry_filter_str.split(',') if x.strip()]
strict_european_only  = snakemake.params.get('strict_european_only', True)

output_summary        = snakemake.output['sensitivity_summary']

os.makedirs(os.path.dirname(output_summary), exist_ok=True)

print(f"\nExcluding top {EXCLUDE_TOP_N} SNPs per trait by |pgs_contribution|")
print(f"Minimum SNPs after exclusion: {MIN_SNPS}")

# ---------------------------------------------------------------------------
# Ancestry filter (identical to main script)
# ---------------------------------------------------------------------------

def apply_pure_european_filter(sample_size_series, strict_mode=True):
    european_keywords = [
        'European','european','British','british','Caucasian','caucasian',
        'White','white','Icelandic','icelandic','Finnish','finnish',
        'Swedish','swedish','Norwegian','norwegian','Danish','danish',
        'German','german','French','french','Italian','italian',
        'Spanish','spanish','Dutch','dutch','Polish','polish',
        'Greek','greek','Portuguese','portuguese','Belgian','belgian',
        'Austrian','austrian','Irish','irish','Scottish','scottish',
        'Welsh','welsh','Swiss','swiss','Czech','czech','Hungarian','hungarian',
        'Romanian','romanian','Bulgarian','bulgarian'
    ]
    non_european_keywords = [
        'Asian','asian','East Asian','east asian','South Asian','south asian',
        'African','african','African American','african american',
        'Afro-Caribbean','afro-caribbean','Sub-Saharan','sub-saharan',
        'Hispanic','hispanic','Latino','latino','Latina','latina',
        'Korean','korean','Japanese','japanese','Chinese','chinese',
        'Han Chinese','han chinese','Filipino','filipino','Vietnamese','vietnamese',
        'Thai','thai','Indian','indian','Pakistani','pakistani',
        'Bangladeshi','bangladeshi','Iranian','iranian','Arab','arab',
        'Middle Eastern','middle eastern','Native American','native american',
        'Indigenous','indigenous','Pacific Islander','pacific islander',
        'Polynesian','polynesian','Maori','maori','Aboriginal','aboriginal',
        'Jamaican','jamaican','Mexican','mexican','Seychellois','seychellois'
    ]
    has_european = sample_size_series.str.contains(
        '|'.join(european_keywords), case=False, na=False, regex=True)
    if not strict_mode:
        return has_european
    has_non_european = sample_size_series.str.contains(
        '|'.join(non_european_keywords), case=False, na=False, regex=True)
    return has_european & ~has_non_european

# ---------------------------------------------------------------------------
# Helper functions (identical to main script)
# ---------------------------------------------------------------------------

def parse_risk_allele(risk_allele_str):
    if pd.isna(risk_allele_str):
        return None
    parts = str(risk_allele_str).split('-')
    if len(parts) < 2:
        return None
    allele = parts[-1].strip().upper()
    if allele in ('?', 'NR') or len(allele) == 0:
        return None
    if len(allele) == 1 and allele in ['A', 'C', 'G', 'T']:
        return allele
    return None

def parse_effect_size(row):
    ci_text    = str(row.get('95% CI (TEXT)', '')).lower()
    or_beta_val = row.get('OR or BETA')
    if pd.isna(or_beta_val) or or_beta_val in ('', 'NR', 'NA'):
        return (None, None, None, None)
    try:
        value = float(or_beta_val)
    except (ValueError, TypeError):
        return (None, None, None, None)
    if 'increase' in ci_text or 'decrease' in ci_text:
        if 'decrease' in ci_text:
            return ('BETA', -abs(value), value, 'decrease')
        return ('BETA', abs(value), value, 'increase')
    else:
        if value > 0:
            return ('OR', np.log(value), value, 'OR')
        return (None, None, None, None)

def extract_genotypes_for_snp(vcf_file, chrom, pos, risk_allele, vcf_samples_ordered, vcf_samples_str):
    cmd = ['bcftools', 'query',
           '-r', f"{chrom}:{pos}-{pos}",
           '-f', '%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n',
           '-s', vcf_samples_str,
           vcf_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}, None, None
    lines = result.stdout.strip().split('\n')
    if not lines or lines[0] == '':
        return {}, None, None
    fields = lines[0].split('\t')
    if len(fields) < 4:
        return {}, None, None
    ref, alt = fields[2], fields[3]
    if risk_allele == ref:
        risk_is_ref = True
    elif risk_allele == alt:
        risk_is_ref = False
    else:
        return {}, ref, alt
    dosages = {}
    for sample, gt in zip(vcf_samples_ordered, fields[4:]):
        if gt in ('./.',  '.|.'):
            dosages[sample] = np.nan
            continue
        try:
            allele_counts = [int(a) for a in gt.replace('|', '/').split('/')]
        except ValueError:
            dosages[sample] = np.nan
            continue
        dosages[sample] = sum(1 for a in allele_counts if a == (0 if risk_is_ref else 1))
    return dosages, ref, alt

# ---------------------------------------------------------------------------
# Load samples
# ---------------------------------------------------------------------------

with open(hg_samples_file) as f:
    hg_samples = [l.strip() for l in f if l.strip()]
with open(ef_samples_file) as f:
    ef_samples = [l.strip() for l in f if l.strip()]

all_samples     = hg_samples + ef_samples
all_samples_set = set(all_samples)
print(f"\nWHG samples: {len(hg_samples)},  EF samples: {len(ef_samples)}")

# VCF sample order
vcf_sample_order_result = subprocess.run(
    ['bcftools', 'query', '-l', vcf_file], capture_output=True, text=True)
vcf_all_samples     = [s.strip() for s in vcf_sample_order_result.stdout.strip().split('\n') if s.strip()]
vcf_samples_ordered = [s for s in vcf_all_samples if s in all_samples_set]
vcf_samples_str     = ','.join(vcf_samples_ordered)
print(f"VCF samples found: {len(vcf_samples_ordered)}")

# ---------------------------------------------------------------------------
# Load traits and GWAS catalog
# ---------------------------------------------------------------------------

trait_enrichment_df = pd.read_csv(trait_enrichment_file, sep='\t')
print(f"\nTraits to process: {len(trait_enrichment_df)}")

gwas_df = pd.read_csv(gwas_catalog_file, sep='\t', low_memory=False)
if ancestry_filter or strict_european_only:
    mask    = apply_pure_european_filter(gwas_df['INITIAL SAMPLE SIZE'], strict_mode=strict_european_only)
    gwas_df = gwas_df[mask].copy()
gwas_df['P-VALUE'] = pd.to_numeric(gwas_df['P-VALUE'], errors='coerce')
gwas_df = gwas_df[gwas_df['P-VALUE'].notna() & (gwas_df['P-VALUE'] < min_pvalue)].copy()
gwas_df = gwas_df[
    gwas_df['STRONGEST SNP-RISK ALLELE'].notna() &
    gwas_df['STRONGEST SNP-RISK ALLELE'].str.contains('-', na=False)
].copy()
gwas_df = gwas_df[gwas_df['CHR_ID'].notna() & gwas_df['CHR_POS'].notna()].copy()
gwas_df['chr_num'] = pd.to_numeric(gwas_df['CHR_ID'], errors='coerce')
gwas_df['CHR_POS'] = pd.to_numeric(gwas_df['CHR_POS'], errors='coerce')
gwas_df = gwas_df[gwas_df['chr_num'].notna() & gwas_df['CHR_POS'].notna()].copy()
gwas_df['chrpos'] = gwas_df['chr_num'].astype(int).astype(str) + ':' + gwas_df['CHR_POS'].astype(int).astype(str)
gwas_df['trait_normalized'] = gwas_df['DISEASE/TRAIT'].str.lower().str.strip()
print(f"GWAS catalog after filters: {len(gwas_df):,}")

# ---------------------------------------------------------------------------
# Main loop — identical to main script + exclusion step
# ---------------------------------------------------------------------------

print(f"\n{'='*80}\nPROCESSING TRAITS\n{'='*80}\n")

all_trait_results = []

for trait_idx, trait_row in trait_enrichment_df.iterrows():
    trait_name    = trait_row['trait']
    category_name = trait_row['category']
    trait_safe    = trait_row['trait_safe']

    print(f"[{trait_idx+1}/{len(trait_enrichment_df)}] {trait_name}")

    # ----------------------------------------------------------------
    # SENSITIVITY STEP: load existing contributions, get top-N to exclude
    # ----------------------------------------------------------------
    category_safe  = sanitize_category_name(category_name)
    contrib_file   = os.path.join(contrib_base_dir, category_safe,
                                   f"{trait_safe}_snp_contributions.txt")

    exclude_positions = set()
    if os.path.exists(contrib_file):
        try:
            contrib_df = pd.read_csv(contrib_file, sep='\t', low_memory=False)
            top_n_rows = contrib_df.nlargest(EXCLUDE_TOP_N, 'abs_pgs_contribution')
            exclude_positions = set(top_n_rows['snp_id'].tolist())
            print(f"  Excluding top {EXCLUDE_TOP_N} SNPs: {sorted(exclude_positions)}")
        except Exception as e:
            print(f"  WARNING: could not read contributions file: {e}")
    else:
        print(f"  WARNING: contributions file not found, no exclusions applied")

    # ----------------------------------------------------------------
    # From here: identical to main script
    # ----------------------------------------------------------------

    trait_gwas = gwas_df[gwas_df['trait_normalized'] == trait_name.lower().strip()].copy()
    if len(trait_gwas) == 0:
        print(f"  SKIP: No GWAS associations found"); continue

    trait_gwas['risk_allele'] = trait_gwas['STRONGEST SNP-RISK ALLELE'].apply(parse_risk_allele)
    effect_results = trait_gwas.apply(parse_effect_size, axis=1)
    trait_gwas['effect_type']    = [r[0] for r in effect_results]
    trait_gwas['beta']           = [r[1] for r in effect_results]
    trait_gwas['effect_original']= [r[2] for r in effect_results]

    trait_gwas_clean = trait_gwas[
        trait_gwas['risk_allele'].notna() & trait_gwas['beta'].notna()].copy()
    if len(trait_gwas_clean) == 0:
        print(f"  SKIP: No SNPs with clear effect allele + size"); continue

    # Duplicate deduplication
    trait_gwas_clean['chrpos'] = (
        trait_gwas_clean['chr_num'].astype(int).astype(str) + ':' +
        trait_gwas_clean['CHR_POS'].astype(int).astype(str))
    trait_gwas_clean['p_numeric'] = pd.to_numeric(trait_gwas_clean['P-VALUE'], errors='coerce')
    trait_gwas_clean = (trait_gwas_clean
                        .sort_values('p_numeric')
                        .drop_duplicates(subset='chrpos', keep='first')
                        .copy())

    # Effect type majority filter
    et_counts = trait_gwas_clean['effect_type'].value_counts()
    dominant_et = et_counts.index[0]
    trait_gwas_clean = trait_gwas_clean[trait_gwas_clean['effect_type'] == dominant_et].copy()
    if len(trait_gwas_clean) == 0:
        print(f"  SKIP: No SNPs after effect type filtering"); continue

    # Load LD blocks
    trait_blocks_file = os.path.join(trait_decomp_dir, category_safe, 'trait_blocks.txt')
    if not os.path.exists(trait_blocks_file):
        print(f"  SKIP: Trait blocks file not found"); continue
    all_trait_blocks_df = pd.read_csv(trait_blocks_file, sep='\t')
    block_members_df    = all_trait_blocks_df[all_trait_blocks_df['trait'] == trait_name].copy()
    if 'is_gwas_snp' in block_members_df.columns:
        block_members_df = block_members_df[block_members_df['is_gwas_snp'] == True].copy()

    # Merge with blocks
    gwas_with_blocks = trait_gwas_clean.merge(
        block_members_df[['snp_id', 'block_id']],
        left_on='chrpos', right_on='snp_id', how='inner')
    if len(gwas_with_blocks) == 0:
        print(f"  SKIP: No SNPs matched to LD blocks"); continue

    # LD deduplication
    gwas_with_blocks['abs_beta'] = gwas_with_blocks['beta'].abs()
    selected_snps = (gwas_with_blocks
                     .sort_values('abs_beta', ascending=False)
                     .groupby('block_id').first()
                     .reset_index())

    # ----------------------------------------------------------------
    # SENSITIVITY STEP: apply exclusion
    # ----------------------------------------------------------------
    selected_snps = selected_snps[~selected_snps['chrpos'].isin(exclude_positions)].copy()
    n_remaining   = len(selected_snps)
    print(f"  SNPs after exclusion: {n_remaining} (excluded {len(exclude_positions)})")

    if n_remaining < MIN_SNPS:
        print(f"  SKIP: {n_remaining} SNPs < minimum {MIN_SNPS} after exclusion"); continue

    # Genotype extraction + PGS (identical to main script)
    print(f"  Extracting genotypes...")
    pgs_dict           = {s: 0.0 for s in all_samples}
    missing_count_dict = {s: 0   for s in all_samples}
    snp_details        = []

    for _, snp_row in selected_snps.iterrows():
        chrom       = str(int(snp_row['chr_num']))
        pos         = int(snp_row['CHR_POS'])
        risk_allele = snp_row['risk_allele']
        beta        = snp_row['beta']

        dosages, vcf_ref, vcf_alt = extract_genotypes_for_snp(
            vcf_file, chrom, pos, risk_allele, vcf_samples_ordered, vcf_samples_str)
        if len(dosages) == 0:
            continue

        whg_d       = [dosages.get(s, np.nan) for s in hg_samples]
        ef_d        = [dosages.get(s, np.nan) for s in ef_samples]
        whg_d_clean = [d for d in whg_d if not np.isnan(d)]
        ef_d_clean  = [d for d in ef_d  if not np.isnan(d)]
        if not whg_d_clean or not ef_d_clean:
            continue

        md_whg = np.mean(whg_d_clean)
        md_ef  = np.mean(ef_d_clean)

        for s in hg_samples:
            d = dosages.get(s, np.nan)
            pgs_dict[s] += d * beta if not np.isnan(d) else md_whg * beta
            if np.isnan(d): missing_count_dict[s] += 1
        for s in ef_samples:
            d = dosages.get(s, np.nan)
            pgs_dict[s] += d * beta if not np.isnan(d) else md_ef * beta
            if np.isnan(d): missing_count_dict[s] += 1

        snp_details.append({'snp_id': snp_row['chrpos']})

    n_snps_used = len(snp_details)
    if n_snps_used == 0:
        print(f"  SKIP: No SNPs extracted from VCF"); continue

    # PGS statistics
    pgs_whg   = [pgs_dict[s] for s in hg_samples]
    pgs_ef    = [pgs_dict[s] for s in ef_samples]
    mean_whg  = np.mean(pgs_whg);  mean_ef   = np.mean(pgs_ef)
    se_whg    = np.std(pgs_whg, ddof=1) / np.sqrt(len(pgs_whg))
    se_ef     = np.std(pgs_ef,  ddof=1) / np.sqrt(len(pgs_ef))
    delta_pgs = mean_ef - mean_whg
    _, tpval  = stats.ttest_ind(pgs_ef, pgs_whg)
    pooled_sd = np.sqrt((np.var(pgs_whg, ddof=1) + np.var(pgs_ef, ddof=1)) / 2)
    cohens_d  = delta_pgs / pooled_sd if pooled_sd > 0 else 0
    direction = "EF_higher" if delta_pgs > 0 else ("WHG_higher" if delta_pgs < 0 else "No_difference")

    avg_miss  = (np.mean([missing_count_dict[s] for s in hg_samples]) +
                 np.mean([missing_count_dict[s] for s in ef_samples])) / 2
    pct_miss  = avg_miss / n_snps_used * 100 if n_snps_used > 0 else 0

    print(f"  PGS WHG: {mean_whg:.4f} ± {se_whg:.4f}")
    print(f"  PGS EF:  {mean_ef:.4f}  ± {se_ef:.4f}")
    print(f"  Delta: {delta_pgs:.4f}, p={tpval:.3e}, direction: {direction}")

    all_trait_results.append({
        'trait':                   trait_name,
        'category':                category_name,
        'n_snps_gwas':             len(trait_gwas_clean),
        'n_snps_in_blocks':        len(gwas_with_blocks),
        'n_snps_after_ld_filter':  len(selected_snps) + len(exclude_positions),
        'n_snps_excluded':         len(exclude_positions),
        'exclude_top_n':           EXCLUDE_TOP_N,
        'n_snps_extracted':        n_snps_used,
        'effect_type_primary':     dominant_et,
        'n_snps_OR':               et_counts.get('OR', 0),
        'n_snps_BETA':             et_counts.get('BETA', 0),
        'PGS_WHG_mean':            mean_whg,
        'PGS_WHG_se':              se_whg,
        'PGS_EF_mean':             mean_ef,
        'PGS_EF_se':               se_ef,
        'delta_PGS':               delta_pgs,
        'ttest_pval':              tpval,
        'cohens_d':                cohens_d,
        'direction':               direction,
        'pct_missing_genotypes':   pct_miss,
        'n_WHG_samples':           len(hg_samples),
        'n_EF_samples':            len(ef_samples),
    })
    print()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

print(f"{'='*80}\nSAVING RESULTS\n{'='*80}")

if all_trait_results:
    out_df = pd.DataFrame(all_trait_results).sort_values('ttest_pval')
    out_df.to_csv(output_summary, sep='\t', index=False)
    print(f"✓ {output_summary}")
    print(f"  Traits: {len(out_df)}")
    print(f"  p < 0.05: {(out_df['ttest_pval'] < 0.05).sum()}")
    same_dir = (out_df['direction'] == out_df['direction']).sum()  # placeholder
    print(f"  EF higher: {(out_df['direction']=='EF_higher').sum()}")
    print(f"  WHG higher: {(out_df['direction']=='WHG_higher').sum()}")
else:
    print("WARNING: No traits processed")
    pd.DataFrame().to_csv(output_summary, sep='\t', index=False)

print("="*80)
print("SENSITIVITY ANALYSIS COMPLETE")
print("="*80)