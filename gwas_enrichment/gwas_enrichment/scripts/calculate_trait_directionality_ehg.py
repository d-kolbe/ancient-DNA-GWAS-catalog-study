#!/usr/bin/env python3
"""
==============================================================================
EHG vs EF TRAIT DIRECTIONALITY (PGS-BASED) -- SECOND-POPULATION ADDITION
==============================================================================

PURPOSE:
    Extend the existing WHG-vs-EF directionality analysis with a second
    ancestry comparison: EHG vs EF, restricted to exactly the trait set
    that appears in the main directionality forest panel (top-N
    categories x top-N traits per category, same selection logic as
    plot_directionality_forest.py / config_analysis.yaml).

KEY DESIGN DECISION:
    Does NOT re-derive SNP sets from the GWAS catalog. The per-trait SNP
    list (chr, pos, risk_allele, beta) was already finalized by the main
    pipeline (calculate_trait_directionality.py) and saved to
    directionality/snp_contributions/{category}/{trait}_snp_contributions.txt.
    This script reads those files directly and only extracts genotypes
    for EF and EHG individuals at those exact positions. This guarantees
    the EHG comparison uses the identical SNP/beta set as the original
    WHG comparison (no risk of divergent LD-block winners), and avoids
    re-parsing the GWAS catalog entirely.

    EF and EHG genotypes are read from the separate VCF given as vcf_ehg in
    config_ehg.yaml (not from the main pipeline's config['vcf']).

SIGN CONVENTION (matches main pipeline):
    delta_PGS = mean(PGS_EF) - mean(PGS_EHG)
    delta_PGS > 0 -> EF_higher
    delta_PGS < 0 -> EHG_higher
    (Same convention as the WHG comparison: positive Cohen's d always
    means "EF higher". This keeps the two comparisons visually and
    numerically comparable on the same x-axis.)

OUTPUTS:
    trait_directionality_summary_ehg.txt
      One row per selected trait: PGS_EF_mean/se, PGS_EHG_mean/se,
      delta_PGS, ttest_pval, cohens_d, direction, sample sizes.

REQUIREMENTS:
    - category_enrichment.txt          (category ranking, for trait selection)
    - trait_directionality_summary.txt (original WHG/EF summary, for trait selection)
    - directionality/snp_contributions/{category}/{trait}_snp_contributions.txt
    - VCF containing EF and EHG genotypes (merged)
    - EF and EHG sample lists
==============================================================================
"""

import os
import re
import sys
import functools
import subprocess
import numpy as np
import pandas as pd
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
        for ch in ["'", "\u2019", "\u2018", "`", "\u00b4"]:
            name = name.replace(ch, '')
        for ch in ['"', '$', '!', ';', '|', '<', '>', '&', '*', '?', '[', ']', '{', '}']:
            name = name.replace(ch, '')
        return name

print("=" * 80)
print("EHG vs EF TRAIT DIRECTIONALITY (PGS-BASED)")
print("=" * 80)

# =============================================================================
# Snakemake I/O
# =============================================================================

enrichment_file     = snakemake.input['enrichment']
directionality_file = snakemake.input['directionality']
vcf_file             = snakemake.input['vcf']
ef_samples_file      = snakemake.input['ef_samples']
ehg_samples_file     = snakemake.input['ehg_samples']

output_summary_ehg   = snakemake.output['directionality_summary_ehg']

contrib_base_dir     = snakemake.params['snp_contributions_dir']
col_category         = str(snakemake.params.get('col_category', 'category'))
col_trait            = str(snakemake.params.get('col_trait', 'trait'))
col_pvalue           = str(snakemake.params.get('col_pvalue', 'ttest_pval'))
col_n_extracted      = str(snakemake.params.get('col_n_extracted', 'n_snps_extracted'))
n_panel_categories   = int(snakemake.params.get('n_panel_categories', 6))
n_panel_traits       = int(snakemake.params.get('n_panel_traits', 10))
min_snps_extracted   = int(snakemake.params.get('min_snps_extracted', 20))
exclude_traits       = list(snakemake.params.get('exclude_traits', []) or [])

os.makedirs(os.path.dirname(output_summary_ehg), exist_ok=True)

print(f"\nInputs:")
print(f"  Enrichment: {enrichment_file}")
print(f"  Directionality (original): {directionality_file}")
print(f"  VCF: {vcf_file}")
print(f"  SNP contributions dir: {contrib_base_dir}")
print(f"  Panel selection: top {n_panel_categories} categories x top {n_panel_traits} traits, "
      f"min_snps_extracted>={min_snps_extracted}, exclude={exclude_traits}")

# =============================================================================
# Helper: replicate the exact trait selection used by
# plot_directionality_forest.py / config_analysis.yaml (directionality_plot)
# =============================================================================

ENR_P_CANDS = ['mannwhitneyu_pval', 'mw_p_value', 'p_value', 'pvalue',
               'mannwhitneyu_pval_twosided', 'kolmogorov_smirnov_pval', 'ttest_pval']

def _norm_str(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def select_panel_traits(enrichment_file, directionality_file):
    enr = pd.read_csv(enrichment_file, sep='\t', comment='#')
    enr_cat = next((c for c in ['category', 'Category', 'trait_category']
                     if c in enr.columns), None)
    if enr_cat is None:
        raise ValueError(f"No category column in {enrichment_file}")
    enr_p = next((c for c in ENR_P_CANDS if c in enr.columns), None)
    if enr_p is None:
        raise ValueError(f"No p-value column in {enrichment_file}")

    top_cats_raw = (enr.dropna(subset=[enr_p])
                        .sort_values(enr_p)
                        .head(n_panel_categories)[enr_cat]
                        .tolist())

    df = pd.read_csv(directionality_file, sep='\t', comment='#')

    if col_n_extracted in df.columns and min_snps_extracted > 0:
        before = len(df)
        df = df[pd.to_numeric(df[col_n_extracted], errors='coerce').fillna(0) >= min_snps_extracted]
        print(f"  min_snps_extracted filter: {before} -> {len(df)}")

    if exclude_traits:
        df = df[~df[col_trait].isin(exclude_traits)]

    actual_cats    = df[col_category].unique()
    norm_to_actual = {_norm_str(c): c for c in actual_cats}
    top_cats = [r for raw in top_cats_raw
                for r in [raw if raw in actual_cats else norm_to_actual.get(_norm_str(raw))]
                if r is not None]
    if not top_cats:
        raise ValueError("No valid categories after normalisation -- check enrichment/directionality category spelling")

    selected = []
    for cat in top_cats[:n_panel_categories]:
        cat_df = (df[df[col_category] == cat]
                    .copy()
                    .assign(**{col_pvalue: lambda x: pd.to_numeric(x[col_pvalue], errors='coerce')})
                    .dropna(subset=[col_pvalue])
                    .nsmallest(n_panel_traits, col_pvalue))
        for _, row in cat_df.iterrows():
            selected.append((str(row[col_trait]), cat))

    return selected

selected_traits = select_panel_traits(enrichment_file, directionality_file)
print(f"\nSelected {len(selected_traits)} (trait, category) pairs for EHG comparison")

# =============================================================================
# Load sample lists
# =============================================================================

with open(ef_samples_file) as f:
    ef_samples = [line.strip() for line in f if line.strip()]
with open(ehg_samples_file) as f:
    ehg_samples = [line.strip() for line in f if line.strip()]

print(f"\nEF samples: {len(ef_samples)}")
print(f"EHG samples: {len(ehg_samples)}")
if len(ehg_samples) < 15:
    print(f"  NOTE: small EHG sample size (n={len(ehg_samples)}) -- expect wide confidence intervals")

all_samples_set = set(ef_samples + ehg_samples)

# =============================================================================
# Get VCF sample order (same care as main pipeline: bcftools -s returns
# samples in VCF column order, not -s order, so we must zip against the
# VCF's own order)
# =============================================================================

res = subprocess.run(['bcftools', 'query', '-l', vcf_file], capture_output=True, text=True)
if res.returncode != 0:
    print(f"ERROR: could not read VCF sample list: {res.stderr}")
    sys.exit(1)

vcf_all_samples     = [s.strip() for s in res.stdout.strip().split('\n') if s.strip()]
vcf_samples_ordered = [s for s in vcf_all_samples if s in all_samples_set]
vcf_samples_str     = ','.join(vcf_samples_ordered)

missing_from_vcf = all_samples_set - set(vcf_samples_ordered)
if missing_from_vcf:
    print(f"  WARNING: {len(missing_from_vcf)} samples not found in VCF:")
    for s in sorted(missing_from_vcf):
        print(f"    {s}")

ef_idx  = np.array([i for i, s in enumerate(vcf_samples_ordered) if s in set(ef_samples)])
ehg_idx = np.array([i for i, s in enumerate(vcf_samples_ordered) if s in set(ehg_samples)])
print(f"  VCF-matched: EF={len(ef_idx)}, EHG={len(ehg_idx)}")

# =============================================================================
# Helpers
# =============================================================================

def extract_genotypes(vcf_file, chrom, pos, risk_allele, vcf_samples_ordered, vcf_samples_str):
    """Extract dosages (0/1/2 copies of risk_allele) from the VCF for one SNP."""
    cmd = ['bcftools', 'query',
           '-r', f"{chrom}:{pos}-{pos}",
           '-f', '%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n',
           '-s', vcf_samples_str,
           vcf_file]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout.strip():
        return {}
    fields = res.stdout.strip().split('\n')[0].split('\t')
    if len(fields) < 4:
        return {}
    ref, alt = fields[2], fields[3]
    if risk_allele == ref:
        risk_is_ref = True
    elif risk_allele == alt:
        risk_is_ref = False
    else:
        return {}
    dosages = {}
    for sample, gt in zip(vcf_samples_ordered, fields[4:]):
        if gt in ('./.', '.|.'):
            dosages[sample] = np.nan
            continue
        try:
            alleles = [int(a) for a in gt.replace('|', '/').split('/')]
        except ValueError:
            dosages[sample] = np.nan
            continue
        dosages[sample] = sum(1 for a in alleles if a == (0 if risk_is_ref else 1))
    return dosages

def cohens_d(arr1, arr2):
    """(mean1 - mean2) / pooled_sd"""
    v1 = np.var(arr1, ddof=1)
    v2 = np.var(arr2, ddof=1)
    ps = np.sqrt((v1 + v2) / 2)
    return (np.mean(arr1) - np.mean(arr2)) / ps if ps > 0 else 0.0

# =============================================================================
# Main loop
# =============================================================================

print(f"\n{'=' * 80}")
print("PROCESSING TRAITS")
print(f"{'=' * 80}\n")

results = []
skip_reasons = {'no_contrib_file': 0, 'empty_contrib': 0, 'no_snps_extracted': 0, 'success': 0}

for trait_name, category_name in selected_traits:
    trait_safe = sanitize_category_name(trait_name)
    cat_safe   = sanitize_category_name(category_name)

    print(f"[{trait_name}] ({category_name})")

    contrib_file = os.path.join(contrib_base_dir, cat_safe, f"{trait_safe}_snp_contributions.txt")
    if not os.path.exists(contrib_file):
        print(f"  SKIP: contributions file not found: {contrib_file}")
        skip_reasons['no_contrib_file'] += 1
        continue

    contrib = pd.read_csv(contrib_file, sep='\t', low_memory=False)
    if len(contrib) == 0:
        print(f"  SKIP: empty contributions file")
        skip_reasons['empty_contrib'] += 1
        continue

    n_snps = len(contrib)
    betas  = np.array(contrib['beta'].values, dtype=float)

    dosage_matrix = np.full((len(vcf_samples_ordered), n_snps), np.nan)

    for j, (_, snp) in enumerate(contrib.iterrows()):
        chrom       = str(int(float(snp['chr'])))
        pos         = int(float(snp['pos']))
        risk_allele = str(snp['risk_allele']).strip()
        dosages     = extract_genotypes(vcf_file, chrom, pos, risk_allele,
                                          vcf_samples_ordered, vcf_samples_str)
        if not dosages:
            continue
        for i, s in enumerate(vcf_samples_ordered):
            dosage_matrix[i, j] = dosages.get(s, np.nan)

    n_snps_extracted = int(np.sum(~np.all(np.isnan(dosage_matrix), axis=0)))
    if n_snps_extracted == 0:
        print(f"  SKIP: no SNPs extracted from VCF")
        skip_reasons['no_snps_extracted'] += 1
        continue

    # Mean-impute missing genotypes per population (same convention as main pipeline)
    missing_count = {s: 0 for s in vcf_samples_ordered}
    for j in range(n_snps):
        col = dosage_matrix[:, j]
        ef_col  = col[ef_idx]
        ehg_col = col[ehg_idx]
        ef_mean  = np.nanmean(ef_col)  if not np.all(np.isnan(ef_col))  else 0.0
        ehg_mean = np.nanmean(ehg_col) if not np.all(np.isnan(ehg_col)) else 0.0
        for i in ef_idx:
            if np.isnan(col[i]):
                dosage_matrix[i, j] = ef_mean
                missing_count[vcf_samples_ordered[i]] += 1
        for i in ehg_idx:
            if np.isnan(col[i]):
                dosage_matrix[i, j] = ehg_mean
                missing_count[vcf_samples_ordered[i]] += 1

    pgs_all = dosage_matrix @ betas
    pgs_ef  = pgs_all[ef_idx]
    pgs_ehg = pgs_all[ehg_idx]

    mean_pgs_ef  = float(np.mean(pgs_ef))
    mean_pgs_ehg = float(np.mean(pgs_ehg))
    se_pgs_ef    = float(np.std(pgs_ef,  ddof=1) / np.sqrt(len(pgs_ef)))
    se_pgs_ehg   = float(np.std(pgs_ehg, ddof=1) / np.sqrt(len(pgs_ehg)))

    delta_pgs = mean_pgs_ef - mean_pgs_ehg
    t_stat, t_pval = stats.ttest_ind(pgs_ef, pgs_ehg)
    d = cohens_d(pgs_ef, pgs_ehg)

    if delta_pgs > 0:
        direction = "EF_higher"
    elif delta_pgs < 0:
        direction = "EHG_higher"
    else:
        direction = "No_difference"

    avg_miss_ef  = np.mean([missing_count[s] for s in vcf_samples_ordered if s in set(ef_samples)])
    avg_miss_ehg = np.mean([missing_count[s] for s in vcf_samples_ordered if s in set(ehg_samples)])
    pct_missing  = (avg_miss_ef + avg_miss_ehg) / (2 * n_snps_extracted) * 100 if n_snps_extracted > 0 else 0.0

    print(f"  SNPs: {n_snps} ({n_snps_extracted} extracted)")
    print(f"  PGS EF:  {mean_pgs_ef:.4f} +/- {se_pgs_ef:.4f}")
    print(f"  PGS EHG: {mean_pgs_ehg:.4f} +/- {se_pgs_ehg:.4f}")
    print(f"  delta_PGS: {delta_pgs:.4f}, p={t_pval:.3e}, d={d:.4f}, direction={direction}")

    results.append({
        'trait': trait_name,
        'category': category_name,
        'n_snps_extracted': n_snps_extracted,
        'PGS_EF_mean':  mean_pgs_ef,  'PGS_EF_se':  se_pgs_ef,
        'PGS_EHG_mean': mean_pgs_ehg, 'PGS_EHG_se': se_pgs_ehg,
        'delta_PGS': delta_pgs,
        'ttest_pval': t_pval,
        'cohens_d': d,
        'direction': direction,
        'pct_missing_genotypes': pct_missing,
        'n_EF_samples': len(ef_samples),
        'n_EHG_samples': len(ehg_samples),
    })
    skip_reasons['success'] += 1
    print()

# =============================================================================
# Save
# =============================================================================

print(f"{'=' * 80}")
print("SAVING RESULTS")
print(f"{'=' * 80}")
print(f"Selected: {len(selected_traits)}  Success: {skip_reasons['success']}")
print(f"Skip reasons: {skip_reasons}")

summary_columns = [
    'trait', 'category', 'n_snps_extracted',
    'PGS_EF_mean', 'PGS_EF_se', 'PGS_EHG_mean', 'PGS_EHG_se',
    'delta_PGS', 'ttest_pval', 'cohens_d', 'direction',
    'pct_missing_genotypes', 'n_EF_samples', 'n_EHG_samples'
]

if results:
    out_df = pd.DataFrame(results).sort_values('ttest_pval')
    out_df.to_csv(output_summary_ehg, sep='\t', index=False)
    print(f"\nSaved: {output_summary_ehg}  ({len(out_df)} traits)")
    print(f"  EF higher:  {(out_df['direction'] == 'EF_higher').sum()}")
    print(f"  EHG higher: {(out_df['direction'] == 'EHG_higher').sum()}")
else:
    print("\nWARNING: no traits processed successfully -- writing empty file with headers")
    pd.DataFrame(columns=summary_columns).to_csv(output_summary_ehg, sep='\t', index=False)

print(f"{'=' * 80}")
print("DONE")
print(f"{'=' * 80}")