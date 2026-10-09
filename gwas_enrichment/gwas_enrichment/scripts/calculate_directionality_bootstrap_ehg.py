#!/usr/bin/env python3
"""
==============================================================================
EHG vs EF DIRECTIONALITY -- BOOTSTRAP CONFIDENCE INTERVALS (Option A only)
==============================================================================

Produces percentile bootstrap confidence intervals for Cohen's d, for the
same trait set already computed by calculate_trait_directionality_ehg.py.

Method (matches Option A in the main pipeline's
calculate_directionality_bootstrap.py exactly):
    Resample EF and EHG individuals WITH replacement, recalculate Cohen's d
    each iteration. Captures uncertainty from small sample sizes
    (especially EHG, n=12).

Does NOT include the main pipeline's Option C (SNP dropout) or Option D
(leave-one-out) -- descoped for this comparison.

Reads:
  - trait_directionality_summary_ehg.txt  (point estimates + trait list;
    already restricted to the panel trait set)
  - snp_contributions/{cat}/{trait}_snp_contributions.txt  (SNP list, shared
    with the main pipeline -- not regenerated here)
  - VCF + EF/EHG sample files

Writes:
  - bootstrap_ci_ehg.txt
    Same rows as trait_directionality_summary_ehg.txt, with added columns:
      cohens_d_reconstructed
      bootstrap_ci_lo, bootstrap_ci_hi  (2.5th/97.5th percentile)
      n_bootstrap
      n_snps_used

CAUTION: with EHG n=12, bootstrap resamples can draw a small, repeated
subset of individuals. Inspect the resulting CIs for degenerate behaviour
(e.g. unusually narrow or bimodal-looking intervals) before trusting them.
==============================================================================
"""

import os
import sys
import functools
import subprocess
import numpy as np
import pandas as pd

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
print("EHG vs EF DIRECTIONALITY -- BOOTSTRAP CI (Option A only)")
print("=" * 80)

# =============================================================================
# Snakemake I/O
# =============================================================================

main_summary_file = snakemake.input['directionality_ehg']
vcf_file           = snakemake.input['vcf']
ef_samples_file    = snakemake.input['ef_samples']
ehg_samples_file   = snakemake.input['ehg_samples']

contrib_base_dir  = snakemake.params['snp_contributions_dir']
N_BOOTSTRAP       = int(snakemake.params.get('n_bootstrap', 1000))
CI_LEVEL          = float(snakemake.params.get('ci_level', 0.95))
SEED              = int(snakemake.params.get('seed', 42))

output_file = snakemake.output['bootstrap_ci_ehg']

os.makedirs(os.path.dirname(output_file), exist_ok=True)
rng = np.random.default_rng(SEED)

alpha  = 1 - CI_LEVEL
lo_pct = alpha / 2 * 100
hi_pct = (1 - alpha / 2) * 100

print(f"\nBootstrap iterations: {N_BOOTSTRAP}")
print(f"CI level: {CI_LEVEL*100:.0f}%  ({lo_pct:.1f}th / {hi_pct:.1f}th percentile)")
print(f"Random seed: {SEED}")

# =============================================================================
# Helpers
# =============================================================================

def cohens_d(arr1, arr2):
    """Point estimate of Cohen's d = (mean1 - mean2) / pooled_sd"""
    v1 = np.var(arr1, ddof=1)
    v2 = np.var(arr2, ddof=1)
    ps = np.sqrt((v1 + v2) / 2)
    return (np.mean(arr1) - np.mean(arr2)) / ps if ps > 0 else 0.0

def extract_genotypes(vcf_file, chrom, pos, risk_allele, vcf_samples_ordered, vcf_samples_str):
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

# =============================================================================
# Load samples + VCF order
# =============================================================================

with open(ef_samples_file) as f:
    ef_samples = [l.strip() for l in f if l.strip()]
with open(ehg_samples_file) as f:
    ehg_samples = [l.strip() for l in f if l.strip()]

all_samples_set = set(ef_samples + ehg_samples)
n_ef  = len(ef_samples)
n_ehg = len(ehg_samples)
print(f"\nEF: {n_ef},  EHG: {n_ehg}")
if n_ehg < 15:
    print(f"  NOTE: small EHG sample size (n={n_ehg}) -- bootstrap resamples may "
          f"repeatedly draw a small subset of individuals; inspect the resulting "
          f"CIs for degenerate behaviour.")

res = subprocess.run(['bcftools', 'query', '-l', vcf_file], capture_output=True, text=True)
vcf_all = [s.strip() for s in res.stdout.strip().split('\n') if s.strip()]
vcf_samples_ordered = [s for s in vcf_all if s in all_samples_set]
vcf_samples_str     = ','.join(vcf_samples_ordered)

ef_idx  = np.array([i for i, s in enumerate(vcf_samples_ordered) if s in set(ef_samples)])
ehg_idx = np.array([i for i, s in enumerate(vcf_samples_ordered) if s in set(ehg_samples)])

# =============================================================================
# Load main summary (already restricted to the panel trait set)
# =============================================================================

main_df = pd.read_csv(main_summary_file, sep='\t')
print(f"Traits to process: {len(main_df)}")

# =============================================================================
# Main loop
# =============================================================================

results = []

for _, trait_row in main_df.iterrows():
    trait_name    = trait_row['trait']
    category_name = trait_row['category']
    trait_safe    = sanitize_category_name(trait_name)
    cat_safe      = sanitize_category_name(category_name)

    print(f"\n{'-'*60}")
    print(f"Trait: {trait_name}")

    contrib_file = os.path.join(contrib_base_dir, cat_safe, f"{trait_safe}_snp_contributions.txt")
    if not os.path.exists(contrib_file):
        print(f"  SKIP: contributions file not found")
        results.append({**trait_row.to_dict(),
                        'cohens_d_reconstructed': np.nan,
                        'bootstrap_ci_lo': np.nan, 'bootstrap_ci_hi': np.nan,
                        'n_bootstrap': N_BOOTSTRAP, 'n_snps_used': 0})
        continue

    contrib = pd.read_csv(contrib_file, sep='\t', low_memory=False)
    n_snps  = len(contrib)
    if n_snps == 0:
        print(f"  SKIP: empty contributions file")
        results.append({**trait_row.to_dict(),
                        'cohens_d_reconstructed': np.nan,
                        'bootstrap_ci_lo': np.nan, 'bootstrap_ci_hi': np.nan,
                        'n_bootstrap': N_BOOTSTRAP, 'n_snps_used': 0})
        continue

    print(f"  SNPs: {n_snps}")
    print(f"  Extracting genotypes from VCF...")

    dosage_matrix = np.full((len(vcf_samples_ordered), n_snps), np.nan)
    betas         = np.array(contrib['beta'].values, dtype=float)

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

    n_snps_used = int(np.sum(~np.all(np.isnan(dosage_matrix), axis=0)))
    if n_snps_used == 0:
        print(f"  SKIP: no SNPs extracted from VCF")
        results.append({**trait_row.to_dict(),
                        'cohens_d_reconstructed': np.nan,
                        'bootstrap_ci_lo': np.nan, 'bootstrap_ci_hi': np.nan,
                        'n_bootstrap': N_BOOTSTRAP, 'n_snps_used': 0})
        continue

    # Mean-impute missing genotypes per population
    for j in range(n_snps):
        col = dosage_matrix[:, j]
        ef_mean  = np.nanmean(col[ef_idx])  if not np.all(np.isnan(col[ef_idx]))  else 0
        ehg_mean = np.nanmean(col[ehg_idx]) if not np.all(np.isnan(col[ehg_idx])) else 0
        for i in ef_idx:
            if np.isnan(col[i]): dosage_matrix[i, j] = ef_mean
        for i in ehg_idx:
            if np.isnan(col[i]): dosage_matrix[i, j] = ehg_mean

    pgs_all = dosage_matrix @ betas
    pgs_ef  = pgs_all[ef_idx]
    pgs_ehg = pgs_all[ehg_idx]
    d_obs   = cohens_d(pgs_ef, pgs_ehg)
    print(f"  Observed Cohen's d (reconstructed): {d_obs:.4f}  "
          f"(point estimate: {trait_row.get('cohens_d', float('nan')):.4f})")

    # -------------------------------------------------------------------
    # Option A: Bootstrap individuals
    # -------------------------------------------------------------------

    print(f"  Running bootstrap ({N_BOOTSTRAP} iterations)...")
    boot_d = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        ef_boot  = pgs_ef[rng.integers(0, n_ef, size=n_ef)]
        ehg_boot = pgs_ehg[rng.integers(0, n_ehg, size=n_ehg)]
        boot_d[b] = cohens_d(ef_boot, ehg_boot)

    boot_lo = float(np.percentile(boot_d, lo_pct))
    boot_hi = float(np.percentile(boot_d, hi_pct))
    print(f"  Bootstrap CI: [{boot_lo:.4f}, {boot_hi:.4f}]")

    results.append({
        **trait_row.to_dict(),
        'cohens_d_reconstructed': d_obs,
        'bootstrap_ci_lo':        boot_lo,
        'bootstrap_ci_hi':        boot_hi,
        'n_bootstrap':            N_BOOTSTRAP,
        'n_snps_used':            n_snps_used,
    })

# =============================================================================
# Save
# =============================================================================

print(f"\n{'=' * 80}\nSAVING\n{'=' * 80}")

out_df = pd.DataFrame(results)
out_df.to_csv(output_file, sep='\t', index=False)
print(f"Saved: {output_file}  ({len(out_df)} traits)")