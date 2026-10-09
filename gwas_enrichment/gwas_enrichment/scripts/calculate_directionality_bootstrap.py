#!/usr/bin/env python3
"""
==============================================================================
DIRECTIONALITY BOOTSTRAP & JACKKNIFE CONFIDENCE INTERVALS
==============================================================================

Produces confidence intervals for Cohen's d for each trait using:

  A) Bootstrap individuals (Option A)
     Resample WHG and EF individuals WITH replacement, recalculate
     Cohen's d each iteration. Captures uncertainty from small sample
     sizes (especially n=40 EF). Does not require SNP re-selection.

  B) Delete-d SNP dropout (Option C)
     Remove a random fraction (default 10%) of SNPs each iteration,
     recalculate Cohen's d on the remainder. More meaningful than
     leave-one-out for traits with many SNPs — a single removal barely
     perturbs the PGS. Answers: does the signal hold when we randomly
     drop 10% of SNPs? snp_dropout_fraction is configurable.

Reads:
  - trait_directionality_summary.txt  (point estimates + trait list)
  - snp_contributions/{cat}/{trait}_snp_contributions.txt  (SNP list)
  - VCF + sample files  (for per-individual PGS reconstruction)

Writes:
  - directionality_bootstrap/bootstrap_ci.txt
    Same rows as trait_directionality_summary.txt, with added columns:
      bootstrap_ci_lo, bootstrap_ci_hi  (2.5th/97.5th percentile)
      snp_dropout_ci_lo, snp_dropout_ci_hi  (delete-d percentile CI)
      n_bootstrap
      n_snps_used

Leaves all existing scripts and outputs untouched.
==============================================================================
"""

import os
import sys
import subprocess
import functools
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

print("="*80)
print("DIRECTIONALITY BOOTSTRAP & JACKKNIFE CI")
print("="*80)

# ---------------------------------------------------------------------------
# Snakemake I/O
# ---------------------------------------------------------------------------

main_summary_file = snakemake.input['main_summary']
vcf_file          = snakemake.input['vcf']
hg_samples_file   = snakemake.input['hg_samples']
ef_samples_file   = snakemake.input['ef_samples']

contrib_base_dir  = snakemake.params['contrib_base_dir']
N_BOOTSTRAP       = int(snakemake.params.get('n_bootstrap', 1000))
CI_LEVEL          = float(snakemake.params.get('ci_level', 0.95))
SEED              = int(snakemake.params.get('seed', 42))
SNP_DROPOUT_FRAC  = float(snakemake.params.get('snp_dropout_fraction', 0.10))

output_file       = snakemake.output['bootstrap_ci']
output_loo        = snakemake.output['snp_leaveoneout']

os.makedirs(os.path.dirname(output_file), exist_ok=True)
rng = np.random.default_rng(SEED)

alpha     = 1 - CI_LEVEL
lo_pct    = alpha / 2 * 100
hi_pct    = (1 - alpha / 2) * 100

print(f"\nBootstrap iterations: {N_BOOTSTRAP}")
print(f"CI level: {CI_LEVEL*100:.0f}%  ({lo_pct:.1f}th / {hi_pct:.1f}th percentile)")
print(f"Random seed: {SEED}")
print(f"SNP dropout fraction: {SNP_DROPOUT_FRAC*100:.0f}%")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cohens_d(arr1, arr2):
    """Point estimate of Cohen's d = (mean1 - mean2) / pooled_sd"""
    v1 = np.var(arr1, ddof=1)
    v2 = np.var(arr2, ddof=1)
    ps = np.sqrt((v1 + v2) / 2)
    return (np.mean(arr1) - np.mean(arr2)) / ps if ps > 0 else 0.0

def extract_genotypes(vcf_file, chrom, pos, risk_allele,
                      vcf_samples_ordered, vcf_samples_str):
    """Extract dosages from VCF. Identical to main script."""
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
    ref, alt     = fields[2], fields[3]
    risk_is_ref  = (risk_allele == ref)
    if risk_allele != ref and risk_allele != alt:
        return {}
    dosages = {}
    for sample, gt in zip(vcf_samples_ordered, fields[4:]):
        if gt in ('./.', '.|.'):
            dosages[sample] = np.nan; continue
        try:
            alleles = [int(a) for a in gt.replace('|', '/').split('/')]
        except ValueError:
            dosages[sample] = np.nan; continue
        dosages[sample] = sum(1 for a in alleles if a == (0 if risk_is_ref else 1))
    return dosages

# ---------------------------------------------------------------------------
# Load samples + VCF order
# ---------------------------------------------------------------------------

with open(hg_samples_file) as f:
    hg_samples = [l.strip() for l in f if l.strip()]
with open(ef_samples_file) as f:
    ef_samples = [l.strip() for l in f if l.strip()]

all_samples_set = set(hg_samples + ef_samples)
n_hg = len(hg_samples); n_ef = len(ef_samples)
print(f"\nWHG: {n_hg},  EF: {n_ef}")

res = subprocess.run(['bcftools', 'query', '-l', vcf_file],
                     capture_output=True, text=True)
vcf_all = [s.strip() for s in res.stdout.strip().split('\n') if s.strip()]
vcf_samples_ordered = [s for s in vcf_all if s in all_samples_set]
vcf_samples_str     = ','.join(vcf_samples_ordered)

# WHG / EF indices into vcf_samples_ordered (for fast array slicing)
hg_idx = np.array([i for i, s in enumerate(vcf_samples_ordered) if s in set(hg_samples)])
ef_idx = np.array([i for i, s in enumerate(vcf_samples_ordered) if s in set(ef_samples)])

# ---------------------------------------------------------------------------
# Load main summary
# ---------------------------------------------------------------------------

main_df = pd.read_csv(main_summary_file, sep='\t')
print(f"Traits to process: {len(main_df)}")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

results  = []
loo_rows = []

for _, trait_row in main_df.iterrows():
    trait_name    = trait_row['trait']
    category_name = trait_row['category']
    trait_safe    = sanitize_category_name(trait_name)
    cat_safe      = sanitize_category_name(category_name)

    print(f"\n{'─'*60}")
    print(f"Trait: {trait_name}")

    contrib_file = os.path.join(contrib_base_dir, cat_safe,
                                f"{trait_safe}_snp_contributions.txt")
    if not os.path.exists(contrib_file):
        print(f"  SKIP: contributions file not found")
        results.append({**trait_row.to_dict(),
                        'bootstrap_ci_lo': np.nan, 'bootstrap_ci_hi': np.nan,
                        'snp_dropout_ci_lo': np.nan, 'snp_dropout_ci_hi': np.nan,
                        'snp_dropout_fraction': SNP_DROPOUT_FRAC,
                        'n_bootstrap': N_BOOTSTRAP, 'n_snps_used': 0})
        continue

    contrib = pd.read_csv(contrib_file, sep='\t', low_memory=False)
    n_snps  = len(contrib)
    print(f"  SNPs: {n_snps}")

    if n_snps == 0:
        print(f"  SKIP: empty contributions file")
        results.append({**trait_row.to_dict(),
                        'bootstrap_ci_lo': np.nan, 'bootstrap_ci_hi': np.nan,
                        'snp_dropout_ci_lo': np.nan, 'snp_dropout_ci_hi': np.nan,
                        'snp_dropout_fraction': SNP_DROPOUT_FRAC,
                        'n_bootstrap': N_BOOTSTRAP, 'n_snps_used': 0})
        continue

    # ------------------------------------------------------------------
    # Step 1: Extract per-individual dosages for all SNPs in this trait
    # Result: dosage_matrix shape (n_vcf_samples, n_snps)
    #         betas: array of length n_snps
    # ------------------------------------------------------------------

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

    # Mean-impute missing genotypes per SNP within each population
    for j in range(n_snps):
        col = dosage_matrix[:, j]
        hg_mean = np.nanmean(col[hg_idx]) if not np.all(np.isnan(col[hg_idx])) else 0
        ef_mean = np.nanmean(col[ef_idx]) if not np.all(np.isnan(col[ef_idx])) else 0
        for i in hg_idx:
            if np.isnan(col[i]): dosage_matrix[i, j] = hg_mean
        for i in ef_idx:
            if np.isnan(col[i]): dosage_matrix[i, j] = ef_mean

    # Per-individual PGS: shape (n_vcf_samples,)
    pgs_all = dosage_matrix @ betas

    pgs_hg = pgs_all[hg_idx]
    pgs_ef = pgs_all[ef_idx]
    d_obs  = cohens_d(pgs_ef, pgs_hg)
    print(f"  Observed Cohen's d (reconstructed): {d_obs:.4f}  "
          f"(main summary: {trait_row['cohens_d']:.4f})")

    # ------------------------------------------------------------------
    # Option A: Bootstrap individuals
    # Resample WHG and EF independently with replacement
    # ------------------------------------------------------------------

    print(f"  Running bootstrap ({N_BOOTSTRAP} iterations)...")
    boot_d = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        hg_boot = pgs_hg[rng.integers(0, n_hg, size=n_hg)]
        ef_boot = pgs_ef[rng.integers(0, n_ef, size=n_ef)]
        boot_d[b] = cohens_d(ef_boot, hg_boot)

    boot_lo = float(np.percentile(boot_d, lo_pct))
    boot_hi = float(np.percentile(boot_d, hi_pct))
    print(f"  Bootstrap CI: [{boot_lo:.4f}, {boot_hi:.4f}]")

    # ------------------------------------------------------------------
    # Option C: Delete-d SNP dropout
    # Remove a random SNP_DROPOUT_FRAC fraction of SNPs each iteration,
    # recalculate Cohen's d on the remainder, take percentile CI.
    # More meaningful than leave-one-out for large SNP sets.
    # ------------------------------------------------------------------

    n_drop = max(1, int(np.floor(n_snps * SNP_DROPOUT_FRAC)))
    print(f"  Running SNP dropout ({N_BOOTSTRAP} iterations, "
          f"dropping {n_drop}/{n_snps} SNPs = {SNP_DROPOUT_FRAC*100:.0f}% each time)...")

    dropout_d = np.empty(N_BOOTSTRAP)
    all_idx   = np.arange(n_snps)
    for b in range(N_BOOTSTRAP):
        keep       = rng.choice(all_idx, size=n_snps - n_drop, replace=False)
        pgs_drop   = dosage_matrix[:, keep] @ betas[keep]
        dropout_d[b] = cohens_d(pgs_drop[ef_idx], pgs_drop[hg_idx])

    jk_lo = float(np.percentile(dropout_d, lo_pct))
    jk_hi = float(np.percentile(dropout_d, hi_pct))
    print(f"  SNP dropout CI: [{jk_lo:.4f}, {jk_hi:.4f}]")

    # ------------------------------------------------------------------
    # Option D: Leave-one-SNP-out
    # ------------------------------------------------------------------

    print(f"  Running leave-one-out ({n_snps} SNPs)...")
    for j in range(n_snps):
        mask    = np.ones(n_snps, dtype=bool); mask[j] = False
        pgs_loo = dosage_matrix[:, mask] @ betas[mask]
        d_loo   = cohens_d(pgs_loo[ef_idx], pgs_loo[hg_idx])
        snp_row = contrib.iloc[j]
        loo_rows.append({
            'trait':             trait_name,
            'category':          category_name,
            'snp_id':            snp_row.get('snp_id', ''),
            'chr':               snp_row.get('chr', ''),
            'pos':               snp_row.get('pos', ''),
            'beta':              float(betas[j]),
            'block_id':          snp_row.get('block_id', ''),
            'MAPPED_GENE':       snp_row.get('MAPPED_GENE', ''),
            'SNPS':              snp_row.get('SNPS', ''),
            'cohens_d_original': d_obs,
            'cohens_d_loo':      d_loo,
            'delta_d':           d_loo - d_obs,
            'crosses_zero':      bool((d_obs > 0) != (d_loo > 0)),
        })
    print(f"  Leave-one-out done")

    results.append({
        **trait_row.to_dict(),
        'cohens_d_reconstructed': d_obs,
        'bootstrap_ci_lo':        boot_lo,
        'bootstrap_ci_hi':        boot_hi,
        'snp_dropout_ci_lo':      jk_lo,
        'snp_dropout_ci_hi':      jk_hi,
        'snp_dropout_fraction':   SNP_DROPOUT_FRAC,
        'n_bootstrap':            N_BOOTSTRAP,
        'n_snps_used':            n_snps,
    })

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

print(f"\n{'='*80}\nSAVING\n{'='*80}")

out_df = pd.DataFrame(results)
out_df.to_csv(output_file, sep='\t', index=False)
print(f"✓ {output_file}  ({len(out_df)} traits)")

loo_df = pd.DataFrame(loo_rows)
loo_df.to_csv(output_loo, sep='\t', index=False)
print(f"✓ {output_loo}  ({len(loo_df)} rows, "
      f"{loo_df['trait'].nunique() if len(loo_df) else 0} traits)")
print(f"  Columns: {list(loo_df.columns) if len(loo_df) else '(empty)'}")
print("="*80)