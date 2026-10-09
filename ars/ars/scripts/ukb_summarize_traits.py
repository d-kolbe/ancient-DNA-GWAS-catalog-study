#!/usr/bin/env python3
"""
ukb_summarize_traits.py
========================
Snakemake script — Step 7 of UKB ARS pipeline.

Concatenates per-trait ars_results.tsv files into a single summary table
and annotates each row with the original human-readable trait name.

Output columns
--------------
trait           : original GWAS catalog trait name
safe_trait      : filesystem-safe version
ancestry        : ancestry label (EF / ST / WHG)
observed_z      : observed z-score (full-sample ARS, z-normalised) — LEGACY
bootstrap_mean_z: mean z-score across bootstrap iterations — LEGACY
ci_lower        : 2.5th percentile of bootstrap z-score distribution — LEGACY
ci_upper        : 97.5th percentile — LEGACY
ci_width        : ci_upper - ci_lower (convenience column for ranking) — LEGACY
observed_raw_ars: full-sample raw ARS (un-normalised) — for reference
n_variants      : variants contributing to this trait's ARS
n_samples       : UKB samples used
n_bootstrap     : bootstrap iterations
n_boot_samples  : individuals drawn per iteration

The following are PER-TRAIT (not per-ancestry — same value repeated across
that trait's 3 ancestry rows), pulled from each trait's
bootstrap_distributions.npz, and reflect the current EF-WHG contrast
pipeline (individual-level bootstrap where available, see bootstrap_method):
bootstrap_method       : 'individual' | 'variant' | 'none'
observed_contrast_std  : (ARS_EF - ARS_WHG) / std_bootstrap
ci_lower_std           : lower 95% CI bound, std_bootstrap-scaled
ci_upper_std           : upper 95% CI bound, std_bootstrap-scaled
std_bootstrap          : bootstrap standard error of the raw contrast
observed_contrast_persd: (ARS_EF - ARS_WHG) / person_sd
ci_lower_persd         : lower 95% CI bound, person_sd-scaled
ci_upper_persd         : upper 95% CI bound, person_sd-scaled
person_sd              : population SD of each individual's own contrast
contrast_p_value       : empirical two-sided bootstrap p-value
contrast_direction     : 'EF_higher' | 'WHG_higher' | 'ns' (from observed_contrast_std)
contrast_significant   : bool, True if ci_lower_std/ci_upper_std exclude zero

The table is sorted by ancestry, then by bootstrap_mean_z descending so
the most differentiated trait × ancestry combinations appear at the top.
A second sheet (wide format) is also written: one row per trait, one column
per ancestry for bootstrap_mean_z and ci_lower/ci_upper, plus the per-trait
contrast columns above (not ancestry-specific, so appear once per trait).
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stderr
)
log = logging.getLogger(__name__)


def load_contrast_fields(npz_path):
    """
    Pull the per-trait EF-WHG contrast fields from one trait's
    bootstrap_distributions.npz. Returns a dict of scalar values (NaN /
    'none' for any field missing, e.g. an npz predating this pipeline
    version). Same empirical p-value formula used by plot_ars_forest.py,
    kept here so this summary table is self-contained and doesn't require
    cross-referencing the plot script to check significance.
    """
    out = {
        'bootstrap_method':        'none',
        'observed_contrast_std':   np.nan,
        'ci_lower_std':            np.nan,
        'ci_upper_std':            np.nan,
        'std_bootstrap':           np.nan,
        'observed_contrast_persd': np.nan,
        'ci_lower_persd':          np.nan,
        'ci_upper_persd':          np.nan,
        'person_sd':               np.nan,
        'contrast_p_value':        np.nan,
        'contrast_direction':      'ns',
        'contrast_significant':    False,
    }
    if not os.path.exists(npz_path):
        return out

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        log.warning(f"  Could not load {npz_path}: {e}")
        return out

    for key in ('observed_contrast_std', 'ci_lower_std', 'ci_upper_std',
                'std_bootstrap', 'observed_contrast_persd', 'ci_lower_persd',
                'ci_upper_persd', 'person_sd'):
        if key in data:
            out[key] = float(data[key])

    if 'bootstrap_method' in data:
        out['bootstrap_method'] = str(data['bootstrap_method'])

    if 'boot_raw_contrast' in data:
        boot = data['boot_raw_contrast']
        valid = boot[~np.isnan(boot)]
        if len(valid) > 0:
            p = 2.0 * min(float(np.mean(valid <= 0)), float(np.mean(valid >= 0)))
            out['contrast_p_value'] = max(p, 1.0 / len(valid))

    ci_lo, ci_hi = out['ci_lower_std'], out['ci_upper_std']
    if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
        out['contrast_significant'] = bool(ci_lo > 0 or ci_hi < 0)
        if out['contrast_significant']:
            d = out['observed_contrast_std']
            out['contrast_direction'] = 'EF_higher' if d > 0 else 'WHG_higher'

    return out

# =============================================================================
# Snakemake interface
# =============================================================================

TRAIT_RESULT_FILES  = snakemake.input.trait_results   # list of per-trait TSVs
BOOTSTRAP_DIST_FILES = snakemake.input.bootstrap_dists  # list of per-trait npz
TRAIT_MANIFEST      = snakemake.input.trait_manifest

OUTPUT_SUMMARY      = snakemake.output.summary

SAFE_TO_ORIG        = snakemake.params.safe_to_orig   # dict safe_trait → original


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(os.path.dirname(OUTPUT_SUMMARY), exist_ok=True)

    # Load manifest for original names and variant counts
    manifest_df = pd.read_csv(TRAIT_MANIFEST, sep='\t')
    manifest_map = dict(zip(manifest_df['safe_trait'], manifest_df['original_trait']))

    # Map safe_trait -> its bootstrap_distributions.npz path
    # (results/{safe_trait}/bootstrap_distributions.npz, same layout as the tsv)
    npz_by_safe = {Path(p).parent.name: p for p in BOOTSTRAP_DIST_FILES}

    all_rows = []

    for result_file in TRAIT_RESULT_FILES:
        path = Path(result_file)
        if not path.exists():
            log.warning(f"Missing result file: {result_file}")
            continue

        try:
            df = pd.read_csv(result_file, sep='\t')
        except Exception as e:
            log.warning(f"Could not read {result_file}: {e}")
            continue

        if df.empty:
            log.info(f"Empty result file: {result_file}")
            continue

        # Infer safe_trait from path structure: results/{safe_trait}/ars_results.tsv
        safe_trait = path.parent.name

        # Get original trait name (fallback to safe_trait if mapping missing)
        orig_name = (
            SAFE_TO_ORIG.get(safe_trait)
            or manifest_map.get(safe_trait)
            or safe_trait
        )

        df.insert(0, 'trait',      orig_name)
        df.insert(1, 'safe_trait', safe_trait)

        # Merge in the per-trait EF-WHG contrast fields (same value repeated
        # across this trait's ancestry rows, since these are trait-level not
        # ancestry-level quantities)
        contrast_fields = load_contrast_fields(npz_by_safe.get(safe_trait, ''))
        for k, v in contrast_fields.items():
            df[k] = v

        all_rows.append(df)
        log.info(f"  [{safe_trait}] {len(df)} rows loaded "
                 f"(bootstrap_method={contrast_fields['bootstrap_method']})")

    if not all_rows:
        log.warning("No trait results found — writing empty summary")
        pd.DataFrame().to_csv(OUTPUT_SUMMARY, sep='\t', index=False)
        return

    summary = pd.concat(all_rows, ignore_index=True)

    # Convenience columns
    summary['ci_width'] = summary['ci_upper'] - summary['ci_lower']

    # Ensure numeric types
    num_cols = ['observed_z', 'bootstrap_mean_z', 'ci_lower', 'ci_upper',
                'ci_width', 'observed_raw_ars', 'bootstrap_mean_raw_ars',
                'n_variants', 'n_samples', 'n_bootstrap', 'n_boot_samples',
                'observed_contrast_std', 'ci_lower_std', 'ci_upper_std',
                'std_bootstrap', 'observed_contrast_persd', 'ci_lower_persd',
                'ci_upper_persd', 'person_sd', 'contrast_p_value']
    for col in num_cols:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors='coerce')

    # Sort: by ancestry, then by abs(bootstrap_mean_z) descending
    summary['_abs_z'] = summary['bootstrap_mean_z'].abs()
    summary = summary.sort_values(
        ['ancestry', '_abs_z'],
        ascending=[True, False]
    ).drop(columns='_abs_z').reset_index(drop=True)

    # Write long-format summary
    summary.to_csv(OUTPUT_SUMMARY, sep='\t', index=False)
    log.info(f"Long-format summary written: {OUTPUT_SUMMARY} ({len(summary):,} rows)")

    # ------------------------------------------------------------------
    # Also write wide-format: one row per trait, columns per ancestry
    # ------------------------------------------------------------------
    wide_out = OUTPUT_SUMMARY.replace('.tsv', '_wide.tsv')
    try:
        wide = summary.pivot_table(
            index=['trait', 'safe_trait', 'n_variants', 'n_samples'],
            columns='ancestry',
            values=['bootstrap_mean_z', 'ci_lower', 'ci_upper', 'ci_width'],
            aggfunc='first'
        )
        wide.columns = [f"{val}_{anc}" for val, anc in wide.columns]
        wide = wide.reset_index()

        # Per-trait EF-WHG contrast columns (not ancestry-specific — same
        # value on every ancestry row for a given trait, so take one row
        # per trait rather than pivoting)
        contrast_cols = ['safe_trait', 'bootstrap_method',
                         'observed_contrast_std', 'ci_lower_std', 'ci_upper_std',
                         'std_bootstrap', 'observed_contrast_persd',
                         'ci_lower_persd', 'ci_upper_persd', 'person_sd',
                         'contrast_p_value', 'contrast_direction',
                         'contrast_significant']
        contrast_cols = [c for c in contrast_cols if c in summary.columns]
        per_trait = summary[contrast_cols].drop_duplicates(subset='safe_trait')
        wide = wide.merge(per_trait, on='safe_trait', how='left')

        wide = wide.sort_values(
            by='observed_contrast_std' if 'observed_contrast_std' in wide.columns
                else ([c for c in wide.columns if c.startswith('bootstrap_mean_z')][0]
                      if any(c.startswith('bootstrap_mean_z') for c in wide.columns)
                      else 'trait'),
            ascending=False,
            na_position='last'
        ).reset_index(drop=True)
        wide.to_csv(wide_out, sep='\t', index=False)
        log.info(f"Wide-format summary written: {wide_out} ({len(wide):,} traits)")
    except Exception as e:
        log.warning(f"Could not write wide-format summary: {e}")

    # ------------------------------------------------------------------
    # Print quick digest to stderr
    # ------------------------------------------------------------------
    log.info("\n=== ARS Summary ===")
    for _, row in summary.iterrows():
        if pd.isna(row.get('bootstrap_mean_z')):
            continue
        log.info(
            f"  {row['trait']:<40s}  {row['ancestry']:<6s}  "
            f"z={row['bootstrap_mean_z']:+.3f}  "
            f"[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]  "
            f"({int(row['n_variants'])} SNPs)"
        )

    log.info("Done.")


if __name__ == '__main__':
    main()