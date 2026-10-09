#!/usr/bin/env python3
"""
compute_concordance.py
=======================
Standalone diagnostic script (not part of the Snakemake pipeline) — compares
the Ancestry Risk Score (ARS) EF-WHG contrast results against the independent
GWAS-Catalog-index (ancient-genome PGS) results, trait by trait.

Inputs
------
--ars-wide      ars_summary_all_traits_wide.tsv (from ukb_summarize_traits.py,
                post bootstrap fix — must contain contrast_direction,
                contrast_significant, observed_contrast_std, ci_lower_std,
                ci_upper_std, contrast_p_value columns; see that script's
                docstring)
--gwas-index    bootstrap_ci.txt (or equivalent) — must contain trait,
                direction, ttest_pval, cohens_d, bootstrap_ci_lo,
                bootstrap_ci_hi columns

Matching is on exact trait-name string match. Trait names that don't match
between the two files are reported separately rather than silently dropped —
check these manually (e.g. "X (UKB data field NNNNN)" suffixes, punctuation,
or genuinely different trait sets) before trusting the concordance numbers.

Output
------
Prints a concordance report to stdout, and writes a merged per-trait
comparison table (--out, default concordance_comparison.tsv) with both
sides' direction/significance/effect-size side by side.
"""

import argparse
import sys
import numpy as np
import pandas as pd


def normalise_trait(s):
    """Light normalisation for matching only (not used for display)."""
    return str(s).strip().lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ars-wide',   required=True)
    p.add_argument('--gwas-index', required=True)
    p.add_argument('--out',        default='concordance_comparison.tsv')
    p.add_argument('--alpha',      type=float, default=0.05,
                   help='Significance threshold applied to both sides '
                        'for the "both significant" concordance check.')
    p.add_argument('--min-snps',   type=int, default=20,
                   help="Minimum variant count required on both sides "
                        "(gwas-index's n_snps_extracted, ars-wide's "
                        "n_variants) — matches the >=20 filter already used "
                        "elsewhere in the pipeline (plot_ars_forest.py's "
                        "trait selection). The GWAS-index file is NOT "
                        "pre-filtered by this, unlike the ARS trait list, "
                        "so skipping this filter here would compare against "
                        "many low-powered/noisy GWAS-index traits.")
    p.add_argument('--no-dedup',   action='store_true', default=False,
                   help="Don't deduplicate rows sharing a trait name (see "
                        "duplicate warning below) — off by default, i.e. "
                        "dedup happens automatically. When multiple rows "
                        "share a trait name, the one with the highest "
                        "n_snps_extracted (gwas-index) or n_variants "
                        "(ars-wide) is kept, on the assumption that it "
                        "reflects the more complete variant set of two "
                        "otherwise-identical reprocessings.")
    args = p.parse_args()

    ars = pd.read_csv(args.ars_wide, sep='\t')
    gwas = pd.read_csv(args.gwas_index, sep='\t')

    required_ars = ['trait', 'contrast_direction', 'contrast_significant',
                    'observed_contrast_std', 'contrast_p_value']
    missing = [c for c in required_ars if c not in ars.columns]
    if missing:
        sys.exit(f"ERROR: --ars-wide is missing columns {missing}. "
                 f"Was it produced by the updated ukb_summarize_traits.py, "
                 f"after rerunning bootstrap_ars with the individual-"
                 f"resampling fix for ALL traits (not just one)?")

    required_gwas = ['trait', 'direction', 'ttest_pval', 'cohens_d']
    missing = [c for c in required_gwas if c not in gwas.columns]
    if missing:
        sys.exit(f"ERROR: --gwas-index is missing columns {missing}.")

    # --- Minimum-SNP filtering ---
    # The ARS trait list (snp_sets_summary.txt) is a curated subset of the
    # GWAS-index trait list (bootstrap_ci.txt) — the latter is NOT filtered
    # by variant count the way the former implicitly is, so without this
    # filter the comparison would include many low-powered GWAS-index
    # traits that were never going to be comparable in the first place.
    if 'n_snps_extracted' in gwas.columns:
        n_before = len(gwas)
        gwas = gwas[gwas['n_snps_extracted'] >= args.min_snps].copy()
        print(f"GWAS-index: filtered n_snps_extracted>={args.min_snps}: "
             f"{n_before} -> {len(gwas)}")
    else:
        print("WARNING: --gwas-index has no n_snps_extracted column — "
             "skipping that filter (results may include very low-powered "
             "GWAS-index traits).")

    if 'n_variants' in ars.columns:
        n_before = len(ars)
        ars = ars[ars['n_variants'] >= args.min_snps].copy()
        print(f"ARS: filtered n_variants>={args.min_snps}: "
             f"{n_before} -> {len(ars)}")

    ars['_match_key']  = ars['trait'].map(normalise_trait)
    gwas['_match_key'] = gwas['trait'].map(normalise_trait)

    # --- Duplicate-key diagnostics (BEFORE merging) ---
    # An inner join silently multiplies rows when either side has duplicate
    # keys — better to surface that explicitly than let it quietly inflate
    # or distort the "matched" count.
    ars_dup_keys  = ars['_match_key'][ars['_match_key'].duplicated(keep=False)]
    gwas_dup_keys = gwas['_match_key'][gwas['_match_key'].duplicated(keep=False)]

    if len(ars_dup_keys) > 0:
        print(f"\n=== WARNING: {ars_dup_keys.nunique()} duplicate trait name(s) "
             f"in --ars-wide ===")
        for key in sorted(ars_dup_keys.unique()):
            rows = ars[ars['_match_key'] == key]
            print(f"  '{rows['trait'].iloc[0]}' appears {len(rows)}x "
                 f"(safe_trait: {', '.join(rows['safe_trait'].astype(str))})"
                 if 'safe_trait' in ars.columns else
                 f"  '{rows['trait'].iloc[0]}' appears {len(rows)}x")

    if len(gwas_dup_keys) > 0:
        print(f"\n=== WARNING: {gwas_dup_keys.nunique()} duplicate trait name(s) "
             f"in --gwas-index ===")
        for key in sorted(gwas_dup_keys.unique()):
            rows = gwas[gwas['_match_key'] == key]
            detail_cols = [c for c in ['direction', 'ttest_pval', 'cohens_d',
                                       'n_snps_extracted', 'category']
                          if c in rows.columns]
            print(f"  '{rows['trait'].iloc[0]}' appears {len(rows)}x:")
            for _, r in rows.iterrows():
                print(f"      {r[detail_cols].to_dict()}")
        print(f"  (each duplicate causes its matching ARS trait to appear "
             f"multiple times in the merged output below — inspect these "
             f"rows to determine whether they're genuinely distinct trait "
             f"definitions that happen to share a name, or true duplicate "
             f"rows that should be deduplicated upstream before comparing.)")

    if not args.no_dedup:
        if len(gwas_dup_keys) > 0 and 'n_snps_extracted' in gwas.columns:
            n_before = len(gwas)
            gwas = (gwas.sort_values('n_snps_extracted', ascending=False)
                        .drop_duplicates(subset='_match_key', keep='first'))
            print(f"\nDeduplicated --gwas-index (kept max n_snps_extracted "
                 f"per trait): {n_before} -> {len(gwas)}")
        if len(ars_dup_keys) > 0 and 'n_variants' in ars.columns:
            n_before = len(ars)
            ars = (ars.sort_values('n_variants', ascending=False)
                      .drop_duplicates(subset='_match_key', keep='first'))
            print(f"Deduplicated --ars-wide (kept max n_variants per trait): "
                 f"{n_before} -> {len(ars)}")
    elif len(ars_dup_keys) > 0 or len(gwas_dup_keys) > 0:
        print("\n--no-dedup set: duplicate rows above were NOT removed — "
             "matched counts and concordance percentages below will "
             "double-count them.")

    merged = ars.merge(gwas, on='_match_key', how='inner',
                       suffixes=('_ars', '_gwas'))
    unmatched_ars  = ars[~ars['_match_key'].isin(gwas['_match_key'])]
    unmatched_gwas = gwas[~gwas['_match_key'].isin(ars['_match_key'])]

    print(f"=== Concordance report ===")
    print(f"ARS traits:        {len(ars):,}")
    print(f"GWAS-index traits: {len(gwas):,}")
    print(f"Matched by name:   {len(merged):,}")
    print(f"Unmatched (ARS only):        {len(unmatched_ars):,}")
    print(f"Unmatched (GWAS-index only): {len(unmatched_gwas):,}")
    if len(unmatched_ars) > 0 or len(unmatched_gwas) > 0:
        print("  (see --out '_unmatched.tsv' siblings for the lists — check "
              "these before trusting the concordance numbers below; trait "
              "name mismatches, not real absence, are the likely cause for "
              "most of them)")

    if len(merged) == 0:
        sys.exit("No traits matched by name — check trait-name formatting "
                 "on both sides before proceeding.")

    # --- Direction concordance (regardless of significance) ---
    gwas_dir = merged['direction'].str.replace('_higher', '', regex=False).str.upper()
    ars_dir  = merged['contrast_direction'].replace(
        {'EF_higher': 'EF', 'WHG_higher': 'WHG', 'ns': np.nan})
    both_have_direction = ars_dir.notna() & gwas_dir.notna()
    direction_agree = (ars_dir == gwas_dir) & both_have_direction
    n_comparable = int(both_have_direction.sum())
    n_agree      = int(direction_agree.sum())
    print(f"\nDirection agreement (ARS significant, excludes ARS 'ns'): "
         f"{n_agree}/{n_comparable} "
         f"({100*n_agree/n_comparable:.1f}%)" if n_comparable else
         "\nDirection agreement: no ARS-significant traits to compare")

    # --- Significant in both, and agree ---
    ars_sig  = merged['contrast_p_value'] < args.alpha
    gwas_sig = merged['ttest_pval'] < args.alpha
    both_sig = ars_sig & gwas_sig
    n_both_sig = int(both_sig.sum())
    n_both_sig_agree = int((both_sig & (ars_dir == gwas_dir)).sum())
    print(f"Significant in both (p<{args.alpha}): {n_both_sig}/{len(merged)}")
    if n_both_sig > 0:
        print(f"  ...of which direction agrees: {n_both_sig_agree}/{n_both_sig} "
             f"({100*n_both_sig_agree/n_both_sig:.1f}%)")

    # --- Discordant traits (both significant, opposite direction) ---
    discordant = merged[both_sig & (ars_dir != gwas_dir) & both_have_direction]
    if len(discordant) > 0:
        print(f"\n=== Discordant traits (both significant, opposite direction) ===")
        for _, row in discordant.iterrows():
            print(f"  {row['trait_ars']}: ARS={row['contrast_direction']} "
                 f"(p={row['contrast_p_value']:.3g}), "
                 f"GWAS-index={row['direction']} (p={row['ttest_pval']:.3g})")

    # --- Effect-size correlation (informational — different scales, but
    #     the sign/rank relationship is still informative) ---
    valid_es = merged['observed_contrast_std'].notna() & merged['cohens_d'].notna()
    if valid_es.sum() > 2:
        r = np.corrcoef(merged.loc[valid_es, 'observed_contrast_std'],
                        merged.loc[valid_es, 'cohens_d'])[0, 1]
        print(f"\nEffect-size correlation (observed_contrast_std vs cohens_d, "
             f"n={int(valid_es.sum())}): r={r:.3f}")
        print("  (informational only — these are different statistics on "
             "different scales; a strong correlation supports convergent "
             "validity, it is not itself a concordance measure)")

    # --- Write full comparison table ---
    out_cols = ['trait_ars', 'category_ars' if 'category_ars' in merged.columns else None,
               'contrast_direction', 'contrast_significant', 'contrast_p_value',
               'observed_contrast_std', 'ci_lower_std', 'ci_upper_std',
               'direction', 'ttest_pval', 'cohens_d',
               'bootstrap_ci_lo', 'bootstrap_ci_hi']
    out_cols = [c for c in out_cols if c and c in merged.columns]
    merged[out_cols].to_csv(args.out, sep='\t', index=False)
    print(f"\nFull comparison table written: {args.out}")

    if len(unmatched_ars) > 0:
        unmatched_ars[['trait']].to_csv(
            args.out.replace('.tsv', '_unmatched_ars.tsv'), sep='\t', index=False)
    if len(unmatched_gwas) > 0:
        unmatched_gwas[['trait']].to_csv(
            args.out.replace('.tsv', '_unmatched_gwas.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    main()