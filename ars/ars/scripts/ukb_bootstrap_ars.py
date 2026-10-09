#!/usr/bin/env python3
"""
ukb_bootstrap_ars.py
=====================
Snakemake script — Step 5 of UKB ARS pipeline.

Computes observed ARS values and bootstrap confidence intervals for one trait.

Algorithm
---------
1. Load merged contribution arrays (n_samples, n_ancestries).

2. Observed ARS:
       ARS_k = Σ_i numerator[i,k] / Σ_i denominator[i,k]
   for each ancestry k. Denominator represents the total ancestry-weighted
   allele exposure across all variants; normalising by it accounts for
   differences in ANP coverage across samples.

3. Z-normalise across ancestries (within the observed values):
       z_k = (ARS_k - mean(ARS)) / std(ARS)
   This allows comparison of ancestries on a common scale regardless of
   trait effect-size units.

4. Bootstrap for the legacy per-ancestry z-score output only (1000 × 10k
   individuals by default):
   For each iteration b:
     a. Draw n_boot_samples indices with replacement from 0..n_samples-1
     b. Recompute ARS_k from resampled rows (fast: numpy sum on index array)
     c. Z-normalise within this iteration
   This gives a distribution of z-scores per ancestry → feeds
   bootstrap_mean_z/ci_lower/ci_upper in ars_results.tsv (kept in the summary
   tables). NOT used for the raw EF-WHG contrast/CI/LOO (see #4b).

4b. Bootstrap for the EF-WHG contrast, its CI, and LOO — proper individual
    resampling (run_individual_bootstrap), used whenever the merged.npz has
    the per-person, per-variant grid (var_person_freq_num/var_person_denom,
    written by ukb_compute_contributions.py). Each iteration resamples
    PEOPLE and, for each variant, recomputes
        F_vk(resample) = Σ_i(resampled) X_i,v,k / Σ_i(resampled) denom_i,v,k
    from only the resampled subset, before summing with beta weights. This
    is the mathematically correct way to resample individuals for this
    ratio-based statistic.

    The earlier (WRONG) approach pooled numerator/denominator across
    everyone, across ALL variants, BEFORE dividing once — a different (and
    biased, ~20x off in testing) quantity than the correct
    Σ_v beta_v × F_vk formula, since dividing per-variant then summing is
    not the same operation as summing then dividing once, when denominators
    vary across variants (they do).

    Falls back to variant-level resampling (run_variant_bootstrap — draws
    this trait's variants with replacement instead of people, using the
    already-correct precomputed F_vk directly) if a trait's merged.npz
    predates the per-person grid. Also unbiased, but answers a different
    question (variant-selection uncertainty rather than person-sampling
    uncertainty). Which one actually ran is recorded in bootstrap_method
    ('individual' | 'variant' | 'none') in the output npz.

5. Also compute a raw (un-z-normalised) ARS for the summary table, so that
   the absolute magnitude is available if needed.

6. Person-level effect-size denominator (added alongside the bootstrap-SE
   approach above, not replacing it):
       R_i,k     = numerator[i,k] / denominator[i,k]   (person i's own ratio)
       person_sd = std(R_i,EF - R_i,WHG, ddof=1)  across the full cohort
   This is an ordinary population SD, computed once directly from all
   ~490k people — unlike std_bootstrap (a resampling standard error), it
   does not shrink as n_boot_samples grows, so the contrast expressed in
   person_sd units is comparable across traits regardless of that knob.
   The bootstrap CI is still built from the same resampled contrast
   distribution as the std_bootstrap approach; only the denominator used
   to express its bounds changes.

Outputs
-------
ars_results.tsv
    ancestry | observed_z | bootstrap_mean_z | ci_lower | ci_upper |
    observed_raw_ars | bootstrap_mean_raw_ars | n_variants | n_samples |
    n_bootstrap | n_boot_samples

bootstrap_distributions.npz
    z_scores               : float32 (n_bootstrap, n_ancestries)  — full distribution
    raw_ars                : float32 (n_bootstrap, n_ancestries)
    std_bootstrap, observed_contrast_std, ci_lower_std, ci_upper_std,
      loo_contrasts                   — resampling-SE-scaled (n_boot_samples-dependent)
    person_sd, observed_contrast_persd, ci_lower_persd, ci_upper_persd,
      loo_contrasts_persd, n_valid_person
                                       — population-SD-scaled (Option B, see above)
    ancestry_names          : str   (n_ancestries,)
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

# =============================================================================
# Snakemake interface
# =============================================================================

MERGED_NPZ      = snakemake.input.merged

OUTPUT_RESULTS  = snakemake.output.results
OUTPUT_BOOT     = snakemake.output.bootstrap_dist

SAFE_TRAIT      = snakemake.params.safe_trait
ANCESTRY_NAMES  = snakemake.params.ancestry_names
N_BOOTSTRAP     = int(snakemake.params.n_bootstrap)
N_BOOT_SAMPLES  = int(snakemake.params.n_samples)
RANDOM_SEED     = int(snakemake.params.random_seed)

N_ANCESTRIES    = len(ANCESTRY_NAMES)


# =============================================================================
# Core functions
# =============================================================================

def compute_ars(num_rows: np.ndarray, den_rows: np.ndarray) -> np.ndarray:
    """
    Compute ARS per ancestry by summing rows and dividing.

    Works for both:
      - Individual bootstrap: num_rows = (n_samples, n_anc)
      - SNP bootstrap:        num_rows = (n_variants, n_anc)

    Returns
    -------
    ars : (n_anc,) — NaN where denominator is zero
    """
    num_sum = num_rows.sum(axis=0)
    den_sum = den_rows.sum(axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        ars = np.where(den_sum > 0, num_sum / den_sum, np.nan)
    return ars


def z_normalise(ars: np.ndarray) -> np.ndarray:
    """
    Z-normalise ARS values across ancestries.
    Returns NaN array if std == 0 or all values are NaN.
    """
    valid = ~np.isnan(ars)
    if valid.sum() < 2:
        return np.full_like(ars, np.nan)
    mu  = ars[valid].mean()
    std = ars[valid].std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.full_like(ars, np.nan)
    z = np.full_like(ars, np.nan)
    z[valid] = (ars[valid] - mu) / std
    return z


def run_bootstrap(numerator: np.ndarray,
                  denominator: np.ndarray,
                  n_bootstrap: int,
                  n_boot_samples: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap resampling of individuals.

    NOTE — scope of this function: kept only to feed the legacy per-ancestry
    z-score output (bootstrap_mean_z / ci_lower / ci_upper in ars_results.tsv,
    kept in the summary tables). z-normalising across the
    3 ancestries happens to cancel out the pooling bias described below (a
    roughly uniform rescaling of all 3 ancestries doesn't change their
    z-scores), so that legacy output is unaffected by the issue.

    Do NOT use this for the raw EF-WHG contrast, its CI, or LOO — see
    run_variant_bootstrap() below for why, and for what replaced it there.

    Returns
    -------
    boot_z   : (n_bootstrap, n_ancestries)
    boot_raw : (n_bootstrap, n_ancestries)
    """
    n_samples = numerator.shape[0]
    n_boot_samples = min(n_boot_samples, n_samples)

    boot_z   = np.full((n_bootstrap, N_ANCESTRIES), np.nan, dtype=np.float32)
    boot_raw = np.full((n_bootstrap, N_ANCESTRIES), np.nan, dtype=np.float32)

    log.info(f"  Bootstrapping (legacy per-ancestry z-scores only): "
             f"{n_bootstrap} iterations × "
             f"{n_boot_samples:,} samples (pool: {n_samples:,})")

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_boot_samples)
        ars_b = compute_ars(numerator[idx], denominator[idx])
        z_b   = z_normalise(ars_b)
        boot_z[b]   = z_b
        boot_raw[b] = ars_b.astype(np.float32)

        if (b + 1) % 200 == 0:
            log.info(f"    {b + 1}/{n_bootstrap} iterations complete")

    return boot_z, boot_raw


def run_variant_bootstrap(F_vk: np.ndarray, betas: np.ndarray,
                          ef_idx: int, whg_idx: int,
                          n_bootstrap: int,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Bootstrap resampling of VARIANTS (SNPs), not individuals.

    Why variants, not people
    ------------------------
    ARS_k = Σ_v beta_v × F_vk, where each F_vk is already correctly computed
    from the full ~490k-person cohort (F_vk = freq_num_v,k / freq_den_v,k,
    divided PER VARIANT). The previous approach instead resampled people and
    pooled their numerator/denominator sums across every variant BEFORE
    dividing:
        Σ_v beta_v×freq_num_v,k  /  Σ_v freq_den_v,k
    which only equals Σ_v beta_v×F_vk if every variant happens to have an
    identical denominator — false in general, since freq_den_v,k varies with
    each variant's ANP coverage/missingness. In testing on a real trait this
    produced bootstrap distributions centred ~20x off from the actual
    point estimate — a real bias, not sampling noise.

    Resampling variants instead sidesteps this entirely: each F_vk is used
    as-is (already correct, never re-aggregated), and each iteration asks
    "how much would the weighted sum shift if a different random slate of
    this trait's variants had come out of the GWAS?". That's also arguably
    the more meaningful source of uncertainty for this design in the first
    place — with a fixed, ~490k-person cohort, person-resampling uncertainty
    is mechanically tiny regardless of the true effect, whereas which
    variants ended up defining the score is a genuine source of variability.

    Returns
    -------
    boot_contrast : (n_valid,) — ARS_EF - ARS_WHG per iteration, NaNs dropped
    """
    n_v = F_vk.shape[0]
    boot_contrast = np.full(n_bootstrap, np.nan, dtype=np.float64)

    log.info(f"  Variant bootstrap: {n_bootstrap} iterations × "
             f"{n_v:,} variants (with replacement, same trait)")

    for b in range(n_bootstrap):
        idx   = rng.integers(0, n_v, size=n_v)
        ars_b = np.nansum(betas[idx, np.newaxis] * F_vk[idx], axis=0)  # (n_anc,)
        boot_contrast[b] = ars_b[ef_idx] - ars_b[whg_idx]

    return boot_contrast[~np.isnan(boot_contrast)]


def run_individual_bootstrap(var_person_freq_num: np.ndarray,
                             var_person_denom: np.ndarray,
                             betas: np.ndarray,
                             ef_idx: int, whg_idx: int,
                             n_bootstrap: int,
                             n_boot_samples: int,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Proper individual-level bootstrap: resamples PEOPLE, and for each
    resampled subset, recomputes F_vk correctly — per variant, using only
    the resampled people's data — before summing with beta weights. This is
    the mathematically correct way to resample individuals, unlike the
    original approach (see run_bootstrap docstring), which pooled
    numerator/denominator across ALL variants before dividing once — a
    different (and ~20x biased, in testing) calculation than the real
    ARS_k = Σ_v beta_v × F_vk formula.

    Requires the full per-person, per-variant grid (var_person_freq_num,
    var_person_denom — see ukb_compute_contributions.py), not just the
    per-person-summed-across-variants numerator/denominator, since F_vk
    must be recomputed one variant at a time for each resampled subset —
    it cannot be derived from arrays that have already collapsed the
    per-variant structure away.

    Parameters
    ----------
    var_person_freq_num : (n_variants, n_samples, n_anc) float32
        X_i,v,k — per-person, per-variant effect-allele-carrying indicator
        weighted by ancestry probability, no beta applied.
    var_person_denom : (n_variants, n_samples, n_anc) float32
        Matching per-person, per-variant total ancestry attribution
        (ANP1+ANP2).
    betas : (n_variants,)

    Returns
    -------
    boot_contrast : (n_valid,) — ARS_EF - ARS_WHG per iteration, NaNs dropped
    """
    n_variants, n_samples, n_anc = var_person_freq_num.shape
    n_boot_samples = min(n_boot_samples, n_samples)
    boot_contrast  = np.full(n_bootstrap, np.nan, dtype=np.float64)

    log.info(f"  Individual bootstrap: {n_bootstrap} iterations × "
             f"{n_boot_samples:,} people resampled per iteration "
             f"(pool: {n_samples:,}), {n_variants:,} variants")

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_boot_samples)

        # Gather only the resampled people for every variant at once, then
        # sum over people -> (n_variants, n_anc) population sums computed
        # ONLY from the resampled subset (repeats counted multiple times,
        # as a bootstrap resample-with-replacement requires).
        freq_num_b = var_person_freq_num[:, idx, :].sum(axis=1)   # (n_variants, n_anc)
        freq_den_b = var_person_denom[:,    idx, :].sum(axis=1)   # (n_variants, n_anc)

        with np.errstate(invalid='ignore', divide='ignore'):
            F_vk_b = np.where(freq_den_b > 0, freq_num_b / freq_den_b, np.nan)

        ars_b = np.nansum(betas[:, np.newaxis] * F_vk_b, axis=0)  # (n_anc,)
        boot_contrast[b] = ars_b[ef_idx] - ars_b[whg_idx]

        if (b + 1) % 200 == 0:
            log.info(f"    {b + 1}/{n_bootstrap} iterations complete")

    return boot_contrast[~np.isnan(boot_contrast)]


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(os.path.dirname(OUTPUT_RESULTS), exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    # -------------------------------------------------------------------------
    # Load merged contributions
    # -------------------------------------------------------------------------
    log.info(f"[{SAFE_TRAIT}] Loading merged contributions: {MERGED_NPZ}")
    data = np.load(MERGED_NPZ, allow_pickle=True)

    numerator   = data['numerator'].astype(np.float64)
    denominator = data['denominator'].astype(np.float64)
    n_variants  = int(data['n_variants'])
    n_samples   = int(data['n_samples']) if 'n_samples' in data else numerator.shape[0]

    # Per-variant arrays
    has_var_arrays = ('var_freq_num' in data and
                      data['var_freq_num'].shape[0] > 0)
    if has_var_arrays:
        var_freq_num    = data['var_freq_num'].astype(np.float64)
        var_denominator = data['var_denominator'].astype(np.float64)
        var_numerator   = data['var_numerator'].astype(np.float64)
        # Stored chrom/pos per variant for positional join with weights
        var_chroms_npz  = data['var_chroms'].astype(np.int32) if 'var_chroms' in data else None
        var_pos_npz     = data['var_pos'].astype(np.int32)    if 'var_pos'    in data else None
        log.info(f"  var arrays: {var_freq_num.shape[0]} variants x {N_ANCESTRIES} ancestries")
    else:
        log.warning(f"  [{SAFE_TRAIT}] No var_freq_num - using pooled ARS only")
        var_chroms_npz = var_pos_npz = None

    # Per-person, per-variant grid — enables the proper individual-level
    # bootstrap (run_individual_bootstrap). Falls back to variant-level
    # resampling (run_variant_bootstrap) below if this trait's merged.npz
    # hasn't been regenerated yet with the updated extraction/merge scripts.
    has_person_arrays = ('var_person_freq_num' in data and
                         data['var_person_freq_num'].shape[0] > 0)
    if has_person_arrays:
        var_person_freq_num = data['var_person_freq_num'].astype(np.float32)
        var_person_denom    = data['var_person_denom'].astype(np.float32)
        log.info(f"  var_person grid: {var_person_freq_num.shape} "
                 f"({var_person_freq_num.nbytes / 1e9:.2f} GB)")
    else:
        var_person_freq_num = var_person_denom = None
        log.warning(f"  [{SAFE_TRAIT}] No var_person_freq_num — this trait's "
                    f"merged.npz predates the individual-bootstrap upgrade; "
                    f"falling back to variant-level resampling for the CI. "
                    f"Rerun compute_contributions/merge_contributions for "
                    f"this trait to get the proper per-person bootstrap.")

    log.info(f"  {n_samples:,} samples | {n_variants:,} variants | "
             f"{N_ANCESTRIES} ancestries")

    if n_samples == 0 or n_variants == 0:
        log.warning(f"[{SAFE_TRAIT}] No data — writing empty results")
        empty_df = pd.DataFrame({
            'ancestry':               ANCESTRY_NAMES,
            'observed_z':             [np.nan] * N_ANCESTRIES,
            'bootstrap_mean_z':       [np.nan] * N_ANCESTRIES,
            'ci_lower':               [np.nan] * N_ANCESTRIES,
            'ci_upper':               [np.nan] * N_ANCESTRIES,
            'observed_raw_ars':       [np.nan] * N_ANCESTRIES,
            'bootstrap_mean_raw_ars': [np.nan] * N_ANCESTRIES,
            'n_variants':             [n_variants] * N_ANCESTRIES,
            'n_samples':              [n_samples] * N_ANCESTRIES,
            'n_bootstrap':            [N_BOOTSTRAP] * N_ANCESTRIES,
            'n_boot_samples':         [N_BOOT_SAMPLES] * N_ANCESTRIES,
        })
        empty_df.to_csv(OUTPUT_RESULTS, sep='\t', index=False)
        np.savez_compressed(
            OUTPUT_BOOT,
            z_scores       = np.full((N_BOOTSTRAP, N_ANCESTRIES), np.nan, dtype=np.float32),
            raw_ars        = np.full((N_BOOTSTRAP, N_ANCESTRIES), np.nan, dtype=np.float32),
            loo_contrasts  = np.array([]),
            loo_chroms     = np.array([]),
            loo_pos        = np.array([]),
            ancestry_names = np.array(ANCESTRY_NAMES)
        )
        return

    # -------------------------------------------------------------------------
    # Load weights to get betas (sorted by CHROM, POS to match var arrays)
    # -------------------------------------------------------------------------
    weights_path = os.path.join(
        os.path.dirname(os.path.dirname(MERGED_NPZ)),
        '..', 'variant_lists', 'weights', f'{SAFE_TRAIT}_weights.tsv'
    )
    weights_path = os.path.normpath(weights_path)
    betas        = None
    wdf          = None
    wdf_matched  = None
    if os.path.exists(weights_path) and has_var_arrays:
        try:
            wdf_all = (pd.read_csv(weights_path, sep='\t')
                         .drop_duplicates(subset=['CHROM', 'POS'])
                         .sort_values(['CHROM', 'POS'])
                         .reset_index(drop=True))
            if var_chroms_npz is not None and var_pos_npz is not None:
                # Join weights to npz variants by CHROM:POS
                npz_keys = set(zip(var_chroms_npz.tolist(), var_pos_npz.tolist()))
                wdf_matched = (wdf_all[wdf_all.apply(
                    lambda r: (int(r['CHROM']), int(r['POS'])) in npz_keys, axis=1
                )].sort_values(['CHROM', 'POS']).reset_index(drop=True))
                if len(wdf_matched) == var_freq_num.shape[0]:
                    betas = wdf_matched['BETA'].values.astype(np.float64)
                    wdf   = wdf_matched
                    log.info(f"  Matched {len(betas)}/{len(wdf_all)} variants by CHROM:POS")
                else:
                    log.warning(f"  CHROM:POS join gave {len(wdf_matched)} vs {var_freq_num.shape[0]}")
            elif len(wdf_all) == var_freq_num.shape[0]:
                betas = wdf_all['BETA'].values.astype(np.float64)
                wdf   = wdf_all
                log.info(f"  Loaded {len(betas)} betas (positional)")
            else:
                log.warning(f"  Cannot align weights ({len(wdf_all)}) to npz ({var_freq_num.shape[0]})")
        except Exception as e:
            log.warning(f"  Could not load weights: {e}")

    # -------------------------------------------------------------------------
    # Observed ARS: ARS_k = Σ_v beta_v × F_vk
    # F_vk = var_freq_num[v,k] / var_denominator[v,k]
    # -------------------------------------------------------------------------
    if has_var_arrays and betas is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            F_vk = np.where(
                var_denominator > 0,
                var_freq_num / var_denominator,
                np.nan
            )  # (n_variants, n_anc)
        # ARS_k = Σ_v beta_v × F_vk
        observed_raw = np.nansum(betas[:, np.newaxis] * F_vk, axis=0)  # (n_anc,)
        log.info(f"  ARS computed as Σ_v beta_v × F_vk (correct formula)")
    else:
        # Fallback: pooled formula
        observed_raw = compute_ars(numerator, denominator)
        log.warning(f"  ARS computed via pooled fallback (no betas available)")
        F_vk = None

    observed_z = z_normalise(observed_raw)

    log.info("  Observed ARS:")
    for k, name in enumerate(ANCESTRY_NAMES):
        log.info(f"    {name}: raw={observed_raw[k]:.6f}  z={observed_z[k]:.4f}")

    # -------------------------------------------------------------------------
    # Individual bootstrap (resample samples with replacement) — legacy
    # per-ancestry z-score output only, see notes below.
    # -------------------------------------------------------------------------
    boot_z, boot_raw = run_bootstrap(
        numerator, denominator, N_BOOTSTRAP, N_BOOT_SAMPLES, rng
    )

    # -------------------------------------------------------------------------
    # Bootstrap raw contrast distribution: ARS_EF - ARS_WHG per iteration.
    #
    # Uses the PROPER individual-level bootstrap when the per-person,
    # per-variant grid is available (run_individual_bootstrap): resamples
    # people, and correctly recomputes F_vk per variant from only the
    # resampled subset before summing with beta weights.
    #
    # boot_z/boot_raw from run_bootstrap() (person-level, above) are kept
    # ONLY for the legacy per-ancestry z-score output; they must NOT be used
    # for the raw contrast, its CI, or LOO — pooling numerator/denominator
    # across resampled people across ALL variants before dividing computes a
    # different (and biased, by ~20x in testing) quantity than the correct
    # Σ_v beta_v × F_vk formula used for the point estimate. That's a
    # distinct, separate calculation from run_individual_bootstrap below,
    # which recomputes F_vk per variant, per resample — not per person,
    # pooled across variants.
    #
    # Falls back to variant-level resampling (run_variant_bootstrap) if this
    # trait's merged.npz predates the per-person grid — still unbiased, just
    # answers a different question (variant-selection uncertainty rather
    # than person-sampling uncertainty). See has_person_arrays warning above.
    # -------------------------------------------------------------------------
    ef_idx  = ANCESTRY_NAMES.index('EF')  if 'EF'  in ANCESTRY_NAMES else 0
    whg_idx = ANCESTRY_NAMES.index('WHG') if 'WHG' in ANCESTRY_NAMES else 2

    bootstrap_method = 'none'
    if (has_person_arrays and betas is not None
            and var_person_freq_num.shape[0] == len(betas)):
        valid_boot = run_individual_bootstrap(
            var_person_freq_num, var_person_denom, betas,
            ef_idx, whg_idx, N_BOOTSTRAP, N_BOOT_SAMPLES, rng
        )
        std_bootstrap    = (float(np.std(valid_boot, ddof=1))
                            if len(valid_boot) > 1 else np.nan)
        bootstrap_method = 'individual'
    elif has_person_arrays and betas is not None:
        log.warning(f"  var_person_freq_num variant count "
                    f"({var_person_freq_num.shape[0]}) != len(betas) "
                    f"({len(betas)}) — cannot align, falling back to "
                    f"variant-level resampling")
    if bootstrap_method == 'none' and has_var_arrays and betas is not None and F_vk is not None:
        valid_boot       = run_variant_bootstrap(F_vk, betas, ef_idx, whg_idx,
                                                 N_BOOTSTRAP, rng)
        std_bootstrap    = (float(np.std(valid_boot, ddof=1))
                            if len(valid_boot) > 1 else np.nan)
        bootstrap_method = 'variant'
    elif bootstrap_method == 'none':
        valid_boot    = np.array([])
        std_bootstrap = np.nan
        log.warning("  Bootstrap skipped — var arrays/betas unavailable; "
                    "CI/std_bootstrap/person_sd CI will be NaN")

    log.info(f"  bootstrap_method = {bootstrap_method}")

    observed_contrast_raw = float(observed_raw[ef_idx] - observed_raw[whg_idx])

    if std_bootstrap > 0 and not np.isnan(std_bootstrap):
        observed_contrast_std = observed_contrast_raw / std_bootstrap
        ci_lower_std = float(np.nanpercentile(valid_boot, 2.5))  / std_bootstrap
        ci_upper_std = float(np.nanpercentile(valid_boot, 97.5)) / std_bootstrap
        # Empirical p-value from raw contrast bootstrap
        p_val = 2.0 * min(float(np.mean(valid_boot <= 0)),
                          float(np.mean(valid_boot >= 0)))
        p_val = max(p_val, 1.0 / len(valid_boot))
    else:
        observed_contrast_std = ci_lower_std = ci_upper_std = p_val = np.nan

    log.info(f"  Observed ARS_EF - ARS_WHG = {observed_contrast_raw:.6f}")
    log.info(f"  std_bootstrap = {std_bootstrap:.6f}")
    log.info(f"  Standardised contrast = {observed_contrast_std:.4f}  "
             f"95% CI [{ci_lower_std:.4f}, {ci_upper_std:.4f}]  p={p_val:.4g}")

    # -------------------------------------------------------------------------
    # Person-level effect-size denominator (Option B).
    #
    # std_bootstrap above is a resampling STANDARD ERROR: it measures how
    # precisely we've estimated the mean contrast, and mechanically shrinks
    # as n_boot_samples grows (~1/sqrt(n_boot_samples)) — so the standardised
    # contrast it produces depends on an arbitrary pipeline knob, not just on
    # biology.
    #
    # person_sd instead measures ordinary person-to-person variability: each
    # individual's own ancestry-weighted ratio,
    #     R_i,k = numerator[i,k] / denominator[i,k]
    # (the same per-person arrays already used for bootstrap resampling,
    # just aggregated per person instead of being resampled), and then
    #     person_sd = std(R_i,EF - R_i,WHG)   across ALL valid people, ddof=1
    # computed once directly from the full ~490k-person cohort — no
    # bootstrap resampling involved, so no dependence on n_boot_samples.
    # This is an effect-size-style denominator (like Cohen's d), comparable
    # across traits regardless of cohort size.
    #
    # The bootstrap CI is still built from the same resampled contrast
    # distribution as before (valid_boot) — only the denominator used to
    # express its bounds changes, from std_bootstrap to person_sd.
    # -------------------------------------------------------------------------
    valid_person = (denominator[:, ef_idx] > 0) & (denominator[:, whg_idx] > 0)
    n_valid_person = int(valid_person.sum())

    if n_valid_person > 1:
        with np.errstate(invalid='ignore', divide='ignore'):
            R_ef  = numerator[valid_person, ef_idx]  / denominator[valid_person, ef_idx]
            R_whg = numerator[valid_person, whg_idx] / denominator[valid_person, whg_idx]
        person_contrast = R_ef - R_whg
        person_contrast = person_contrast[~np.isnan(person_contrast)]
        person_sd = (float(np.std(person_contrast, ddof=1))
                     if len(person_contrast) > 1 else np.nan)
    else:
        person_sd = np.nan

    if person_sd > 0 and not np.isnan(person_sd):
        observed_contrast_persd = observed_contrast_raw / person_sd
        ci_lower_persd = float(np.nanpercentile(valid_boot, 2.5))  / person_sd
        ci_upper_persd = float(np.nanpercentile(valid_boot, 97.5)) / person_sd
    else:
        observed_contrast_persd = ci_lower_persd = ci_upper_persd = np.nan

    log.info(f"  person_sd (n={n_valid_person:,} people with valid ANP coverage "
             f"at both EF and WHG) = {person_sd:.6f}")
    log.info(f"  Person-SD-standardised contrast = {observed_contrast_persd:.4f}  "
             f"95% CI [{ci_lower_persd:.4f}, {ci_upper_persd:.4f}]")

    # -------------------------------------------------------------------------
    # Leave-one-out (LOO): same denominator as CI
    # LOO_v = (ARS_EF_loo - ARS_WHG_loo) / std_bootstrap
    #       = observed_contrast_std - contrast_contrib_v / std_bootstrap
    # -------------------------------------------------------------------------
    loo_contrasts = np.array([])
    loo_contrasts_persd = np.array([])
    loo_chroms    = np.array([], dtype=np.int32)
    loo_pos       = np.array([], dtype=np.int32)

    if has_var_arrays and betas is not None and F_vk is not None and not np.isnan(std_bootstrap):
        n_v = F_vk.shape[0]
        total_bF = np.nansum(betas[:, np.newaxis] * F_vk, axis=0)

        loo_contrasts = np.full(n_v, np.nan, dtype=np.float64)
        loo_contrasts_persd = np.full(n_v, np.nan, dtype=np.float64)
        for v in range(n_v):
            ars_loo = total_bF - betas[v] * F_vk[v]
            loo_raw = float(ars_loo[ef_idx] - ars_loo[whg_idx])
            loo_contrasts[v] = loo_raw / std_bootstrap   # same denominator as CI
            if person_sd > 0 and not np.isnan(person_sd):
                loo_contrasts_persd[v] = loo_raw / person_sd

        if wdf is not None:
            loo_chroms = wdf['CHROM'].values.astype(np.int32)
            loo_pos    = wdf['POS'].values.astype(np.int32)
        log.info(f"  LOO range: [{np.nanmin(loo_contrasts):.4f}, "
                 f"{np.nanmax(loo_contrasts):.4f}]  "
                 f"(baseline={observed_contrast_std:.4f})")

        # Diagnostic TSV
        if wdf is not None:
            diag_rows = []
            for v in range(n_v):
                row = {
                    'rsID':             str(wdf.iloc[v].get('rsID', '')),
                    'gene':             str(wdf.iloc[v].get('gene', '')),
                    'chr':              int(wdf.iloc[v]['CHROM']),
                    'pos':              int(wdf.iloc[v]['POS']),
                    'effect_allele':    str(wdf.iloc[v]['EFFECT_ALLELE']),
                    'beta':             float(betas[v]),
                }
                for k, name in enumerate(ANCESTRY_NAMES):
                    row[f'F_{name}']           = round(float(F_vk[v, k]),             6)
                    row[f'ars_contrib_{name}']  = round(float(betas[v] * F_vk[v, k]), 8)
                contrast_contrib_v = float(betas[v] * F_vk[v, ef_idx]) - \
                                     float(betas[v] * F_vk[v, whg_idx])
                row['contrast_contrib']     = round(contrast_contrib_v, 8)
                row['contrast_contrib_std'] = round(contrast_contrib_v / std_bootstrap, 6)
                row['loo_contrast_std']     = round(float(loo_contrasts[v]), 6)
                if person_sd > 0 and not np.isnan(person_sd):
                    row['contrast_contrib_persd'] = round(contrast_contrib_v / person_sd, 6)
                    row['loo_contrast_persd']      = round(float(loo_contrasts_persd[v]), 6)
                else:
                    row['contrast_contrib_persd'] = np.nan
                    row['loo_contrast_persd']      = np.nan
                diag_rows.append(row)

            diag_df   = (pd.DataFrame(diag_rows)
                           .sort_values('contrast_contrib', ascending=False)
                           .reset_index(drop=True))
            diag_dir  = os.path.join(os.path.dirname(OUTPUT_RESULTS), 'diagnostics')
            os.makedirs(diag_dir, exist_ok=True)
            diag_path = os.path.join(diag_dir, f'{SAFE_TRAIT}_snp_contributions.tsv')
            header_lines = [
                f"# Trait: {SAFE_TRAIT}",
                f"# ARS_EF={observed_raw[ef_idx]:.8f}  "
                f"ARS_WHG={observed_raw[whg_idx]:.8f}",
                f"# observed_contrast_raw={observed_contrast_raw:.8f}",
                f"# std_bootstrap={std_bootstrap:.8f}  "
                f"(resampling SE — scales with n_boot_samples, see script docstring)",
                f"# standardised_contrast (std_bootstrap)={observed_contrast_std:.6f}  "
                f"95%CI=[{ci_lower_std:.4f}, {ci_upper_std:.4f}]  p={p_val:.4g}",
                f"# person_sd={person_sd:.8f}  (n={n_valid_person:,} people; "
                f"population SD of each individual's own ancestry-weighted "
                f"contrast — does not depend on n_boot_samples)",
                f"# standardised_contrast (person_sd)={observed_contrast_persd:.6f}  "
                f"95%CI=[{ci_lower_persd:.4f}, {ci_upper_persd:.4f}]",
                f"# Formula: ARS_k=Σ_v beta_v×F_vk  "
                f"F_vk=Σ_i[indicator×ANP]/Σ_i[ANP1+ANP2]",
                f"# contrast_contrib_std   = contrast_contrib / std_bootstrap",
                f"# contrast_contrib_persd = contrast_contrib / person_sd",
                f"# loo_contrast_std       = (ARS_EF_loo - ARS_WHG_loo) / std_bootstrap",
                f"# loo_contrast_persd     = (ARS_EF_loo - ARS_WHG_loo) / person_sd",
                "#",
            ]
            with open(diag_path, 'w') as fh:
                fh.write('\n'.join(header_lines) + '\n')
                diag_df.to_csv(fh, sep='\t', index=False)
            log.info(f"  Diagnostic TSV: {diag_path}")
    else:
        log.warning("  LOO skipped — var arrays/betas/std_bootstrap not available")
        if np.isnan(person_sd):
            log.warning("  person_sd unavailable — persd-scaled fields will be NaN")

    # -------------------------------------------------------------------------
    # Summarise results — write BOTH formats
    # -------------------------------------------------------------------------

    # Per-ancestry z-score format (summary tables)
    rows = []
    for k, name in enumerate(ANCESTRY_NAMES):
        z_dist   = boot_z[:, k]
        raw_dist = boot_raw[:, k]
        valid_z  = z_dist[~np.isnan(z_dist)]
        valid_raw= raw_dist[~np.isnan(raw_dist)]
        rows.append({
            'ancestry':               name,
            'observed_z':             float(observed_z[k]),
            'bootstrap_mean_z':       float(np.mean(valid_z))   if len(valid_z)  > 0 else np.nan,
            'ci_lower':               float(np.percentile(valid_z,  2.5)) if len(valid_z) > 0 else np.nan,
            'ci_upper':               float(np.percentile(valid_z, 97.5)) if len(valid_z) > 0 else np.nan,
            'observed_raw_ars':       float(observed_raw[k]),
            'bootstrap_mean_raw_ars': float(np.mean(valid_raw)) if len(valid_raw) > 0 else np.nan,
            'n_variants':             n_variants,
            'n_samples':              n_samples,
            'n_bootstrap':            N_BOOTSTRAP,
            'n_boot_samples':         min(N_BOOT_SAMPLES, n_samples),
        })
    pd.DataFrame(rows).to_csv(OUTPUT_RESULTS, sep='\t', index=False)
    log.info(f"Results written: {OUTPUT_RESULTS}")

    np.savez_compressed(
        OUTPUT_BOOT,
        z_scores              = boot_z,                # (n_bootstrap, n_ancestries) legacy
        raw_ars               = boot_raw,              # (n_bootstrap, n_ancestries)
        boot_raw_contrast     = valid_boot,            # (n_bootstrap,) EF-WHG per iteration
        std_bootstrap         = np.array(std_bootstrap),
        observed_contrast_std = np.array(observed_contrast_std),
        ci_lower_std          = np.array(ci_lower_std),
        ci_upper_std          = np.array(ci_upper_std),
        loo_contrasts         = loo_contrasts,         # (n_variants,) std_bootstrap-scaled
        # Person-level effect-size denominator (Option B) — see comments above.
        # Does not depend on n_boot_samples.
        person_sd               = np.array(person_sd),
        n_valid_person           = np.array(n_valid_person),
        observed_contrast_persd  = np.array(observed_contrast_persd),
        ci_lower_persd           = np.array(ci_lower_persd),
        ci_upper_persd           = np.array(ci_upper_persd),
        loo_contrasts_persd      = loo_contrasts_persd,   # (n_variants,) person_sd-scaled
        loo_chroms            = loo_chroms,
        loo_pos               = loo_pos,
        bootstrap_method      = np.array(bootstrap_method),  # 'individual' | 'variant' | 'none'
        ancestry_names        = np.array(ANCESTRY_NAMES)
    )
    log.info(f"Bootstrap distributions written: {OUTPUT_BOOT}")
    log.info("Done.")


if __name__ == '__main__':
    main()