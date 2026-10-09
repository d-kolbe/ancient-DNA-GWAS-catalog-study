#!/usr/bin/env python3
"""
ukb_merge_contributions.py
===========================
Snakemake script — Step 4 of UKB ARS pipeline.

Merges per-individual contribution arrays across all 22 chromosomes for a
single trait. Numerators and denominators are summed element-wise. Sample IDs
are taken from the first non-empty chromosome file and validated for
consistency across chromosomes.

Input
-----
One .npz per chromosome (from ukb_compute_contributions.py):
  numerator    : float32 (n_samples, n_ancestries)  — may be (0, n_anc) if empty
  denominator  : float32 (n_samples, n_ancestries)
  sample_ids   : object  (n_samples,)
  n_variants   : scalar int
  ancestry_names: str (n_ancestries,)

Output
------
merged.npz with:
  numerator    : float32 (n_samples, n_ancestries)   — sum across 22 chroms
  denominator  : float32 (n_samples, n_ancestries)
  sample_ids   : object  (n_samples,)
  n_variants   : scalar int                          — total across 22 chroms
  ancestry_names: str (n_ancestries,)
  n_samples    : scalar int
  chroms_used  : int array — chromosomes that had ≥1 variant
"""

import os
import sys
import logging
import numpy as np
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

CHROM_FILES    = snakemake.input.chrom_files   # list of .npz paths, all 22 chroms
OUTPUT_MERGED  = snakemake.output.merged
SAFE_TRAIT     = snakemake.params.safe_trait
ANCESTRY_NAMES = snakemake.params.ancestry_names


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(os.path.dirname(OUTPUT_MERGED), exist_ok=True)

    n_ancestries     = len(ANCESTRY_NAMES)
    merged_numerator = None
    merged_denominator = None
    reference_sample_ids = None
    total_variants   = 0
    chroms_used      = []
    # Variant arrays: concatenated across chromosomes (not summed — different variants)
    var_numerator_chunks   = []
    var_denominator_chunks = []
    var_freq_num_chunks    = []
    var_chroms_chunks      = []
    var_pos_chunks         = []
    # Per-person, per-variant grid (needed for correct individual-level
    # bootstrap resampling downstream — see ukb_bootstrap_ars.py). Same
    # concatenation pattern as the other variant arrays: each chromosome
    # contributes a different set of variants (axis 0), same people (axis 1).
    var_person_freq_num_chunks = []
    var_person_denom_chunks    = []

    # Sort chrom files to process 1..22 in order
    chrom_files_sorted = sorted(CHROM_FILES, key=lambda p: int(
        Path(p).parent.name.replace('chr', '')
    ))

    for npz_path in chrom_files_sorted:
        chrom_str = Path(npz_path).parent.name  # e.g. 'chr3'
        try:
            chrom_num = int(chrom_str.replace('chr', ''))
        except ValueError:
            chrom_num = -1

        if not os.path.exists(npz_path):
            log.warning(f"Missing file (skipping): {npz_path}")
            continue

        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            log.warning(f"Could not load {npz_path}: {e} — skipping")
            continue

        num  = data['numerator']    # (n_samples, n_anc) or (0, n_anc)
        den  = data['denominator']
        sids = data['sample_ids']
        n_v  = int(data['n_variants'])

        # Skip empty chromosome files (no variants for this trait on this chrom)
        if num.shape[0] == 0 or n_v == 0:
            log.info(f"  {chrom_str}: 0 variants — skipping")
            continue

        # Validate ancestry count
        if num.shape[1] != n_ancestries:
            log.error(
                f"{chrom_str}: ancestry count mismatch "
                f"({num.shape[1]} vs expected {n_ancestries})"
            )
            sys.exit(1)

        # Initialise accumulators from first non-empty file
        if merged_numerator is None:
            merged_numerator   = np.zeros_like(num, dtype=np.float64)
            merged_denominator = np.zeros_like(den, dtype=np.float64)
            reference_sample_ids = sids
            log.info(f"  {chrom_str}: initialised accumulators — "
                     f"{num.shape[0]:,} samples, {n_v:,} variants")
        else:
            # Validate sample count consistency
            if num.shape[0] != merged_numerator.shape[0]:
                log.error(
                    f"{chrom_str}: sample count mismatch "
                    f"({num.shape[0]} vs {merged_numerator.shape[0]})"
                )
                sys.exit(1)

            if reference_sample_ids is not None and len(sids) == len(reference_sample_ids):
                if not np.array_equal(sids[:5], reference_sample_ids[:5]):
                    log.warning(
                        f"{chrom_str}: sample IDs differ from reference — "
                        "continuing but results may be unreliable"
                    )

        # Individual arrays: sum across chromosomes
        merged_numerator   += num.astype(np.float64)
        merged_denominator += den.astype(np.float64)

        # Variant arrays: collect for concatenation
        if 'var_numerator' in data and data['var_numerator'].shape[0] > 0:
            var_numerator_chunks.append(data['var_numerator'].astype(np.float32))
            var_denominator_chunks.append(data['var_denominator'].astype(np.float32))
        if 'var_freq_num' in data and data['var_freq_num'].shape[0] > 0:
            var_freq_num_chunks.append(data['var_freq_num'].astype(np.float32))
        if ('var_person_freq_num' in data and
                data['var_person_freq_num'].shape[0] > 0):
            var_person_freq_num_chunks.append(
                data['var_person_freq_num'].astype(np.float32))
            var_person_denom_chunks.append(
                data['var_person_denom'].astype(np.float32))
        if 'var_chroms' in data and len(data['var_chroms']) > 0:
            var_chroms_chunks.append(data['var_chroms'].astype(np.int32))
            var_pos_chunks.append(data['var_pos'].astype(np.int32))

        total_variants     += n_v
        chroms_used.append(chrom_num)
        log.info(f"  {chrom_str}: +{n_v:,} variants (running total: {total_variants:,})")

    if merged_numerator is None:
        log.warning(f"[{SAFE_TRAIT}] No variants found across any chromosome — writing zeros")
        n_samples_fallback = 0
        merged_numerator   = np.zeros((n_samples_fallback, n_ancestries), dtype=np.float32)
        merged_denominator = np.zeros((n_samples_fallback, n_ancestries), dtype=np.float32)
        reference_sample_ids = np.array([], dtype=object)
    else:
        n_samples_fallback = merged_numerator.shape[0]

    # Concatenate per-variant arrays across chromosomes
    if var_numerator_chunks:
        merged_var_numerator   = np.concatenate(var_numerator_chunks, axis=0)
        merged_var_denominator = np.concatenate(var_denominator_chunks, axis=0)
    else:
        merged_var_numerator   = np.zeros((0, n_ancestries), dtype=np.float32)
        merged_var_denominator = np.zeros((0, n_ancestries), dtype=np.float32)

    log.info(
        f"[{SAFE_TRAIT}] Merge complete — "
        f"{n_samples_fallback:,} samples | "
        f"{total_variants:,} variants | "
        f"{len(chroms_used)} chromosomes with data"
    )

    if var_freq_num_chunks:
        merged_var_freq_num = np.concatenate(var_freq_num_chunks, axis=0)
    else:
        merged_var_freq_num = np.zeros((0, n_ancestries), dtype=np.float32)

    # Per-person, per-variant grid — concatenated along the variant axis
    # (axis 0), same as var_numerator/var_freq_num. Same people (axis 1) in
    # every chunk since it's the same UKB cohort on every chromosome.
    if var_person_freq_num_chunks:
        merged_var_person_freq_num = np.concatenate(var_person_freq_num_chunks, axis=0)
        merged_var_person_denom    = np.concatenate(var_person_denom_chunks,    axis=0)
        log.info(f"[{SAFE_TRAIT}]   var_person grid: {merged_var_person_freq_num.shape} "
                 f"({merged_var_person_freq_num.nbytes / 1e9:.2f} GB)")
    else:
        merged_var_person_freq_num = np.zeros((0, n_samples_fallback, n_ancestries), dtype=np.float32)
        merged_var_person_denom    = np.zeros((0, n_samples_fallback, n_ancestries), dtype=np.float32)

    merged_var_chroms = (np.concatenate(var_chroms_chunks)
                         if var_chroms_chunks else np.array([], dtype=np.int32))
    merged_var_pos    = (np.concatenate(var_pos_chunks)
                         if var_pos_chunks else np.array([], dtype=np.int32))

    np.savez_compressed(
        OUTPUT_MERGED,
        numerator       = merged_numerator.astype(np.float32),
        denominator     = merged_denominator.astype(np.float32),
        var_numerator   = merged_var_numerator,
        var_denominator = merged_var_denominator,
        var_freq_num    = merged_var_freq_num,
        var_person_freq_num = merged_var_person_freq_num,
        var_person_denom    = merged_var_person_denom,
        var_chroms      = merged_var_chroms,
        var_pos         = merged_var_pos,
        sample_ids      = reference_sample_ids,
        n_variants      = np.array(total_variants),
        ancestry_names  = np.array(ANCESTRY_NAMES),
        n_samples       = np.array(n_samples_fallback),
        chroms_used     = np.array(sorted(chroms_used))
    )
    log.info(f"Written: {OUTPUT_MERGED}")


if __name__ == '__main__':
    main()