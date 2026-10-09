#!/usr/bin/env python3
"""
ukb_compute_contributions.py
=============================
Snakemake script — Step 3 of UKB ARS pipeline.

Computes per-individual ARS contribution vectors for every configured trait.

PERFORMANCE NOTE
----------------
The original Python VCF parser iterated over 490k samples per variant line
in a Python for-loop — 150M+ iterations per chromosome, hitting 6h time limits.

This version uses bcftools query to extract numerical fields at C speed,
then fully vectorized numpy operations. Expected speedup: 100-1000x.

Strategy
--------
1. bcftools query extracts AN1+AN2 (dosage) and ANP1/ANP2 arrays into flat
   tab-separated text — one row per variant, all samples concatenated.
2. pandas.read_csv reads the flat text at C speed (~500 MB/s).
3. All contribution calculations are numpy matrix operations.
   No Python loop over samples anywhere.

Output per trait per chromosome
--------------------------------
numerator            float32 (n_samples, n_ancestries)  per-individual, all variants pooled
denominator          float32 (n_samples, n_ancestries)  per-individual, all variants pooled
var_numerator        float32 (n_variants, n_ancestries) per-variant population aggregate
var_denominator      float32 (n_variants, n_ancestries) per-variant population aggregate
var_person_freq_num  float32 (n_variants, n_samples, n_ancestries)  X_i,v,k, no beta —
                     per-person, per-variant "does this person's haplotype carry the
                     effect allele, weighted by that haplotype's ancestry probability".
                     Needed downstream (ukb_bootstrap_ars.py) to correctly resample
                     INDIVIDUALS and recompute F_vk per variant from only the resampled
                     subset, rather than pooling across all variants before dividing
                     (which is a different, biased calculation — see that script).
var_person_denom     float32 (n_variants, n_samples, n_ancestries)  ANP1+ANP2 per person,
                     per variant — the matching denominator for var_person_freq_num.
"""

import os
import sys
import io
import logging
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# =============================================================================
# Snakemake interface
# =============================================================================

EXTRACTED_VCF  = snakemake.input.extracted_vcf
WEIGHTS_DIR    = snakemake.input.weights_dir
TRAIT_MANIFEST = snakemake.input.trait_manifest

SAFE_TRAITS    = snakemake.params.safe_traits
ANCESTRY_NAMES = snakemake.params.ancestry_names
OUT_DIR        = snakemake.params.out_dir
WEIGHTS_DIR_P  = snakemake.params.weights_dir

N_ANCESTRIES   = len(ANCESTRY_NAMES)

AMBIGUOUS_PAIRS = {('A', 'T'), ('T', 'A'), ('C', 'G'), ('G', 'C')}


# =============================================================================
# Load weights
# =============================================================================

def load_all_weights(weights_dir, safe_traits):
    weights = {}
    for st in safe_traits:
        wf = os.path.join(weights_dir, f"{st}_weights.tsv")
        if not os.path.exists(wf):
            log.warning(f"Weights file not found: {wf}")
            continue
        df = pd.read_csv(wf, sep="\t")
        if df.empty:
            continue
        df["CHROM"] = df["CHROM"].astype(int)
        df["POS"]   = df["POS"].astype(int)
        df = df.set_index(["CHROM", "POS"])
        weights[st] = df
        log.info(f"  [{st}] {len(df):,} variants loaded")
    return weights


# =============================================================================
# bcftools-based VCF data extraction (vectorized)
# =============================================================================

def _run_bcftools(args, vcf_path):
    """Run bcftools and return stdout as bytes. Raises on failure."""
    cmd = ["bcftools"] + args + [vcf_path]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout


def get_sample_ids(vcf_path):
    out = _run_bcftools(["query", "-l"], vcf_path)
    lines = out.decode().strip().split("\n")
    return [l for l in lines if l]


def get_variant_info(vcf_path):
    """Returns list of (chrom_int, pos_int, ref, alt) for all variants."""
    out = _run_bcftools(["query", "-f", "%CHROM\t%POS\t%REF\t%ALT\n"], vcf_path)
    variants = []
    for line in out.decode().strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            chrom = int(parts[0].replace("chr", "").replace("Chr", ""))
            pos   = int(parts[1])
            ref   = parts[2].upper()
            alt   = parts[3].upper()
            # Skip non-biallelic, non-SNP, strand-ambiguous
            if "," in alt or len(ref) != 1 or len(alt) != 1:
                continue
            if (ref, alt) in AMBIGUOUS_PAIRS:
                continue
            variants.append((chrom, pos, ref, alt))
        except ValueError:
            continue
    return variants


def _write_regions_file(variants, tmpdir):
    """Write a bcftools regions file for the filtered variant list."""
    import tempfile
    fh = tempfile.NamedTemporaryFile(mode='w', suffix='.tsv',
                                      dir=tmpdir, delete=False)
    for chrom, pos, ref, alt in variants:
        fh.write(f"{chrom}\t{pos}\n")
    fh.close()
    return fh.name


def get_haplotype_allele_counts(vcf_path, n_samples, n_variants, variants, tmpdir='/tmp'):
    """
    Extract per-haplotype allele counts from GT for FILTERED variants only.
    Writes a regions file so bcftools outputs exactly one row per filtered variant.

    Returns
    -------
    an1_mat : (n_variants, n_samples) float32
    an2_mat : (n_variants, n_samples) float32
    """
    regions_file = _write_regions_file(variants, tmpdir)
    try:
        out = _run_bcftools(["query", "-R", regions_file,
                             "-f", "[%GT\t]\n"], vcf_path)
    finally:
        os.unlink(regions_file)

    an1_mat = np.full((n_variants, n_samples), np.nan, dtype=np.float32)
    an2_mat = np.full((n_variants, n_samples), np.nan, dtype=np.float32)

    for i, line in enumerate(out.decode().strip().split("\n")):
        if i >= n_variants or not line.strip():
            continue
        line_bytes = (line.rstrip("\t") + "\t").encode()
        arr = np.frombuffer(line_bytes, dtype=np.uint8)
        if len(arr) < 4 * n_samples:
            continue
        hap1 = arr[0::4].astype(np.float32) - 48
        hap2 = arr[2::4].astype(np.float32) - 48
        if len(hap1) != n_samples or len(hap2) != n_samples:
            continue
        hap1[hap1 < 0] = np.nan
        hap2[hap2 < 0] = np.nan
        an1_mat[i] = hap1
        an2_mat[i] = hap2

    return an1_mat, an2_mat


def get_anp_matrix(vcf_path, n_samples, n_variants, n_anc, variants, tmpdir='/tmp'):
    """
    Extract ANP1 and ANP2 for FILTERED variants only via bcftools -R.

    Returns
    -------
    anp1_mat : (n_variants, n_samples, n_anc) float32
    anp2_mat : (n_variants, n_samples, n_anc) float32
    """
    regions_file = _write_regions_file(variants, tmpdir)
    try:
        out = _run_bcftools(["query", "-R", regions_file,
                             "-f", "[%ANP1\t%ANP2\t]\n"], vcf_path)
    finally:
        os.unlink(regions_file)

    anp1_mat = np.full((n_variants, n_samples, n_anc), np.nan, dtype=np.float32)
    anp2_mat = np.full((n_variants, n_samples, n_anc), np.nan, dtype=np.float32)

    for i, line in enumerate(out.decode().strip().split("\n")):
        if i >= n_variants or not line.strip():
            continue
        flat = line.rstrip("\t").replace(",", "\t")
        vals = np.fromstring(flat, dtype=np.float32, sep="\t")
        expected = n_samples * 2 * n_anc
        if len(vals) != expected:
            continue
        shaped = vals.reshape(n_samples, 2, n_anc)
        anp1_mat[i] = shaped[:, 0, :]
        anp2_mat[i] = shaped[:, 1, :]

    return anp1_mat, anp2_mat


# =============================================================================
# Vectorized contribution computation
# =============================================================================

def compute_contributions(variants, an1_mat, an2_mat, anp1_mat, anp2_mat, weights):
    """
    Compute per-individual and per-variant contribution arrays.

    Correct haplotype-specific formula (following Barrie et al. 2024):

        numerator[i, k]   = Σ_v (AN1_vi × ANP1_vik + AN2_vi × ANP2_vik) × beta_v
        denominator[i, k] = Σ_v (ANP1_vik + ANP2_vik)

    This correctly pairs each haplotype's allele count with that haplotype's
    ancestry probability. The previous form (dosage × mean_anp) incorrectly
    averages ancestry across both haplotypes before weighting, misattributing
    ancestry for heterozygotes where the effect allele sits on only one haplotype.

    For effect == REF allele, haplotype allele counts are flipped:
        an1_effect = 1 - AN1,  an2_effect = 1 - AN2

    All operations are fully vectorized — no Python loops over samples.
    """
    n_samples      = an1_mat.shape[1]
    n_variants_all = len(variants)

    results = {
        st: {
            "numerator":       np.zeros((n_samples, N_ANCESTRIES), dtype=np.float64),
            "denominator":     np.zeros((n_samples, N_ANCESTRIES), dtype=np.float64),
            "var_numerator":   [],   # WITH beta (for bootstrap approximation)
            "var_denominator": [],   # no beta = freq_den
            "var_freq_num":    [],   # NO beta = painting freq numerator
            "var_chroms":      [],   # chrom for each variant
            "var_pos":         [],   # pos for each variant
            "var_person_freq_num": [],  # NEW: (n_samples, n_anc) per variant, no beta
            "var_person_denom":    [],  # NEW: (n_samples, n_anc) per variant
            "n_variants":      0,
        }
        for st in weights
    }

    for v_idx, (chrom, pos, ref, alt) in enumerate(variants):
        key = (chrom, pos)

        trait_hits = []
        for st, df in weights.items():
            if key in df.index:
                row           = df.loc[key]
                effect_allele = str(row["EFFECT_ALLELE"]).upper()
                beta          = float(row["BETA"])
                trait_hits.append((st, effect_allele, beta))

        if not trait_hits:
            continue

        # Per-haplotype allele counts: (n_samples,) — NaN = missing GT
        an1_v = an1_mat[v_idx]   # haplotype 1 allele count (0 or 1)
        an2_v = an2_mat[v_idx]   # haplotype 2 allele count (0 or 1)

        # Valid mask: both haplotypes must be non-missing
        valid = ~(np.isnan(an1_v) | np.isnan(an2_v))

        # Per-haplotype ancestry probabilities: (n_samples, n_anc)
        anp1_v = np.nan_to_num(anp1_mat[v_idx], nan=0.0)
        anp2_v = np.nan_to_num(anp2_mat[v_idx], nan=0.0)

        # Denominator is the same regardless of which allele is the effect allele:
        # ANP1 + ANP2 = total ancestry attribution across both haplotypes
        denom_v = anp1_v + anp2_v                           # (n_samples, n_anc)
        denom_v[~valid] = 0.0

        for st, effect_allele, beta in trait_hits:
            # Haplotype allele counts for the EFFECT allele
            if effect_allele == alt:
                an1_eff = np.where(valid, an1_v, 0.0)       # (n_samples,)
                an2_eff = np.where(valid, an2_v, 0.0)
            elif effect_allele == ref:
                # Flip: reference allele count = 1 - ALT allele count
                an1_eff = np.where(valid, 1.0 - an1_v, 0.0)
                an2_eff = np.where(valid, 1.0 - an2_v, 0.0)
            else:
                continue

            # Frequency numerator (NO beta): Σ_i(indicator × ANP)
            # Used for exact F_vk = freq_num / freq_den
            freq_num = (
                an1_eff[:, np.newaxis] * anp1_v +
                an2_eff[:, np.newaxis] * anp2_v
            )

            # Beta-weighted numerator (for individual bootstrap approximation)
            contrib_num = freq_num * beta

            # Per-individual accumulation
            results[st]["numerator"]   += contrib_num
            results[st]["denominator"] += denom_v

            # Per-variant population aggregates
            results[st]["var_numerator"].append(contrib_num.sum(axis=0))
            results[st]["var_denominator"].append(denom_v.sum(axis=0))
            results[st]["var_freq_num"].append(freq_num.sum(axis=0))   # no beta
            results[st]["var_chroms"].append(chrom)
            results[st]["var_pos"].append(pos)

            # Per-person, per-variant grid (needed for correct individual-level
            # bootstrap resampling downstream — see ukb_bootstrap_ars.py).
            # Stored WITHOUT beta so it can be recombined with beta at
            # resample-time after F_vk is recomputed from just the resampled
            # people.
            results[st]["var_person_freq_num"].append(freq_num.astype(np.float32))
            results[st]["var_person_denom"].append(denom_v.astype(np.float32))

            results[st]["n_variants"] += 1

    for st, r in results.items():
        log.info(f"  [{st}] variants used: {r['n_variants']:,}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(EXTRACTED_VCF):
        log.error(f"Extracted VCF not found: {EXTRACTED_VCF}")
        sys.exit(1)

    weights = load_all_weights(WEIGHTS_DIR_P, SAFE_TRAITS)

    def write_empty(st, reason=""):
        out_path = os.path.join(OUT_DIR, f"{st}_contributions.npz")
        log.info(f"  [{st}] Writing empty placeholder{' — ' + reason if reason else ''}")
        np.savez_compressed(
            out_path,
            numerator       = np.zeros((0, N_ANCESTRIES), dtype=np.float32),
            denominator     = np.zeros((0, N_ANCESTRIES), dtype=np.float32),
            var_numerator   = np.zeros((0, N_ANCESTRIES), dtype=np.float32),
            var_denominator = np.zeros((0, N_ANCESTRIES), dtype=np.float32),
            var_person_freq_num = np.zeros((0, 0, N_ANCESTRIES), dtype=np.float32),
            var_person_denom    = np.zeros((0, 0, N_ANCESTRIES), dtype=np.float32),
            sample_ids      = np.array([], dtype=object),
            n_variants      = np.array(0),
            ancestry_names  = np.array(ANCESTRY_NAMES)
        )

    if not weights:
        log.warning("No weights loaded — writing empty contribution files")
        for st in SAFE_TRAITS:
            write_empty(st)
        return

    # ── Step 1: get sample IDs and variant list ────────────────────────────
    log.info("Querying sample IDs...")
    sample_ids = get_sample_ids(EXTRACTED_VCF)
    n_samples  = len(sample_ids)
    log.info(f"  {n_samples:,} samples")

    if n_samples == 0:
        for st in SAFE_TRAITS:
            write_empty(st, "0 samples in VCF")
        return

    log.info("Querying variant info...")
    variants = get_variant_info(EXTRACTED_VCF)
    n_variants_all = len(variants)
    log.info(f"  {n_variants_all:,} variants (biallelic SNPs, non-ambiguous)")

    if n_variants_all == 0:
        for st in SAFE_TRAITS:
            write_empty(st, "0 variants in VCF")
        return

    # ── Step 2: extract numerical matrices via bcftools (C speed) ──────────
    log.info("Extracting per-haplotype allele counts (GT) via bcftools query...")
    an1_mat, an2_mat = get_haplotype_allele_counts(
        EXTRACTED_VCF, n_samples, n_variants_all, variants)
    log.info(f"  an1_mat shape: {an1_mat.shape}")

    log.info("Extracting ANP matrices via bcftools query...")
    anp1_mat, anp2_mat = get_anp_matrix(
        EXTRACTED_VCF, n_samples, n_variants_all, N_ANCESTRIES, variants)
    log.info(f"  anp1_mat shape: {anp1_mat.shape}")

    # ── Step 3: vectorized contribution computation ────────────────────────
    log.info("Computing contributions (haplotype-specific formula)...")
    results = compute_contributions(variants, an1_mat, an2_mat, anp1_mat, anp2_mat, weights)

    # ── Step 4: write per-trait npz files ──────────────────────────────────
    for st in SAFE_TRAITS:
        out_path = os.path.join(OUT_DIR, f"{st}_contributions.npz")
        if st in results and results[st]["n_variants"] > 0:
            r       = results[st]
            var_num = (np.array(r["var_numerator"],   dtype=np.float32)
                       if r["var_numerator"]
                       else np.zeros((0, N_ANCESTRIES), dtype=np.float32))
            var_den = (np.array(r["var_denominator"], dtype=np.float32)
                       if r["var_denominator"]
                       else np.zeros((0, N_ANCESTRIES), dtype=np.float32))
            var_freq_num = (np.array(r["var_freq_num"], dtype=np.float32)
                            if r["var_freq_num"]
                            else np.zeros((0, N_ANCESTRIES), dtype=np.float32))
            # (n_variants, n_samples, n_anc) — stack, don't concatenate, since
            # each list entry is a whole (n_samples, n_anc) per-variant slice
            var_person_freq_num = (np.stack(r["var_person_freq_num"], axis=0)
                                   if r["var_person_freq_num"]
                                   else np.zeros((0, n_samples, N_ANCESTRIES), dtype=np.float32))
            var_person_denom    = (np.stack(r["var_person_denom"], axis=0)
                                   if r["var_person_denom"]
                                   else np.zeros((0, n_samples, N_ANCESTRIES), dtype=np.float32))
            np.savez_compressed(
                out_path,
                numerator       = r["numerator"].astype(np.float32),
                denominator     = r["denominator"].astype(np.float32),
                var_numerator   = var_num,
                var_denominator = var_den,
                var_freq_num    = var_freq_num,
                var_person_freq_num = var_person_freq_num,
                var_person_denom    = var_person_denom,
                var_chroms      = np.array(r["var_chroms"], dtype=np.int32),
                var_pos         = np.array(r["var_pos"],    dtype=np.int32),
                sample_ids      = np.array(sample_ids, dtype=object),
                n_variants      = np.array(r["n_variants"]),
                ancestry_names  = np.array(ANCESTRY_NAMES)
            )
            log.info(f"  [{st}] Written ({r['n_variants']} variants): {out_path}")
        else:
            write_empty(st, "no variants on this chrom")

    log.info("Done.")


if __name__ == "__main__":
    main()