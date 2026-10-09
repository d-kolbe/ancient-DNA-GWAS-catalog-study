#!/usr/bin/env python3
"""
ukb_build_variant_list.py
=========================
Snakemake script — Step 1 of UKB ARS pipeline.

Reads the snp_sets_summary.txt from the main enrichment pipeline and
loads each per-trait SNP file directly. No GWAS catalog access, no
trait_blocks scanning — the SNP selection has already been done upstream.

Input
-----
snp_sets_summary.txt  (tab-separated)
    category  trait  n_snps  file
    where 'file' points to a per-trait SNP file with columns:
        chr  pos  effect_allele  beta

Outputs
-------
variant_lists/weights/{safe_trait}_weights.tsv
    CHROM  POS  EFFECT_ALLELE  BETA  TRAIT

variant_lists/chr_positions/chr{N}_positions.tsv
    CHROM  POS   (no header — bcftools -R format)

variant_lists/trait_manifest.tsv
    safe_trait  original_trait  category  n_variants  file

variant_lists/chr_positions.done   (sentinel)
"""

import os
import re
import sys
import logging
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

SNP_SETS_SUMMARY      = snakemake.input.snp_sets_summary
WEIGHTS_DIR           = snakemake.output.weights_dir
TRAIT_MANIFEST        = snakemake.output.trait_manifest
CHR_POSITIONS_DONE    = snakemake.output.chr_positions_sentinel

SAFE_TRAITS           = snakemake.params.safe_traits
CHR_POSITIONS_DIR     = snakemake.params.chr_positions_dir
OUT_DIR               = snakemake.params.out_dir

# GWAS catalog for rsID / gene lookup (GRCh37 liftover)
GWAS_CATALOG          = getattr(snakemake.params, 'gwas_catalog', '')


# =============================================================================
# GWAS catalog loader
# =============================================================================

def load_gwas_catalog(catalog_path):
    """
    Load GWAS catalog and build a (CHR, POS) -> (rsID, gene) lookup dict.
    Returns empty dict if path not provided or file not found.
    """
    if not catalog_path or not os.path.exists(catalog_path):
        log.warning(f"GWAS catalog not found: {catalog_path} — rsID/gene will be empty")
        return {}

    log.info(f"Loading GWAS catalog for rsID/gene lookup: {catalog_path}")
    try:
        cols_needed = ['CHR_ID', 'CHR_POS', 'SNPS', 'MAPPED_GENE']
        cat = pd.read_csv(catalog_path, sep='\t', usecols=cols_needed,
                          low_memory=False)
        cat['CHR_ID']  = pd.to_numeric(cat['CHR_ID'],  errors='coerce')
        cat['CHR_POS'] = pd.to_numeric(cat['CHR_POS'], errors='coerce')
        cat = cat.dropna(subset=['CHR_ID', 'CHR_POS'])
        cat['CHR_ID']  = cat['CHR_ID'].astype(int)
        cat['CHR_POS'] = cat['CHR_POS'].astype(int)
        cat['SNPS']        = cat['SNPS'].fillna('').astype(str)
        cat['MAPPED_GENE'] = cat['MAPPED_GENE'].fillna('').astype(str)

        lookup = {}
        for _, row in cat.iterrows():
            key = (int(row['CHR_ID']), int(row['CHR_POS']))
            if key not in lookup:  # keep first match
                rsid = row['SNPS'].split(';')[0].strip()
                gene = row['MAPPED_GENE'].split(',')[0].strip()
                lookup[key] = (rsid, gene)

        log.info(f"  GWAS catalog: {len(lookup):,} unique chr:pos entries")
        return lookup
    except Exception as e:
        log.warning(f"Could not load GWAS catalog: {e}")
        return {}


# =============================================================================
# Helpers
# =============================================================================

def resolve_snp_file(path):
    """The 'file' column of snp_sets_summary.txt can be relative to the working
    directory of the enrichment project. If it is not found as given, look in
    <summary dir>/<category>/<file name>."""
    if os.path.exists(path):
        return path
    alt = os.path.join(os.path.dirname(os.path.abspath(SNP_SETS_SUMMARY)),
                       *Path(path).parts[-2:])
    return alt if os.path.exists(alt) else path


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(CHR_POSITIONS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load GWAS catalog for rsID/gene lookup
    gwas_lookup = load_gwas_catalog(GWAS_CATALOG)

    # Load summary
    log.info(f"Reading SNP sets summary: {SNP_SETS_SUMMARY}")
    summary = pd.read_csv(SNP_SETS_SUMMARY, sep="\t")
    log.info(f"  {len(summary):,} traits across {summary['category'].nunique()} categories")

    # Accumulate positions per chromosome across all traits
    all_positions: dict[int, set[int]] = {c: set() for c in range(1, 23)}
    manifest_rows = []

    for _, row in summary.iterrows():
        trait_orig = str(row["trait"])
        category   = str(row["category"])
        snp_file   = resolve_snp_file(str(row["file"]))
        safe_trait = SAFE_TRAITS.get(trait_orig)

        if safe_trait is None:
            log.warning(f"  Trait not in SAFE_TRAITS dict: '{trait_orig}' — skipping")
            continue

        log.info(f"  [{safe_trait}] Loading: {snp_file}")

        if not os.path.exists(snp_file):
            log.warning(f"  [{safe_trait}] File not found — skipping: {snp_file}")
            continue

        try:
            snp_df = pd.read_csv(snp_file, sep="\t")
        except Exception as e:
            log.warning(f"  [{safe_trait}] Could not read file: {e} — skipping")
            continue

        # Rename columns to pipeline standard
        rename = {"chr": "CHROM", "pos": "POS",
                  "effect_allele": "EFFECT_ALLELE", "beta": "BETA"}
        snp_df = snp_df.rename(columns=rename)

        # Validate required columns
        required = {"CHROM", "POS", "EFFECT_ALLELE", "BETA"}
        missing = required - set(snp_df.columns)
        if missing:
            log.warning(f"  [{safe_trait}] Missing columns {missing} — skipping")
            continue

        # Clean
        snp_df["CHROM"] = pd.to_numeric(snp_df["CHROM"], errors="coerce")
        snp_df["POS"]   = pd.to_numeric(snp_df["POS"],   errors="coerce")
        snp_df["BETA"]  = pd.to_numeric(snp_df["BETA"],  errors="coerce")
        snp_df["EFFECT_ALLELE"] = snp_df["EFFECT_ALLELE"].str.upper().str.strip()

        snp_df = snp_df.dropna(subset=["CHROM", "POS", "BETA", "EFFECT_ALLELE"])
        snp_df["CHROM"] = snp_df["CHROM"].astype(int)
        snp_df["POS"]   = snp_df["POS"].astype(int)

        # Restrict to autosomes
        snp_df = snp_df[(snp_df["CHROM"] >= 1) & (snp_df["CHROM"] <= 22)]

        # Deduplicate on CHROM:POS — keep first (already LD-pruned upstream)
        snp_df = snp_df.drop_duplicates(subset=["CHROM", "POS"])

        n_final = len(snp_df)
        log.info(f"  [{safe_trait}] {n_final:,} variants")

        if n_final == 0:
            log.warning(f"  [{safe_trait}] No valid variants — skipping")
            continue

        # Add rsID and gene from GWAS catalog lookup
        snp_df["rsID"] = snp_df.apply(
            lambda r: gwas_lookup.get((int(r["CHROM"]), int(r["POS"])), ('', ''))[0],
            axis=1
        )
        snp_df["gene"] = snp_df.apply(
            lambda r: gwas_lookup.get((int(r["CHROM"]), int(r["POS"])), ('', ''))[1],
            axis=1
        )
        n_with_rsid = (snp_df["rsID"] != '').sum()
        log.info(f"  [{safe_trait}] rsID found for {n_with_rsid}/{len(snp_df)} variants")

        # Add trait column and write weights file
        snp_df["TRAIT"] = trait_orig
        weights_path = os.path.join(WEIGHTS_DIR, f"{safe_trait}_weights.tsv")
        snp_df[["CHROM", "POS", "EFFECT_ALLELE", "BETA", "TRAIT",
                "rsID", "gene"]].to_csv(weights_path, sep="\t", index=False)

        # Accumulate positions per chromosome
        for chrom, pos in zip(snp_df["CHROM"], snp_df["POS"]):
            all_positions[int(chrom)].add(int(pos))

        manifest_rows.append({
            "safe_trait":     safe_trait,
            "original_trait": trait_orig,
            "category":       category,
            "n_variants":     n_final,
            "file":           snp_file,
        })

    # Write per-chromosome positions files (CHROM\tPOS, no header)
    total_positions = 0
    for chrom in range(1, 23):
        positions = sorted(all_positions[chrom])
        pos_file  = os.path.join(CHR_POSITIONS_DIR, f"chr{chrom}_positions.tsv")
        with open(pos_file, "w") as f:
            for pos in positions:
                f.write(f"{chrom}\t{pos}\n")
        total_positions += len(positions)
        if positions:
            log.info(f"  chr{chrom}: {len(positions):,} unique positions")

    log.info(f"Total unique positions across all chromosomes: {total_positions:,}")

    # Write manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(TRAIT_MANIFEST, sep="\t", index=False)
    log.info(f"Trait manifest: {len(manifest_df)} traits → {TRAIT_MANIFEST}")

    # Sentinel
    Path(CHR_POSITIONS_DONE).touch()
    log.info("Done.")


if __name__ == "__main__":
    main()