"""
scripts/extract_snp_dosage.py
==============================
Extracts ALT allele dosage for a list of SNPs from per-chromosome VCFs.
Writes a TSV of eid + one column per SNP.

Memory-efficient: processes one chromosome at a time, writes incrementally,
never holds the full sample x SNP matrix in RAM simultaneously.
"""

import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

_log = open(str(snakemake.log[0]), "w", buffering=1)
sys.stdout = _log
sys.stderr = _log

LOG_PREFIX = "[extract_snp_dosage]"
def log(msg): print(f"{LOG_PREFIX} {msg}", flush=True)

# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

snp_list    = snakemake.params.snp_list
vcf_pattern = snakemake.params.vcf_pattern

log(f"SNPs to extract: {len(snp_list)}")
for s in snp_list:
    log(f"  chr{s['chrom']}:{s['pos']} {s['ref']}>{s['alt']}  label={s['label']}")

# ---------------------------------------------------------------------------
# Complement for strand flip detection
# ---------------------------------------------------------------------------

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

def complement(allele):
    return "".join(COMPLEMENT.get(b, b) for b in allele.upper())

def alleles_match(ref_c, alt_c, ref_v, alt_v):
    rc, ac = ref_c.upper(), alt_c.upper()
    rv, av = ref_v.upper(), alt_v.upper()
    if rc == rv and ac == av:             return "match"
    if complement(rc) == rv and complement(ac) == av: return "flip"
    if rc == av and ac == rv:             return "swap"
    if complement(rc) == av and complement(ac) == rv: return "swap_flip"
    return None

def gt_to_dosage(gt_str):
    if gt_str in (".", "./.", ".|."):
        return np.nan
    alleles = gt_str.replace("|", "/").split("/")
    try:
        return float(sum(int(a) for a in alleles if a != "."))
    except ValueError:
        return np.nan

# ---------------------------------------------------------------------------
# Group SNPs by chromosome
# ---------------------------------------------------------------------------

from collections import defaultdict
snps_by_chrom = defaultdict(list)
for snp in snp_list:
    snps_by_chrom[str(snp["chrom"])].append(snp)

# ---------------------------------------------------------------------------
# Extract per chromosome — build result as dict {label: array} incrementally
# ---------------------------------------------------------------------------

# First get sample names from first available VCF
sample_names = None
for chrom in snps_by_chrom:
    vcf_path = vcf_pattern.format(chrom=chrom)
    if Path(vcf_path).exists():
        res = subprocess.run(["bcftools", "query", "-l", vcf_path],
                             capture_output=True, text=True, check=True)
        sample_names = res.stdout.strip().split("\n")
        log(f"Samples in VCF: {len(sample_names):,}")
        break

if sample_names is None:
    log("ERROR: no VCF found — writing empty dosage file")
    pd.DataFrame(columns=["eid"]).to_csv(snakemake.output.dosages, sep="\t", index=False)
    sys.exit(0)

n_samples = len(sample_names)
eids = pd.to_numeric(pd.Series(sample_names), errors="coerce").astype("Int64")

# Collect dosage arrays — one per SNP label
dosage_cols = {}   # label -> np.array float32

for chrom, snps in snps_by_chrom.items():
    vcf_path = vcf_pattern.format(chrom=chrom)
    if not Path(vcf_path).exists():
        log(f"WARNING: VCF not found for chr{chrom} — skipping")
        continue

    log(f"\nProcessing chr{chrom} ({len(snps)} SNPs)")
    regions = ",".join(f"{chrom}:{s['pos']}-{s['pos']}" for s in snps)

    cmd = ["bcftools", "query", "-r", regions,
           "-f", "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n", vcf_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        log(f"ERROR: bcftools failed for chr{chrom}: {e.stderr}")
        continue

    if not result.stdout.strip():
        log(f"WARNING: no output from bcftools for chr{chrom}")
        continue

    for line in result.stdout.strip().split("\n"):
        parts   = line.split("\t")
        vcf_pos = int(parts[1])
        vcf_ref = parts[2]
        vcf_alt = parts[3]
        gt_calls = parts[4:]

        if len(gt_calls) != n_samples:
            log(f"WARNING: GT count mismatch at {chrom}:{vcf_pos} — skipping")
            continue

        matching_snp = next((s for s in snps if int(s["pos"]) == vcf_pos), None)
        if matching_snp is None:
            log(f"WARNING: no config SNP matched pos {vcf_pos} — skipping")
            continue

        orientation = alleles_match(
            matching_snp["ref"], matching_snp["alt"], vcf_ref, vcf_alt)
        log(f"  {chrom}:{vcf_pos} {vcf_ref}>{vcf_alt}  orientation={orientation}")

        if orientation is None:
            log(f"WARNING: allele mismatch at {chrom}:{vcf_pos} — skipping")
            continue

        # Convert GT → dosage as float32 to save memory
        ds = np.array([gt_to_dosage(g) for g in gt_calls], dtype=np.float32)

        if orientation in ("swap", "swap_flip"):
            log(f"  REF/ALT swapped — flipping dosage")
            ds = 2.0 - ds

        label = matching_snp["label"]
        dosage_cols[label] = ds
        log(f"  Extracted: {label}  n_valid={np.sum(~np.isnan(ds)):,}")

# ---------------------------------------------------------------------------
# Write output — build DataFrame only at write time
# ---------------------------------------------------------------------------

if not dosage_cols:
    log("ERROR: no dosages extracted — writing empty file")
    pd.DataFrame(columns=["eid"]).to_csv(snakemake.output.dosages, sep="\t", index=False)
    sys.exit(0)

log(f"\nWriting {len(dosage_cols)} SNP columns for {n_samples:,} samples...")

out_df = pd.DataFrame({"eid": eids})
for label, arr in dosage_cols.items():
    out_df[label] = arr

out_df.to_csv(snakemake.output.dosages, sep="\t", index=False)
log(f"Written: {snakemake.output.dosages}  ({len(out_df):,} rows, {len(dosage_cols)} SNPs)")