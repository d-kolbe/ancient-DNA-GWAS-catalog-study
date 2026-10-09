"""
Snakemake Pipeline for Local Ancestry Inference (LAI) - UK Biobank 500k
========================================================================

Pipeline overview:
  1. Split all samples into batches (configurable size, e.g. 5000)
  2. For each batch × chromosome:
       2a. extract_batch_chr_vcf    — subset UKB VCF to batch samples + FLARE positions, SNPs only
       2b. normalize_batch_chr_vcf  — left-normalise, split multiallelics, remove duplicates
       2c. filter_missing_batch_chr_vcf — add F_MISSING tag, drop variants with any missingness
       2d. prepare_batches_complete — barrier sentinel before FLARE starts
  3. Run FLARE per batch × chromosome
  4. Hierarchical merge:
       a. Merge groups of N batches per chr  →  intermediate VCFs (~100k samples)
       b. Merge intermediates per chr        →  stable final VCF (~500k samples)

Run from the project folder with:
    snakemake -s Snakefile_lai.py --configfile config/config_lai.local.yaml \\
              --profile ../profiles/slurm_lai
"""

import os
import sys
import math
import random

# ============================================================================
# SETUP
# ============================================================================

# Defaults live in config/config_lai.yaml (relative paths, run from this folder).
# Site-specific paths go into a second file passed with --configfile.
configfile: "config/config_lai.yaml"

# Scratch space for bcftools/FLARE temporary files
TEMP_DIR = os.path.abspath(config.get("tmp_dir", "results/tmp"))
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ['TMPDIR'] = TEMP_DIR
os.environ['TEMP']   = TEMP_DIR
os.environ['TMP']    = TEMP_DIR

shell.prefix(
    f"export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; "
    f"export TMPDIR={TEMP_DIR}; export TEMP={TEMP_DIR}; export TMP={TEMP_DIR}; "
)

# Conda environment for all rules. Default: envs/lai.yaml. To use an existing
# environment (with bcftools and Java), set conda_env to its directory path.
LAI_ENV     = config.get("conda_env", "envs/lai.yaml")
CHROMOSOMES = [str(i) for i in range(1, 23)]

OUT_DIR     = config["output_dir"]
STABLE_DIR  = config["stable_flare_outputs"]["dir"]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_sample_batches():
    """
    Read the single sample list, shuffle reproducibly, and split into batches.
    Returns dict: {batch_id (int) -> [sample, ...]}
    """
    sample_file = config["sample_list"]
    batch_size  = config.get("flare_batch_size", 5000)
    seed        = config.get("random_seed", 42)

    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"Sample list not found: {sample_file} (config: sample_list)")

    with open(sample_file) as fh:
        samples = [line.strip() for line in fh if line.strip()]

    random.seed(seed)
    random.shuffle(samples)

    batches = {}
    for batch_id, start in enumerate(range(0, len(samples), batch_size), start=1):
        batches[batch_id] = samples[start : start + batch_size]

    return batches


def get_merge_groups(n_batches, group_size):
    """
    Partition batch IDs 1..n_batches into groups of `group_size`.
    Returns list of lists of batch IDs.
    E.g. 100 batches, group_size=20 → 5 groups of 20.
    """
    all_batches = list(range(1, n_batches + 1))
    groups = []
    for start in range(0, len(all_batches), group_size):
        groups.append(all_batches[start : start + group_size])
    return groups


# ---- Initialise global constants ----------------------------------------

SAMPLE_BATCHES   = get_sample_batches()
N_BATCHES        = len(SAMPLE_BATCHES)

MERGE_GROUP_SIZE = config.get("merge_group_size", 20)
MERGE_GROUPS     = get_merge_groups(N_BATCHES, MERGE_GROUP_SIZE)
N_MERGE_GROUPS   = len(MERGE_GROUPS)

# Map group_id -> batch list (1-indexed group IDs)
MERGE_GROUP_MAP  = {gid: batches for gid, batches in enumerate(MERGE_GROUPS, start=1)}

# ---- Debug summary -------------------------------------------------------

print("=" * 80)
print("PIPELINE SUMMARY")
print(f"  Sample list    : {config['sample_list']}")
if SAMPLE_BATCHES:
    total = sum(len(v) for v in SAMPLE_BATCHES.values())
    print(f"  Total samples  : {total}")
    print(f"  Batch size     : {config.get('flare_batch_size', 5000)}")
    print(f"  N batches      : {N_BATCHES}")
    print(f"  N chromosomes  : {len(CHROMOSOMES)}")
    print(f"  Total FLARE jobs: {N_BATCHES * len(CHROMOSOMES)}")
    print(f"  Merge group sz : {MERGE_GROUP_SIZE}")
    print(f"  N merge groups : {N_MERGE_GROUPS}")
else:
    print("  WARNING: No samples loaded — check sample_list path in config!")
print("=" * 80)


# ============================================================================
# FINAL TARGET
# ============================================================================

rule all:
    input:
        # Batch sample files
        expand(OUT_DIR + "/sample_batches/batch{batch}.txt",
               batch=range(1, N_BATCHES + 1)),

        # Stable per-chromosome FLARE outputs
        expand(STABLE_DIR + "/chr{chr}.anc.vcf.gz", chr=CHROMOSOMES),
        expand(STABLE_DIR + "/chr{chr}.anc.vcf.gz.csi", chr=CHROMOSOMES),


# ============================================================================
# STEP 1 — Create sample batch files
# ============================================================================

rule create_sample_batch_files:
    """
    Randomly shuffle all samples (reproducibly) and write one file per batch.
    These files drive every downstream batch × chr job.
    """
    input:
        sample_list = config["sample_list"]
    output:
        batch_files = expand(OUT_DIR + "/sample_batches/batch{batch}.txt",
                             batch=range(1, N_BATCHES + 1))
    params:
        outdir = OUT_DIR + "/sample_batches"
    threads: 1
    resources:
        mem_mb   = 4000,
        runtime  = "30m"
    run:
        # SAMPLE_BATCHES (see get_sample_batches) is the single definition of the
        # shuffle + split, so the batch files always match the DAG.
        os.makedirs(params.outdir, exist_ok=True)
        for batch_id, chunk in SAMPLE_BATCHES.items():
            with open(f"{params.outdir}/batch{batch_id}.txt", "w") as out:
                out.write("\n".join(chunk) + "\n")


# ============================================================================
# STEP 2 — Prepare per-batch × per-chr VCF for FLARE input (3 sub-steps)
# ============================================================================

rule extract_batch_chr_vcf:
    """
    STEP 2a: Extract batch samples + FLARE variant positions from the raw UKB VCF.
    Writes a positions file (chr pos) and a raw subset VCF (SNPs only).
    """
    input:
        ukb_vcf    = lambda w: os.path.join(
                         config["ukb_vcf_dir"],
                         config["ukb_vcf_pattern"].format(chr=w.chr)
                     ),
        batch_file = OUT_DIR + "/sample_batches/batch{batch}.txt",
        flare_vars = config["flare_variants_file"]
    output:
        vcf       = OUT_DIR + "/flare_input/chr{chr}/batch{batch}.extracted.vcf.gz",
        csi       = OUT_DIR + "/flare_input/chr{chr}/batch{batch}.extracted.vcf.gz.csi",
        positions = OUT_DIR + "/flare_input/chr{chr}/batch{batch}.positions.txt"
    wildcard_constraints:
        chr   = "[0-9]+",
        batch = "[0-9]+"
    threads: 16
    conda: LAI_ENV
    resources:
        mem_mb          = 120000,
        runtime         = "72h"
    log: OUT_DIR + "/logs/extract/chr{chr}_batch{batch}.log"
    shell:
        """
        set -euo pipefail
        mkdir -p $(dirname {output.vcf}) $(dirname {log})

        echo "[$(date)] Building positions file for chr{wildcards.chr}..." | tee {log}

        awk -v chr="{wildcards.chr}" '$1 == chr {{print $1"\\t"$2}}' \
            {input.flare_vars} | sort -k1,1V -k2,2n > {output.positions}

        N_POS=$(wc -l < {output.positions})
        echo "[$(date)] Positions for chr{wildcards.chr}: $N_POS" | tee -a {log}

        if [ "$N_POS" -eq 0 ]; then
            echo "WARNING: No FLARE variants found for chr{wildcards.chr} — creating empty output." | tee -a {log}
            bcftools view -h {input.ukb_vcf} -Oz -o {output.vcf} 2>> {log}
            bcftools index -f {output.vcf} 2>> {log}
            exit 0
        fi

        echo "[$(date)] Extracting samples + variants from UKB VCF..." | tee -a {log}
        bcftools view \
            --threads {threads} \
            --force-samples \
            -R {output.positions} \
            -S {input.batch_file} \
            -v snps \
            -Oz -o {output.vcf} \
            {input.ukb_vcf} \
            2>> {log}

        bcftools index -f {output.vcf} 2>> {log}

        N_VAR=$(bcftools index -n {output.vcf} 2>/dev/null || echo 0)
        N_SAM=$(bcftools query -l {output.vcf} 2>/dev/null | wc -l || echo 0)
        echo "[$(date)] Extracted: $N_VAR variants, $N_SAM samples" | tee -a {log}
        """


rule normalize_batch_chr_vcf:
    """
    STEP 2b: Left-normalise and remove duplicate variants.
    bcftools norm -m -any splits multiallelics; -d none removes exact duplicates.
    """
    input:
        vcf = OUT_DIR + "/flare_input/chr{chr}/batch{batch}.extracted.vcf.gz"
    output:
        vcf = OUT_DIR + "/flare_input/chr{chr}/batch{batch}.norm.vcf.gz",
        csi = OUT_DIR + "/flare_input/chr{chr}/batch{batch}.norm.vcf.gz.csi"
    wildcard_constraints:
        chr   = "[0-9]+",
        batch = "[0-9]+"
    threads: 16
    conda: LAI_ENV
    resources:
        mem_mb          = 120000,
        runtime         = "24h"
    log: OUT_DIR + "/logs/normalize/chr{chr}_batch{batch}.log"
    shell:
        """
        set -euo pipefail
        mkdir -p $(dirname {log})

        echo "[$(date)] Normalizing chr{wildcards.chr} batch{wildcards.batch}..." | tee {log}

        if [ ! -s {input.vcf} ]; then
            echo "WARNING: Input VCF is empty — passing through." | tee -a {log}
            cp {input.vcf} {output.vcf}
            bcftools index -f {output.vcf} 2>> {log}
            exit 0
        fi

        bcftools norm \
            --threads {threads} \
            -m -any \
            {input.vcf} \
            2>> {log} \
        | bcftools norm \
            --threads {threads} \
            -d none \
            -Oz -o {output.vcf} \
            2>> {log}

        bcftools index -f {output.vcf} 2>> {log}

        N_VAR=$(bcftools index -n {output.vcf} 2>/dev/null || echo 0)
        echo "[$(date)] After normalisation: $N_VAR variants" | tee -a {log}
        """


rule filter_missing_batch_chr_vcf:
    """
    STEP 2c: Add F_MISSING tag and remove any variant with missing genotypes
    in this batch (F_MISSING = 0). This is the final FLARE-ready VCF.
    """
    input:
        vcf = OUT_DIR + "/flare_input/chr{chr}/batch{batch}.norm.vcf.gz"
    output:
        vcf = OUT_DIR + "/flare_input/chr{chr}/batch{batch}.vcf.gz",
        csi = OUT_DIR + "/flare_input/chr{chr}/batch{batch}.vcf.gz.csi"
    wildcard_constraints:
        chr   = "[0-9]+",
        batch = "[0-9]+"
    threads: 16
    conda: LAI_ENV
    resources:
        mem_mb          = 120000,
        runtime         = "24h"
    log: OUT_DIR + "/logs/filter_missing/chr{chr}_batch{batch}.log"
    shell:
        """
        set -euo pipefail
        mkdir -p $(dirname {log})

        echo "[$(date)] Filtering missing genotypes for chr{wildcards.chr} batch{wildcards.batch}..." | tee {log}

        if [ ! -s {input.vcf} ]; then
            echo "WARNING: Input VCF is empty — passing through." | tee -a {log}
            cp {input.vcf} {output.vcf}
            bcftools index -f {output.vcf} 2>> {log}
            exit 0
        fi

        bcftools +fill-tags \
            --threads {threads} \
            {input.vcf} \
            -- -t F_MISSING \
            2>> {log} \
        | bcftools view \
            --threads {threads} \
            -i 'F_MISSING = 0' \
            -Oz -o {output.vcf} \
            2>> {log}

        bcftools index -f {output.vcf} 2>> {log}

        N_VAR=$(bcftools index -n {output.vcf} 2>/dev/null || echo 0)
        N_SAM=$(bcftools query -l {output.vcf} 2>/dev/null | wc -l || echo 0)
        echo "[$(date)] Final FLARE input: $N_VAR variants, $N_SAM samples" | tee -a {log}
        """


# ============================================================================
# STEP 3 — Run FLARE per batch × chromosome
# ============================================================================

rule run_flare_on_batch:
    """
    Run FLARE on a single batch (≤5000 samples) for one full chromosome.
    Full-chromosome runs give better HMM performance than chunked runs.
    Outputs are kept for inspection / re-merging.
    """
    input:
        vcf_query   = OUT_DIR + "/flare_input/chr{chr}/batch{batch}.vcf.gz",
        sentinel    = OUT_DIR + "/flare_input/chr{chr}/prepare_complete.sentinel",
        vcf_ref     = config["vcf_reference"],
        sample_map  = config["sample_map"],
        genetic_map = config["genetic_map"],
        flare_jar   = config["flare_jar"]
    output:
        vcf      = OUT_DIR + "/flare_output/chr{chr}_batch{batch}.anc.vcf.gz",
        csi      = OUT_DIR + "/flare_output/chr{chr}_batch{batch}.anc.vcf.gz.csi",
        flare_log = OUT_DIR + "/flare_output/chr{chr}_batch{batch}.flare.log"
    params:
        outdir  = OUT_DIR + "/flare_output",
        prefix  = lambda w: OUT_DIR + f"/flare_output/chr{w.chr}_batch{w.batch}",
        gen     = config["generations"],
        nthreads = config.get("flare_threads", 20),
        min_maf  = config["min_maf"],
        min_mac  = config["min_mac"],
        memory   = config.get("flare_java_memory", "90g")
    wildcard_constraints:
        chr   = "[0-9]+",
        batch = "[0-9]+"
    threads: config.get("flare_threads", 20)
    conda: LAI_ENV
    resources:
        mem_mb          = config.get("flare_mem_mb", 120000),
        runtime         = "24h"
    log: OUT_DIR + "/logs/flare/chr{chr}_batch{batch}.log"
    shell:
        """
        set -euo pipefail
        mkdir -p {params.outdir} $(dirname {log})

        echo "================================================================================" | tee {log}
        echo "FLARE — chr{wildcards.chr}  batch{wildcards.batch}" | tee -a {log}
        echo "Started: $(date)" | tee -a {log}
        echo "================================================================================" | tee -a {log}

        # Handle empty input (e.g. chr had no FLARE variants for this batch)
        if [ ! -s {input.vcf_query} ]; then
            echo "WARNING: Input VCF is empty — skipping FLARE, creating empty outputs." | tee -a {log}
            touch {output.vcf} {output.csi} {output.flare_log}
            exit 0
        fi

        N_VAR=$(bcftools index -n {input.vcf_query} 2>/dev/null || echo 0)
        N_SAM=$(bcftools query -l   {input.vcf_query} 2>/dev/null | wc -l || echo 0)
        echo "Input: $N_SAM samples, $N_VAR variants" | tee -a {log}

        if [ "$N_VAR" -eq 0 ]; then
            echo "WARNING: No variants — skipping FLARE." | tee -a {log}
            touch {output.vcf} {output.csi} {output.flare_log}
            exit 0
        fi

        echo "Running FLARE..." | tee -a {log}
        java -Xmx{params.memory} -jar {input.flare_jar} \
            ref={input.vcf_ref} \
            ref-panel={input.sample_map} \
            gt={input.vcf_query} \
            map={input.genetic_map} \
            out={params.prefix} \
            probs=true \
            gen={params.gen} \
            nthreads={params.nthreads} \
            min-maf={params.min_maf} \
            min-mac={params.min_mac} \
            em=false \
            > {output.flare_log} 2>&1

        echo "FLARE exit code: $?" | tee -a {log}

        if [ ! -f {output.vcf} ]; then
            echo "ERROR: FLARE did not produce output VCF — see {output.flare_log}" | tee -a {log}
            cat {output.flare_log} >> {log}
            exit 1
        fi

        bcftools index -f {output.vcf} 2>> {log}

        N_OUT=$(bcftools index -n {output.vcf} 2>/dev/null || echo 0)
        echo "Output variants: $N_OUT" | tee -a {log}
        echo "Completed: $(date)" | tee -a {log}
        """


# ============================================================================
# STEP 2d — Barrier: all filter_missing jobs done for a chromosome
# ============================================================================

rule prepare_batches_complete:
    """
    Aggregation barrier: all prepare_batch_chr_vcf outputs for one chromosome
    must exist before any FLARE job starts on that chromosome.
    """
    input:
        vcfs = lambda w: expand(
            OUT_DIR + "/flare_input/chr{chr}/batch{batch}.vcf.gz",
            chr   = w.chr,
            batch = range(1, N_BATCHES + 1)
        )
    output:
        sentinel = OUT_DIR + "/flare_input/chr{chr}/prepare_complete.sentinel"
    wildcard_constraints:
        chr = "[0-9]+"
    threads: 1
    resources:
        mem_mb          = 1000,
        runtime         = "10m"
    shell:
        """
        echo "All prepare jobs complete for chr{wildcards.chr} at $(date)" > {output.sentinel}
        """




def batches_for_group(wildcards):
    """Return list of FLARE output VCF paths for one merge group."""
    gid     = int(wildcards.group_id)
    batches = MERGE_GROUP_MAP[gid]
    return expand(
        OUT_DIR + "/flare_output/chr{chr}_batch{batch}.anc.vcf.gz",
        chr   = wildcards.chr,
        batch = batches
    )


rule merge_flare_batches_intermediate:
    """
    Merge one group of ≤20 FLARE batch outputs per chromosome.
    Each intermediate contains ≤100k samples.
    Skips empty/invalid VCFs automatically.
    """
    input:
        vcfs     = batches_for_group
    output:
        vcf = OUT_DIR + "/flare_merged/chr{chr}_group{group_id}.anc.vcf.gz",
        csi = OUT_DIR + "/flare_merged/chr{chr}_group{group_id}.anc.vcf.gz.csi"
    wildcard_constraints:
        chr      = "[0-9]+",
        group_id = "[0-9]+"
    threads: 20
    conda: LAI_ENV
    resources:
        mem_mb          = 120000,
        runtime         = "24h"
    log: OUT_DIR + "/logs/merge_intermediate/chr{chr}_group{group_id}.log"
    shell:
        """
        set -euo pipefail
        mkdir -p $(dirname {output.vcf}) $(dirname {log})

        echo "[$(date)] Collecting valid VCFs for chr{wildcards.chr} group{wildcards.group_id}..." | tee {log}

        VALID=""
        for vcf in {input.vcfs}; do
            if [ -s "$vcf" ]; then
                N=$(bcftools index -n "$vcf" 2>/dev/null || echo 0)
                if [ "$N" -gt 0 ]; then
                    VALID="$VALID $vcf"
                else
                    echo "  Skipping (no variants): $vcf" | tee -a {log}
                fi
            else
                echo "  Skipping (empty file): $vcf" | tee -a {log}
            fi
        done

        N_VALID=$(echo $VALID | wc -w)
        echo "[$(date)] Valid VCFs: $N_VALID" | tee -a {log}

        if [ -z "$VALID" ]; then
            echo "WARNING: No valid VCFs in this group — creating empty output." | tee -a {log}
            touch {output.vcf} {output.csi}
            exit 0
        fi

        if [ "$N_VALID" -eq 1 ]; then
            # Only one valid file — just copy rather than merge
            echo "[$(date)] Single valid VCF — copying directly." | tee -a {log}
            cp $VALID {output.vcf}
            bcftools index -f {output.vcf} 2>> {log}
            cp {output.vcf}.csi {output.csi} 2>/dev/null || bcftools index -f {output.vcf} 2>> {log}
            exit 0
        fi

        echo "[$(date)] Merging $N_VALID VCFs..." | tee -a {log}
        bcftools merge \
            --threads {threads} \
            -m none \
            -Oz -o {output.vcf} \
            $VALID \
            2>> {log}

        bcftools index -f {output.vcf} 2>> {log}

        N_SAM=$(bcftools query -l {output.vcf} | wc -l)
        N_VAR=$(bcftools index -n  {output.vcf} 2>/dev/null || echo 0)
        echo "[$(date)] Intermediate merge done: $N_SAM samples, $N_VAR variants" | tee -a {log}
        """


# ============================================================================
# STEP 4b — Final merge: all intermediate groups → stable per-chr output
# ============================================================================

rule merge_flare_chr_final:
    """
    Merge all intermediate group VCFs for one chromosome into the stable output.
    This is the last rule in the pipeline.
    With 500k samples this will produce very large files (~hundreds of GB).
    Memory: 120 GB, runtime: 72h (chr1-7 exceeded 24h).
    """
    input:
        intermediates = expand(
            OUT_DIR + "/flare_merged/chr{{chr}}_group{group_id}.anc.vcf.gz",
            group_id = range(1, N_MERGE_GROUPS + 1)
        )
    output:
        vcf = STABLE_DIR + "/chr{chr}.anc.vcf.gz",
        csi = STABLE_DIR + "/chr{chr}.anc.vcf.gz.csi"
    wildcard_constraints:
        chr = "[0-9]+"
    threads: 20
    conda: LAI_ENV
    resources:
        mem_mb          = 120000,
        runtime         = "72h"
    log: STABLE_DIR + "/logs/chr{chr}_final_merge.log"
    shell:
        """
        set -euo pipefail
        mkdir -p {STABLE_DIR}/logs

        echo "[$(date)] Final merge for chr{wildcards.chr}" | tee {log}

        VALID=""
        for vcf in {input.intermediates}; do
            if [ -s "$vcf" ]; then
                N=$(bcftools index -n "$vcf" 2>/dev/null || echo 0)
                if [ "$N" -gt 0 ]; then
                    VALID="$VALID $vcf"
                else
                    echo "  Skipping (no variants): $vcf" | tee -a {log}
                fi
            else
                echo "  Skipping (empty/missing): $vcf" | tee -a {log}
            fi
        done

        N_VALID=$(echo $VALID | wc -w)
        echo "[$(date)] Valid intermediate VCFs: $N_VALID" | tee -a {log}

        if [ -z "$VALID" ]; then
            echo "WARNING: No valid intermediates — empty final output." | tee -a {log}
            touch {output.vcf} {output.csi}
            exit 0
        fi

        if [ "$N_VALID" -eq 1 ]; then
            echo "[$(date)] Single intermediate — copying directly." | tee -a {log}
            cp $VALID {output.vcf}
            bcftools index -f {output.vcf} 2>> {log}
            exit 0
        fi

        echo "[$(date)] Merging $N_VALID intermediate VCFs into stable output..." | tee -a {log}
        bcftools merge \
            --threads {threads} \
            -m none \
            -Oz -o {output.vcf} \
            $VALID \
            2>> {log}

        bcftools index -f {output.vcf} 2>> {log}

        N_SAM=$(bcftools query -l {output.vcf} | wc -l)
        N_VAR=$(bcftools index -n  {output.vcf} 2>/dev/null || echo 0)
        echo "[$(date)] FINAL chr{wildcards.chr}: $N_SAM samples, $N_VAR variants" | tee -a {log}
        echo "[$(date)] Done: $(date)" | tee -a {log}
        """
