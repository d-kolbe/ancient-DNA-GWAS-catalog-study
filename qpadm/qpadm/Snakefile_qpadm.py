"""
Snakefile_qpadm.py
==================
Global ancestry of UK Biobank individuals with qpAdm, one qpAdm run per individual.
Author: Daniel Kolbe

Input: one merged PLINK set (config: plink_prefix). Its first n_reference_samples
individuals are the reference individuals used for the left and right populations,
followed by the UK Biobank individuals.

The UKB individuals are processed in blocks of subset_size ("subset_0", "subset_1", ...):
  make_keep_list               reference individuals + one block of UKB individuals
  make_plink_subsets           plink --keep
  convert_plink_to_eigenstrat  convertf
  finalize_eigenstrat_files    population labels: reference individuals from reference_ind,
                               every UKB individual its own population (UKB1..UKBn);
                               participant IDs without the FID prefix
  extract_f2_qpadm             per UKB individual: extract_f2 + qpadm (admixtools),
                               left = leftpops, right = rightpops
  aggregate_qpadm_subset       weights of all individuals of the subset -> one table
                               (eid + one column per left population)
  concat_aggregated_qpadm      all subset tables -> final_output/aggregated_qpadm_all.tsv

Run once per PLINK set (different plink_prefix and output_dir), from the project folder:
    snakemake -s Snakefile_qpadm.py --configfile config/config_qpadm.local.yaml \\
              --profile ../profiles/slurm
The final tables of the runs have the same columns; Snakefile_analysis.py concatenates them
(qpadm_tables in config/config_analysis.yaml).
"""

import os
import math
import pandas as pd

configfile: "config/config_qpadm.yaml"

# ---------------------------------------------------------------- settings
plink_prefix = config["plink_prefix"]
base_path    = config["output_dir"].rstrip("/") + "/"     # output folder; the paths below are relative to it
N_REF        = int(config.get("n_reference_samples", 123))
SUBSET_SIZE  = int(config.get("subset_size", 1000))
LEFTPOPS     = config["leftpops"]
RIGHTPOPS    = config["rightpops"]
REFERENCE_IND = config["reference_ind"]

# Conda environment for all rules. Default: envs/qpadm.yaml. To use an existing
# environment, set qpadm_conda_env to its directory path.
QPADM_ENV = config.get("qpadm_conda_env", "envs/qpadm.yaml")

wildcard_constraints:
    sub    = r"subset_\d+",
    ukbpop = r"UKB\d+"

# ---------------------------------------------------------------- subsets
# Block i = UKB individuals [i*subset_size, (i+1)*subset_size) after the reference individuals.
if not os.path.exists(plink_prefix + ".fam"):
    raise FileNotFoundError(f"PLINK fam file not found: {plink_prefix}.fam (config: plink_prefix)")
FAM = pd.read_csv(plink_prefix + ".fam", sep=r"\s+", header=None)
N_UKB = len(FAM) - N_REF
if N_UKB <= 0:
    raise ValueError(f"{plink_prefix}.fam has {len(FAM)} individuals, not more than n_reference_samples={N_REF}")

SUBSET_POP_COMBOS = []      # (subset name, population label of one UKB individual)
for i in range(math.ceil(N_UKB / SUBSET_SIZE)):
    n_in_subset = min(SUBSET_SIZE, N_UKB - i * SUBSET_SIZE)
    for k in range(n_in_subset):
        SUBSET_POP_COMBOS.append((f"subset_{i}", f"UKB{k+1}"))
SUBS = sorted(set(x[0] for x in SUBSET_POP_COMBOS), key=lambda x: int(x.split("_")[1]))
print(f"[qpadm] {N_UKB:,} UKB individuals in {len(SUBS)} subsets of up to {SUBSET_SIZE}; "
      f"{len(SUBSET_POP_COMBOS):,} qpAdm runs")


##### Rule all #####
rule all:
    input:
        base_path + "final_output/aggregated_qpadm_all.tsv"


rule make_keep_list:
    """Keep list of one subset: the reference individuals + one block of UKB individuals."""
    output:
        base_path + "keep_lists/{sub}.keep"
    resources:
        mem_mb = 4000,
        runtime = "30m"
    run:
        i = int(wildcards.sub.split("_")[1])
        chunk = FAM.iloc[N_REF + i * SUBSET_SIZE : N_REF + (i + 1) * SUBSET_SIZE]
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        pd.concat([FAM.iloc[:N_REF], chunk])[[0, 1]].to_csv(output[0], sep="\t", header=False, index=False)


rule make_plink_subsets:
    input:
        bed=plink_prefix + ".bed",
        bim=plink_prefix + ".bim",
        fam=plink_prefix + ".fam",
        keep=base_path + "keep_lists/{sub}.keep",
    output:
        bed=base_path + "plink_subsets/{sub}/region.bed",
        bim=base_path + "plink_subsets/{sub}/region.bim",
        fam=base_path + "plink_subsets/{sub}/region.fam",
    params:
        outdir=base_path + "plink_subsets/{sub}/region",
        plink_prefix=plink_prefix
    resources:
        mem_mb = 128000, # Mb
        runtime = "12h", # in hours
    conda: QPADM_ENV
    shell:
        """
        mkdir -p $(dirname {output.bed})
        plink --bfile {params.plink_prefix} \
              --keep {input.keep} \
              --allow-no-sex \
              --make-bed --out {params.outdir}
        """


rule convert_plink_to_eigenstrat:
    input:
        bed=base_path + "plink_subsets/{sub}/region.bed",
        bim=base_path + "plink_subsets/{sub}/region.bim",
        fam=base_path + "plink_subsets/{sub}/region.fam",
    output:
        geno=base_path + "plink_subsets/{sub}/eigenstrat_temp.geno",
        snp=base_path + "plink_subsets/{sub}/eigenstrat_temp.snp",
        ind=base_path + "plink_subsets/{sub}/eigenstrat_temp.ind",
    params:
        parfile=base_path + "plink_subsets/{sub}/parfile.par",
    resources:
        mem_mb = 16000, # Mb
        runtime = "8h", # in hours
    conda: QPADM_ENV
    shell:
        """
        mkdir -p $(dirname {output.geno})

        cat > {params.parfile} <<EOF
genotypename:    {input.bed}
snpname:         {input.bim}
indivname:       {input.fam}
outputformat:    EIGENSTRAT
genotypeoutname: {output.geno}
snpoutname:      {output.snp}
indivoutname:    {output.ind}
EOF

        convertf -p {params.parfile}
        """


rule finalize_eigenstrat_files:
    input:
        geno=base_path + "plink_subsets/{sub}/eigenstrat_temp.geno",
        snp=base_path + "plink_subsets/{sub}/eigenstrat_temp.snp",
        ind=base_path + "plink_subsets/{sub}/eigenstrat_temp.ind",
        ref=REFERENCE_IND,
    output:
        geno=base_path + "plink_subsets/{sub}/eigenstrat.geno",
        snp=base_path + "plink_subsets/{sub}/eigenstrat.snp",
        ind=base_path + "plink_subsets/{sub}/eigenstrat.ind",
    resources:
        mem_mb = 16000, # Mb
        runtime = "8h", # in hours
    run:
        import pandas as pd
        import shutil

        # Rename geno and snp
        shutil.copy(input.geno, output.geno)
        shutil.copy(input.snp, output.snp)

        # Fix .ind file
        df = pd.read_csv(input.ind, sep=r"\s+", names=["eid", "sex", "groupID"])
        ref = pd.read_csv(input.ref, sep=r"\s+", names=["eid", "sex", "groupID"])

        df['eid'] = df['eid'].str.replace(r'^.*?:', '', regex=True)

        ref_subset = df.iloc[:N_REF].merge(ref[['eid', 'sex', 'groupID']], on=['eid', 'sex'], how='left', suffixes=('', '_ref'))
        df.loc[:N_REF - 1, 'groupID'] = ref_subset['groupID_ref']
        df.loc[N_REF:, 'groupID'] = pd.Series([f"UKB{i+1}" for i in range(len(df) - N_REF)]).values

        # Strip any spaces (belt-and-braces)
        df['groupID'] = df['groupID'].astype(str).str.strip().str.replace(r"\s+", "", regex=True)

        df.to_csv(output.ind, sep="\t", index=False, header=False)


rule extract_f2_qpadm:
    input:
        geno=base_path + "plink_subsets/{sub}/eigenstrat.geno",
        snp=base_path + "plink_subsets/{sub}/eigenstrat.snp",
        ind=base_path + "plink_subsets/{sub}/eigenstrat.ind",
        leftpops=LEFTPOPS,
        rightpops=RIGHTPOPS,
    output:
        result=base_path + "plink_subsets/{sub}/qpadm/{ukbpop}/results/{ukbpop}_qpadm.Rdata"
    params:
        f2_dir=base_path + "plink_subsets/{sub}/qpadm/{ukbpop}/f2",
        qpadm_dir=base_path + "plink_subsets/{sub}/qpadm/{ukbpop}/results",
        prefix=base_path + "plink_subsets/{sub}/eigenstrat",
        pop_in="{ukbpop}"
    resources:
        mem_mb = 16000, # Mb
        runtime = "2h", # in hours
    log:
        base_path + "logs/extract_f2_qpadm/{sub}_{ukbpop}.log"
    conda: QPADM_ENV
    script:
        "scripts/extract_f2_block.R"


rule aggregate_qpadm_subset:
    input:
        wait_for = lambda wildcards: expand(
            base_path + "plink_subsets/{sub}/qpadm/{ukbpop}/results/{ukbpop}_qpadm.Rdata",
            sub=wildcards.sub,
            ukbpop=[x[1] for x in SUBSET_POP_COMBOS if x[0] == wildcards.sub]
        ),
        leftpops=LEFTPOPS,
    output:
        base_path + "plink_subsets/{sub}/aggregated_qpadm.tsv"
    params:
        sub="{sub}"
    log:
        base_path + "logs/aggregate_qpadm_subset/{sub}.log"
    resources:
        mem_mb = 32000,
        runtime = "12h",
    conda: QPADM_ENV
    script:
        "scripts/summarize_qpadm_subset.R"


rule concat_aggregated_qpadm:
    input:
        expand(base_path + "plink_subsets/{sub}/aggregated_qpadm.tsv", sub=SUBS)
    output:
        base_path + "final_output/aggregated_qpadm_all.tsv"
    resources:
        mem_mb = 60000, # Mb
        runtime = "12h", # in hours
    run:
        import pandas as pd

        input_files = list(input)
        dfs = [pd.read_csv(f, sep="\t") for f in input_files]
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(output[0], sep="\t", index=False)