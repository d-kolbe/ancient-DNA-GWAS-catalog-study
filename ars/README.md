# ars

Ancestry risk score (ARS) analysis in all ~500k UK Biobank participants: local ancestry inference with FLARE, then ARS for GWAS Catalog traits.

| workflow | purpose |
|---|---|
| `Snakefile_lai.py` | FLARE local ancestry, genome-wide, in batches of 5000 samples |
| `Snakefile_ars.py` | ARS per trait, forest plot, concordance with the ancient-genome results |

## Setup

Snakemake 8 or newer and conda/mamba (`--use-conda` builds the environments in `envs/`; to use existing ones, set `conda_env` or `ars_conda_env` in your local config). FLARE 0.5.1 (`flare.jar`, https://github.com/browning-lab/flare) goes to `software/flare.jar`. Large inputs go under `data/`; paths for your system go into `config/*.local.yaml`. Run Snakemake from this folder. The environment that launches `Snakefile_ars.py` also needs pandas. SLURM profiles are in `../profiles/slurm_lai/` and `../profiles/slurm/`.

## Input data

| input | config key |
|---|---|
| UK Biobank phased VCFs (GRCh37), one per chromosome | `ukb_vcf_dir`, `ukb_vcf_pattern` |
| Participant list | `sample_list` |
| FLARE inputs: variants, reference panel, sample map (in `resources/`), genetic map | `flare_variants_file`, `vcf_reference`, `sample_map`, `genetic_map` |
| Trait SNP sets and enrichment results from the `gwas_enrichment` project | `ukb.snp_sets_summary`, `ukb.main_pipeline_*` |
| GWAS Catalog (GRCh37), for rsID and gene labels | `ukb.gwas_catalog` |

## Running

The `gwas_enrichment` results must exist first.

```
snakemake -s Snakefile_lai.py --configfile config/config_lai.local.yaml --profile ../profiles/slurm_lai
snakemake -s Snakefile_ars.py --configfile config/config_ars.local.yaml --profile ../profiles/slurm
```

`Snakefile_ars.py` reads the FLARE output folder from `stable_flare_outputs.dir` in the LAI config (default `results/lai/stable`); if you change it in `config_lai.local.yaml`, set the same value in `config_ars.local.yaml`. `ukb.ancestry_names` must be in the order of the ancestry labels in the FLARE VCF header.

## What it does

**Local ancestry.** The participants are split into batches of 5000. Per batch and chromosome, the FLARE variants are extracted from the UK Biobank VCF and FLARE is run against the ancient reference panel. The batch results are merged into one ancestry VCF per chromosome.

**ARS.** For each trait, the trait's variants are extracted from the ancestry VCFs and, per ancestry, the effect-allele burden on haplotypes of that ancestry is computed, weighted by the effect sizes. The contrast between EF and WHG is tested with a bootstrap over individuals. The results are summarised in one table, a forest plot for the same trait panel as the ancient-genome directionality figure, and a concordance table against the ancient-genome results.

## Outputs

| path | content |
|---|---|
| `results/lai/stable/chr{N}.anc.vcf.gz` | FLARE ancestry VCFs |
| `results/ars/results/<trait>/ars_results.tsv` | ARS results per trait |
| `results/ars/summary/ars_summary_all_traits.tsv` | results for all traits |
| `results/ars/summary/ars_forest.pdf` | forest plot |
| `results/ars/summary/concordance_comparison.tsv` | concordance with the ancient-genome results |
