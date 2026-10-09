# qpadm

Global ancestry (qpAdm) of UK Biobank individuals and its association with phenotypes.

| workflow | purpose |
|---|---|
| `Snakefile_qpadm.py` | qpAdm for every individual |
| `Snakefile_analysis.py` | associations with phenotypes, parental lifespan analysis, forest plots |

## Setup

Snakemake 8 or newer and conda/mamba (`--use-conda` builds the environments in `envs/`). The qpAdm step needs R with admixtools, PLINK and EIGENSOFT. Large inputs go under `data/`; paths for your system go into `config/*.local.yaml`. Run Snakemake from this folder.

## Input data

| input | config key |
|---|---|
| Merged PLINK set: reference individuals first, then the UK Biobank individuals | `plink_prefix`, `n_reference_samples` |
| Left and right populations, reference labels | `leftpops`, `rightpops`, `reference_ind` (in `resources/`) |
| UK Biobank phenotype files | `phenotype_files` |
| UK Biobank VCFs (SNP-conditioned analysis only) | `vcf_pattern` |

## Running

```
snakemake -s Snakefile_qpadm.py --configfile config/config_qpadm.local.yaml --profile ../profiles/slurm
snakemake -s Snakefile_analysis.py --configfile config/config_analysis.local.yaml --profile ../profiles/slurm
```

Run the first workflow for each PLINK set and list the resulting tables in `qpadm_tables` (`config_analysis.yaml`); they are concatenated into `admixture_file`.

## What it does

**qpAdm.** The UK Biobank individuals are processed in subsets of 1000. Every individual is its own population and is run through `extract_f2` and `qpadm` with the left and right populations from `resources/`. The weights are collected in one table with a column per left population.

**Analysis.** For each trait and covariate set, the ancestry proportions are tested against the trait, one model per ancestry component: logistic regression and Mann-Whitney U for case/control traits, linear regression and Pearson correlation for continuous traits. The same models are run with the dosages of the trait's top SNPs (from the `gwas_enrichment` project, listed under `traits` in the config) as extra covariates. Parental lifespan is analysed with log-rank tests and Cox regression. Lifestyle principal components serve as covariates. Forest plots are drawn per trait category.

## Outputs

Under `results_dir` (default `results/analysis`):

| path | content |
|---|---|
| `<field id>/<covariate set>/stats.tsv` | association results |
| `category_plots*/` | forest plots (unadjusted, SNP-conditioned, combined) |
| `survival/` | parental lifespan results and plots |
| `summary/all_traits.tsv` | all results in one table |
