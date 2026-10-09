# gwas_enrichment

Screen of GWAS Catalog traits for differential allele-frequency divergence between Western Hunter-Gatherers (WHG) and Early Farmers (EF), with trait-level directionality analysis and an Eastern Hunter-Gatherer (EHG) comparison.

| workflow | purpose |
|---|---|
| `Snakefile_gwas_enrichment.py` | main pipeline: enrichment, directionality, robustness |
| `Snakefile_analysis.py` | figures |
| `Snakefile_ehg.py` | EF vs EHG for the same traits |

## Setup

Snakemake 8 or newer and conda/mamba (`--use-conda` builds the environments in `envs/`; to use existing ones, set `conda_env` or `plots_conda_env` in your local config). Large inputs go under `data/`; paths for your system go into `config/*.local.yaml`. Run Snakemake from this folder. VCFs must be bgzipped and indexed, with contigs named `1` to `22`. The environment that launches `Snakefile_analysis.py` also needs pandas. A SLURM profile is in `../profiles/slurm/`.

## Input data

| input | config key |
|---|---|
| Ancient genotypes (imputed, phased VCF) | `vcf` |
| Sample lists (in `resources/`) | `hg_samples`, `ef_samples`; `EHG_samples.txt` for the EHG workflow |
| VCF with EF and EHG samples | `vcf_ehg` |
| Positions used for Fst | `positions` |
| GWAS Catalog v1.0.2, release `e115_r2026-04-27`, GRCh37 | `gwas.catalog_file` |
| LD reference (1000 Genomes CEU, PLINK format) | `ld.ref_panel` |
| Trait categories (`resources/gwas_categories_v2.yaml`) | `categories_yaml` |

## Running

```
snakemake -s Snakefile_gwas_enrichment.py --configfile config/config.local.yaml --profile ../profiles/slurm
snakemake -s Snakefile_analysis.py --configfile config/config_analysis.local.yaml --use-conda --cores 4
snakemake -s Snakefile_ehg.py --configfile config/config_ehg.local.yaml --use-conda --cores 4
```

The main pipeline has to finish first. The local configs of the other two set `analysis_output_dir` to its `output_dir`.

## What it does

**Main pipeline.** GWAS Catalog traits are grouped into curated categories. For each category, LD blocks around the GWAS SNPs are built from the CEU panel and tested for elevated WHG-vs-EF Fst against all category blocks. For the top categories, a GWAS Catalog-derived trait index (PGS in the code) is computed per ancient individual and compared between WHG and EF, with a top-SNP exclusion analysis and bootstrap confidence intervals.

**Figures.** `Snakefile_analysis.py` draws the enrichment volcano plots, the directionality forest panel, the SNP chord panel, and the SNP overlap and contribution plots. Panel selection is set in `config/config_analysis.yaml`.

**EHG.** `Snakefile_ehg.py` repeats the directionality comparison for EF vs EHG on the trait set of the main forest panel, using the same SNP sets.

## Outputs

Under `output_dir` of the main pipeline:

| path | content |
|---|---|
| `enrichment/category_enrichment.txt` | category-level results |
| `directionality/trait_directionality_summary.txt` | trait-level results |
| `directionality_sensitivity/`, `directionality_bootstrap/` | robustness analyses |
| `snp_sets/` | per-trait SNP sets |
| `report/` | HTML report |
| `plots/`, `ehg/` | figures; EHG comparison |
