# Ancient DNA / GWAS Catalog study

Code for an ancient-DNA archaeogenomics study testing whether Western Hunter-Gatherer (WHG)
and Early Farmer (EF) ancestry in Neolithic Europe left a differential imprint on trait and
disease variation in present-day Europeans, with Eastern Hunter-Gatherers (EHG) included as
an independent temporal replicate. Phenotypic validation uses UK Biobank (~458,000
participants); ancestry-trait associations are screened across the GWAS Catalog.

## Repository layout

Each folder is self-contained (own README, config, conda envs, Snakefiles):

- **`gwas_enrichment/`** — GWAS Catalog category enrichment and trait-directionality
  pipeline. Includes `curation/`, the tooling that builds and validates the trait category
  vocabulary from the raw GWAS Catalog.
- **`ars/`** — Local ancestry inference (FLARE) followed by ancestry-specific regional score
  (ARS) analysis, genome-wide across the UK Biobank cohort.
- **`qpadm/`** — qpAdm ancestry proportion modeling.

See each folder's `README.md` for setup, inputs, and run commands.

## Requirements

- Snakemake >= 8 with the SLURM executor plugin, conda
- Run on a SLURM cluster; each project ships a `profiles/slurm` config (partition, resources)
- Python (environments defined per project under `envs/`)

## Data

Ancient DNA VCFs, UK Biobank phenotypes, and the GWAS Catalog associations file are not
included; paths are set per project in `config/config.local.yaml`.
