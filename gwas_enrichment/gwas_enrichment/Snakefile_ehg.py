"""
Snakefile_ehg.py
=================
Side-pipeline (part of the gwas_enrichment project). Adds EHG as a second
comparison population (vs EF) for exactly the trait set already shown in the
main directionality forest panel.

Reads the main pipeline's existing outputs (enrichment, directionality,
snp_contributions) read-only. Does not touch them or config.yaml. Fst is not
recalculated. Reuses the already-finalized per-trait SNP/beta sets from
snp_contributions/ -- no GWAS catalog re-parsing.

The trait/panel selection (directionality_plot) and analysis_output_dir are
taken from config/config_analysis.yaml, so the EHG comparison always uses the
same trait set as the main forest figure. config/config_ehg.yaml only adds the
EHG-specific settings.

Run from the project folder with:
    snakemake -s Snakefile_ehg.py \\
              --configfile config/config_ehg.local.yaml \\
              --use-conda --cores 4

Note: the EHG comparison reads EF and EHG genotypes from the separate VCF
vcf_ehg, not from the main pipeline's vcf.
"""

import os

# Shared panel selection and analysis_output_dir first, EHG-specific settings on top
configfile: "config/config_analysis.yaml"
configfile: "config/config_ehg.yaml"

ADIR = config['analysis_output_dir']     # read-only: main pipeline's output folder
ODIR = config.get('ehg_output_dir') or os.path.join(ADIR, 'ehg')

# Conda environments. Defaults: definitions in envs/. To use existing
# environments instead, set conda_env / plots_conda_env to their directory paths.
PYTOOLS_ENV = config.get('conda_env', 'envs/pytools.yaml')
PLOTS_ENV   = config.get('plots_conda_env', 'envs/plots.yaml')

DP = config['directionality_plot']

# --- Existing pipeline outputs (read-only inputs) ---------------------------
ENRICHMENT_FILE       = os.path.join(ADIR, 'enrichment', 'category_enrichment.txt')
DIRECTIONALITY_FILE   = os.path.join(ADIR, 'directionality', 'trait_directionality_summary.txt')
SNP_CONTRIBUTIONS_DIR = os.path.join(ADIR, 'directionality', 'snp_contributions')

# ---------------------------------------------------------------------------
# Target rule
# ---------------------------------------------------------------------------

rule all:
    input:
        os.path.join(ODIR, 'trait_directionality_summary_ehg.txt'),
        os.path.join(ODIR, 'bootstrap_ci_ehg.txt'),
        os.path.join(ODIR, 'plots', 'directionality_panel_ehg.pdf'),
        os.path.join(ODIR, 'plots', 'directionality_panel_ehg_only.pdf'),
        # full trait set (rules 4 and 4b) for the supplementary table
        os.path.join(ODIR, 'trait_directionality_summary_ehg_all.txt'),
        os.path.join(ODIR, 'bootstrap_ci_ehg_all.txt')

# ---------------------------------------------------------------------------
# Rule 1: EHG vs EF directionality, restricted to the panel trait set
# ---------------------------------------------------------------------------

rule calculate_trait_directionality_ehg:
    input:
        enrichment     = ENRICHMENT_FILE,
        directionality = DIRECTIONALITY_FILE,
        vcf            = config['vcf_ehg'],
        ef_samples     = config['ef_samples'],
        ehg_samples    = config['ehg_samples']
    output:
        directionality_summary_ehg = os.path.join(ODIR, 'trait_directionality_summary_ehg.txt')
    params:
        snp_contributions_dir = SNP_CONTRIBUTIONS_DIR,
        col_category          = DP['col_category'],
        col_trait              = DP['col_trait'],
        col_pvalue             = DP['col_pvalue'],
        col_n_extracted        = DP['col_n_extracted'],
        n_panel_categories     = DP['n_panel_categories'],
        n_panel_traits         = DP['n_panel_traits'],
        min_snps_extracted     = DP['min_snps_extracted'],
        exclude_traits         = DP.get('exclude_traits', [])
    resources:
        mem_mb          = 24000,
        runtime         = 180
    conda:
        PYTOOLS_ENV
    script:
        "scripts/calculate_trait_directionality_ehg.py"

# ---------------------------------------------------------------------------
# Rule 1b: Bootstrap CI for the EHG comparison (Option A only -- resample
# individuals, 1000x. No SNP dropout, no leave-one-out.)
# ---------------------------------------------------------------------------

rule calculate_directionality_bootstrap_ehg:
    input:
        directionality_ehg = os.path.join(ODIR, 'trait_directionality_summary_ehg.txt'),
        vcf                 = config['vcf_ehg'],
        ef_samples          = config['ef_samples'],
        ehg_samples         = config['ehg_samples']
    output:
        bootstrap_ci_ehg = os.path.join(ODIR, 'bootstrap_ci_ehg.txt')
    params:
        snp_contributions_dir = SNP_CONTRIBUTIONS_DIR,
        n_bootstrap            = config.get('n_bootstrap', 1000),
        ci_level                = config.get('ci_level', 0.95),
        seed                     = config.get('bootstrap_seed', 42)
    resources:
        mem_mb          = 24000,
        runtime         = 240
    conda:
        PYTOOLS_ENV
    script:
        "scripts/calculate_directionality_bootstrap_ehg.py"

# ---------------------------------------------------------------------------
# Rule 2: Forest panel -- WHG-vs-EF and EHG-vs-EF dodged per trait row
# ---------------------------------------------------------------------------

rule plot_directionality_panel_ehg:
    input:
        enrichment         = ENRICHMENT_FILE,
        directionality     = DIRECTIONALITY_FILE,
        directionality_ehg = os.path.join(ODIR, 'trait_directionality_summary_ehg.txt')
    output:
        plot = os.path.join(ODIR, 'plots', 'directionality_panel_ehg.pdf')
    params:
        **DP,
        ehg_y_offset = config.get('ehg_y_offset', 0.18)
    resources:
        mem_mb          = 12000,
        runtime         = 60
    conda:
        PLOTS_ENV
    script:
        "scripts/plots/plot_directionality_forest_ehg.py"

# ---------------------------------------------------------------------------
# Rule 3: EF-vs-EHG-only forest panel (no WHG series)
# ---------------------------------------------------------------------------

rule plot_directionality_panel_ehg_only:
    input:
        bootstrap_ci_ehg = os.path.join(ODIR, 'bootstrap_ci_ehg.txt')
    output:
        plot = os.path.join(ODIR, 'plots', 'directionality_panel_ehg_only.pdf')
    params:
        **DP,
        n_bootstrap = config.get('n_bootstrap', 1000)
    resources:
        mem_mb          = 8000,
        runtime         = 30
    conda:
        PLOTS_ENV
    script:
        "scripts/plots/plot_directionality_forest_ehg_only.py"

# ---------------------------------------------------------------------------
# Rule 4: EHG vs EF directionality, ALL traits (not just the top-10/category
# panel) -- for the supplementary table, not the figure. Same script as
# Rule 1, same inputs, only n_panel_traits is overridden to effectively
# unlimited so every trait per category is kept. exclude_traits is left
# empty here (Rule 1 excludes IBD_combined for plot space; the table wants
# it included, to match trait_directionality_summary.txt).
# ---------------------------------------------------------------------------

rule calculate_trait_directionality_ehg_all:
    input:
        enrichment     = ENRICHMENT_FILE,
        directionality = DIRECTIONALITY_FILE,
        vcf            = config['vcf_ehg'],
        ef_samples     = config['ef_samples'],
        ehg_samples    = config['ehg_samples']
    output:
        directionality_summary_ehg = os.path.join(ODIR, 'trait_directionality_summary_ehg_all.txt')
    params:
        snp_contributions_dir = SNP_CONTRIBUTIONS_DIR,
        col_category          = DP['col_category'],
        col_trait              = DP['col_trait'],
        col_pvalue             = DP['col_pvalue'],
        col_n_extracted        = DP['col_n_extracted'],
        n_panel_categories     = DP['n_panel_categories'],
        n_panel_traits         = 9999,  # effectively unlimited -- all traits per category
        min_snps_extracted     = DP['min_snps_extracted'],
        exclude_traits         = []
    resources:
        mem_mb          = 24000,
        runtime         = 180
    conda:
        PYTOOLS_ENV
    script:
        "scripts/calculate_trait_directionality_ehg.py"

# ---------------------------------------------------------------------------
# Rule 4b: Bootstrap CI for the full (uncapped) EHG trait set
# ---------------------------------------------------------------------------

rule calculate_directionality_bootstrap_ehg_all:
    input:
        directionality_ehg = os.path.join(ODIR, 'trait_directionality_summary_ehg_all.txt'),
        vcf                 = config['vcf_ehg'],
        ef_samples          = config['ef_samples'],
        ehg_samples         = config['ehg_samples']
    output:
        bootstrap_ci_ehg = os.path.join(ODIR, 'bootstrap_ci_ehg_all.txt')
    params:
        snp_contributions_dir = SNP_CONTRIBUTIONS_DIR,
        n_bootstrap            = config.get('n_bootstrap', 1000),
        ci_level                = config.get('ci_level', 0.95),
        seed                     = config.get('bootstrap_seed', 42)
    resources:
        mem_mb          = 24000,
        runtime         = 240
    conda:
        PYTOOLS_ENV
    script:
        "scripts/calculate_directionality_bootstrap_ehg.py"