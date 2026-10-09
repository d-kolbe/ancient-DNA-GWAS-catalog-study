"""
Snakefile_analysis.py
=====================
Visualization pipeline for the Ancient DNA / GWAS enrichment project.
Runs independently of the main Snakefile — reads its output files as inputs.

Run from the project folder with:
    snakemake -s Snakefile_analysis.py \
              --configfile config/config_analysis.local.yaml \
              --use-conda --cores 4
(config/config_analysis.yaml is loaded automatically; the local file only
overrides site-specific paths.)

All plot parameters are controlled via config_analysis.yaml.

Note: the list of categories for the SNP-contribution plots is read from the main
pipeline's category_enrichment.txt while the DAG is built, so the main run must be
finished first and pandas must be available in the environment that launches Snakemake.
"""

import os

configfile: "config/config_analysis.yaml"

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------

ADIR  = config['analysis_output_dir']
PDIR  = config['plot_output_dir']

# Conda environment for all plotting rules. Default: envs/plots.yaml.
# To use an existing environment, set plots_conda_env to its directory path.
PLOTS_ENV = config.get('plots_conda_env', 'envs/plots.yaml')

THEMES                   = ['dark', 'light']

ENRICHMENT_FILE          = os.path.join(ADIR, 'enrichment',         'category_enrichment.txt')
DIRECTIONALITY_FILE      = os.path.join(ADIR, 'directionality',     'trait_directionality_summary.txt')
DIRECTIONALITY_CATS_FILE = os.path.join(ADIR, 'trait_decomposition', 'directionality_categories.txt')
SNP_CONTRIBUTIONS_DIR    = os.path.join(ADIR, 'directionality',     'snp_contributions')

# Optional result files of the main pipeline used by the forest plot.
# Configured relative to ADIR; an empty value disables the corresponding panel.
DIRPLOT = dict(config['directionality_plot'])
for _key in ('sensitivity_directionality_file', 'bootstrap_ci_file', 'loo_file'):
    _rel = DIRPLOT.get(_key, '')
    DIRPLOT[_key] = os.path.join(ADIR, _rel) if _rel else ''
OPTIONAL_FOREST_INPUTS = [DIRPLOT[k] for k in
                          ('sensitivity_directionality_file', 'bootstrap_ci_file', 'loo_file')
                          if DIRPLOT[k]]

# ---------------------------------------------------------------------------
# Helper: read top-N category names from checkpoint output
# ---------------------------------------------------------------------------

def get_directionality_categories():
    cats = []
    with open(DIRECTIONALITY_CATS_FILE) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#'):
                cats.append(line)
    return cats


def get_snp_contribution_categories():
    """
    Top-N categories ranked by configured p-value column.
    Read at DAG-build time from category_enrichment.txt.
    """
    import pandas as pd, re as _re

    top_n   = int(config.get('snp_contribution_plot', {}).get('top_n_categories', 10))
    rank_by = config.get('snp_contribution_plot', {}).get('rank_by', 'mannwhitneyu_pval')

    enr = pd.read_csv(ENRICHMENT_FILE, sep='\t', comment='#')
    cat_col = next((c for c in ['category', 'Category', 'trait_category']
                    if c in enr.columns), None)
    if cat_col is None:
        raise ValueError(f"No category column in {ENRICHMENT_FILE}")

    if rank_by not in enr.columns:
        P_CANDS = ['mannwhitneyu_pval', 'mw_p_value', 'p_value', 'pvalue',
                   'mannwhitneyu_pval_twosided', 'kolmogorov_smirnov_pval']
        rank_by = next((c for c in P_CANDS if c in enr.columns), None)
        if rank_by is None:
            raise ValueError(f"No p-value column found in {ENRICHMENT_FILE}")

    ascending = bool(_re.search(r'p.?val', rank_by, flags=_re.IGNORECASE))
    top_cats  = (enr.dropna(subset=[rank_by])
                    .sort_values(rank_by, ascending=ascending)
                    .head(top_n)[cat_col]
                    .tolist())

    def to_safe(name):
        return _re.sub(r'[^\w\-]', '_', str(name)).strip('_')

    return [to_safe(c) for c in top_cats]


# ---------------------------------------------------------------------------
# Target rule
# ---------------------------------------------------------------------------

rule all:
    input:
        # Plot 1 — category enrichment volcano
        expand(os.path.join(PDIR, '{theme}', 'enrichment_volcano.pdf'),     theme=THEMES),
        expand(os.path.join(PDIR, '{theme}', 'enrichment_volcano_fdr.pdf'), theme=THEMES),

        # Plot 2 — 2×3 panel directionality forest
        os.path.join(PDIR, 'directionality_panel.pdf'),
        os.path.join(PDIR, 'directionality_panel_sensitivity.pdf'),
        os.path.join(PDIR, 'directionality_panel_bootstrap.pdf'),

        # Plot 2b — SNP contribution chord diagrams
        os.path.join(PDIR, 'snp_chord_panel.pdf'),

        # Plot 3 — SNP overlap heatmaps
        expand(os.path.join(PDIR, '{theme}', 'snp_overlap', 'jaccard_all_snps.pdf'),      theme=THEMES),
        expand(os.path.join(PDIR, '{theme}', 'snp_overlap', 'jaccard_top_snps.pdf'),      theme=THEMES),
        expand(os.path.join(PDIR, '{theme}', 'snp_overlap', 'overlap_summary_table.tsv'), theme=THEMES),

        # Plot 4 — SNP contribution Manhattan plots
        expand(
            os.path.join(PDIR, '{theme}', 'snp_contributions', '{category}'),
            theme=THEMES, category=get_snp_contribution_categories()
        )


# ---------------------------------------------------------------------------
# Rule 1: Category Enrichment Overview
# ---------------------------------------------------------------------------

rule plot_category_enrichment:
    input:
        enrichment = ENRICHMENT_FILE
    output:
        plot_volcano     = os.path.join(PDIR, '{theme}', 'enrichment_volcano.pdf'),
        plot_volcano_fdr = os.path.join(PDIR, '{theme}', 'enrichment_volcano_fdr.pdf')
    params:
        theme = '{theme}',
        **config['enrichment_plot'],
        **config['volcano_plot']
    resources:
        mem_mb          = 8000,
        runtime         = 30
    conda:
        PLOTS_ENV
    script:
        "scripts/plots/plot_category_enrichment.py"


# ---------------------------------------------------------------------------
# Rule 2: 2×3 Panel Directionality Forest
# ---------------------------------------------------------------------------

rule plot_directionality_panel:
    input:
        enrichment                   = ENRICHMENT_FILE,
        directionality               = DIRECTIONALITY_FILE,
        snp_contributions_dir        = SNP_CONTRIBUTIONS_DIR,
        optional_results             = OPTIONAL_FOREST_INPUTS
    output:
        plot             = os.path.join(PDIR, 'directionality_panel.pdf'),
        plot_sensitivity = os.path.join(PDIR, 'directionality_panel_sensitivity.pdf'),
        plot_bootstrap   = os.path.join(PDIR, 'directionality_panel_bootstrap.pdf')
    params:
        **DIRPLOT
    resources:
        mem_mb          = 16000,
        runtime         = 240
    conda:
        PLOTS_ENV
    script:
        "scripts/plots/plot_directionality_forest.py"


# ---------------------------------------------------------------------------
# Rule 2b: SNP Contribution Chord Panel
# ---------------------------------------------------------------------------

rule plot_snp_chord:
    input:
        enrichment            = ENRICHMENT_FILE,
        directionality        = DIRECTIONALITY_FILE,
        snp_contributions_dir = SNP_CONTRIBUTIONS_DIR
    output:
        plot = os.path.join(PDIR, 'snp_chord_panel.pdf')
    params:
        **DIRPLOT
    resources:
        mem_mb          = 16000,
        runtime         = 360
    retries: 1
    conda:
        PLOTS_ENV
    script:
        "scripts/plots/plot_snp_chord.py"


# ---------------------------------------------------------------------------
# Rule 3: SNP Overlap Heatmaps
# ---------------------------------------------------------------------------

rule plot_snp_overlap:
    input:
        directionality        = DIRECTIONALITY_FILE,
        dir_cats              = DIRECTIONALITY_CATS_FILE,
        snp_contributions_dir = SNP_CONTRIBUTIONS_DIR
    output:
        heatmap_all   = os.path.join(PDIR, '{theme}', 'snp_overlap', 'jaccard_all_snps.pdf'),
        heatmap_top   = os.path.join(PDIR, '{theme}', 'snp_overlap', 'jaccard_top_snps.pdf'),
        summary_table = os.path.join(PDIR, '{theme}', 'snp_overlap', 'overlap_summary_table.tsv')
    params:
        theme = '{theme}',
        **config['snp_overlap_plot']
    resources:
        mem_mb          = 16000,
        runtime         = 60
    conda:
        PLOTS_ENV
    script:
        "scripts/plots/plot_snp_overlap_heatmap.py"


# ---------------------------------------------------------------------------
# Rule 4: SNP Contribution Manhattan Plots
# ---------------------------------------------------------------------------

rule plot_snp_contributions:
    input:
        contributions_dir = SNP_CONTRIBUTIONS_DIR,
        dir_cats          = DIRECTIONALITY_CATS_FILE,
        enrichment        = ENRICHMENT_FILE,
        directionality    = DIRECTIONALITY_FILE
    output:
        plot_dir = directory(os.path.join(PDIR, '{theme}', 'snp_contributions', '{category}'))
    params:
        category = '{category}',
        theme    = '{theme}',
        **config['snp_contribution_plot']
    resources:
        mem_mb          = 8000,
        runtime         = 30
    conda:
        PLOTS_ENV
    script:
        "scripts/plots/plot_snp_contributions.py"