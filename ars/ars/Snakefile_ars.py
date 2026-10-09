"""
UKB Ancestry Risk Score (ARS) Pipeline
======================================

Computes ancestry-weighted risk scores in UK Biobank FLARE data for GWAS traits
from the gwas_enrichment project. Tests whether risk allele burden at trait-associated
loci is predominantly carried on EF, ST, or WHG haplotypes in UKB.

Steps:
  1  build_ukb_variant_list     SNP sets -> variant weights + bcftools position files
  2  extract_flare_variants     bcftools -R per chromosome (22 parallel jobs, I/O-bound)
  3  compute_contributions      parse extracted VCF -> per-individual contribution npz per trait
  4  merge_contributions        sum contributions across chr1-22 per trait
  5  bootstrap_ars              bootstrap CIs on the EF-WHG contrast, per trait
  6  summarize_all_traits       combined TSV across all traits
  7  plot_ars_forest            forest plot (same panels as the ancient-genome directionality plot)
  8  compute_concordance        direction concordance with the ancient-genome results

Config
------
  config/config_lai.yaml        loaded first; only stable_flare_outputs.dir (the FLARE output of
                                Snakefile_lai.py) is used from it
  config/config_ars.yaml        ARS settings (under 'ukb:'), inputs from the gwas_enrichment
                                project, forest plot, bootstrap
  *.local.yaml                  site-specific overrides (not committed)

Run from the project folder with:
    snakemake -s Snakefile_ars.py --configfile config/config_ars.local.yaml \\
              --profile ../profiles/slurm

Inputs
------
  FLARE output of Snakefile_lai.py (chr{N}.anc.vcf.gz + .csi), and from the gwas_enrichment
  project: snp_sets_summary.txt, category_enrichment.txt, trait_directionality_summary.txt
  and bootstrap_ci.txt (the last one only for the concordance step).

Scripts (all in scripts/)
-------------------------
  ukb_build_variant_list.py, ukb_compute_contributions.py, ukb_merge_contributions.py,
  ukb_bootstrap_ars.py, ukb_summarize_traits.py, plot_ars_forest.py, compute_concordance.py
"""

import os
import re
import pandas as pd

# FLARE output location comes from the LAI config, ARS settings from config_ars.yaml
configfile: "config/config_lai.yaml"
configfile: "config/config_ars.yaml"

# =============================================================================
# Part A setup
# =============================================================================

CHROMOSOMES = list(range(1, 23))
UKB_OUT     = config['ukb']['output_dir']
# FLARE output of Snakefile_lai.py (chr{N}.anc.vcf.gz); ukb.flare_dir overrides it
FLARE_DIR   = config['ukb'].get('flare_dir') or config['stable_flare_outputs']['dir']

# Conda environment for all rules. Default: envs/ars.yaml. To use an existing
# environment, set ars_conda_env to its directory path.
ARS_ENV     = config.get('ars_conda_env', 'envs/ars.yaml')

# Read all traits from the snp_sets_summary.txt produced by the main pipeline.
_summary_path = config['ukb']['snp_sets_summary']
_summary_df   = pd.read_csv(_summary_path, sep='	')
UKB_TRAITS    = _summary_df['trait'].tolist()
print(f"[ARS] Loaded {len(UKB_TRAITS)} traits from {_summary_path}")


def safe_name(trait: str) -> str:
    """Filesystem-safe trait name: lower-case, runs of non-alphanumerics -> '_'.
    (Not the same function as sanitize_category_name() of the enrichment project.)"""
    s = trait.lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_')


SAFE_TRAITS  = {t: safe_name(t) for t in UKB_TRAITS}
SAFE_TO_ORIG = {v: k for k, v in SAFE_TRAITS.items()}
ALL_SAFE     = list(SAFE_TRAITS.values())

# Sanity checks on the trait list
_n_repeat = len(UKB_TRAITS) - len(set(UKB_TRAITS))
if _n_repeat:
    print(f"[ARS] NOTE: {_n_repeat} repeated trait rows in the summary (same trait in several "
          f"categories); the weights file of a trait is written by the last row for that trait.")
_by_safe = {}
for _t, _sn in SAFE_TRAITS.items():
    _by_safe.setdefault(_sn, []).append(_t)
_collisions = {k: v for k, v in _by_safe.items() if len(v) > 1}
if _collisions:
    print(f"[ARS] WARNING: {len(_collisions)} safe names shared by different traits "
          f"(e.g. {list(_collisions.items())[0]}); these traits are merged.")


def results_path(*parts):
    return os.path.join(UKB_OUT, *parts)


# =============================================================================
# Rule all
# =============================================================================

rule all:
    input:
        results_path('summary', 'ars_summary_all_traits.tsv'),
        results_path('summary', 'ars_forest.pdf'),
        *([results_path('summary', 'concordance_comparison.tsv')]
          if config['ukb'].get('main_pipeline_bootstrap_ci') else []),


# =============================================================================
# Rules
# =============================================================================

rule build_ukb_variant_list:
    input:
        snp_sets_summary = config['ukb']['snp_sets_summary']
    output:
        weights_dir            = directory(results_path('variant_lists', 'weights')),
        trait_manifest         = results_path('variant_lists', 'trait_manifest.tsv'),
        chr_positions_sentinel = results_path('variant_lists', 'chr_positions.done')
    params:
        safe_traits       = SAFE_TRAITS,
        gwas_catalog      = config['ukb']['gwas_catalog'],
        chr_positions_dir = results_path('variant_lists', 'chr_positions'),
        out_dir           = results_path('variant_lists')
    resources:
        mem_mb=16000, runtime=360
    conda:
        ARS_ENV
    script:
        "scripts/ukb_build_variant_list.py"


rule extract_flare_variants:
    input:
        vcf            = os.path.join(FLARE_DIR, 'chr{chrom}.anc.vcf.gz'),
        vcf_index      = os.path.join(FLARE_DIR, 'chr{chrom}.anc.vcf.gz.csi'),
        positions_done = results_path('variant_lists', 'chr_positions.done')
    output:
        extracted_vcf   = results_path('extracted', 'chr{chrom}.anc.vcf.gz'),
        extracted_index = results_path('extracted', 'chr{chrom}.anc.vcf.gz.csi')
    params:
        positions_file = results_path('variant_lists', 'chr_positions', 'chr{chrom}_positions.tsv')
    resources:
        mem_mb=8000, runtime=360
    conda:
        ARS_ENV
    shell:
        """
        set -euo pipefail
        if [[ ! -s {params.positions_file} ]]; then
            echo "No positions for chr{wildcards.chrom} - writing header-only VCF"
            bcftools view --header-only --output-type z \
                --output {output.extracted_vcf} {input.vcf}
            bcftools index --csi {output.extracted_vcf}
            exit 0
        fi
        N_POS=$(wc -l < {params.positions_file})
        echo "Extracting $N_POS positions from chr{wildcards.chrom}..."
        bcftools view --regions-file {params.positions_file} \
            --output-type z --output {output.extracted_vcf} {input.vcf}
        bcftools index --csi {output.extracted_vcf}
        N_OUT=$(bcftools view --no-header {output.extracted_vcf} | wc -l)
        echo "Done: $N_OUT variants written"
        """


rule compute_contributions:
    input:
        extracted_vcf  = results_path('extracted', 'chr{chrom}.anc.vcf.gz'),
        weights_dir    = results_path('variant_lists', 'weights'),
        trait_manifest = results_path('variant_lists', 'trait_manifest.tsv')
    output:
        contrib_files = expand(
            results_path('contributions', 'chr{{chrom}}', '{safe_trait}_contributions.npz'),
            safe_trait=ALL_SAFE
        )
    params:
        safe_traits    = ALL_SAFE,
        ancestry_names = config['ukb']['ancestry_names'],
        out_dir        = results_path('contributions', 'chr{chrom}'),
        weights_dir    = results_path('variant_lists', 'weights')
    resources:
        mem_mb=128000, runtime=360
    conda:
        ARS_ENV
    script:
        "scripts/ukb_compute_contributions.py"


rule merge_contributions:
    input:
        chrom_files = expand(
            results_path('contributions', 'chr{chrom}', '{{safe_trait}}_contributions.npz'),
            chrom=CHROMOSOMES
        )
    output:
        merged = results_path('contributions_merged', '{safe_trait}', 'merged.npz')
    params:
        safe_trait='{safe_trait}', ancestry_names=config['ukb']['ancestry_names']
    resources:
        mem_mb=64000, runtime=360
    conda:
        ARS_ENV
    script:
        "scripts/ukb_merge_contributions.py"


rule bootstrap_ars:
    input:
        merged = results_path('contributions_merged', '{safe_trait}', 'merged.npz')
    output:
        results        = results_path('results', '{safe_trait}', 'ars_results.tsv'),
        bootstrap_dist = results_path('results', '{safe_trait}', 'bootstrap_distributions.npz')
    params:
        safe_trait='{safe_trait}', ancestry_names=config['ukb']['ancestry_names'],
        n_bootstrap=config['ukb']['bootstrap']['n_iterations'],
        n_samples=config['ukb']['bootstrap']['n_samples'],
        random_seed=config['ukb']['bootstrap']['random_seed']
    resources:
        mem_mb=48000, runtime=360
    conda:
        ARS_ENV
    script:
        "scripts/ukb_bootstrap_ars.py"


rule summarize_all_traits:
    input:
        trait_results  = expand(
            results_path('results', '{safe_trait}', 'ars_results.tsv'), safe_trait=ALL_SAFE
        ),
        bootstrap_dists = expand(
            results_path('results', '{safe_trait}', 'bootstrap_distributions.npz'), safe_trait=ALL_SAFE
        ),
        trait_manifest = results_path('variant_lists', 'trait_manifest.tsv')
    output:
        summary      = results_path('summary', 'ars_summary_all_traits.tsv'),
        summary_wide = results_path('summary', 'ars_summary_all_traits_wide.tsv')
    params:
        safe_to_orig=SAFE_TO_ORIG
    resources:
        mem_mb=8000, runtime=360
    conda:
        ARS_ENV
    script:
        "scripts/ukb_summarize_traits.py"

rule plot_ars_forest:
    input:
        # Depend on summary TSV so all bootstrap jobs must complete first
        ars_summary         = results_path('summary', 'ars_summary_all_traits.tsv'),
        trait_manifest      = results_path('variant_lists', 'trait_manifest.tsv'),
        enrichment_file     = config['ukb']['main_pipeline_enrichment'],
        directionality_file = config['ukb']['main_pipeline_directionality']
    output:
        plot = results_path('summary', 'ars_forest.pdf'),
        svg  = results_path('summary', 'ars_forest.svg')
    params:
        # Pass results dir as param — not an input so Snakemake won't check existence
        results_dir     = results_path('results'),
        n_cats          = config['ukb'].get('forest_n_cats',          6),
        n_traits        = config['ukb'].get('forest_n_traits',        10),
        ci_level        = config['ukb'].get('forest_ci_level',        0.95),
        fig_width       = config['ukb'].get('forest_fig_width',       13.0),
        fig_height      = config['ukb'].get('forest_fig_height',      None),
        row_height      = config['ukb'].get('forest_row_height',      0.52),
        max_label       = config['ukb'].get('forest_max_label',       40),
        label_fontsize  = config['ukb'].get('forest_label_fontsize',  6.5),
        exclude_traits  = config['ukb'].get('forest_exclude_traits',  []),
        show_loo        = config['ukb'].get('forest_show_loo',        False),
        denominator     = config['ukb'].get('forest_denominator',     'std')
    resources:
        mem_mb=8000, runtime=360
    conda:
        ARS_ENV
    script:
        "scripts/plot_ars_forest.py"



rule compute_concordance:
    """Direction concordance between the ARS EF-WHG contrast and the ancient-genome
    trait indices (bootstrap_ci.txt of the gwas_enrichment project)."""
    input:
        ars_wide   = results_path('summary', 'ars_summary_all_traits_wide.tsv'),
        gwas_index = config['ukb'].get('main_pipeline_bootstrap_ci', '')
    output:
        table  = results_path('summary', 'concordance_comparison.tsv'),
        report = results_path('summary', 'concordance_report.txt')
    resources:
        mem_mb=8000, runtime=60
    conda:
        ARS_ENV
    shell:
        """
        set -euo pipefail
        python scripts/compute_concordance.py \\
            --ars-wide {input.ars_wide} --gwas-index {input.gwas_index} \\
            --out {output.table} > {output.report}
        """
