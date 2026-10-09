"""
Ancient Population Genetics GWAS Enrichment Pipeline (SIMPLIFIED)
Snakemake workflow for quantitative trait enrichment analysis
Uses exact trait matching from gwas_categories_v2.yaml
"""

import os

# =============================================================================
# Configuration
# =============================================================================

# Defaults live in config/config.yaml (relative paths, run from this folder).
# Site-specific paths go into a second file passed with --configfile
# (see config/config.local.yaml); it overrides the defaults key by key.
configfile: "config/config.yaml"

# Conda environment used by all rules. Default: the env definition shipped in
# envs/. To use an already existing environment instead, set conda_env in the
# local config to its directory path.
PYTOOLS_ENV = config.get('conda_env', 'envs/pytools.yaml')

# =============================================================================
# Helper Functions
# =============================================================================

def get_categories(wildcards):
    """Read category list from checkpoint output"""
    with open(checkpoints.extract_gwas_categories.get(**wildcards).output['category_list']) as f:
        return [line.strip() for line in f]

def get_significant_categories(wildcards):
    """
    Read top-N categories for trait decomposition and directionality analysis.
    Uses identify_directionality_categories checkpoint (top N by p-value),
    which is independent of FDR significance.
    """
    cat_file = checkpoints.identify_directionality_categories.get(**wildcards).output['directionality_categories']
    with open(cat_file) as f:
        cats = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return cats

# =============================================================================
# Target Rules
# =============================================================================

rule all:
    input:
        os.path.join(config['output_dir'], 'enrichment', 'category_enrichment.txt'),
        os.path.join(config['output_dir'], 'trait_decomposition', 'all_traits_enrichment.txt'),
        os.path.join(config['output_dir'], 'directionality', 'trait_directionality_summary.txt'),
        os.path.join(config['output_dir'], 'report', 'analysis_report.html'),
        os.path.join(config['output_dir'], 'snp_sets', 'snp_sets_summary.txt'),
        os.path.join(config['output_dir'], 'directionality_sensitivity',
                     'trait_directionality_summary_excl'
                     + str(config.get('sensitivity', {}).get('exclude_top_n', 3))
                     + '.txt'),
        os.path.join(config['output_dir'], 'directionality_bootstrap', 'bootstrap_ci.txt'),
        os.path.join(config['output_dir'], 'directionality_bootstrap', 'snp_leaveoneout.txt'),
        os.path.join(config['output_dir'], 'summary', 'pipeline_summary.txt')


# =============================================================================
# Report
# =============================================================================

rule generate_report:
    input:
        category_enrichment    = os.path.join(config['output_dir'], 'enrichment', 'category_enrichment.txt'),
        directionality_summary = os.path.join(config['output_dir'], 'directionality', 'trait_directionality_summary.txt'),
        all_traits_enrichment  = os.path.join(config['output_dir'], 'trait_decomposition', 'all_traits_enrichment.txt')
    output:
        report = os.path.join(config['output_dir'], 'report', 'analysis_report.html')
    resources:
        mem_mb          = 8000,
        runtime         = 60
    conda:
        PYTOOLS_ENV
    script:
        "scripts/generate_report.py"


# =============================================================================
# Step 1: Extract GWAS Categories from Catalog
# =============================================================================

checkpoint extract_gwas_categories:
    input:
        gwas_catalog    = config['gwas']['catalog_file'],
        categories_yaml = config['categories_yaml']
    output:
        category_list = os.path.join(config['output_dir'], 'categories', 'category_list.txt'),
        category_snps = os.path.join(config['output_dir'], 'categories', 'category_snps.txt')
    params:
        min_pvalue            = config['gwas']['min_pvalue'],
        ancestry_filter       = ','.join(config['gwas'].get('ancestry_filter', [])),
        min_snps_per_category = config['gwas'].get('min_snps_per_category', 30)
    resources:
        mem_mb          = 20000,
        runtime         = 120
    conda:
        PYTOOLS_ENV
    script:
        "scripts/extract_gwas_categories.py"

# =============================================================================
# Step 2: Generate LD Blocks Per Category
# =============================================================================

rule per_category_ld_blocks:
    input:
        vcf             = config['vcf'],
        gwas_catalog    = config['gwas']['catalog_file'],
        categories_yaml = config['categories_yaml'],
        category_list   = os.path.join(config['output_dir'], 'categories', 'category_list.txt')
    output:
        ld_blocks     = os.path.join(config['output_dir'], 'ld_blocks_per_category', '{category}', 'ld_blocks.txt'),
        block_members = os.path.join(config['output_dir'], 'ld_blocks_per_category', '{category}', 'block_members.txt'),
        stats         = os.path.join(config['output_dir'], 'ld_blocks_per_category', '{category}', 'stats.txt')
    params:
        category        = '{category}',
        stable_dir      = config['stable_dir'],
        r2_threshold    = config['ld']['r2_threshold'],
        window_kb       = config['ld']['window_kb'],
        ref_panel       = config['ld']['ref_panel'],
        keep_temp       = config['ld'].get('keep_temp', False),
        min_pvalue      = config['gwas']['min_pvalue'],
        ancestry_filter = ','.join(config['gwas'].get('ancestry_filter', []))
    resources:
        mem_mb          = 48000,
        runtime         = 360
    conda:
        PYTOOLS_ENV
    script:
        "scripts/generate_category_ld_blocks.py"

# =============================================================================
# Step 3: Merge Category Blocks
# =============================================================================

def get_category_block_files(wildcards):
    categories = get_categories(wildcards)
    return expand(
        os.path.join(config['output_dir'], 'ld_blocks_per_category', '{category}', 'ld_blocks.txt'),
        category=categories
    )

def get_category_member_files(wildcards):
    categories = get_categories(wildcards)
    return expand(
        os.path.join(config['output_dir'], 'ld_blocks_per_category', '{category}', 'block_members.txt'),
        category=categories
    )

rule merge_category_blocks:
    input:
        block_files  = get_category_block_files,
        member_files = get_category_member_files
    output:
        merged_blocks  = os.path.join(config['output_dir'], 'ld_blocks_merged', 'all_blocks.txt'),
        merged_members = os.path.join(config['output_dir'], 'ld_blocks_merged', 'all_block_members.txt')
    resources:
        mem_mb          = 20000,
        runtime         = 120
    conda:
        PYTOOLS_ENV
    script:
        "scripts/merge_category_blocks.py"

# =============================================================================
# Step 4: Calculate Fst (HG vs EF)
# =============================================================================

rule calculate_fst:
    input:
        vcf        = config['vcf'],
        hg_samples = config['hg_samples'],
        ef_samples = config['ef_samples'],
        positions  = config['positions']
    output:
        fst_results = os.path.join(config['output_dir'], 'fst', 'fst_results.txt')
    resources:
        mem_mb          = 24000,
        runtime         = 180
    conda:
        PYTOOLS_ENV
    script:
        "scripts/calculate_fst.py"

# =============================================================================
# Step 5: Quantitative Category Enrichment
# =============================================================================

rule quantitative_enrichment:
    input:
        fst_results    = os.path.join(config['output_dir'], 'fst', 'fst_results.txt'),
        merged_blocks  = os.path.join(config['output_dir'], 'ld_blocks_merged', 'all_blocks.txt'),
        merged_members = os.path.join(config['output_dir'], 'ld_blocks_merged', 'all_block_members.txt')
    output:
        enrichment      = os.path.join(config['output_dir'], 'enrichment', 'category_enrichment.txt'),
        distributions   = os.path.join(config['output_dir'], 'enrichment', 'fst_distributions.txt'),
        blocks_with_fst = os.path.join(config['output_dir'], 'enrichment', 'blocks_with_fst.txt')
    params:
        min_blocks       = config['gwas'].get('min_blocks_per_category', 5),
        block_fst_metric = config.get('enrichment', {}).get('block_fst_metric', 'max'),
        seed             = config.get('enrichment', {}).get('seed', 42)
    resources:
        mem_mb          = 24000,
        runtime         = 120
    conda:
        PYTOOLS_ENV
    script:
        "scripts/quantitative_category_enrichment.py"

# =============================================================================
# Step 6a: Identify FDR-Significant Categories (for enrichment report only)
# =============================================================================

checkpoint identify_significant_categories:
    input:
        enrichment_results = os.path.join(config['output_dir'], 'enrichment', 'category_enrichment.txt')
    output:
        significant_categories = os.path.join(config['output_dir'], 'trait_decomposition', 'significant_categories.txt')
    params:
        fdr_threshold = config.get('trait_decomposition', {}).get('category_fdr_threshold', 0.05)
    resources:
        mem_mb          = 20000,
        runtime         = 120
    conda:
        PYTOOLS_ENV
    script:
        "scripts/identify_significant_categories.py"

# =============================================================================
# Step 6b: Identify Top-N Categories for Directionality Analysis
# =============================================================================

checkpoint identify_directionality_categories:
    input:
        enrichment_results = os.path.join(config['output_dir'], 'enrichment', 'category_enrichment.txt')
    output:
        directionality_categories = os.path.join(config['output_dir'], 'trait_decomposition', 'directionality_categories.txt')
    params:
        top_n   = config.get('trait_decomposition', {}).get('directionality_top_n', 10),
        rank_by = config.get('trait_decomposition', {}).get('directionality_rank_by', 'mannwhitneyu_pval')
    resources:
        mem_mb          = 8000,
        runtime         = 30
    conda:
        PYTOOLS_ENV
    script:
        "scripts/identify_directionality_categories.py"

# =============================================================================
# Step 7: Decompose Top-N Categories into Trait-Level Block Assignments
#
# Simplified: reads pre-computed category LD blocks (block_members.txt) and
# writes trait_blocks.txt mapping each trait's GWAS SNPs to their block IDs.
# No PLINK calls, no FST calculation — filtering happens downstream in
# calculate_trait_directionality.py.
# =============================================================================

rule decompose_category_traits:
    input:
        gwas_catalog       = config['gwas']['catalog_file'],
        categories_yaml    = config['categories_yaml'],
        block_members      = os.path.join(config['output_dir'], 'ld_blocks_per_category', '{category}', 'block_members.txt'),
        directionality_cats = os.path.join(config['output_dir'], 'trait_decomposition', 'directionality_categories.txt')
    output:
        trait_enrichment    = os.path.join(config['output_dir'], 'trait_decomposition', '{category}', 'trait_enrichment.txt'),
        trait_blocks        = os.path.join(config['output_dir'], 'trait_decomposition', '{category}', 'trait_blocks.txt')
    params:
        category        = '{category}',
        min_pvalue      = config['gwas']['min_pvalue'],
        ancestry_filter = ','.join(config['gwas'].get('ancestry_filter', []))
    resources:
        mem_mb          = 8000,
        runtime         = 60
    conda:
        PYTOOLS_ENV
    script:
        "scripts/decompose_category_traits.py"

# =============================================================================
# Step 8: Merge Trait Results from Top-N Categories
# =============================================================================

def get_trait_enrichment_files(wildcards):
    """Get trait enrichment files for all directionality (top-N) categories"""
    categories = get_significant_categories(wildcards)
    return expand(
        os.path.join(config['output_dir'], 'trait_decomposition', '{category}', 'trait_enrichment.txt'),
        category=categories
    )

rule merge_trait_enrichment:
    input:
        trait_files = get_trait_enrichment_files
    output:
        summary = os.path.join(config['output_dir'], 'trait_decomposition', 'all_traits_enrichment.txt')
    resources:
        mem_mb          = 20000,
        runtime         = 120
    conda:
        PYTOOLS_ENV
    script:
        "scripts/merge_trait_enrichment.py"

# =============================================================================
# Step 9: Calculate Trait Directionality (PGS-based)
# =============================================================================

rule calculate_trait_directionality:
    input:
        trait_enrichment = os.path.join(config['output_dir'], 'trait_decomposition', 'all_traits_enrichment.txt'),
        gwas_catalog     = config['gwas']['catalog_file'],
        vcf              = config['vcf'],
        hg_samples       = config['hg_samples'],
        ef_samples       = config['ef_samples']
    output:
        directionality_summary = os.path.join(config['output_dir'], 'directionality', 'trait_directionality_summary.txt'),
        directionality_per_snp = os.path.join(config['output_dir'], 'directionality', 'trait_directionality_per_snp.txt')
    log:
        os.path.join(config['output_dir'], 'directionality', 'logs', 'calculate_trait_directionality.log')
    params:
        min_pvalue            = config['gwas']['min_pvalue'],
        ancestry_filter       = ','.join(config['gwas'].get('ancestry_filter', [])),
        strict_european_only  = config['gwas'].get('strict_european_only', True),
        trait_decomp_dir      = os.path.join(config['output_dir'], 'trait_decomposition'),
        snp_contributions_dir = os.path.join(config['output_dir'], 'directionality', 'snp_contributions'),
        trait_groups          = config.get('trait_groups', [])
    resources:
        mem_mb          = 72000,
        runtime         = 1080
    conda:
        PYTOOLS_ENV
    script:
        "scripts/calculate_trait_directionality.py"


# =============================================================================
# Step 10: Export SNP Sets for Downstream Analysis
#
# One file per trait (chr, pos, effect_allele, beta) for all traits from
# FDR-significant categories (mannwhitneyu_fdr < 0.05) with >= min_snps SNPs.
# SNPs are exactly those used in the directionality calculation.
# =============================================================================

rule export_snp_sets:
    input:
        enrichment       = os.path.join(config['output_dir'], 'enrichment', 'category_enrichment.txt'),
        directionality   = os.path.join(config['output_dir'], 'directionality', 'trait_directionality_summary.txt')
    output:
        summary = os.path.join(config['output_dir'], 'snp_sets', 'snp_sets_summary.txt')
    params:
        run_dir          = config['output_dir'],
        contrib_base_dir = os.path.join(config['output_dir'], 'directionality', 'snp_contributions'),
        output_dir       = os.path.join(config['output_dir'], 'snp_sets'),
        min_snps         = config.get('snp_sets', {}).get('min_snps', 10)
    resources:
        mem_mb          = 8000,
        runtime         = 30
    conda:
        PYTOOLS_ENV
    script:
        "scripts/export_snp_sets.py"


# =============================================================================
# Step 11: Sensitivity Analysis — Directionality with Top-N SNPs Excluded
#
# Reruns the directionality summary after removing the top N SNPs per trait
# (by |pgs_contribution|) to verify the signal is not driven by a handful
# of high-leverage SNPs. Same trait set as the main directionality analysis.
# =============================================================================

rule directionality_sensitivity:
    input:
        trait_enrichment = os.path.join(config['output_dir'], 'trait_decomposition', 'all_traits_enrichment.txt'),
        gwas_catalog     = config['gwas']['catalog_file'],
        vcf              = config['vcf'],
        hg_samples       = config['hg_samples'],
        ef_samples       = config['ef_samples']
    output:
        sensitivity_summary = os.path.join(
            config['output_dir'], 'directionality_sensitivity',
            'trait_directionality_summary_excl'
            + str(config.get('sensitivity', {}).get('exclude_top_n', 3))
            + '.txt'
        )
    params:
        trait_decomp_dir         = os.path.join(config['output_dir'], 'trait_decomposition'),
        contrib_base_dir         = os.path.join(config['output_dir'], 'directionality', 'snp_contributions'),
        output_dir               = os.path.join(config['output_dir'], 'directionality_sensitivity'),
        min_pvalue               = config['gwas']['min_pvalue'],
        ancestry_filter          = ','.join(config['gwas'].get('ancestry_filter', [])),
        strict_european_only     = config['gwas'].get('strict_european_only', True),
        exclude_top_n            = config.get('sensitivity', {}).get('exclude_top_n', 3),
        min_snps_after_exclusion = config.get('sensitivity', {}).get('min_snps_after_exclusion', 5)
    resources:
        mem_mb          = 72000,
        runtime         = 1080
    conda:
        PYTOOLS_ENV
    script:
        "scripts/calculate_trait_directionality_sensitivity.py"


# =============================================================================
# Step 12: Bootstrap & Jackknife Confidence Intervals for Cohen's d
#
# Produces two types of CIs per trait:
#   A) Bootstrap individuals (resample WHG/EF with replacement)
#   B) Delete-d SNP dropout (fraction set by bootstrap.snp_dropout_fraction)
# plus a leave-one-SNP-out table (snp_leaveoneout.txt).
# Output has same columns as trait_directionality_summary.txt + CI columns.
# =============================================================================

rule directionality_bootstrap:
    input:
        main_summary = os.path.join(config['output_dir'], 'directionality',
                                    'trait_directionality_summary.txt'),
        vcf          = config['vcf'],
        hg_samples   = config['hg_samples'],
        ef_samples   = config['ef_samples']
    output:
        bootstrap_ci    = os.path.join(config['output_dir'],
                                       'directionality_bootstrap', 'bootstrap_ci.txt'),
        snp_leaveoneout = os.path.join(config['output_dir'],
                                       'directionality_bootstrap', 'snp_leaveoneout.txt')
    params:
        contrib_base_dir = os.path.join(config['output_dir'], 'directionality',
                                        'snp_contributions'),
        n_bootstrap          = config.get('bootstrap', {}).get('n_bootstrap', 1000),
        ci_level             = config.get('bootstrap', {}).get('ci_level', 0.95),
        seed                 = config.get('bootstrap', {}).get('seed', 42),
        snp_dropout_fraction = config.get('bootstrap', {}).get('snp_dropout_fraction', 0.10)
    resources:
        mem_mb          = 48000,
        runtime         = 360
    conda:
        PYTOOLS_ENV
    script:
        "scripts/calculate_directionality_bootstrap.py"


# =============================================================================
# Step 13: Pipeline Summary Statistics
#
# Unique SNPs, unique traits, mean/median SNPs per category, across
# multiple pipeline stages (genome-wide LD blocks, trait decomposition,
# final PGS/directionality analysis).
# =============================================================================

rule pipeline_summary_stats:
    input:
        merged_members          = os.path.join(config['output_dir'], 'ld_blocks_merged', 'all_block_members.txt'),
        all_traits_enrichment   = os.path.join(config['output_dir'], 'trait_decomposition', 'all_traits_enrichment.txt'),
        directionality_summary  = os.path.join(config['output_dir'], 'directionality', 'trait_directionality_summary.txt'),
        directionality_per_snp  = os.path.join(config['output_dir'], 'directionality', 'trait_directionality_per_snp.txt')
    output:
        summary_txt       = os.path.join(config['output_dir'], 'summary', 'pipeline_summary.txt'),
        per_category_tsv  = os.path.join(config['output_dir'], 'summary', 'pipeline_summary_per_category.tsv')
    resources:
        mem_mb          = 16000,
        runtime         = 240
    conda:
        PYTOOLS_ENV
    script:
        "scripts/pipeline_summary_stats.py"

# =============================================================================
# Clean Rule (Optional)
# =============================================================================

rule clean:
    shell:
        """
        rm -rf {config[output_dir]}/categories
        rm -rf {config[output_dir]}/ld_blocks_per_category
        rm -rf {config[output_dir]}/ld_blocks_merged
        rm -rf {config[output_dir]}/enrichment
        rm -rf {config[output_dir]}/fst
        rm -rf {config[output_dir]}/trait_decomposition
        rm -rf {config[output_dir]}/directionality
        """