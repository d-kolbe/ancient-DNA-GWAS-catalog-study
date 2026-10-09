#!/usr/bin/env python3
"""
==============================================================================
TRAIT DIRECTIONALITY ANALYSIS USING POLYGENIC SCORES (PGS)
==============================================================================

PURPOSE:
    Determine which population (WHG vs EF) has higher genetic predisposition
    for each trait by calculating and comparing Polygenic Scores.

APPROACH:
    1. For each trait, identify GWAS SNPs with clear effect sizes
    2. Convert all effect sizes to beta scale (handle OR vs BETA correctly)
    3. Deduplicate SNPs within LD blocks (one SNP per block)
    4. Extract genotypes from ancient DNA VCF
    5. Calculate PGS per individual: sum(dosage × beta)
    6. Compare PGS distributions between WHG and EF (t-test)

KEY METHODOLOGICAL DECISIONS:

    OR vs BETA Parsing:
    -------------------
    GWAS catalog "OR or BETA" column contains mixed effect types.
    We identify the type by examining "95% CI (TEXT)" column:
    
    - If contains "increase/decrease" → BETA (continuous trait)
      - "unit increase" → positive beta
      - "unit decrease" → negative beta (FLIP SIGN!)
    
    - If just numbers [X-Y] → OR (case-control trait)
      - Convert to log scale: beta = log(OR)
    
    This is CRITICAL - most lipid/anthropometry traits are BETAs,
    most IBD traits are ORs.

    LD Block Handling:
    ------------------
    To avoid pseudo-replication from correlated SNPs:
    - Only use GWAS SNPs (ignore proxy SNPs from LD expansion)
    - Within each LD block, select ONE SNP with largest |beta|
    - This gives independent genetic signals for PGS calculation
    
    PGS Calculation:
    ----------------
    For each individual:
        PGS = Σ (dosage_i × beta_i) for all selected SNPs
    
    Where:
        dosage_i = number of risk allele copies (0, 1, or 2)
        beta_i = effect size (signed for BETA, log(OR) for OR)
    
    Missing genotypes are skipped (conservative approach).
    
    Statistical Testing:
    --------------------
    Independent samples t-test on PGS distributions:
        H0: mean(PGS_WHG) = mean(PGS_EF)
        HA: mean(PGS_WHG) ≠ mean(PGS_EF)
    
    Direction determined by sign of delta_PGS = mean(EF) - mean(WHG):
        delta_PGS > 0 → EF has higher genetic predisposition
        delta_PGS < 0 → WHG has higher genetic predisposition

OUTPUTS:
    1. trait_directionality_summary.txt
       - One row per trait
       - PGS statistics (mean, SE, delta, p-value)
       - Direction (EF_higher vs WHG_higher)
    
    2. trait_directionality_per_snp.txt
       - One row per SNP used in PGS calculation
       - Effect sizes, frequencies, LD block assignments

REQUIREMENTS:
    - Trait enrichment results (all_traits_enrichment.txt)
    - GWAS catalog with effect sizes
    - Ancient DNA VCF with genotypes
    - Trait LD blocks (from decomposition analysis)
    - Sample lists (WHG and EF)

AUTHOR: Claude (with human oversight)
DATE: 2026-02-03
==============================================================================
"""

# IMMEDIATELY write a test file to confirm script is running
import os
os.makedirs('os.path.join(os.path.dirname(output_summary), "debug")', exist_ok=True)
with open('os.path.join(os.path.dirname(output_summary), "debug")/SCRIPT_STARTED.txt', 'w') as f:
    f.write(f"Script started at {os.getpid()}\n")

import pandas as pd
import numpy as np
import sys
import subprocess
from scipy import stats

# Import category utilities for sanitizing names
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
try:
    from category_utils import sanitize_category_name
except ImportError:
    # Fallback if import fails
    def sanitize_category_name(name):
        """Sanitize category name for use in filenames"""
        name = (name.replace(' - ', '_-_')
                    .replace(' & ', '_and_')
                    .replace(' ', '_')
                    .replace('/', '-')
                    .replace('(', '')
                    .replace(')', ''))
        for apostrophe in ["'", "'", "'", "`", "´", "'"]:
            name = name.replace(apostrophe, '')
        for char in ['"', '"', '"', '$', '!', ';', '|', '<', '>', '&', '*', '?', '[', ']', '{', '}']:
            name = name.replace(char, '')
        return name


def apply_pure_european_filter(sample_size_series, strict_mode=True):
    """
    Filter GWAS entries to pure European ancestry studies
    
    Two-step filter:
    1. MUST contain European ancestry keywords
    2. MUST NOT contain non-European ancestry keywords (if strict_mode=True)
    
    Args:
        sample_size_series: pandas Series of 'INITIAL SAMPLE SIZE' text
        strict_mode: If True, exclude studies with ANY non-European samples
                    If False, keep any study containing European samples (permissive)
    
    Returns:
        Boolean mask for entries to keep
    """
    
    # European ancestry keywords (positive filter)
    european_keywords = [
        'European', 'european',
        'British', 'british',
        'Caucasian', 'caucasian',
        'White', 'white',
        # Specific European populations
        'Icelandic', 'icelandic',
        'Finnish', 'finnish',
        'Swedish', 'swedish',
        'Norwegian', 'norwegian',
        'Danish', 'danish',
        'German', 'german',
        'French', 'french',
        'Italian', 'italian',
        'Spanish', 'spanish',
        'Dutch', 'dutch',
        'Polish', 'polish',
        'Greek', 'greek',
        'Portuguese', 'portuguese',
        'Belgian', 'belgian',
        'Austrian', 'austrian',
        'Irish', 'irish',
        'Scottish', 'scottish',
        'Welsh', 'welsh',
        'Swiss', 'swiss',
        'Czech', 'czech',
        'Hungarian', 'hungarian',
        'Romanian', 'romanian',
        'Bulgarian', 'bulgarian'
    ]
    
    # Non-European ancestry keywords (negative filter)
    non_european_keywords = [
        'Asian', 'asian',
        'East Asian', 'east asian',
        'South Asian', 'south asian',
        'African', 'african',
        'African American', 'african american',
        'Afro-Caribbean', 'afro-caribbean', 'Afro Caribbean',
        'Sub-Saharan', 'sub-saharan',
        'Hispanic', 'hispanic',
        'Latino', 'latino',
        'Latina', 'latina',
        'Korean', 'korean',
        'Japanese', 'japanese',
        'Chinese', 'chinese',
        'Han Chinese', 'han chinese',
        'Filipino', 'filipino',
        'Vietnamese', 'vietnamese',
        'Thai', 'thai',
        'Indian', 'indian',
        'Pakistani', 'pakistani',
        'Bangladeshi', 'bangladeshi',
        'Iranian', 'iranian',
        'Arab', 'arab',
        'Middle Eastern', 'middle eastern',
        'Native American', 'native american',
        'Indigenous', 'indigenous',
        'Pacific Islander', 'pacific islander',
        'Polynesian', 'polynesian',
        'Maori', 'maori',
        'Aboriginal', 'aboriginal',
        'Jamaican', 'jamaican',
        'Mexican', 'mexican',
        'Seychellois', 'seychellois'
    ]
    
    # Step 1: Check for European keywords
    has_european = sample_size_series.str.contains(
        '|'.join(european_keywords),
        case=False,
        na=False,
        regex=True
    )
    
    if not strict_mode:
        # Permissive mode: just need European ancestry present
        return has_european
    
    # Step 2: Check for non-European keywords (strict mode)
    has_non_european = sample_size_series.str.contains(
        '|'.join(non_european_keywords),
        case=False,
        na=False,
        regex=True
    )
    
    # Keep only if has European AND does not have non-European
    return has_european & ~has_non_european

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# Print immediately to verify script is running
print("="*80)
print("TRAIT DIRECTIONALITY ANALYSIS (PGS-BASED)")
print("="*80)
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Script PID: {os.getpid()}")

try:
    # Get inputs
    trait_enrichment_file = snakemake.input['trait_enrichment']
    gwas_catalog_file = snakemake.input['gwas_catalog']
    vcf_file = snakemake.input['vcf']
    hg_samples_file = snakemake.input['hg_samples']
    ef_samples_file = snakemake.input['ef_samples']

    # Get trait decomposition directory from params
    trait_decomp_dir = snakemake.params['trait_decomp_dir']
    snp_contributions_dir = snakemake.params['snp_contributions_dir']
    try:
        trait_groups = snakemake.params['trait_groups']
        if trait_groups is None:
            trait_groups = []
    except (KeyError, AttributeError):
        trait_groups = []

    # Outputs
    output_summary = snakemake.output['directionality_summary']
    output_per_snp = snakemake.output['directionality_per_snp']

    # Parameters
    min_pvalue = float(snakemake.params['min_pvalue'])
    ancestry_filter_str = snakemake.params.get('ancestry_filter', '')
    ancestry_filter = [x.strip() for x in ancestry_filter_str.split(',') if x.strip()]
    strict_european_only = snakemake.params.get('strict_european_only', True)

    print(f"\nInputs:")
    print(f"  Trait enrichment: {trait_enrichment_file}")
    print(f"  GWAS catalog: {gwas_catalog_file}")
    print(f"  VCF: {vcf_file}")
    print(f"  Trait decomp dir: {trait_decomp_dir}")
    print(f"  SNP contributions dir: {snp_contributions_dir}")
    print(f"  Trait groups: {len(trait_groups)} defined")
    
    _debug_dir = os.path.join(os.path.dirname(output_summary), 'debug')
    os.makedirs(_debug_dir, exist_ok=True)
    
    with open(os.path.join(os.path.dirname(output_summary), 'debug', 'INPUTS_OK.txt'), 'w') as f:
        f.write(f"Inputs loaded successfully\n")
        f.write(f"Trait enrichment: {trait_enrichment_file}\n")
    
except Exception as e:
    print(f"\n!!! ERROR DURING INITIALIZATION !!!")
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    with open(os.path.join(os.path.dirname(output_summary), 'debug', 'INIT_ERROR.txt'), 'w') as f:
        f.write(f"Error: {str(e)}\n")
        traceback.print_exc(file=f)
    sys.exit(1)

# =========================================================================
# Load sample lists
# =========================================================================

print(f"\n{'='*80}")
print("LOADING SAMPLE LISTS")
print(f"{'='*80}")

with open(hg_samples_file) as f:
    hg_samples = [line.strip() for line in f if line.strip()]

with open(ef_samples_file) as f:
    ef_samples = [line.strip() for line in f if line.strip()]

print(f"  WHG samples: {len(hg_samples)}")
print(f"  EF samples: {len(ef_samples)}")

all_samples = hg_samples + ef_samples
all_samples_set = set(all_samples)
sample_to_pop = {s: 'WHG' for s in hg_samples}
sample_to_pop.update({s: 'EF' for s in ef_samples})

# =========================================================================
# CRITICAL: Get VCF sample order
# =========================================================================
# bcftools -s filters samples but returns them in VCF column order,
# NOT in the order specified by -s. If we zip(all_samples, genotypes)
# assuming our order matches VCF order, dosages get assigned to the
# wrong individuals — flipping population directions entirely.
#
# Fix: query VCF sample list once at startup, store the VCF order,
# and use that for all genotype parsing.
# =========================================================================

print(f"\n{'='*80}")
print("GETTING VCF SAMPLE ORDER")
print(f"{'='*80}")

vcf_sample_order_result = subprocess.run(
    ['bcftools', 'query', '-l', vcf_file],
    capture_output=True, text=True
)
if vcf_sample_order_result.returncode != 0:
    print(f"ERROR: Could not get VCF sample list: {vcf_sample_order_result.stderr}")
    sys.exit(1)

vcf_all_samples = [s.strip() for s in vcf_sample_order_result.stdout.strip().split('\n') if s.strip()]
# Keep only samples we care about, preserving VCF order
vcf_samples_ordered = [s for s in vcf_all_samples if s in all_samples_set]

missing_from_vcf = all_samples_set - set(vcf_samples_ordered)
if missing_from_vcf:
    print(f"  WARNING: {len(missing_from_vcf)} samples in sample files not found in VCF:")
    for s in sorted(missing_from_vcf):
        print(f"    {s}")

print(f"  VCF total samples: {len(vcf_all_samples)}")
print(f"  Our samples found in VCF: {len(vcf_samples_ordered)}")
print(f"  WHG: {sum(1 for s in vcf_samples_ordered if s in hg_samples)}, "
      f"EF: {sum(1 for s in vcf_samples_ordered if s in ef_samples)}")

# Build comma-separated sample string for bcftools -s (VCF-ordered)
vcf_samples_str = ','.join(vcf_samples_ordered)

# =========================================================================
# Load traits to analyze
# =========================================================================

print(f"\n{'='*80}")
print("LOADING TRAITS")
print(f"{'='*80}")

trait_enrichment_df = pd.read_csv(trait_enrichment_file, sep='\t')
print(f"  Total traits: {len(trait_enrichment_df)}")

if len(trait_enrichment_df) == 0:
    print("\nERROR: Trait enrichment file is empty!")
    print(f"File: {trait_enrichment_file}")
    # Create empty outputs and exit
    os.makedirs(os.path.dirname(output_summary), exist_ok=True)
    pd.DataFrame(columns=['trait', 'category']).to_csv(output_summary, sep='\t', index=False)
    pd.DataFrame(columns=['trait', 'snp_id']).to_csv(output_per_snp, sep='\t', index=False)
    sys.exit(0)

print(f"\nFirst few traits:")
for idx, row in trait_enrichment_df.head(3).iterrows():
    print(f"  - {row['trait']} ({row['category']})")

# =========================================================================
# Load GWAS catalog
# =========================================================================

print(f"\n{'='*80}")
print("LOADING GWAS CATALOG")
print(f"{'='*80}")

gwas_df = pd.read_csv(gwas_catalog_file, sep='\t', low_memory=False)
print(f"  Total associations: {len(gwas_df):,}")

# =====================================================================
# APPLY ANCESTRY FILTERING
# =====================================================================
# Filter to European ancestry studies only
# Two modes:
#   - Strict (default): Only pure European studies (excludes mixed populations)
#   - Permissive: Any study containing European samples
# =====================================================================

if ancestry_filter or strict_european_only:
    print(f"\n  Applying ancestry filter...")
    print(f"    Mode: {'STRICT (pure European only)' if strict_european_only else 'PERMISSIVE (any European)'}")
    
    initial_count = len(gwas_df)
    
    # Apply pure European filter
    ancestry_mask = apply_pure_european_filter(
        gwas_df['INITIAL SAMPLE SIZE'],
        strict_mode=strict_european_only
    )
    
    # Show examples of what gets excluded
    excluded = gwas_df[~ancestry_mask]
    if len(excluded) > 0:
        print(f"    Excluded {len(excluded):,} entries with non-European or mixed samples")
        
        # Show first 3 examples
        print(f"    Examples of excluded entries:")
        for idx, row in excluded.head(3).iterrows():
            sample_text = str(row['INITIAL SAMPLE SIZE'])[:70]
            trait = str(row['DISEASE/TRAIT'])[:40]
            print(f"      {trait:40s} | {sample_text}")
    
    # Apply filter
    gwas_df = gwas_df[ancestry_mask].copy()
    print(f"    After ancestry filter: {len(gwas_df):,} entries ({len(gwas_df)/initial_count*100:.1f}% retained)")

# Apply p-value filter
gwas_df['P-VALUE'] = pd.to_numeric(gwas_df['P-VALUE'], errors='coerce')
gwas_df = gwas_df[gwas_df['P-VALUE'].notna() & (gwas_df['P-VALUE'] < min_pvalue)].copy()
print(f"  After p-value filter: {len(gwas_df):,}")

# Extract positions
gwas_df = gwas_df[gwas_df['CHR_ID'].notna() & gwas_df['CHR_POS'].notna()].copy()
gwas_df['chr_num'] = pd.to_numeric(gwas_df['CHR_ID'], errors='coerce')
gwas_df = gwas_df[gwas_df['chr_num'].notna()].copy()
gwas_df['CHR_POS'] = pd.to_numeric(gwas_df['CHR_POS'], errors='coerce')
gwas_df = gwas_df[gwas_df['CHR_POS'].notna()].copy()
gwas_df['chrpos'] = gwas_df['chr_num'].astype(int).astype(str) + ':' + gwas_df['CHR_POS'].astype(int).astype(str)

print(f"  Final with positions: {len(gwas_df):,}")

# =========================================================================
# Helper functions
# =========================================================================

def parse_risk_allele(risk_allele_str):
    """Extract risk allele from STRONGEST SNP-RISK ALLELE column"""
    if pd.isna(risk_allele_str):
        return None
    
    # Format: rs123-A or rs123-?
    parts = str(risk_allele_str).split('-')
    if len(parts) < 2:
        return None
    
    allele = parts[-1].strip().upper()
    
    # Skip if ambiguous
    if allele == '?' or allele == 'NR' or len(allele) == 0:
        return None
    
    # Only keep single nucleotide alleles for now
    if len(allele) == 1 and allele in ['A', 'C', 'G', 'T']:
        return allele
    
    return None

def parse_effect_size(row):
    """
    Parse effect size from GWAS catalog and convert to beta scale
    
    CRITICAL: The "OR or BETA" column contains different effect types:
    - BETA (regression coefficients) for continuous traits
    - OR (odds ratios) for case-control traits
    
    We identify which type by examining the "95% CI (TEXT)" column:
    - If contains "increase" or "decrease" → BETA
    - If just numbers [X-Y] → OR
    
    For BETAs:
    - "unit increase" → positive beta
    - "unit decrease" → negative beta (flip sign!)
    
    For ORs:
    - Convert to log scale: beta = log(OR)
    
    Returns: (effect_type, beta_value, original_value, direction_text)
        effect_type: 'OR' or 'BETA' or None
        beta_value: converted to beta scale (log for OR, signed for BETA)
        original_value: raw value from GWAS catalog
        direction_text: 'increase', 'decrease', or 'OR'
    """
    # Get CI text to determine effect type
    ci_text = str(row.get('95% CI (TEXT)', '')).lower()
    
    # Get the value from OR or BETA column
    or_beta_val = row.get('OR or BETA')
    
    # Check if value exists and is parseable
    if pd.isna(or_beta_val) or or_beta_val == '' or or_beta_val == 'NR' or or_beta_val == 'NA':
        return (None, None, None, None)
    
    try:
        value = float(or_beta_val)
    except (ValueError, TypeError):
        return (None, None, None, None)
    
    # STEP 1: Determine if this is a BETA or OR
    # BETAs have "increase" or "decrease" in CI text
    # ORs have just numbers in brackets
    
    if 'increase' in ci_text or 'decrease' in ci_text:
        # ============================================================
        # This is a BETA (continuous trait)
        # ============================================================
        effect_type = 'BETA'
        
        # CRITICAL: Check direction
        # The value in "OR or BETA" column is ALWAYS positive
        # Direction is ONLY specified in CI text
        
        if 'decrease' in ci_text:
            # Risk allele DECREASES the trait
            # Flip sign to negative
            beta = -abs(value)
            direction = 'decrease'
        else:  # 'increase' in ci_text
            # Risk allele INCREASES the trait
            # Keep positive
            beta = abs(value)
            direction = 'increase'
        
        return ('BETA', beta, value, direction)
    
    else:
        # ============================================================
        # This is an OR (case-control trait)
        # ============================================================
        # Convert OR to log scale for PGS calculation
        # OR > 1 → positive beta (risk increases)
        # OR < 1 → negative beta (protective)
        
        if value > 0:
            beta = np.log(value)
            return ('OR', beta, value, 'OR')
        else:
            # Invalid OR (must be > 0)
            return (None, None, None, None)

def extract_genotypes_for_snp(vcf_file, chrom, pos, risk_allele, vcf_samples_ordered, vcf_samples_str):
    """
    Extract genotypes for a specific SNP from VCF.

    CRITICAL: vcf_samples_ordered must be in VCF column order (obtained via
    'bcftools query -l'), NOT in arbitrary user order. bcftools -s filters
    to those samples but outputs them in VCF column order regardless of the
    order specified in -s. Zipping genotypes with vcf_samples_ordered ensures
    each dosage is assigned to the correct individual.

    Returns: (dosages_dict, ref, alt)
    """
    cmd = [
        'bcftools', 'query',
        '-r', f"{chrom}:{pos}-{pos}",
        '-f', '%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n',
        '-s', vcf_samples_str,
        vcf_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return {}, None, None
    
    lines = result.stdout.strip().split('\n')
    if len(lines) == 0 or lines[0] == '':
        return {}, None, None
    
    # Parse first line (should only be one SNP)
    fields = lines[0].split('\t')
    if len(fields) < 4:
        return {}, None, None
    
    vcf_chrom = fields[0]
    vcf_pos = fields[1]
    ref = fields[2]
    alt = fields[3]
    
    # Determine which allele is the risk allele
    # risk_allele should match either REF or ALT
    if risk_allele == ref:
        risk_is_ref = True
    elif risk_allele == alt:
        risk_is_ref = False
    else:
        # Risk allele doesn't match REF or ALT - strand issue or multiallelic
        return {}, ref, alt
    
    # Parse genotypes
    genotypes = fields[4:]
    
    dosages = {}
    for sample, gt in zip(vcf_samples_ordered, genotypes):  # VCF-ordered zip is correct
        if gt == './.' or gt == '.|.':
            # Missing genotype
            dosages[sample] = np.nan
            continue
        
        # Parse genotype (e.g., "0/1" or "0|1")
        gt_alleles = gt.replace('|', '/').split('/')
        
        try:
            allele_counts = [int(a) for a in gt_alleles]
        except ValueError:
            # Invalid genotype
            dosages[sample] = np.nan
            continue
        
        # Count copies of risk allele
        if risk_is_ref:
            # Risk allele is REF (0)
            dosage = sum(1 for a in allele_counts if a == 0)
        else:
            # Risk allele is ALT (1)
            dosage = sum(1 for a in allele_counts if a == 1)
        
        dosages[sample] = dosage
    
    return dosages, ref, alt

# =========================================================================
# Process each trait
# =========================================================================

print(f"\n{'='*80}")
print("PROCESSING TRAITS")
print(f"{'='*80}\n")

all_trait_results = []
all_snp_details = []

# Track skip reasons
skip_reasons = {
    'no_gwas': 0,
    'no_effect_alleles': 0,
    'no_blocks_file': 0,
    'no_blocks_match': 0,
    'no_snps_after_ld': 0,
    'no_snps_extracted': 0,
    'success': 0
}

for trait_idx, trait_row in trait_enrichment_df.iterrows():
    trait_name = trait_row['trait']
    category_name = trait_row['category']
    trait_safe = trait_row['trait_safe']
    
    print(f"[{trait_idx+1}/{len(trait_enrichment_df)}] {trait_name}")
    
    # =====================================================================
    # Load GWAS SNPs for this trait
    # =====================================================================
    
    # Filter GWAS catalog for this exact trait
    gwas_df['trait_normalized'] = gwas_df['DISEASE/TRAIT'].str.lower().str.strip()
    trait_normalized = trait_name.lower().strip()
    trait_gwas = gwas_df[gwas_df['trait_normalized'] == trait_normalized].copy()
    
    if len(trait_gwas) == 0:
        print(f"  SKIP: No GWAS associations found")
        skip_reasons['no_gwas'] += 1
        continue
    
    # =====================================================================
    # Parse risk alleles and effect sizes
    # =====================================================================
    
    print(f"  Parsing effect sizes...")
    
    # Parse risk alleles from "STRONGEST SNP-RISK ALLELE" column
    trait_gwas['risk_allele'] = trait_gwas['STRONGEST SNP-RISK ALLELE'].apply(parse_risk_allele)
    
    # Parse effect sizes (returns 4-tuple: type, beta, original, direction)
    effect_results = trait_gwas.apply(parse_effect_size, axis=1)
    trait_gwas['effect_type'] = [r[0] for r in effect_results]
    trait_gwas['beta'] = [r[1] for r in effect_results]
    trait_gwas['effect_original'] = [r[2] for r in effect_results]
    trait_gwas['direction_text'] = [r[3] for r in effect_results]  # 'increase', 'decrease', or 'OR'
    
    # Filter to SNPs with clear risk allele AND effect size
    trait_gwas_clean = trait_gwas[
        trait_gwas['risk_allele'].notna() & 
        trait_gwas['beta'].notna()
    ].copy()
    
    print(f"  GWAS SNPs: {len(trait_gwas):,}")
    print(f"  With clear effect allele + size: {len(trait_gwas_clean):,}")
    
    if len(trait_gwas_clean) == 0:
        print(f"  SKIP: No SNPs with clear effect allele and size")
        skip_reasons['no_effect_alleles'] += 1
        continue
    
    # Report effect type breakdown
    effect_type_counts = trait_gwas_clean['effect_type'].value_counts()
    direction_counts = trait_gwas_clean['direction_text'].value_counts()
    print(f"  Effect types: {dict(effect_type_counts)}")
    print(f"  Directions: {dict(direction_counts)}")
    
    # =====================================================================
    # HANDLE DUPLICATE SNPs (same chr:pos)
    # =====================================================================
    # Same SNP can appear multiple times due to:
    # - Different studies (discovery, replication, meta-analysis)
    # - Different populations (though we filter to European)
    # - Updated analyses with larger samples
    #
    # Selection priority:
    # 1. MUST have clear risk allele (not "?")
    # 2. Among those, select lowest p-value (strongest evidence)
    #
    # This ensures we use the most reliable, well-powered estimate
    # =====================================================================
    
    # Create chr:pos identifier for duplicate detection
    trait_gwas_clean['chrpos'] = (
        trait_gwas_clean['chr_num'].astype(int).astype(str) + ':' + 
        trait_gwas_clean['CHR_POS'].astype(int).astype(str)
    )
    
    # Check for duplicates
    dup_mask = trait_gwas_clean['chrpos'].duplicated(keep=False)
    n_duplicate_positions = trait_gwas_clean.loc[dup_mask, 'chrpos'].nunique()
    n_duplicate_entries = dup_mask.sum()
    
    if n_duplicate_positions > 0:
        print(f"  Found {n_duplicate_entries} duplicate entries at {n_duplicate_positions} positions")
        
        # Show example of duplicate selection (first duplicate position)
        first_dup_pos = trait_gwas_clean[dup_mask]['chrpos'].iloc[0]
        first_dup_group = trait_gwas_clean[trait_gwas_clean['chrpos'] == first_dup_pos]
        if len(first_dup_group) > 1:
            print(f"    Example duplicate at {first_dup_pos}:")
            for idx, row in first_dup_group.iterrows():
                # Convert to string safely, handling NaN and float values
                snp_id = str(row['SNPS']) if pd.notna(row['SNPS']) else 'NA'
                risk = str(row['risk_allele']) if pd.notna(row['risk_allele']) else '?'
                pval = str(row['P-VALUE']) if pd.notna(row['P-VALUE']) else 'NA'
                sample_size = str(row['INITIAL SAMPLE SIZE'])[:40] if pd.notna(row['INITIAL SAMPLE SIZE']) else 'NA'
                print(f"      {snp_id:12s} | risk={risk:3s} | p={pval:10s} | {sample_size}")
        
        # For each chr:pos with duplicates, select best entry
        def select_best_duplicate(group):
            """
            Select best entry from duplicates at same chr:pos
            
            Priority hierarchy:
            1. MUST have clear risk allele (not "?" or missing)
            2. Among those with risk alleles, select lowest p-value
            3. If p-values tied, select largest sample size
            
            Rationale:
            - Without risk allele, we can't calculate PGS (dosage meaningless)
            - Lowest p-value = strongest statistical evidence
            - Largest N = most precise estimate (tiebreaker)
            """
            # Step 1: Filter to entries with clear risk alleles
            # (should already be done, but safety check)
            has_risk = group['risk_allele'].notna()
            group_with_risk = group[has_risk].copy()
            
            if len(group_with_risk) == 0:
                # No entries have risk alleles - skip entire group
                return None
            
            # Step 2: Among entries with risk alleles, select by p-value
            group_with_risk['p_value_numeric'] = pd.to_numeric(
                group_with_risk['P-VALUE'], errors='coerce'
            )
            
            # Step 3: Sort by p-value (lowest first), then by sample size (largest first)
            # Sample size parsing for tiebreaker
            def parse_sample_size(text):
                """Extract total sample size from text like '10,000 cases, 5,000 controls'"""
                import re
                if pd.isna(text):
                    return 0
                numbers = re.findall(r'[\d,]+', str(text))
                if not numbers:
                    return 0
                # Filter out empty strings after removing commas to avoid ValueError
                return sum(int(n.replace(',', '')) for n in numbers if n.replace(',', ''))
            
            group_with_risk['sample_size'] = group_with_risk['INITIAL SAMPLE SIZE'].apply(
                parse_sample_size
            )
            
            # Sort: lowest p-value first, then largest sample size
            group_sorted = group_with_risk.sort_values(
                ['p_value_numeric', 'sample_size'],
                ascending=[True, False]
            )
            
            # Return best entry
            return group_sorted.iloc[[0]]
        
        # Apply deduplication
        deduplicated = trait_gwas_clean.groupby('chrpos', group_keys=False).apply(
            select_best_duplicate
        )
        
        # Remove any None results (shouldn't happen, but safety)
        deduplicated = deduplicated.dropna(subset=['risk_allele'])
        
        n_after = len(deduplicated)
        n_removed = len(trait_gwas_clean) - n_after
        
        print(f"  After deduplication: {n_after} SNPs ({n_removed} duplicates removed)")
        
        trait_gwas_clean = deduplicated
    
    # =====================================================================
    # CHECK EFFECT TYPE CONSISTENCY
    # =====================================================================
    # If trait has mixed OR and BETA entries, use whichever is more common
    # This ensures we don't mix incompatible effect scales in same PGS
    # =====================================================================
    
    effect_type_counts = trait_gwas_clean['effect_type'].value_counts()
    n_or = effect_type_counts.get('OR', 0)
    n_beta = effect_type_counts.get('BETA', 0)
    total_effects = n_or + n_beta
    
    if total_effects > 0 and n_or > 0 and n_beta > 0:
        # Mixed types - use majority
        
        if n_or > n_beta:
            # More ORs - use only ORs
            trait_gwas_clean = trait_gwas_clean[trait_gwas_clean['effect_type'] == 'OR'].copy()
            or_pct = n_or / total_effects
            print(f"  Effect type filtering: Using ORs only ({n_or} SNPs, {or_pct:.1%} of total)")
            print(f"    Filtered out {n_beta} BETA entries")
            
        elif n_beta > n_or:
            # More BETAs - use only BETAs
            trait_gwas_clean = trait_gwas_clean[trait_gwas_clean['effect_type'] == 'BETA'].copy()
            beta_pct = n_beta / total_effects
            print(f"  Effect type filtering: Using BETAs only ({n_beta} SNPs, {beta_pct:.1%} of total)")
            print(f"    Filtered out {n_or} OR entries")
            
        else:
            # Exactly tied - prefer BETAs for continuous traits, ORs for diseases
            # Default to BETA since most traits are continuous
            trait_gwas_clean = trait_gwas_clean[trait_gwas_clean['effect_type'] == 'BETA'].copy()
            print(f"  Effect type filtering: Tied ({n_or} OR, {n_beta} BETA) - using BETAs by default")
            print(f"    Filtered out {n_or} OR entries")
    
    if len(trait_gwas_clean) == 0:
        print(f"  SKIP: No SNPs after effect type filtering")
        skip_reasons['no_effect_alleles'] += 1
        continue
    
    # =====================================================================
    # Load LD blocks for this trait
    # =====================================================================
    
    # Sanitize category name for directory path
    category_safe = sanitize_category_name(category_name)
    
    # Find trait blocks file for this category
    trait_blocks_file = os.path.join(
        trait_decomp_dir,
        category_safe,
        'trait_blocks.txt'
    )
    
    print(f"  Looking for: {trait_blocks_file}")
    
    if not os.path.exists(trait_blocks_file):
        print(f"  SKIP: Trait blocks file not found")
        skip_reasons['no_blocks_file'] += 1
        continue
    
    try:
        all_trait_blocks_df = pd.read_csv(trait_blocks_file, sep='\t')
        # Filter to this specific trait
        block_members_df = all_trait_blocks_df[all_trait_blocks_df['trait'] == trait_name].copy()
    except Exception as e:
        print(f"  SKIP: Failed to load trait blocks: {e}")
        skip_reasons['no_blocks_file'] += 1
        continue
    
    # Keep only actual GWAS SNPs (not proxy SNPs)
    if 'is_gwas_snp' in block_members_df.columns:
        gwas_block_members = block_members_df[block_members_df['is_gwas_snp'] == True].copy()
    else:
        gwas_block_members = block_members_df.copy()
    
    print(f"  LD blocks: {block_members_df['block_id'].nunique()}")
    print(f"  GWAS SNPs in blocks: {len(gwas_block_members):,}")
    
    # =====================================================================
    # Merge GWAS data with block assignments
    # =====================================================================
    
    # Merge on chr:pos
    gwas_with_blocks = trait_gwas_clean.merge(
        gwas_block_members[['snp_id', 'block_id']],
        left_on='chrpos',
        right_on='snp_id',
        how='inner'
    )
    
    print(f"  SNPs matched to blocks: {len(gwas_with_blocks):,}")
    
    if len(gwas_with_blocks) == 0:
        print(f"  SKIP: No SNPs matched to LD blocks")
        skip_reasons['no_blocks_match'] += 1
        continue
    
    # =====================================================================
    # Select one SNP per LD block (largest |beta|)
    # =====================================================================
    
    gwas_with_blocks['abs_beta'] = gwas_with_blocks['beta'].abs()
    
    # For each block, keep SNP with largest |beta|
    selected_snps = gwas_with_blocks.sort_values('abs_beta', ascending=False).groupby('block_id').first().reset_index()
    
    print(f"  After LD deduplication: {len(selected_snps):,} SNPs")
    
    if len(selected_snps) == 0:
        print(f"  SKIP: No SNPs after LD filtering")
        skip_reasons['no_snps_after_ld'] += 1
        continue
    
    # =====================================================================
    # EXTRACT GENOTYPES AND CALCULATE PGS
    # =====================================================================
    # For each selected SNP:
    # 1. Query VCF to get genotypes for all individuals (WHG + EF)
    # 2. Convert genotypes to dosage (0, 1, 2 copies of risk allele)
    # 3. Add weighted contribution to each individual's PGS
    #
    # PGS formula: sum(dosage × beta) across all SNPs
    # 
    # Example:
    #   Individual has genotypes: 0/1, 1/1, 0/0 at 3 SNPs
    #   Dosages (risk allele copies): 1, 2, 0
    #   Betas: +0.4, -0.2, +0.3
    #   PGS = (1 × 0.4) + (2 × -0.2) + (0 × 0.3) = 0.4 - 0.4 + 0 = 0
    # =====================================================================
    
    print(f"  Extracting genotypes from VCF...")
    
    # Initialize PGS arrays for each individual
    pgs_dict = {sample: 0.0 for sample in all_samples}
    missing_count_dict = {sample: 0 for sample in all_samples}
    
    snp_details = []
    
    for snp_idx, snp_row in selected_snps.iterrows():
        # Get SNP details
        chrom = str(int(snp_row['chr_num']))
        pos = int(snp_row['CHR_POS'])
        risk_allele = snp_row['risk_allele']
        beta = snp_row['beta']  # Already on correct scale (log for OR, signed for BETA)
        
        # ================================================================
        # EXTRACT GENOTYPES FROM VCF
        # ================================================================
        # Query the ancient DNA VCF file for this SNP position
        # Returns dosages (0, 1, 2) for the risk allele per individual
        # Missing genotypes are returned as NaN
        # ================================================================
        
        dosages, vcf_ref, vcf_alt = extract_genotypes_for_snp(vcf_file, chrom, pos, risk_allele, vcf_samples_ordered, vcf_samples_str)
        
        if len(dosages) == 0:
            # Failed to extract (strand mismatch, missing data, etc.)
            continue
        
        # ================================================================
        # CALCULATE ALLELE FREQUENCIES PER POPULATION
        # ================================================================
        # Split dosages by population
        # Calculate risk allele frequency (dosage/2 for diploid)
        # ================================================================
        
        whg_dosages = [dosages.get(s, np.nan) for s in hg_samples]
        ef_dosages = [dosages.get(s, np.nan) for s in ef_samples]
        
        # Remove missing genotypes for frequency calculation
        whg_dosages_clean = [d for d in whg_dosages if not np.isnan(d)]
        ef_dosages_clean = [d for d in ef_dosages if not np.isnan(d)]
        
        if len(whg_dosages_clean) == 0 or len(ef_dosages_clean) == 0:
            # No data for one or both populations
            continue
        
        # Calculate allele frequencies from non-missing individuals only
        # Frequency = mean(dosage) / 2 because diploid
        freq_whg = np.mean(whg_dosages_clean) / 2.0
        freq_ef = np.mean(ef_dosages_clean) / 2.0
        delta_freq = freq_ef - freq_whg
        
        # ================================================================
        # ADD TO POLYGENIC SCORE (PGS) FOR EACH INDIVIDUAL
        # ================================================================
        # For each individual: PGS += dosage × beta
        #
        # MISSING GENOTYPE HANDLING: mean imputation per population
        # Missing individuals receive their population's mean dosage for
        # this SNP. This is neutral — it adds exactly the average
        # contribution and keeps every individual's PGS comparable
        # regardless of missingness.
        #
        # Ancient DNA missingness is substantial and non-random (better
        # preserved samples cluster by population/time), so substituting 0
        # would systematically bias the population mean.
        # ================================================================
        
        # Population mean dosages from genotyped individuals (used for imputation)
        mean_dosage_whg_pgs = np.mean(whg_dosages_clean)
        mean_dosage_ef_pgs  = np.mean(ef_dosages_clean)
        
        for sample in hg_samples:
            d = dosages.get(sample, np.nan)
            if not np.isnan(d):
                pgs_dict[sample] += d * beta
            else:
                pgs_dict[sample] += mean_dosage_whg_pgs * beta  # impute with WHG mean
                missing_count_dict[sample] += 1
        
        for sample in ef_samples:
            d = dosages.get(sample, np.nan)
            if not np.isnan(d):
                pgs_dict[sample] += d * beta
            else:
                pgs_dict[sample] += mean_dosage_ef_pgs * beta   # impute with EF mean
                missing_count_dict[sample] += 1
        
        # Store SNP details
        snp_details.append({
            'trait': trait_name,
            'category': category_name,
            'snp_id': snp_row['chrpos'],
            'chr': chrom,
            'pos': pos,
            'vcf_ref': vcf_ref,
            'vcf_alt': vcf_alt,
            'risk_allele': risk_allele,
            'risk_allele_is_ref': (risk_allele == vcf_ref),
            'effect_type': snp_row['effect_type'],
            'effect_original': snp_row['effect_original'],
            'beta': beta,
            'block_id': snp_row['block_id'],
            'freq_WHG': freq_whg,
            'freq_EF': freq_ef,
            'delta_freq': delta_freq,
            'mean_dosage_WHG': mean_dosage_whg_pgs,
            'mean_dosage_EF': mean_dosage_ef_pgs,
            'n_WHG_genotyped': len(whg_dosages_clean),
            'n_EF_genotyped': len(ef_dosages_clean),
            'n_WHG_total': len(hg_samples),
            'n_EF_total': len(ef_samples)
        })
    
    all_snp_details.extend(snp_details)
    
    n_snps_used = len(snp_details)
    print(f"  SNPs successfully extracted: {n_snps_used}")
    
    if n_snps_used == 0:
        print(f"  SKIP: No SNPs extracted from VCF")
        skip_reasons['no_snps_extracted'] += 1
        continue
    
    # =====================================================================
    # CALCULATE PGS STATISTICS AND TEST POPULATION DIFFERENCE
    # =====================================================================
    # At this point, each individual has a PGS value:
    #   PGS = sum(dosage_i × beta_i) across all SNPs
    #
    # For continuous traits (BETA):
    #   - Higher PGS = genetically predisposed to higher trait value
    #   - Can be positive or negative depending on effect directions
    #
    # For disease traits (OR):  
    #   - Higher PGS = genetically higher disease risk
    #   - Usually positive (risk alleles increase risk)
    #
    # We compare the DISTRIBUTION of PGS values between populations
    # using a t-test to determine if there's a significant difference
    # =====================================================================
    
    skip_reasons['success'] += 1
    
    # Get PGS arrays for each population
    pgs_whg = [pgs_dict[s] for s in hg_samples]
    pgs_ef = [pgs_dict[s] for s in ef_samples]
    
    # Calculate summary statistics
    mean_pgs_whg = np.mean(pgs_whg)
    mean_pgs_ef = np.mean(pgs_ef)
    se_pgs_whg = np.std(pgs_whg, ddof=1) / np.sqrt(len(pgs_whg))
    se_pgs_ef = np.std(pgs_ef, ddof=1) / np.sqrt(len(pgs_ef))
    
    # Difference between populations
    delta_pgs = mean_pgs_ef - mean_pgs_whg
    
    # =====================================================================
    # STATISTICAL TEST: Independent samples t-test
    # =====================================================================
    # Null hypothesis: PGS distributions are the same
    # Alternative: PGS differs between WHG and EF
    #
    # The t-test compares the MEANS of two independent groups
    # accounting for within-group variance
    # =====================================================================
    
    t_stat, t_pval = stats.ttest_ind(pgs_ef, pgs_whg)
    
    # =====================================================================
    # EFFECT SIZE: Cohen's d
    # =====================================================================
    # Standardized difference between groups
    # d = (mean1 - mean2) / pooled_std
    #
    # Interpretation:
    #   |d| < 0.2  = small effect
    #   |d| < 0.5  = medium effect  
    #   |d| ≥ 0.8  = large effect
    # =====================================================================
    
    pooled_std = np.sqrt((np.var(pgs_whg, ddof=1) + np.var(pgs_ef, ddof=1)) / 2)
    cohens_d = delta_pgs / pooled_std if pooled_std > 0 else 0
    
    # =====================================================================
    # DETERMINE DIRECTION
    # =====================================================================
    # Which population has higher genetic predisposition for this trait?
    #
    # delta_PGS > 0: EF has higher PGS (higher trait value / risk)
    # delta_PGS < 0: WHG has higher PGS
    # =====================================================================
    
    if delta_pgs > 0:
        direction = "EF_higher"
    elif delta_pgs < 0:
        direction = "WHG_higher"
    else:
        direction = "No_difference"
    
    # Report missing data
    avg_missing_whg = np.mean([missing_count_dict[s] for s in hg_samples])
    avg_missing_ef = np.mean([missing_count_dict[s] for s in ef_samples])
    pct_missing = (avg_missing_whg + avg_missing_ef) / (2 * n_snps_used) * 100 if n_snps_used > 0 else 0
    
    print(f"  PGS WHG: {mean_pgs_whg:.4f} ± {se_pgs_whg:.4f}")
    print(f"  PGS EF:  {mean_pgs_ef:.4f} ± {se_pgs_ef:.4f}")
    print(f"  Delta: {delta_pgs:.4f}, p-value: {t_pval:.3e}, direction: {direction}")
    print(f"  Missing genotypes: {pct_missing:.1f}%")
    
    # =====================================================================
    # CREATE DETAILED SNP CONTRIBUTION RANKING
    # =====================================================================
    # Calculate contribution metrics and create per-trait file
    # =====================================================================
    
    if n_snps_used > 0 and len(snp_details) > 0:
        # Create DataFrame from snp_details (already has PGS-consistent mean dosages)
        snp_contrib_df = pd.DataFrame(snp_details)
        
        # delta_dosage using PGS-consistent means (sum/N_total, missing=0)
        # This ensures: sum(pgs_contribution) == delta_PGS  (sanity check below)
        snp_contrib_df['delta_dosage'] = snp_contrib_df['mean_dosage_EF'] - snp_contrib_df['mean_dosage_WHG']
        
        # Contribution metrics
        # pgs_contribution: exact contribution of this SNP to delta_PGS
        #   Positive → SNP pushes EF higher (agrees with overall direction if EF_higher)
        #   Negative → SNP pushes WHG higher (opposes overall direction if EF_higher)
        snp_contrib_df['pgs_contribution'] = snp_contrib_df['beta'] * snp_contrib_df['delta_dosage']
        snp_contrib_df['abs_pgs_contribution'] = snp_contrib_df['pgs_contribution'].abs()
        
        # impact_score: magnitude of effect × magnitude of freq difference (always ≥ 0)
        # High impact = large effect AND large frequency difference between populations
        snp_contrib_df['impact_score'] = snp_contrib_df['beta'].abs() * snp_contrib_df['delta_freq'].abs()
        
        # Rank by impact_score descending
        snp_contrib_df = snp_contrib_df.sort_values('impact_score', ascending=False).reset_index(drop=True)
        snp_contrib_df['contribution_rank'] = range(1, len(snp_contrib_df) + 1)
        
        # Sanity check: sum(pgs_contribution) should equal delta_PGS
        sum_contributions = snp_contrib_df['pgs_contribution'].sum()
        print(f"  Sanity check: sum(pgs_contribution)={sum_contributions:.4f}, delta_PGS={delta_pgs:.4f} "
              f"(diff={abs(sum_contributions - delta_pgs):.6f})")
        
        # Merge with GWAS catalog for annotations
        # CRITICAL: deduplicate gwas_for_merge on chrpos FIRST (keep lowest p-value)
        # so we get exactly one annotation row per SNP — no fan-out duplicates
        gwas_for_merge = gwas_df[gwas_df['DISEASE/TRAIT'] == trait_name].copy()
        gwas_for_merge['p_numeric'] = pd.to_numeric(gwas_for_merge['P-VALUE'], errors='coerce')
        gwas_for_merge = (gwas_for_merge
                          .sort_values('p_numeric')
                          .drop_duplicates(subset='chrpos', keep='first'))
        
        annotation_cols = [c for c in [
            'chrpos', 'SNPS', 'REPORTED GENE(S)', 'MAPPED_GENE', 'CONTEXT',
            'INTERGENIC', 'SNP_GENE_IDS', 'UPSTREAM_GENE_ID', 'DOWNSTREAM_GENE_ID',
            'UPSTREAM_GENE_DISTANCE', 'DOWNSTREAM_GENE_DISTANCE',
            'P-VALUE', 'PUBMEDID', 'FIRST AUTHOR', 'DATE', 'JOURNAL',
            'STUDY', 'INITIAL SAMPLE SIZE', 'RISK ALLELE FREQUENCY',
            '95% CI (TEXT)', 'PLATFORM [SNPS PASSING QC]'
        ] if c in gwas_for_merge.columns]
        
        snp_contrib_annotated = snp_contrib_df.merge(
            gwas_for_merge[annotation_cols],
            left_on='snp_id',
            right_on='chrpos',
            how='left'
        )
        
        # Select and order output columns
        output_cols = [
            'contribution_rank', 'trait', 'category',
            'SNPS', 'snp_id', 'chr', 'pos',
            'vcf_ref', 'vcf_alt', 'risk_allele', 'risk_allele_is_ref',
            'impact_score', 'pgs_contribution', 'abs_pgs_contribution',
            'effect_type', 'effect_original', 'beta', '95% CI (TEXT)',
            'freq_WHG', 'freq_EF', 'delta_freq',
            'mean_dosage_WHG', 'mean_dosage_EF', 'delta_dosage',
            'n_WHG_genotyped', 'n_WHG_total', 'n_EF_genotyped', 'n_EF_total',
            'block_id',
            'P-VALUE', 'RISK ALLELE FREQUENCY',
            'REPORTED GENE(S)', 'MAPPED_GENE', 'CONTEXT', 'INTERGENIC',
            'SNP_GENE_IDS', 'UPSTREAM_GENE_ID', 'DOWNSTREAM_GENE_ID',
            'UPSTREAM_GENE_DISTANCE', 'DOWNSTREAM_GENE_DISTANCE',
            'PUBMEDID', 'FIRST AUTHOR', 'DATE', 'JOURNAL', 'STUDY',
            'INITIAL SAMPLE SIZE', 'PLATFORM [SNPS PASSING QC]'
        ]
        output_cols_available = [c for c in output_cols if c in snp_contrib_annotated.columns]
        snp_contrib_final = snp_contrib_annotated[output_cols_available]
        
        # Write per-trait SNP contribution file
        trait_safe = sanitize_category_name(trait_name)
        category_safe = sanitize_category_name(category_name)
        trait_snp_dir = os.path.join(snp_contributions_dir, category_safe)
        os.makedirs(trait_snp_dir, exist_ok=True)
        trait_snp_file = os.path.join(trait_snp_dir, f"{trait_safe}_snp_contributions.txt")
        snp_contrib_final.to_csv(trait_snp_file, sep='\t', index=False)
        print(f"  ✓ SNP contributions saved: {trait_snp_file} ({len(snp_contrib_final)} SNPs)")
    
    # =====================================================================
    # STORE TRAIT-LEVEL RESULTS
    # =====================================================================
    
    # Store results
    all_trait_results.append({
        'trait': trait_name,
        'category': category_name,
        'n_snps_gwas': len(trait_gwas_clean),
        'n_snps_in_blocks': len(gwas_with_blocks),
        'n_snps_after_ld_filter': len(selected_snps),
        'n_snps_extracted': n_snps_used,
        'effect_type_primary': effect_type_counts.index[0] if len(effect_type_counts) > 0 else 'NA',
        'n_snps_OR': effect_type_counts.get('OR', 0),
        'n_snps_BETA': effect_type_counts.get('BETA', 0),
        'PGS_WHG_mean': mean_pgs_whg,
        'PGS_WHG_se': se_pgs_whg,
        'PGS_EF_mean': mean_pgs_ef,
        'PGS_EF_se': se_pgs_ef,
        'delta_PGS': delta_pgs,
        'ttest_pval': t_pval,
        'cohens_d': cohens_d,
        'direction': direction,
        'pct_missing_genotypes': pct_missing,
        'n_WHG_samples': len(hg_samples),
        'n_EF_samples': len(ef_samples)
    })
    
    print()

# =========================================================================
# TRAIT GROUPS: COMBINED PGS ANALYSES
# =========================================================================
# For each user-defined group, pool GWAS SNPs from all member traits,
# deduplicate, LD-deduplicate, and run a combined PGS analysis.
# Results are appended to all_trait_results and all_snp_details.
# =========================================================================

print(f"\n{'='*80}")
print("TRAIT GROUPS - COMBINED PGS ANALYSES")
print(f"{'='*80}")
print(f"  trait_groups type: {type(trait_groups)}")
print(f"  trait_groups value: {trait_groups}")
print(f"  Number of groups: {len(trait_groups) if trait_groups else 0}")

# Write debug to file regardless of stdout routing
_debug_path = os.path.join(os.path.dirname(output_summary), 'debug', 'TRAIT_GROUPS_DEBUG.txt')
os.makedirs(os.path.dirname(_debug_path), exist_ok=True)
with open(_debug_path, 'w') as _dbf:
    _dbf.write(f"trait_groups type: {type(trait_groups)}\n")
    _dbf.write(f"trait_groups value: {trait_groups}\n")
    _dbf.write(f"trait_groups len: {len(trait_groups) if trait_groups else 0}\n")
    _dbf.write(f"trait_groups is None: {trait_groups is None}\n")
    _dbf.write(f"bool(trait_groups): {bool(trait_groups)}\n")

# Ensure trait_normalized exists (set in main loop, but add safety fallback)
if 'trait_normalized' not in gwas_df.columns:
    gwas_df['trait_normalized'] = gwas_df['DISEASE/TRAIT'].str.lower().str.strip()

for group in (trait_groups or []):
    group_name     = group['name']
    group_category = group['category']
    group_traits   = group['traits']

    print(f"\n[GROUP] {group_name}")
    print(f"  Category: {group_category}")
    print(f"  Member traits: {group_traits}")

    # File-based progress tracking
    with open(_debug_path, 'a') as _dbf:
        _dbf.write(f"\nProcessing group: {group_name}\n")
        _dbf.write(f"  category: {group_category}\n")
        _dbf.write(f"  traits: {group_traits}\n")

    # ------------------------------------------------------------------
    # Step 1: Pool GWAS SNPs from all member traits
    # ------------------------------------------------------------------
    pool_frames = []
    for t in group_traits:
        t_norm = t.lower().strip()
        t_rows = gwas_df[gwas_df['trait_normalized'] == t_norm].copy()
        if len(t_rows) == 0:
            print(f"  WARNING: No GWAS rows found for '{t}'")
        else:
            print(f"  '{t}': {len(t_rows)} GWAS rows")
            pool_frames.append(t_rows)
    with open(_debug_path, 'a') as _dbf: _dbf.write(f"  GWAS rows for trait: {len(t_rows)}\n")

    if not pool_frames:
        print(f"  SKIP: No GWAS data for any member trait")
        continue

    pooled = pd.concat(pool_frames, ignore_index=True)
    with open(_debug_path, 'a') as _dbf: _dbf.write(f"  Step1 pooled: {len(pooled)}\n")
    print(f"  Pooled: {len(pooled)} rows before effect-size parsing")

    # ------------------------------------------------------------------
    # Step 2: Parse risk alleles and effect sizes (same as per-trait)
    # ------------------------------------------------------------------
    parsed = pooled.apply(lambda row: pd.Series({
        'risk_allele':    parse_risk_allele(row.get('STRONGEST SNP-RISK ALLELE', '')),
        'effect_type':    parse_effect_size(row)[0],
        'beta':           parse_effect_size(row)[1],
        'effect_original':parse_effect_size(row)[2],
    }), axis=1)
    pooled = pd.concat([pooled, parsed], axis=1)

    pool_clean = pooled[
        pooled['risk_allele'].notna() &
        pooled['effect_type'].notna() &
        pooled['beta'].notna()
    ].copy()
    print(f"  With clear effect allele + size: {len(pool_clean)}")

    if len(pool_clean) == 0:
        print(f"  SKIP: No SNPs with parsed effects")
        continue

    # Determine dominant effect type (OR vs BETA)
    et_counts = pool_clean['effect_type'].value_counts()
    dominant_et = et_counts.index[0]
    pool_clean = pool_clean[pool_clean['effect_type'] == dominant_et].copy()
    with open(_debug_path, 'a') as _dbf: _dbf.write(f"  Step2 pool_clean after et filter: {len(pool_clean)}, et={dominant_et}\n")
    print(f"  Using effect type: {dominant_et} ({len(pool_clean)} SNPs)")

    # ------------------------------------------------------------------
    # Step 3: Deduplicate on chrpos — keep lowest p-value per position
    # ------------------------------------------------------------------
    pool_clean['p_numeric'] = pd.to_numeric(pool_clean['P-VALUE'], errors='coerce')
    pool_clean = (pool_clean
                  .sort_values('p_numeric')
                  .drop_duplicates(subset='chrpos', keep='first')
                  .copy())
    print(f"  After chrpos deduplication: {len(pool_clean)} unique positions")

    # ------------------------------------------------------------------
    # Step 4: Load LD blocks for the group's category
    # ------------------------------------------------------------------
    category_safe_grp = sanitize_category_name(group_category)
    blocks_file = os.path.join(trait_decomp_dir, category_safe_grp, 'trait_blocks.txt')
    with open(_debug_path, 'a') as _dbf: _dbf.write(f"  Step4 blocks_file={blocks_file}, exists={os.path.exists(blocks_file)}\n")

    if not os.path.exists(blocks_file):
        print(f"  SKIP: Blocks file not found: {blocks_file}")
        continue

    try:
        all_blocks_df = pd.read_csv(blocks_file, sep='\t')
    except Exception as e:
        print(f"  SKIP: Failed to load blocks: {e}")
        continue

    # Pool block members across all member traits in this group
    grp_block_members = all_blocks_df[all_blocks_df['trait'].isin(group_traits)].copy()
    if 'is_gwas_snp' in grp_block_members.columns:
        grp_block_members = grp_block_members[grp_block_members['is_gwas_snp'] == True].copy()
    grp_block_members = grp_block_members.drop_duplicates(subset='snp_id').copy()
    with open(_debug_path, 'a') as _dbf: _dbf.write(f"  Step4 grp_block_members: {len(grp_block_members)}\n")
    print(f"  LD block SNPs (across member traits): {len(grp_block_members)}")

    # ------------------------------------------------------------------
    # Step 5: Merge GWAS pool with block assignments → LD-deduplicate
    # ------------------------------------------------------------------
    pool_with_blocks = pool_clean.merge(
        grp_block_members[['snp_id', 'block_id']],
        left_on='chrpos', right_on='snp_id', how='inner'
    )
    with open(_debug_path, 'a') as _dbf: _dbf.write(f"  Step5 matched to blocks: {len(pool_with_blocks)}\n")
    print(f"  SNPs matched to blocks: {len(pool_with_blocks)}")

    if len(pool_with_blocks) == 0:
        print(f"  SKIP: No pooled SNPs matched to LD blocks")
        continue

    pool_with_blocks['abs_beta'] = pool_with_blocks['beta'].abs()
    selected_grp = (pool_with_blocks
                    .sort_values('abs_beta', ascending=False)
                    .groupby('block_id').first()
                    .reset_index())
    print(f"  After LD deduplication: {len(selected_grp)} SNPs")

    # ------------------------------------------------------------------
    # Step 6: Extract genotypes and calculate combined PGS
    # ------------------------------------------------------------------
    pgs_dict_grp     = {s: 0.0 for s in all_samples}
    missing_cnt_grp  = {s: 0   for s in all_samples}
    snp_details_grp  = []

    print(f"  Extracting genotypes from VCF...")

    for _, snp_row in selected_grp.iterrows():
        chrom      = str(int(snp_row['chr_num']))
        pos        = int(snp_row['CHR_POS'])
        risk_allele = snp_row['risk_allele']
        beta        = snp_row['beta']

        dosages, vcf_ref, vcf_alt = extract_genotypes_for_snp(
            vcf_file, chrom, pos, risk_allele,
            vcf_samples_ordered, vcf_samples_str
        )
        if len(dosages) == 0:
            continue

        whg_d = [dosages.get(s, np.nan) for s in hg_samples]
        ef_d  = [dosages.get(s, np.nan) for s in ef_samples]
        whg_d_clean = [d for d in whg_d if not np.isnan(d)]
        ef_d_clean  = [d for d in ef_d  if not np.isnan(d)]

        if not whg_d_clean or not ef_d_clean:
            continue

        freq_whg   = np.mean(whg_d_clean) / 2.0
        freq_ef    = np.mean(ef_d_clean)  / 2.0
        delta_freq = freq_ef - freq_whg

        # Mean imputation per population (same as main loop)
        md_whg = np.mean(whg_d_clean)
        md_ef  = np.mean(ef_d_clean)

        for s in hg_samples:
            d = dosages.get(s, np.nan)
            pgs_dict_grp[s] += d * beta if not np.isnan(d) else md_whg * beta
            if np.isnan(d): missing_cnt_grp[s] += 1

        for s in ef_samples:
            d = dosages.get(s, np.nan)
            pgs_dict_grp[s] += d * beta if not np.isnan(d) else md_ef * beta
            if np.isnan(d): missing_cnt_grp[s] += 1

        snp_details_grp.append({
            'trait': group_name, 'category': group_category,
            'snp_id': snp_row['chrpos'], 'chr': chrom, 'pos': pos,
            'vcf_ref': vcf_ref, 'vcf_alt': vcf_alt,
            'risk_allele': risk_allele,
            'risk_allele_is_ref': (risk_allele == vcf_ref),
            'effect_type': snp_row['effect_type'],
            'effect_original': snp_row['effect_original'],
            'beta': beta, 'block_id': snp_row['block_id'],
            'freq_WHG': freq_whg, 'freq_EF': freq_ef, 'delta_freq': delta_freq,
            'mean_dosage_WHG': md_whg, 'mean_dosage_EF': md_ef,
            'n_WHG_genotyped': len(whg_d_clean), 'n_EF_genotyped': len(ef_d_clean),
            'n_WHG_total': len(hg_samples), 'n_EF_total': len(ef_samples),
        })

    n_grp_used = len(snp_details_grp)
    print(f"  SNPs successfully extracted: {n_grp_used}")

    if n_grp_used == 0:
        print(f"  SKIP: No SNPs extracted from VCF")
        continue

    # ------------------------------------------------------------------
    # Step 7: PGS statistics (same as main loop)
    # ------------------------------------------------------------------
    pgs_whg_grp = [pgs_dict_grp[s] for s in hg_samples]
    pgs_ef_grp  = [pgs_dict_grp[s] for s in ef_samples]

    mean_pgs_whg_grp = np.mean(pgs_whg_grp)
    mean_pgs_ef_grp  = np.mean(pgs_ef_grp)
    se_pgs_whg_grp   = np.std(pgs_whg_grp, ddof=1) / np.sqrt(len(pgs_whg_grp))
    se_pgs_ef_grp    = np.std(pgs_ef_grp,  ddof=1) / np.sqrt(len(pgs_ef_grp))
    delta_pgs_grp    = mean_pgs_ef_grp - mean_pgs_whg_grp

    t_stat_grp, t_pval_grp = stats.ttest_ind(pgs_ef_grp, pgs_whg_grp)
    pooled_std_grp   = np.sqrt((np.var(pgs_whg_grp, ddof=1) + np.var(pgs_ef_grp, ddof=1)) / 2)
    cohens_d_grp     = delta_pgs_grp / pooled_std_grp if pooled_std_grp > 0 else 0
    direction_grp    = "EF_higher" if delta_pgs_grp > 0 else ("WHG_higher" if delta_pgs_grp < 0 else "No_difference")

    avg_miss_whg = np.mean([missing_cnt_grp[s] for s in hg_samples])
    avg_miss_ef  = np.mean([missing_cnt_grp[s] for s in ef_samples])
    pct_missing_grp = (avg_miss_whg + avg_miss_ef) / (2 * n_grp_used) * 100

    print(f"  PGS WHG: {mean_pgs_whg_grp:.4f} ± {se_pgs_whg_grp:.4f}")
    print(f"  PGS EF:  {mean_pgs_ef_grp:.4f} ± {se_pgs_ef_grp:.4f}")
    print(f"  Delta: {delta_pgs_grp:.4f}, p-value: {t_pval_grp:.3e}, direction: {direction_grp}")
    print(f"  Missing genotypes: {pct_missing_grp:.1f}%")

    # ------------------------------------------------------------------
    # Step 8: SNP contribution file (same as main loop)
    # ------------------------------------------------------------------
    contrib_df = pd.DataFrame(snp_details_grp)
    contrib_df['delta_dosage']       = contrib_df['mean_dosage_EF'] - contrib_df['mean_dosage_WHG']
    contrib_df['pgs_contribution']   = contrib_df['beta'] * contrib_df['delta_dosage']
    contrib_df['abs_pgs_contribution'] = contrib_df['pgs_contribution'].abs()
    contrib_df['impact_score']       = contrib_df['beta'].abs() * contrib_df['delta_freq'].abs()
    contrib_df = contrib_df.sort_values('impact_score', ascending=False).reset_index(drop=True)
    contrib_df['contribution_rank']  = range(1, len(contrib_df) + 1)

    sum_contributions_grp = contrib_df['pgs_contribution'].sum()
    print(f"  Sanity: sum(pgs_contribution)={sum_contributions_grp:.4f}, delta_PGS={delta_pgs_grp:.4f} "
          f"(diff={abs(sum_contributions_grp - delta_pgs_grp):.6f})")

    # Annotate from GWAS catalog (best entry per chrpos across all member traits)
    gwas_grp_annot = gwas_df[gwas_df['DISEASE/TRAIT'].isin(group_traits)].copy()
    gwas_grp_annot['p_numeric'] = pd.to_numeric(gwas_grp_annot['P-VALUE'], errors='coerce')
    gwas_grp_annot = (gwas_grp_annot
                      .sort_values('p_numeric')
                      .drop_duplicates(subset='chrpos', keep='first'))

    annot_cols = [c for c in [
        'chrpos', 'SNPS', 'REPORTED GENE(S)', 'MAPPED_GENE', 'CONTEXT',
        'INTERGENIC', 'SNP_GENE_IDS', 'UPSTREAM_GENE_ID', 'DOWNSTREAM_GENE_ID',
        'UPSTREAM_GENE_DISTANCE', 'DOWNSTREAM_GENE_DISTANCE',
        'P-VALUE', 'PUBMEDID', 'FIRST AUTHOR', 'DATE', 'JOURNAL',
        'STUDY', 'INITIAL SAMPLE SIZE', 'RISK ALLELE FREQUENCY',
        '95% CI (TEXT)', 'PLATFORM [SNPS PASSING QC]'
    ] if c in gwas_grp_annot.columns]

    contrib_annot = contrib_df.merge(
        gwas_grp_annot[annot_cols],
        left_on='snp_id', right_on='chrpos', how='left'
    )

    out_cols = [
        'contribution_rank', 'trait', 'category',
        'SNPS', 'snp_id', 'chr', 'pos',
        'vcf_ref', 'vcf_alt', 'risk_allele', 'risk_allele_is_ref',
        'impact_score', 'pgs_contribution', 'abs_pgs_contribution',
        'effect_type', 'effect_original', 'beta', '95% CI (TEXT)',
        'freq_WHG', 'freq_EF', 'delta_freq',
        'mean_dosage_WHG', 'mean_dosage_EF', 'delta_dosage',
        'n_WHG_genotyped', 'n_WHG_total', 'n_EF_genotyped', 'n_EF_total',
        'block_id', 'P-VALUE', 'RISK ALLELE FREQUENCY',
        'REPORTED GENE(S)', 'MAPPED_GENE', 'CONTEXT', 'INTERGENIC',
        'SNP_GENE_IDS', 'UPSTREAM_GENE_ID', 'DOWNSTREAM_GENE_ID',
        'UPSTREAM_GENE_DISTANCE', 'DOWNSTREAM_GENE_DISTANCE',
        'PUBMEDID', 'FIRST AUTHOR', 'DATE', 'JOURNAL', 'STUDY',
        'INITIAL SAMPLE SIZE', 'PLATFORM [SNPS PASSING QC]'
    ]
    out_cols_avail = [c for c in out_cols if c in contrib_annot.columns]
    contrib_final = contrib_annot[out_cols_avail]

    grp_snp_dir = os.path.join(snp_contributions_dir, category_safe_grp)
    os.makedirs(grp_snp_dir, exist_ok=True)
    grp_snp_file = os.path.join(grp_snp_dir, f"{group_name}_snp_contributions.txt")
    contrib_final.to_csv(grp_snp_file, sep='\t', index=False)
    print(f"  ✓ SNP contributions saved: {grp_snp_file} ({len(contrib_final)} SNPs)")

    # ------------------------------------------------------------------
    # Step 9: Append to global results
    # ------------------------------------------------------------------
    et_counts_grp = pd.Series([s['effect_type'] for s in snp_details_grp]).value_counts()
    all_trait_results.append({
        'trait': group_name, 'category': group_category,
        'n_snps_gwas': len(pool_clean),
        'n_snps_in_blocks': len(pool_with_blocks),
        'n_snps_after_ld_filter': len(selected_grp),
        'n_snps_extracted': n_grp_used,
        'effect_type_primary': et_counts_grp.index[0] if len(et_counts_grp) > 0 else 'NA',
        'n_snps_OR':   et_counts_grp.get('OR', 0),
        'n_snps_BETA': et_counts_grp.get('BETA', 0),
        'PGS_WHG_mean': mean_pgs_whg_grp, 'PGS_WHG_se': se_pgs_whg_grp,
        'PGS_EF_mean':  mean_pgs_ef_grp,  'PGS_EF_se':  se_pgs_ef_grp,
        'delta_PGS': delta_pgs_grp, 'ttest_pval': t_pval_grp,
        'cohens_d': cohens_d_grp, 'direction': direction_grp,
        'pct_missing_genotypes': pct_missing_grp,
        'n_WHG_samples': len(hg_samples), 'n_EF_samples': len(ef_samples)
    })
    all_snp_details.extend(snp_details_grp)
    print(f"  ✓ Appended to summary results")

# =========================================================================
# Processing Summary
# =========================================================================

print(f"{'='*80}")
print("PROCESSING SUMMARY")
print(f"{'='*80}\n")

print(f"Total traits loaded: {len(trait_enrichment_df)}")
print(f"\nResults:")
print(f"  Successfully analyzed: {skip_reasons['success']}")
print(f"\nSkip reasons:")
print(f"  No GWAS associations: {skip_reasons['no_gwas']}")
print(f"  No SNPs with effect alleles: {skip_reasons['no_effect_alleles']}")
print(f"  Block members file missing: {skip_reasons['no_blocks_file']}")
print(f"  SNPs not in LD blocks: {skip_reasons['no_blocks_match']}")
print(f"  No SNPs after LD filter: {skip_reasons['no_snps_after_ld']}")
print(f"  No SNPs extracted from VCF: {skip_reasons['no_snps_extracted']}")

# =========================================================================
# Save results
# =========================================================================

print(f"{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

os.makedirs(os.path.dirname(output_summary), exist_ok=True)

# Define output columns
summary_columns = [
    'trait', 'category', 'n_snps_gwas', 'n_snps_in_blocks', 'n_snps_after_ld_filter',
    'n_snps_extracted', 'effect_type_primary', 'n_snps_OR', 'n_snps_BETA',
    'PGS_WHG_mean', 'PGS_WHG_se', 'PGS_EF_mean', 'PGS_EF_se', 'delta_PGS',
    'ttest_pval', 'cohens_d', 'direction', 'pct_missing_genotypes',
    'n_WHG_samples', 'n_EF_samples'
]

snp_columns = [
    'trait', 'category', 'snp_id', 'chr', 'pos',
    'vcf_ref', 'vcf_alt', 'risk_allele', 'risk_allele_is_ref',
    'effect_type', 'effect_original', 'beta', 'block_id',
    'freq_WHG', 'freq_EF', 'delta_freq',
    'mean_dosage_WHG', 'mean_dosage_EF',
    'n_WHG_genotyped', 'n_WHG_total', 'n_EF_genotyped', 'n_EF_total'
]

# Save summary
if len(all_trait_results) > 0:
    summary_df = pd.DataFrame(all_trait_results)
    summary_df = summary_df.sort_values('ttest_pval')
    summary_df.to_csv(output_summary, sep='\t', index=False)
    
    print(f"\n✓ Trait directionality summary: {output_summary}")
    print(f"  Traits analyzed: {len(summary_df)}")
    print(f"  Significant (p < 0.05): {(summary_df['ttest_pval'] < 0.05).sum()}")
    print(f"  EF higher: {(summary_df['direction'] == 'EF_higher').sum()}")
    print(f"  WHG higher: {(summary_df['direction'] == 'WHG_higher').sum()}")
else:
    print("\nWARNING: No traits analyzed successfully")
    print("Creating empty output file with headers...")
    pd.DataFrame(columns=summary_columns).to_csv(output_summary, sep='\t', index=False)

# Save SNP details
if len(all_snp_details) > 0:
    snp_df = pd.DataFrame(all_snp_details)
    snp_df.to_csv(output_per_snp, sep='\t', index=False)
    print(f"✓ SNP details: {output_per_snp}")
    print(f"  Total SNPs: {len(snp_df):,}")
else:
    print("\nWARNING: No SNPs extracted")
    print("Creating empty SNP file with headers...")
    pd.DataFrame(columns=snp_columns).to_csv(output_per_snp, sep='\t', index=False)

print(f"\n{'='*80}")
print("SCRIPT COMPLETED SUCCESSFULLY")
print(f"{'='*80}")
print(f"Output files created:")
print(f"  1. {output_summary}")
print(f"  2. {output_per_snp}")
print(f"  3. SNP contributions: {snp_contributions_dir}/")
sys.exit(0)