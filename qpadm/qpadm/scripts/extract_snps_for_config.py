"""
extract_snp_config.py
=====================
Reads the top N SNPs from each GWAS trait SNP contribution file and
outputs the snps: config block for config.yaml, ready to paste in.

Run from the project folder (helper, not part of the workflow; its output was pasted into
config/config_analysis.yaml):
  python scripts/extract_snps_for_config.py --top 10
  python scripts/extract_snps_for_config.py /path/to/other/snp_contributions/ --top 10   # override default
"""

import os
import sys
import argparse
import pandas as pd

DEFAULT_SNP_DIR = "../gwas_enrichment/results/v8/directionality/snp_contributions/"

parser = argparse.ArgumentParser()
parser.add_argument("snp_dir", nargs="?", default=DEFAULT_SNP_DIR,
                     help=f"Path to snp_contributions root directory (default: {DEFAULT_SNP_DIR})")
parser.add_argument("--top", type=int, default=10, help="Number of top SNPs per trait")
args = parser.parse_args()

print(f"# Using snp_dir: {args.snp_dir}", file=sys.stderr)

SNP_DIR = args.snp_dir
TOP_N   = args.top

# Mapping: UKB field_id -> (label, subdir, filename)
MAPPINGS = [
    # Fatty acids
    ("23442", "Total fatty acids",
     "Lipids_-_Fatty_Acids", "Omega-3_fatty_acids_to_total_fatty_acids_percentage_snp_contributions.txt"),
    ("23444", "Omega-3",
     "Lipids_-_Fatty_Acids", "Omega-3_fatty_acid_levels_snp_contributions.txt"),
    ("23445", "Omega-6",
     "Lipids_-_Fatty_Acids", "Total_omega-6_fatty_acid_levels_snp_contributions.txt"),
    ("23448", "Saturated FA",
     "Lipids_-_Fatty_Acids", "Saturated_fatty_acids_levels_snp_contributions.txt"),
    ("23450", "DHA",
     "Lipids_-_Fatty_Acids", "Docosahexaenoic_acid_levels_snp_contributions.txt"),
    ("23451", "Omega-3 / Total FA %",
     "Lipids_-_Fatty_Acids", "Omega-3_fatty_acids_to_total_fatty_acids_percentage_snp_contributions.txt"),
    ("23452", "Omega-6 / Total FA %",
     "Lipids_-_Fatty_Acids", "Omega-6_fatty_acids_to_total_fatty_acids_percentage_snp_contributions.txt"),
    ("23457", "DHA / Total FA %",
     "Lipids_-_Fatty_Acids", "Docosahexaenoic_acid_to_total_fatty_acids_percentage_snp_contributions.txt"),
    ("23459", "Omega-6 / Omega-3",
     "Lipids_-_Fatty_Acids", "Omega-6_fatty_acids_to_omega-3_fatty_acids_ratio_snp_contributions.txt"),
    # Anthropometry
    ("21001", "BMI",
     "Anthropometry_-_Weight_and_BMI", "Body_mass_index_UKB_data_field_21001_snp_contributions.txt"),
    ("23099", "Body fat %",
     "Anthropometry_-_Body_Composition_and_Fat_Distribution", "Body_fat_percentage_UKB_data_field_23099_snp_contributions.txt"),
    ("waist_hip_ratio", "Waist-hip ratio",
     "Anthropometry_-_Body_Composition_and_Fat_Distribution", "Waist-hip_ratio_snp_contributions.txt"),
    ("49", "Hip circumference",
     "Anthropometry_-_Body_Composition_and_Fat_Distribution", "Hip_circumference_UKB_data_field_49_snp_contributions.txt"),
    # IBD
    ("131626", "Crohn's disease",
     "Gastrointestinal_-_Inflammatory_Bowel_Disease", "Crohns_disease_snp_contributions.txt"),
    ("131628", "Ulcerative colitis",
     "Gastrointestinal_-_Inflammatory_Bowel_Disease", "Ulcerative_colitis_snp_contributions.txt"),
    ("IBD", "IBD combined",
     "Gastrointestinal_-_Inflammatory_Bowel_Disease", "Inflammatory_bowel_disease_snp_contributions.txt"),
    # Neuropsychiatric
    ("130874", "Schizophrenia",
     "Neuropsychiatric_-_Psychiatric_Disorders", "Autism_spectrum_disorder_or_schizophrenia_snp_contributions.txt"),
    ("130892", "Bipolar disorder",
     "Neuropsychiatric_-_Psychiatric_Disorders", "Bipolar_disorder_snp_contributions.txt"),
    ("130896", "Recurrent depressive disorder",
     "Neuropsychiatric_-_Psychiatric_Disorders", "Bipolar_disorder_or_major_depressive_disorder_snp_contributions.txt"),
    ("130904", "Phobic anxiety disorder",
     "Neuropsychiatric_-_Psychiatric_Disorders", "Anxiety_disorder_snp_contributions.txt"),
    ("130906", "Other anxiety disorder",
     "Neuropsychiatric_-_Psychiatric_Disorders", "Anxiety_disorder_snp_contributions.txt"),
    ("130910", "Reaction to severe stress",
     "Neuropsychiatric_-_Psychiatric_Disorders", "Post-traumatic_stress_disorder_snp_contributions.txt"),
    # Hearing
    ("131258", "Conductive/sensorineural hearing loss",
     "Sensory_-_Hearing", "Age-related_hearing_impairment_snp_contributions.txt"),
    ("131260", "Other hearing loss",
     "Sensory_-_Hearing", "Age-related_hearing_impairment_snp_contributions.txt"),
    # Sleep & behavioural
    ("1180", "Chronotype",
     "Neuropsychiatric_-_Neurological_Sleep_and_Behavior", "Chronotype_snp_contributions.txt"),
    ("1200", "Insomnia",
     "Neuropsychiatric_-_Neurological_Sleep_and_Behavior",
     "Insomnia_symptoms_never-rarely_vs._usually_snp_contributions.txt"),
    ("1980", "Worrier / anxious feelings",
     # TENTATIVE — confirm against Worry_snp_contributions.txt before running.
     # Both files plausibly derive from UKB field 1980; disambiguate via
     # study/PMID metadata in the file header before trusting this mapping.
     "Neuropsychiatric_-_Neurological_Sleep_and_Behavior", "Feeling_worry_snp_contributions.txt"),
]

print("# ─── SNP conditioning — top SNPs per trait ──────────────────────────────────")
print("# Generated by extract_snp_config.py")
print("# Paste these snps: blocks under the matching field_id entries in traits:")
print()

for fid, label, subdir, fname in MAPPINGS:
    fpath = os.path.join(SNP_DIR, subdir, fname)
    if not os.path.exists(fpath):
        # Try without subdir (flat structure)
        fpath = os.path.join(SNP_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  # WARNING: file not found for {fid} ({label}): {fname}")
        print()
        continue

    try:
        df = pd.read_csv(fpath, sep='\t')
        df = df.sort_values('contribution_rank').head(TOP_N)
    except Exception as e:
        print(f"  # ERROR reading {fname}: {e}")
        print()
        continue

    print(f"  # field_id: {fid}  ({label})")
    print(f"  # Source: {fname}")
    print(f"    snps:")
    for _, row in df.iterrows():
        chrom  = int(row['chr'])
        pos    = int(row['pos'])
        ref    = str(row['vcf_ref']).strip()
        alt    = str(row['vcf_alt']).strip()
        snp_id = str(row['snp_id']).strip()   # this is "chr:pos" format
        rsid   = str(row.get('SNPS', '')).strip()  # rsID column
        label  = rsid if rsid and rsid not in ('nan', '') else snp_id
        gene   = str(row.get('MAPPED_GENE', '')).strip()
        gene_s = f" # {gene}" if gene and gene not in ('nan', '') else ''
        print(f"      - {{chrom: {chrom}, pos: {pos}, ref: {ref}, alt: {alt}, label: \"{label}\"}}{gene_s}")
    print()