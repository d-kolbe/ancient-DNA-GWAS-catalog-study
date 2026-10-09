# =============================================================================
# UKB global ancestry (qpAdm) x phenotype analysis
# =============================================================================
# Associates the qpAdm ancestry proportions of UK Biobank individuals (output of
# Snakefile_qpadm.py) with phenotypes: linear / logistic regression per ancestry
# component, SNP-conditioned analyses, parental lifespan (survival) analyses,
# and forest plots per trait category.
#
# Run from the project folder with:
#   snakemake -s Snakefile_analysis.py --configfile config/config_analysis.local.yaml \\
#             --profile ../profiles/slurm
#
# Config: config/config_analysis.yaml (defaults) + optional *.local.yaml with paths.
# =============================================================================

import os
import re
import json
import pandas as pd
from pathlib import Path

configfile: "config/config_analysis.yaml"


# ─── Constants from config ────────────────────────────────────────────────────

RESULTS_DIR = config["results_dir"]

# Conda environment for all rules. Default: envs/analysis.yaml. To use an existing
# environment, set analysis_conda_env to its directory path.
ANALYSIS_ENV = config.get("analysis_conda_env", "envs/analysis.yaml")


# ─── Build FieldID → source lookup at startup ─────────────────────────────────

def build_trait_lookup(file_groups: dict) -> dict:
    """
    Parse every chars file and map FieldID (int) → metadata dict.
    Supports TSV (standard UKB) and JSON ({FieldID_str: field_name}) chars formats.

    Duplicate FieldIDs:
      - Same data file → silently keep first.
      - Different data files → one summary warning per group-pair, last-writer wins.
    """
    lookup = {}
    cross_file_conflicts: dict = {}

    for group, paths in file_groups.items():
        chars_path = paths.get("chars")
        if not chars_path:
            continue
        if not Path(chars_path).exists():
            print(f"[WARNING] chars file not found for group '{group}': {chars_path}")
            continue

        ext  = Path(chars_path).suffix.lower()
        rows = []

        if ext == ".json":
            import json as _json
            with open(chars_path) as fh:
                raw = _json.load(fh)
            if isinstance(raw, dict):
                for fid_str, name in raw.items():
                    try:
                        rows.append((int(fid_str), str(name), "Continuous"))
                    except ValueError:
                        continue
            elif isinstance(raw, list):
                for entry in raw:
                    try:
                        rows.append((int(entry["FieldID"]),
                                     str(entry.get("Field", "")),
                                     str(entry.get("ValueType", "Continuous"))))
                    except (KeyError, ValueError):
                        continue
        else:
            chars = pd.read_csv(chars_path, sep="\t", comment="#")
            for _, row in chars.iterrows():
                try:
                    rows.append((int(row["FieldID"]), str(row["Field"]), str(row["ValueType"])))
                except (KeyError, ValueError):
                    continue

        for fid, field_name, value_type in rows:
            if fid in lookup:
                prev = lookup[fid]
                if prev["data_file"] == paths["data"]:
                    continue
                cross_file_conflicts.setdefault((prev["group"], group), []).append(fid)
            lookup[fid] = {
                "group":       group,
                "data_file":   paths["data"],
                "chars_file":  chars_path,
                "cbkeys_file": paths.get("cbkeys"),
                "value_type":  value_type,
                "field_name":  field_name,
            }

    for (g1, g2), fids in cross_file_conflicts.items():
        preview = fids[:5]
        dots    = "..." if len(fids) > 5 else ""
        print(f"[WARNING] {len(fids)} FieldID(s) in both '{g1}' and '{g2}' "
              f"with different data files — using '{g2}'. "
              f"First few: {preview}{dots}")

    return lookup


TRAIT_LOOKUP = build_trait_lookup(config["phenotype_files"])


# ─── Validate + enrich requested traits ───────────────────────────────────────

TRAITS = []
for entry in config["traits"]:
    fid        = entry["field_id"]
    trait_type = entry.get("type", "standard")

    if trait_type == "standard":
        # Normal UKB field — must exist in the chars lookup
        if fid not in TRAIT_LOOKUP:
            raise ValueError(
                f"FieldID {fid} not found in any chars file. "
                "Check your phenotype_files paths in config.yaml."
            )
        merged_entry = {**TRAIT_LOOKUP[fid], **entry}
        if "value_type" in entry:
            merged_entry["value_type"] = entry["value_type"]
        TRAITS.append(merged_entry)

    elif trait_type == "combined_cases":
        # Synthetic case/control: cases from ANY component, controls from NONE.
        # Data file inferred from the first component field.
        components = entry["components"]
        for c in components:
            if c not in TRAIT_LOOKUP:
                raise ValueError(
                    f"combined_cases component FieldID {c} (in trait '{fid}') "
                    "not found in any chars file."
                )
        TRAITS.append({
            "field_id":    fid,
            "field_name":  entry.get("label", str(fid)),
            "value_type":  "Date",
            "group":       "combined",
            "data_file":   TRAIT_LOOKUP[components[0]]["data_file"],
            "cbkeys_file": None,
            "type":        "combined_cases",
            "components":  components,
        })

    elif trait_type == "computed":
        # Derived continuous variable from an arbitrary formula over named fields.
        # Config must specify:
        #   source_group: which phenotype file group the data lives in
        #   fields:       {variable_name: FieldID} — all inputs to the formula
        #   formula:      a pandas-eval-compatible expression using those variable names
        #                 e.g. "waist / hip" or "hip / (height / 100 * weight)"
        source_group = entry["source_group"]
        if source_group not in config["phenotype_files"]:
            raise ValueError(
                f"computed trait '{fid}': source_group '{source_group}' "
                "not in phenotype_files."
            )
        fields = entry["fields"]   # {name: FieldID}
        for var_name, comp_fid in fields.items():
            if comp_fid not in TRAIT_LOOKUP:
                raise ValueError(
                    f"computed trait '{fid}': field '{var_name}' (FieldID {comp_fid}) "
                    "not found in any chars file."
                )
        TRAITS.append({
            "field_id":     fid,
            "field_name":   entry.get("label", str(fid)),
            "value_type":   "Continuous",
            "group":        source_group,
            "data_file":    config["phenotype_files"][source_group]["data"],
            "cbkeys_file":  None,
            "type":         "computed",
            "fields":       fields,
            "formula":      entry["formula"],
            "source_group": source_group,
        })

    elif trait_type == "survival":
        # Survival traits are handled by separate rules (extract_survival,
        # analyze_survival, plot_survival) — not the standard pipeline.
        # Register in SURVIVAL_MAP only; skip TRAIT_LOOKUP entirely.
        TRAITS.append({
            "field_id":      fid,
            "field_name":    entry.get("label", str(fid)),
            "label":         entry.get("label", str(fid)),
            "value_type":    "survival",
            "group":         "survival",
            "data_file":     None,
            "cbkeys_file":   None,
            "type":          "survival",
            "survival_type": entry.get("survival_type", "father"),
        })

    else:
        raise ValueError(f"Unknown trait type '{trait_type}' for field_id '{fid}'.")

# Standard traits (exclude survival — those use separate rules)
STANDARD_TRAITS = [t for t in TRAITS if t.get("type") != "survival"]
FIELD_IDS = [str(t["field_id"]) for t in STANDARD_TRAITS]
TRAIT_MAP  = {str(t["field_id"]): t for t in STANDARD_TRAITS}


# ─── SNP conditioning setup ──────────────────────────────────────────────────
# Traits with SNPs listed in config get an extra analysis with those SNPs
# added as covariates. Outputs go to {fid}/age_sex_lifestyle_pcs_snp_cond/.

VCF_PATTERN = config["vcf_pattern"]     # per-chromosome UKB VCFs, {chrom} = chromosome number

# Map fid -> snp list for traits that have SNPs defined.
# Read directly from raw config entries because the trait parsing loop
# does not forward the 'snps' key into the processed TRAITS dicts.
TRAITS_WITH_SNPS = {
    str(entry["field_id"]): entry["snps"]
    for entry in config.get("traits", [])
    if isinstance(entry, dict) and entry.get("snps")
}
SNP_COND_COVSET  = config.get("snp_cond_covset", "age_sex_lifestyle_pcs")
SNP_COND_COVSET_OUT = SNP_COND_COVSET + "_snp_cond"

# Wildcard pattern for SNP-conditioned fids
FID_SNP_PATTERN = (
    "(" + "|".join(re.escape(f) for f in TRAITS_WITH_SNPS) + ")"
    if TRAITS_WITH_SNPS else "[^/]+"
)

# SNP-conditioned plot categories (defined in config under snp_cond_plot_categories)
SNP_COND_PLOT_CATEGORIES = config.get("snp_cond_plot_categories", {})
SNP_COND_CATEGORY_IDS    = list(SNP_COND_PLOT_CATEGORIES.keys())

SNP_COND_PLOT_TARGETS = [
    f
    for cat_id in SNP_COND_CATEGORY_IDS
    for f in [
        os.path.join(RESULTS_DIR, "category_plots_snp_cond", f"{cat_id}.png"),
        os.path.join(RESULTS_DIR, "category_plots_snp_cond", f"{cat_id}.pdf"),
        os.path.join(RESULTS_DIR, "category_plots_snp_cond", f"{cat_id}.svg"),
    ]
]

# Combined plot targets (standard + SNP-conditioned overlaid, one per category)
COMBINED_PLOT_TARGETS = [
    f
    for cat_id in SNP_COND_CATEGORY_IDS
    for f in [
        os.path.join(RESULTS_DIR, "category_plots_combined", f"{cat_id}.png"),
        os.path.join(RESULTS_DIR, "category_plots_combined", f"{cat_id}.pdf"),
        os.path.join(RESULTS_DIR, "category_plots_combined", f"{cat_id}.svg"),
    ]
]

# ─── Covariate sets ───────────────────────────────────────────────────────────
# Each covariate can be specified as:
#   simple:       cov_name: field_id
#   with options: cov_name: {field_id: X, categorical: true}
# Categorical covariates are wrapped in C() in the regression formula
# (proper dummy coding, lowest value as reference).

def _parse_covariate_set(covs_raw):
    field_ids    = {}
    lookup       = {}
    categoricals = []
    for cov_name, spec in covs_raw.items():
        if isinstance(spec, dict):
            raw_fid = spec["field_id"]
            is_cat  = bool(spec.get("categorical", False))
        else:
            raw_fid = spec
            is_cat  = False

        # Lifestyle PC sentinel — not a real FieldID, loaded from PCA output file
        if str(raw_fid) == "lifestyle_pc":
            field_ids[cov_name] = "lifestyle_pc"
            # No TRAIT_LOOKUP entry needed — handled separately in extract_merge
            continue

        cov_fid = int(raw_fid)
        if cov_fid not in TRAIT_LOOKUP:
            raise ValueError(
                f"Covariate '{cov_name}' (FieldID {cov_fid}) not found in any chars file."
            )
        field_ids[cov_name] = cov_fid
        lookup[cov_name]    = TRAIT_LOOKUP[cov_fid]
        if is_cat:
            categoricals.append(cov_name)
    return field_ids, lookup, categoricals

COVARIATE_SETS_RAW            = config["covariate_sets"]
COVSET_IDS                    = list(COVARIATE_SETS_RAW.keys())
COVARIATE_SETS                = {}
COVARIATE_LOOKUP_BY_SET       = {}
CATEGORICAL_COVARIATES_BY_SET = {}

for _sname, _sraw in COVARIATE_SETS_RAW.items():
    _fids, _lkp, _cats = _parse_covariate_set(_sraw)
    COVARIATE_SETS[_sname]                = _fids
    COVARIATE_LOOKUP_BY_SET[_sname]       = _lkp
    CATEGORICAL_COVARIATES_BY_SET[_sname] = _cats


# ─── Ancestry setup ───────────────────────────────────────────────────────────

ANCESTRY_RENAME = config["ancestry_rename"]
ANCESTRY_COLS   = list(ANCESTRY_RENAME.values())


# ─── Survival traits setup ───────────────────────────────────────────────────

SURVIVAL_TRAITS = [t for t in TRAITS if t.get("type") == "survival"]
SURVIVAL_IDS    = [str(t["field_id"]) for t in SURVIVAL_TRAITS]
SURVIVAL_MAP    = {str(t["field_id"]): t for t in SURVIVAL_TRAITS}


# ─── Wildcard constraints ────────────────────────────────────────────────────
# Prevent Snakemake from ambiguously matching survival paths with standard rules.
# Standard rules only fire for FIELD_IDS x COVSET_IDS.
# Survival rules only fire for SURVIVAL_IDS.

FID_PATTERN    = "(" + "|".join(re.escape(str(f)) for f in FIELD_IDS)    + ")" if FIELD_IDS    else "NOMATCH"
COVSET_PATTERN = "(" + "|".join(re.escape(str(c)) for c in COVSET_IDS)   + ")" if COVSET_IDS   else "NOMATCH"
SID_PATTERN    = "(" + "|".join(re.escape(str(s)) for s in SURVIVAL_IDS) + ")" if SURVIVAL_IDS else "NOMATCH"


# ─── Lifestyle PCA setup ──────────────────────────────────────────────────────

LIFESTYLE_PCA_CONFIG = config.get("lifestyle_pca", {})
USE_LIFESTYLE_PCA    = bool(LIFESTYLE_PCA_CONFIG)

if USE_LIFESTYLE_PCA:
    LIFESTYLE_PCS_FILE   = os.path.join(RESULTS_DIR, "lifestyle_pca", "lifestyle_pcs.tsv")
    LIFESTYLE_N_PCS      = LIFESTYLE_PCA_CONFIG.get("n_pcs_to_use", 10)
    LIFESTYLE_PC_COLS    = [f"lifestyle_PC{i+1}" for i in range(LIFESTYLE_N_PCS)]

    # Inject lifestyle PC covariate set — PC columns come from the PCA output file
    # rather than a UKB phenotype file, so we register a special sentinel entry.
    # extract_merge.py detects this and loads from the PCA output file instead.
    if "age_sex_lifestyle_pcs" in COVARIATE_SETS_RAW:
        # User defined it in config — validate
        pass
    # The actual covariate set is defined in config.yaml by the user;
    # we just need to make the PCA file available as an input to extract_and_merge.


# ─── Category plot setup ──────────────────────────────────────────────────────

PLOT_CATEGORIES = config.get("plot_categories", {})
CATEGORY_IDS    = list(PLOT_CATEGORIES.keys())
FOCUS_COVSET    = config.get("focus_covset", COVSET_IDS[0] if COVSET_IDS else "age_sex")
FOCUS_ANCESTRY  = config.get("focus_ancestry", list(ANCESTRY_COLS))

CATEGORY_PLOT_TARGETS = [
    f
    for cat_id in CATEGORY_IDS
    for f in [
        os.path.join(RESULTS_DIR, "category_plots", f"{cat_id}.png"),
        os.path.join(RESULTS_DIR, "category_plots", f"{cat_id}.pdf"),
        os.path.join(RESULTS_DIR, "category_plots", f"{cat_id}.svg"),
    ]
]


# ─── Rules ────────────────────────────────────────────────────────────────────

wildcard_constraints:
    fid    = FID_PATTERN,
    covset = COVSET_PATTERN,
    sid    = SID_PATTERN,


rule all:
    input:
        # Stats for all standard traits x covariate sets
        expand(
            os.path.join(RESULTS_DIR, "{fid}", "{covset}", "stats.tsv"),
            fid=FIELD_IDS, covset=COVSET_IDS,
        ),
        # Category forest plots (main visual output)
        CATEGORY_PLOT_TARGETS,
        # Summary TSV
        os.path.join(RESULTS_DIR, "summary", "all_traits.tsv"),
        # Lifestyle PCA outputs
        *([os.path.join(RESULTS_DIR, "lifestyle_pca", "lifestyle_pca_variance.png")]
          if USE_LIFESTYLE_PCA else []),
        # Survival analysis
        expand(os.path.join(RESULTS_DIR, "survival", "{sid}", "km_plot.png"),
               sid=SURVIVAL_IDS),
        expand(os.path.join(RESULTS_DIR, "survival", "{sid}", "cox_plot.png"),
               sid=SURVIVAL_IDS),
        # SNP-conditioned category plots
        *SNP_COND_PLOT_TARGETS,
        # Combined category plots (standard + SNP-conditioned overlaid)
        *COMBINED_PLOT_TARGETS,


if config.get("qpadm_tables"):

    rule merge_qpadm_tables:
        """Concatenate the qpAdm tables of all PLINK sets (Snakefile_qpadm.py, one run per set)
        into the admixture file used below. Skipped when qpadm_tables is not set."""
        input:
            tables = config["qpadm_tables"],
        output:
            merged = config["admixture_file"],
        resources:
            mem_mb  = 16000,
            runtime = "1h"
        run:
            frames = [pd.read_csv(f, sep="\t") for f in input.tables]
            cols = list(frames[0].columns)
            for f, df in zip(input.tables, frames):
                if list(df.columns) != cols:
                    raise ValueError(f"{f}: columns {list(df.columns)} differ from {cols}")
            merged = pd.concat(frames, ignore_index=True)
            n_dup = int(merged["eid"].duplicated().sum())
            if n_dup:
                raise ValueError(f"{n_dup} eids occur in more than one qpAdm table")
            os.makedirs(os.path.dirname(output.merged), exist_ok=True)
            merged.to_csv(output.merged, sep="\t", index=False)
            print(f"[merge_qpadm_tables] {len(merged):,} individuals from {len(frames)} table(s)")


rule extract_and_merge:
    """
    Extract trait + covariates, merge with admixture.
    Output lives in results/{fid}/{covset}/ so each covariate set is independent.
    Date fields: left join (absent eid = no disease = control).
    Continuous:  inner join (absent eid = no measurement, excluded).
    """
    input:
        admixture    = config["admixture_file"],
        data         = lambda wc: TRAIT_MAP[wc.fid]["data_file"],
        lifestyle_pcs = (lambda wc: os.path.join(RESULTS_DIR, "lifestyle_pca", "lifestyle_pcs.tsv")
                         if USE_LIFESTYLE_PCA else []),
    output:
        merged = os.path.join(RESULTS_DIR, "{fid}", "{covset}", "merged.tsv"),
        meta   = os.path.join(RESULTS_DIR, "{fid}", "{covset}", "trait_meta.json"),
    params:
        trait_info        = lambda wc: TRAIT_MAP[wc.fid],
        ancestry_rename   = ANCESTRY_RENAME,
        covariates_config      = lambda wc: COVARIATE_SETS[wc.covset],
        covariate_lookup       = lambda wc: COVARIATE_LOOKUP_BY_SET[wc.covset],
        categorical_covariates = lambda wc: CATEGORICAL_COVARIATES_BY_SET[wc.covset],
        covset_name            = lambda wc: wc.covset,
        lifestyle_pcs_file     = (os.path.join(RESULTS_DIR, "lifestyle_pca", "lifestyle_pcs.tsv")
                                  if USE_LIFESTYLE_PCA else None),
        lifestyle_pc_cols      = lambda wc: (
                                  LIFESTYLE_PC_COLS
                                  if USE_LIFESTYLE_PCA and any(
                                      str(v) == "lifestyle_pc"
                                      for v in COVARIATE_SETS[wc.covset].values()
                                  ) else []
                              ),
    threads: 20
    resources:
        mem_mb          = 16000,
        runtime         = "8h",
    log:
        os.path.join(RESULTS_DIR, "{fid}", "{covset}", "extract_merge.log")
    conda:
        ANALYSIS_ENV
    script:
        "scripts/extract_merge.py"


rule analyze:
    """
    Statistical tests per ancestry component, separate model per component.
    case_control → Mann-Whitney U + logistic regression (adj. for covset covariates)
    continuous   → Pearson r     + linear regression   (adj. for covset covariates)
    """
    input:
        merged = os.path.join(RESULTS_DIR, "{fid}", "{covset}", "merged.tsv"),
        meta   = os.path.join(RESULTS_DIR, "{fid}", "{covset}", "trait_meta.json"),
    output:
        stats = os.path.join(RESULTS_DIR, "{fid}", "{covset}", "stats.tsv"),
    params:
        ancestry_cols          = ANCESTRY_COLS,
        covariates_config      = lambda wc: COVARIATE_SETS[wc.covset],
        categorical_covariates = lambda wc: CATEGORICAL_COVARIATES_BY_SET[wc.covset],
    threads: 20
    resources:
        mem_mb          = 16000,
        runtime         = "8h",
    log:
        os.path.join(RESULTS_DIR, "{fid}", "{covset}", "analyze.log")
    conda:
        ANALYSIS_ENV
    script:
        "scripts/analyze.py"


rule extract_survival:
    """
    Build survival outcome (time + event) for parental lifespan traits.
    Handles father, mother, mean parental lifespan, and max parental lifespan.
    """
    input:
        admixture  = config["admixture_file"],
        touchscreen = config["phenotype_files"]["touchscreen"]["data"],
        popchar     = config["phenotype_files"]["popchar"]["data"],
    output:
        merged = os.path.join(RESULTS_DIR, "survival", "{sid}", "merged.tsv"),
        meta   = os.path.join(RESULTS_DIR, "survival", "{sid}", "trait_meta.json"),
    params:
        trait_info      = lambda wc: SURVIVAL_MAP[wc.sid],
        ancestry_rename = ANCESTRY_RENAME,
        covariates      = {},   # handled internally in script
    threads: 20
    resources:
        mem_mb          = 16000,
        runtime         = "8h",
    log:
        os.path.join(RESULTS_DIR, "survival", "{sid}", "extract_survival.log")
    conda:
        ANALYSIS_ENV
    script:
        "scripts/extract_survival.py"


rule analyze_survival:
    """
    Log-rank test (tertiles) + Cox proportional hazards regression.
    One model per ancestry component.
    """
    input:
        merged = os.path.join(RESULTS_DIR, "survival", "{sid}", "merged.tsv"),
        meta   = os.path.join(RESULTS_DIR, "survival", "{sid}", "trait_meta.json"),
    output:
        stats = os.path.join(RESULTS_DIR, "survival", "{sid}", "stats.tsv"),
    params:
        ancestry_cols = ANCESTRY_COLS,
    threads: 20
    resources:
        mem_mb          = 16000,
        runtime         = "8h",
    log:
        os.path.join(RESULTS_DIR, "survival", "{sid}", "analyze_survival.log")
    conda:
        ANALYSIS_ENV
    script:
        "scripts/analyze_survival.py"


rule plot_survival:
    """
    Figure 1: Kaplan-Meier curves (tertiles, one panel per ancestry component).
    Figure 2: Cox forest plot (HR per SD, all ancestry components).
    """
    input:
        merged = os.path.join(RESULTS_DIR, "survival", "{sid}", "merged.tsv"),
        meta   = os.path.join(RESULTS_DIR, "survival", "{sid}", "trait_meta.json"),
        stats  = os.path.join(RESULTS_DIR, "survival", "{sid}", "stats.tsv"),
    output:
        km_plot  = os.path.join(RESULTS_DIR, "survival", "{sid}", "km_plot.png"),
        cox_plot = os.path.join(RESULTS_DIR, "survival", "{sid}", "cox_plot.png"),
    params:
        ancestry_cols = ANCESTRY_COLS,
    threads: 20
    resources:
        mem_mb          = 16000,
        runtime         = "8h",
    log:
        os.path.join(RESULTS_DIR, "survival", "{sid}", "plot_survival.log")
    conda:
        ANALYSIS_ENV
    script:
        "scripts/plot_survival.py"


rule plot_category:
    """
    Forest plot for a category of traits.
    Two panels: one per focal ancestry component (EF, HG).
    Rows = traits sorted by EF effect size.
    Continuous: standardised β (per unit ancestry / SD of trait).
    Case/control: log(OR) with OR tick labels.
    Filled dot = p<0.05, open = non-significant.
    """
    input:
        stats_files  = lambda wc: [
            os.path.join(RESULTS_DIR,
                         str(t["field_id"]),
                         FOCUS_COVSET,
                         "stats.tsv")
            for t in PLOT_CATEGORIES[wc.cat_id]["traits"]
        ],
        merged_files = lambda wc: [
            os.path.join(RESULTS_DIR,
                         str(t["field_id"]),
                         FOCUS_COVSET,
                         "merged.tsv")
            for t in PLOT_CATEGORIES[wc.cat_id]["traits"]
        ],
    output:
        plot_png = os.path.join(RESULTS_DIR, "category_plots", "{cat_id}.png"),
        plot_pdf = os.path.join(RESULTS_DIR, "category_plots", "{cat_id}.pdf"),
        plot_svg = os.path.join(RESULTS_DIR, "category_plots", "{cat_id}.svg"),
    params:
        category_id    = lambda wc: wc.cat_id,
        category_label = lambda wc: PLOT_CATEGORIES[wc.cat_id]["label"],
        trait_list     = lambda wc: PLOT_CATEGORIES[wc.cat_id]["traits"],
        focus_covset   = FOCUS_COVSET,
        focus_ancestry = FOCUS_ANCESTRY,
        results_dir    = RESULTS_DIR,
    threads: 4
    resources:
        mem_mb          = 8000,
        runtime         = "6h",
    log:
        os.path.join(RESULTS_DIR, "category_plots", "{cat_id}.log")
    conda:
        ANALYSIS_ENV
    script:
        "scripts/plot_category.py"


rule compute_lifestyle_pca:
    """
    Build lifestyle principal components from diet, smoking, alcohol, income fields.
    Runs once globally; reruns automatically if lifestyle_pca config changes.
    Outputs:
      - lifestyle_pcs.tsv       : eid + PC1..PCN scores
      - lifestyle_pca_variance.png : scree plot
      - lifestyle_pca_loadings.tsv : variable loadings
    """
    input:
        admixture = config["admixture_file"],
    output:
        pcs           = os.path.join(RESULTS_DIR, "lifestyle_pca", "lifestyle_pcs.tsv"),
        variance_plot = os.path.join(RESULTS_DIR, "lifestyle_pca", "lifestyle_pca_variance.png"),
        pca_plot      = os.path.join(RESULTS_DIR, "lifestyle_pca", "lifestyle_pca_pc1_pc2.png"),
        loadings      = os.path.join(RESULTS_DIR, "lifestyle_pca", "lifestyle_pca_loadings.tsv"),
    params:
        pca_config      = LIFESTYLE_PCA_CONFIG,
        phenotype_files = config["phenotype_files"],
    threads: 20
    resources:
        mem_mb          = 32000,
        runtime         = "8h",
    log:
        os.path.join(RESULTS_DIR, "lifestyle_pca", "compute_lifestyle_pca.log")
    conda:
        ANALYSIS_ENV
    script:
        "scripts/compute_lifestyle_pca.py"


rule summarize:
    """Concatenate all per-trait stats across all covariate sets into one TSV."""
    input:
        stats = expand(
            os.path.join(RESULTS_DIR, "{fid}", "{covset}", "stats.tsv"),
            fid=FIELD_IDS, covset=COVSET_IDS,
        ),
        metas = expand(
            os.path.join(RESULTS_DIR, "{fid}", "{covset}", "trait_meta.json"),
            fid=FIELD_IDS, covset=COVSET_IDS,
        ),
    output:
        summary = os.path.join(RESULTS_DIR, "summary", "all_traits.tsv"),
    threads: 20
    resources:
        mem_mb          = 16000,
        runtime         = "8h",
    conda:
        ANALYSIS_ENV
    run:
        frames = []
        for stats_f, meta_f in zip(input.stats, input.metas):
            with open(meta_f) as fh:
                meta = json.load(fh)
            if os.path.getsize(stats_f) == 0:
                continue
            try:
                df = pd.read_csv(stats_f, sep="\t")
            except Exception:
                continue
            if df.empty:
                continue
            df.insert(0, "covset",        meta.get("covset", "unknown"))
            df.insert(0, "field_name",    meta["field_name"])
            df.insert(0, "field_id",      meta["field_id"])
            df.insert(0, "analysis_type", meta["analysis_type"])
            frames.append(df)
        if not frames:
            raise RuntimeError("[summarize] No valid stats files found.")
        os.makedirs(os.path.dirname(output.summary), exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(output.summary, sep="\t", index=False)
        print(f"[summarize] Written {len(frames)} result(s) to {output.summary}")

# =============================================================================
# SNP-conditioning rules
# =============================================================================


rule extract_snp_dosage:
    """
    Extract ALT allele dosage for trait-specific SNPs from per-chromosome VCFs.
    Output: one TSV per trait with columns eid + one per SNP label.
    Only fires for traits that have 'snps' defined in config.
    """
    input:
        admixture = config["admixture_file"],  # anchor — VCFs found at runtime via vcf_pattern
    output:
        dosages = os.path.join(RESULTS_DIR, "{fid}", "snp_dosages.tsv"),
    log:
        os.path.join(RESULTS_DIR, "{fid}", "snp_dosages.log"),
    params:
        snp_list    = lambda wc: TRAITS_WITH_SNPS.get(wc.fid, []),
        vcf_pattern = lambda wc: VCF_PATTERN,
    wildcard_constraints:
        fid = FID_SNP_PATTERN,
    threads: 2
    resources:
        mem_mb          = 24000,
        runtime         = "2h",
    conda:
        ANALYSIS_ENV
    script:
        "scripts/extract_snp_dosage.py"


rule analyze_snp_cond:
    """
    Re-run association analysis with SNP dosages added as covariates.
    Reuses analyze.py — the SNP columns are merged into the standard merged.tsv
    and passed through as extra continuous covariates.
    """
    input:
        merged   = os.path.join(RESULTS_DIR, "{fid}", SNP_COND_COVSET, "merged.tsv"),
        meta     = os.path.join(RESULTS_DIR, "{fid}", SNP_COND_COVSET, "trait_meta.json"),
        dosages  = os.path.join(RESULTS_DIR, "{fid}", "snp_dosages.tsv"),
    output:
        stats = os.path.join(RESULTS_DIR, "{fid}", SNP_COND_COVSET_OUT, "stats.tsv"),
    log:
        os.path.join(RESULTS_DIR, "{fid}", SNP_COND_COVSET_OUT, "analyze.log"),
    params:
        ancestry_cols          = ANCESTRY_COLS,
        covariates_config      = lambda wc: COVARIATE_SETS[SNP_COND_COVSET],
        categorical_covariates = lambda wc: CATEGORICAL_COVARIATES_BY_SET[SNP_COND_COVSET],
        dosages_file           = lambda wc: os.path.join(
                                     RESULTS_DIR, wc.fid, "snp_dosages.tsv"),
    wildcard_constraints:
        fid = FID_SNP_PATTERN,
    threads: 4
    resources:
        mem_mb          = 8000,
        runtime         = "2h",
    conda:
        ANALYSIS_ENV
    script:
        "scripts/analyze_snp_cond.py"


rule plot_category_snp_cond:
    """
    Category forest plot for SNP-conditioned results.
    Identical to plot_category but reads from SNP_COND_COVSET_OUT directories
    and writes to category_plots_snp_cond/.
    """
    input:
        stats_files  = lambda wc: [
            os.path.join(RESULTS_DIR, str(t["field_id"]),
                         SNP_COND_COVSET_OUT, "stats.tsv")
            for t in SNP_COND_PLOT_CATEGORIES[wc.cat_id]["traits"]
        ],
        merged_files = lambda wc: [
            os.path.join(RESULTS_DIR, str(t["field_id"]),
                         SNP_COND_COVSET, "merged.tsv")
            for t in SNP_COND_PLOT_CATEGORIES[wc.cat_id]["traits"]
        ],
    output:
        plot_png = os.path.join(RESULTS_DIR, "category_plots_snp_cond", "{cat_id}.png"),
        plot_pdf = os.path.join(RESULTS_DIR, "category_plots_snp_cond", "{cat_id}.pdf"),
        plot_svg = os.path.join(RESULTS_DIR, "category_plots_snp_cond", "{cat_id}.svg"),
    log:
        os.path.join(RESULTS_DIR, "category_plots_snp_cond", "{cat_id}.log"),
    params:
        category_id    = lambda wc: wc.cat_id,
        category_label = lambda wc: SNP_COND_PLOT_CATEGORIES[wc.cat_id]["label"],
        trait_list     = lambda wc: SNP_COND_PLOT_CATEGORIES[wc.cat_id]["traits"],
        focus_covset   = SNP_COND_COVSET_OUT,
        focus_ancestry = FOCUS_ANCESTRY,
        results_dir    = RESULTS_DIR,
    wildcard_constraints:
        cat_id = "[^/]+",
    threads: 4
    resources:
        mem_mb          = 8000,
        runtime         = "2h",
    conda:
        ANALYSIS_ENV
    script:
        "scripts/plot_category.py"


rule plot_category_combined:
    """
    Combined forest plot: standard and SNP-conditioned results overlaid,
    one circle (standard) and one diamond (SNP-conditioned) per trait row.
    Only runs for categories defined in snp_cond_plot_categories.
    """
    input:
        stats_files = lambda wc: [
            os.path.join(RESULTS_DIR, str(t["field_id"]),
                         SNP_COND_COVSET, "stats.tsv")
            for t in SNP_COND_PLOT_CATEGORIES[wc.cat_id]["traits"]
        ],
        stats_files_snp_cond = lambda wc: [
            os.path.join(RESULTS_DIR, str(t["field_id"]),
                         SNP_COND_COVSET_OUT, "stats.tsv")
            for t in SNP_COND_PLOT_CATEGORIES[wc.cat_id]["traits"]
        ],
        merged_files = lambda wc: [
            os.path.join(RESULTS_DIR, str(t["field_id"]),
                         SNP_COND_COVSET, "merged.tsv")
            for t in SNP_COND_PLOT_CATEGORIES[wc.cat_id]["traits"]
        ],
    output:
        plot_png = os.path.join(RESULTS_DIR, "category_plots_combined", "{cat_id}.png"),
        plot_pdf = os.path.join(RESULTS_DIR, "category_plots_combined", "{cat_id}.pdf"),
        plot_svg = os.path.join(RESULTS_DIR, "category_plots_combined", "{cat_id}.svg"),
    log:
        os.path.join(RESULTS_DIR, "category_plots_combined", "{cat_id}.log"),
    params:
        category_id     = lambda wc: wc.cat_id,
        category_label  = lambda wc: SNP_COND_PLOT_CATEGORIES[wc.cat_id]["label"],
        trait_list      = lambda wc: SNP_COND_PLOT_CATEGORIES[wc.cat_id]["traits"],
        focus_covset    = SNP_COND_COVSET,
        focus_ancestry  = FOCUS_ANCESTRY,
        results_dir     = RESULTS_DIR,
        snp_cond_covset = SNP_COND_COVSET_OUT,
    wildcard_constraints:
        cat_id = "[^/]+",
    threads: 4
    resources:
        mem_mb          = 8000,
        runtime         = "2h",
    conda:
        ANALYSIS_ENV
    script:
        "scripts/plot_category_combined.py"