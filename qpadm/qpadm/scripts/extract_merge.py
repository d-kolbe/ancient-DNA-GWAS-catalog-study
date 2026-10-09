"""
scripts/extract_merge.py
========================
Called by the `extract_and_merge` Snakemake rule.

Responsibilities
----------------
1. Load admixture file, rename ancestry columns to short names.
2. Find and load the correct column for the requested trait FieldID.
3. Load covariate columns (age, sex) from whichever file they live in.
4. Merge everything on eid.
5. For Date-type fields (FirstOccurrences): derive a binary case_control column.
6. For continuous fields: replace UKB special missing codes with NaN.
7. Recode sex: UKB 1 (Male) → 0, 2 (Female) → 1.
8. Write merged.tsv and trait_meta.json.
"""

import sys as _sys
_log = open(str(snakemake.log[0]), "w", buffering=1)
_sys.stdout = _log
_sys.stderr = _log

import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UKB_MISSING_CODES = {-1, -3, -7, -10, -11, -13}

LOG_PREFIX = "[extract_merge]"

def log(msg):
    print(f"{LOG_PREFIX} {msg}", flush=True)

def log_count(label, df, extra=""):
    """Print a labelled row count, optionally with extra context."""
    msg = f"  {label}: {len(df):,} rows"
    if extra:
        msg += f"  ({extra})"
    print(msg, flush=True)


def get_eid_col(columns) -> str:
    if "eid" in columns:
        return "eid"
    if "f_eid" in columns:
        return "f_eid"
    raise KeyError(f"No eid column found. Available columns (first 10): {list(columns)[:10]}")


def find_column(columns, field_id: int, instance: int = 0) -> str:
    cols = list(columns)
    candidates = [
        f"f_{field_id}_{instance}_0",
        f"f_{field_id}_{instance}",
    ]
    for c in candidates:
        if c in cols:
            return c
    prefix  = f"f_{field_id}_"
    matches = [c for c in cols if c.startswith(prefix)]
    if matches:
        return matches[0]
    raise KeyError(
        f"FieldID {field_id} not found in data file. "
        f"Tried: {candidates}. "
        f"Columns starting with 'f_{field_id}': {matches}"
    )


def load_columns(filepath: str, cols: list) -> pd.DataFrame:
    return pd.read_csv(filepath, sep="\t", usecols=cols, low_memory=False)


# ---------------------------------------------------------------------------
# Snakemake params
# ---------------------------------------------------------------------------

trait_info        = snakemake.params.trait_info
ancestry_rename   = snakemake.params.ancestry_rename
covariates_config      = snakemake.params.covariates_config
covariate_lookup       = snakemake.params.covariate_lookup
categorical_covariates = snakemake.params.categorical_covariates
sex_field_id           = covariates_config.get("sex")
covset_name            = snakemake.params.covset_name
lifestyle_pcs_file     = snakemake.params.lifestyle_pcs_file   # path or None
lifestyle_pc_cols      = snakemake.params.lifestyle_pc_cols    # e.g. ["lifestyle_PC1", ...]

# Identify which covariates are lifestyle PCs (loaded from PCA file, not UKB files)
# These have field_id == "lifestyle_pc" sentinel in covariates_config
lifestyle_pc_covnames = [
    cov_name for cov_name, fid in covariates_config.items()
    if str(fid) == "lifestyle_pc"
]
regular_covariates_config = {
    k: v for k, v in covariates_config.items()
    if str(v) != "lifestyle_pc"
}

field_id   = trait_info["field_id"]
instance   = trait_info.get("instance", 0)
value_type = trait_info["value_type"]
trait_type = trait_info.get("type", "standard")   # standard | combined_cases | computed
data_file  = snakemake.input.data

log(f"FieldID {field_id} ({trait_info['field_name']}) — type={trait_type}, starting merge pipeline")

# ---------------------------------------------------------------------------
# 1. Admixture
# ---------------------------------------------------------------------------

admix = pd.read_csv(snakemake.input.admixture, sep="\t")
admix = admix.rename(columns=ancestry_rename)
ancestry_cols = list(ancestry_rename.values())

if "eid" not in admix.columns:
    raise KeyError(f"No 'eid' column in admixture file. Columns: {admix.columns.tolist()}")

log_count("Admixture — loaded", admix)

# Remove out-of-bounds qpadm proportions (values must be in [0, 1])
valid_mask = pd.Series(True, index=admix.index)
for anc in ancestry_cols:
    out = ~admix[anc].between(0, 1)
    n_out = out.sum()
    if n_out > 0:
        log(f"  {anc}: {n_out:,} samples with proportion outside [0, 1]")
    valid_mask &= ~out
admix = admix[valid_mask].reset_index(drop=True)
log_count("Admixture — after [0,1] filter", admix)

# ---------------------------------------------------------------------------
# 2. Trait column(s)
# ---------------------------------------------------------------------------

header_df = pd.read_csv(data_file, sep="\t", nrows=0)
eid_col   = get_eid_col(header_df.columns)

trait_file_nrows = pd.read_csv(data_file, sep="\t", usecols=[eid_col]).shape[0]
log(f"  Trait data file: {trait_file_nrows:,} rows")

if trait_type == "combined_cases":
    # Resolve a column for every component FieldID
    component_cols = []
    for comp_fid in trait_info["components"]:
        try:
            col = find_column(header_df.columns, comp_fid, instance)
            component_cols.append((comp_fid, col))
            log(f"  Component FieldID {comp_fid} → '{col}'")
        except KeyError as e:
            sys.exit(f"{LOG_PREFIX} ERROR: {e}")
    trait_col = None   # not used directly for combined

elif trait_type == "computed":
    # Resolve all field columns from the fields dict {var_name: FieldID}
    computed_fields = trait_info["fields"]    # {var_name: FieldID}
    computed_formula = trait_info["formula"]  # e.g. "hip / (height / 100 * weight)"
    computed_cols = {}                        # {var_name: actual_col_name}
    for var_name, comp_fid in computed_fields.items():
        try:
            col = find_column(header_df.columns, comp_fid, instance)
            computed_cols[var_name] = col
            log(f"  Field '{var_name}' (FieldID {comp_fid}) → '{col}'")
        except KeyError as e:
            sys.exit(f"{LOG_PREFIX} ERROR resolving computed field '{var_name}': {e}")
    log(f"  Formula: {computed_formula}")
    trait_col = None

else:  # standard
    try:
        trait_col = find_column(header_df.columns, field_id, instance)
    except KeyError as e:
        sys.exit(f"{LOG_PREFIX} ERROR: {e}")
    log(f"  Trait column resolved to: '{trait_col}' in {Path(data_file).name}")

# ---------------------------------------------------------------------------
# 3. Covariates — may live in the same or different data files
# ---------------------------------------------------------------------------

cov_files: dict[str, list[tuple]] = {}

for cov_name, cov_fid in regular_covariates_config.items():
    cov_entry     = covariate_lookup[cov_name]
    cov_data_file = cov_entry["data_file"]
    cov_header    = pd.read_csv(cov_data_file, sep="\t", nrows=0)
    cov_eid       = get_eid_col(cov_header.columns)
    try:
        cov_col = find_column(cov_header.columns, cov_fid, 0)
    except KeyError as e:
        sys.exit(f"{LOG_PREFIX} Covariate '{cov_name}' ERROR: {e}")
    # Log row count of covariate file — this is a common bottleneck
    cov_file_eids = pd.read_csv(cov_data_file, sep="\t", usecols=[cov_eid])[cov_eid]
    cov_overlap   = admix["eid"].isin(cov_file_eids).sum()
    log(f"  Covariate '{cov_name}' (FieldID {cov_fid}) → '{cov_col}' "
        f"in {Path(cov_data_file).name} "
        f"({len(cov_file_eids):,} rows, "
        f"{cov_overlap:,} overlap with admixture  ← ceiling for this covariate)")
    cov_files.setdefault(cov_data_file, []).append((cov_name, cov_fid, cov_col, cov_eid))

# ---------------------------------------------------------------------------
# 4. Load trait (possibly with covariates if they share the same file)
# ---------------------------------------------------------------------------

same_file_covs = []

if trait_type == "combined_cases":
    # Load eid + all component date columns
    comp_raw_cols = [col for _, col in component_cols]
    cols_to_load  = [eid_col] + comp_raw_cols
    if data_file in cov_files:
        for cov_name, cov_fid, cov_col, _cov_eid in cov_files[data_file]:
            if cov_col not in cols_to_load:
                cols_to_load.append(cov_col)
            same_file_covs.append((cov_name, cov_col))
    trait_df = load_columns(data_file, cols_to_load)
    trait_df = trait_df.rename(columns={eid_col: "eid"})
    # Case = has a date in ANY component; store as combined flag
    trait_df["trait_value"] = trait_df[comp_raw_cols].notna().any(axis=1).map({True: "case", False: np.nan})
    for cov_name, cov_col in same_file_covs:
        trait_df = trait_df.rename(columns={cov_col: cov_name})
    trait_df = trait_df[["eid", "trait_value"] + [c for c, _ in same_file_covs]]

elif trait_type == "computed":
    # Load eid + all formula field columns
    raw_cols  = list(computed_cols.values())
    cols_to_load = list(dict.fromkeys([eid_col] + raw_cols))
    if data_file in cov_files:
        for cov_name, cov_fid, cov_col, _cov_eid in cov_files[data_file]:
            if cov_col not in cols_to_load:
                cols_to_load.append(cov_col)
            same_file_covs.append((cov_name, cov_col))
    trait_df = load_columns(data_file, cols_to_load)
    trait_df = trait_df.rename(columns={eid_col: "eid"})
    # Build a clean numeric frame with user-friendly variable names for eval
    eval_df = pd.DataFrame({"eid": trait_df["eid"]})
    for var_name, raw_col in computed_cols.items():
        eval_df[var_name] = pd.to_numeric(trait_df[raw_col], errors="coerce")
    try:
        trait_df["trait_value"] = eval_df.eval(computed_formula)
    except Exception as e:
        sys.exit(f"{LOG_PREFIX} Formula eval failed for '{computed_formula}': {e}")
    for cov_name, cov_col in same_file_covs:
        trait_df = trait_df.rename(columns={cov_col: cov_name})
    trait_df = trait_df[["eid", "trait_value"] + [c for c, _ in same_file_covs]]

else:  # standard
    cols_to_load = [eid_col, trait_col]
    if data_file in cov_files:
        for cov_name, cov_fid, cov_col, _cov_eid in cov_files[data_file]:
            if cov_col not in cols_to_load:
                cols_to_load.append(cov_col)
            same_file_covs.append((cov_name, cov_col))
    trait_df = load_columns(data_file, cols_to_load)
    trait_df = trait_df.rename(columns={eid_col: "eid", trait_col: "trait_value"})
    for cov_name, cov_col in same_file_covs:
        if cov_col == trait_col:
            # Covariate shares the same column as the trait (e.g. BMI as both
            # trait and covariate). trait_value already holds the data — alias it.
            trait_df[cov_name] = trait_df["trait_value"]
            log(f"  Covariate '{cov_name}' aliases trait_value (same FieldID/column)")
        else:
            trait_df = trait_df.rename(columns={cov_col: cov_name})

log_count("Trait df — loaded", trait_df)
log(f"  Non-null trait values: {trait_df['trait_value'].notna().sum():,}")

# ---------------------------------------------------------------------------
# 5. Load covariates from other files
# ---------------------------------------------------------------------------

cov_dfs        = []
already_loaded = {cov_name for cov_name, _ in same_file_covs}

for cov_file, entries in cov_files.items():
    if cov_file == data_file:
        continue
    load_cols  = []
    rename_map = {}
    for cov_name, cov_fid, cov_col, cov_eid in entries:
        if cov_name in already_loaded:
            continue
        load_cols.append(cov_eid)
        load_cols.append(cov_col)
        rename_map[cov_eid] = "eid"
        rename_map[cov_col] = cov_name
    if not load_cols:
        continue
    cov_df = load_columns(cov_file, list(dict.fromkeys(load_cols)))
    cov_df = cov_df.rename(columns=rename_map)
    log_count(f"Covariate file {Path(cov_file).name} — loaded", cov_df)
    cov_dfs.append(cov_df)

# ---------------------------------------------------------------------------
# 6. Merge — with counts at every step
# ---------------------------------------------------------------------------

merged = admix[["eid"] + ancestry_cols].copy()
log_count("Start merge (admixture)", merged)

# Date fields (FirstOccurrences): LEFT join — an eid absent from the trait file
# means no disease recorded, which is identical to an NA date → control.
# Continuous / categorical: INNER join — missing means no measurement, exclude.
trait_merge_how = "left" if (value_type == "Date" or trait_type == "combined_cases") else "inner"
n_before_trait_merge = len(merged)
merged = merged.merge(trait_df, on="eid", how=trait_merge_how)
n_in_trait   = merged["trait_value"].notna().sum() if "trait_value" in merged.columns else 0
n_lost       = n_before_trait_merge - len(merged)
log_count(f"After {trait_merge_how} merge with trait file", merged,
          f"lost {n_lost:,} eids not in trait file" if n_lost > 0 else
          f"{merged['trait_value'].isna().sum():,} NAs in trait col → will become controls")

for cov_df in cov_dfs:
    n_before = len(merged)
    merged   = merged.merge(cov_df, on="eid", how="left")
    cov_cols = [c for c in cov_df.columns if c != "eid"]
    log_count(f"After left merge with covariate(s) {cov_cols}", merged)

# ── Lifestyle PCs (from PCA output file) ──────────────────────────────────
if lifestyle_pc_covnames and lifestyle_pcs_file:
    log(f"  Loading lifestyle PCs from {Path(lifestyle_pcs_file).name}")
    pcs_df = pd.read_csv(lifestyle_pcs_file, sep="\t", usecols=["eid"] + lifestyle_pc_cols)
    merged = merged.merge(pcs_df, on="eid", how="left")
    n_missing_pcs = merged[lifestyle_pc_cols[0]].isna().sum() if lifestyle_pc_cols else 0
    log_count(f"After left merge with lifestyle PCs", merged,
              f"{n_missing_pcs:,} samples without PC scores")

# ---------------------------------------------------------------------------
# 7. Post-processing
# ---------------------------------------------------------------------------

if "sex" in merged.columns:
    unique_vals = set(merged["sex"].dropna().unique())
    if unique_vals <= {0, 1}:
        log("  Sex column already 0/1 coded — skipping recode")
    elif unique_vals <= {1, 2}:
        merged["sex"] = merged["sex"].map({1: 0, 2: 1})
        log("  Sex recoded from UKB 1/2 → 0/1")
    else:
        log(f"  WARNING: unexpected sex values {unique_vals} — mapping 1→0, 2→1, others→NaN")
        merged["sex"] = merged["sex"].map({1: 0, 2: 1})

# Replace UKB negative special codes in ALL covariate columns with NaN.
# This covers: -1 (do not know), -3 (prefer not to answer), -7 (none of above) etc.
# For categorical covariates this also removes invalid responses before C() coding.
for cov_col in regular_covariates_config.keys():
    if cov_col in merged.columns and cov_col != "sex":
        n_before = merged[cov_col].notna().sum()
        merged[cov_col] = pd.to_numeric(merged[cov_col], errors="coerce")
        merged.loc[merged[cov_col] < 0, cov_col] = np.nan
        n_after = merged[cov_col].notna().sum()
        if n_before > n_after:
            log(f"  Covariate '{cov_col}': set {n_before - n_after:,} negative codes to NaN")

# ── Optional trait-level recode ────────────────────────────────────────────
# For traits whose raw UKB coding needs remapping and/or exclusion of certain
# response categories before typing (e.g. dichotomizing an ordinal field).
# Config: recode: {raw_value: new_value, ...} — any raw value NOT listed as a
# key is set to NaN (excluded), including UKB's own -1/-3 special codes if
# they aren't explicitly mapped.
recode_map = trait_info.get("recode")
if recode_map:
    merged["trait_value"] = pd.to_numeric(merged["trait_value"], errors="coerce")
    recode_map_num = {float(k): v for k, v in recode_map.items()}
    n_before_recode = merged["trait_value"].notna().sum()
    merged["trait_value"] = merged["trait_value"].map(recode_map_num)
    n_after_recode = merged["trait_value"].notna().sum()
    log(f"  Applied recode {recode_map_num}: "
        f"{n_before_recode:,} -> {n_after_recode:,} non-null "
        f"({n_before_recode - n_after_recode:,} excluded as not in recode map)")

# ── Optional analysis-type override ────────────────────────────────────────
# Bypasses the chars-derived value_type dispatch below. Use when a field's
# UKB ValueType doesn't match how it should actually be analysed (e.g. a
# binary Categorical field that should run through the case_control path
# rather than the generic categorical/continuous one).
force_type = trait_info.get("force_analysis_type")  # "case_control" | "continuous" | None

if trait_type == "combined_cases":
    # trait_value is "case" for any-component-positive, NaN otherwise
    merged["case_control"] = (merged["trait_value"] == "case").astype(int)
    analysis_type = "case_control"
    log(f"  Combined case/control: "
        f"cases={merged['case_control'].sum():,} (any component), "
        f"controls={(merged['case_control']==0).sum():,} (none of the components)")

elif force_type == "case_control":
    merged["trait_value"] = pd.to_numeric(merged["trait_value"], errors="coerce")
    # Strip UKB special missing codes (-1, -3, etc.) even when no recode: is
    # defined — case_control must end up strictly {0, 1, NaN}, never e.g. -1,
    # or statsmodels.logit will choke on (or silently misfit) a non-binary outcome.
    merged.loc[merged["trait_value"].isin(UKB_MISSING_CODES), "trait_value"] = np.nan
    merged["case_control"] = merged["trait_value"]
    merged.loc[~merged["case_control"].isin([0, 1]), "case_control"] = np.nan
    analysis_type = "case_control"
    log(f"  Case/control forced via config (force_analysis_type): "
        f"cases={int((merged['case_control']==1).sum()):,}, "
        f"controls={int((merged['case_control']==0).sum()):,}, "
        f"excluded={int(merged['case_control'].isna().sum()):,}")

elif force_type == "continuous":
    merged["trait_value"] = pd.to_numeric(merged["trait_value"], errors="coerce")
    merged.loc[merged["trait_value"].isin(UKB_MISSING_CODES), "trait_value"] = np.nan
    analysis_type = "continuous"
    log(f"  Continuous forced via config (force_analysis_type): "
        f"n_valid={int(merged['trait_value'].notna().sum()):,}")

elif value_type == "Date":
    merged["case_control"] = merged["trait_value"].notna().astype(int)
    analysis_type = "case_control"
    log(f"  Case/control derived from Date: "
        f"cases={merged['case_control'].sum():,}, "
        f"controls={(merged['case_control']==0).sum():,}")

elif value_type in ("Continuous", "Integer", "Continuous auto-converted"):
    merged["trait_value"] = pd.to_numeric(merged["trait_value"], errors="coerce")
    merged.loc[merged["trait_value"].isin(UKB_MISSING_CODES), "trait_value"] = np.nan
    analysis_type = "continuous"

else:
    merged["trait_value"] = pd.to_numeric(merged["trait_value"], errors="coerce")
    merged.loc[merged["trait_value"].isin(UKB_MISSING_CODES), "trait_value"] = np.nan
    analysis_type = "categorical"

# Drop rows with missing ancestry or covariates
required_complete = (
    ["eid"] + ancestry_cols
    + list(regular_covariates_config.keys())
    + lifestyle_pc_cols
)
n_before = len(merged)

# Per-column missing counts before dropping — shows exactly what is limiting sample size
log(f"  Sample completeness before dropna ({n_before:,} rows total):")
for col in required_complete:
    n_missing = merged[col].isna().sum()
    n_present = n_before - n_missing
    src = covariate_lookup.get(col, {}).get("data_file", "")
    src_label = f"  ← from {Path(src).name}" if src else ""
    status = "OK" if n_missing == 0 else f"*** {n_missing:,} missing — only {n_present:,} usable{src_label}"
    log(f"    {col}: {status}")

merged  = merged.dropna(subset=required_complete)
n_after = len(merged)
log_count("After dropna on ancestry + covariates", merged,
          f"dropped {n_before - n_after:,} — limiting column is likely whichever has most missing above")

# ---------------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------------

merged.to_csv(snakemake.output.merged, sep="\t", index=False)

meta = {
    "field_id":      field_id,
    "field_name":    trait_info["field_name"],
    "value_type":    value_type,
    "analysis_type": analysis_type,
    "group":         trait_info["group"],
    "instance":      instance,
    "covset":        covset_name,
    "n_total":       int(n_after),
}

if analysis_type == "case_control":
    meta["n_cases"]    = int(merged["case_control"].sum())
    meta["n_controls"] = int((merged["case_control"] == 0).sum())
else:
    n_valid = int(merged["trait_value"].notna().sum())
    meta["n_trait_valid"]   = n_valid
    meta["n_trait_missing"] = int(n_after - n_valid)

with open(snakemake.output.meta, "w") as fh:
    json.dump(meta, fh, indent=2)

log(f"DONE — FieldID {field_id}, analysis_type={analysis_type}, final n={n_after:,}")
if analysis_type == "case_control":
    log(f"  cases={meta['n_cases']:,}, controls={meta['n_controls']:,}")