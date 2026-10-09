"""
scripts/analyze_snp_cond.py
============================
SNP-conditioned association analysis.

Loads the standard merged.tsv + trait_meta.json from the base covariate set,
merges in SNP dosage columns, then runs the same linear/logistic regression
as analyze.py with the SNP dosages added as extra continuous covariates.

This allows direct comparison of effect sizes before and after conditioning
on trait-associated variants.
"""

import sys as _sys
_log = open(str(snakemake.log[0]), "w", buffering=1)
_sys.stdout = _log
_sys.stderr = _log

import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

LOG_PREFIX = "[analyze_snp_cond]"
def log(msg): print(f"{LOG_PREFIX} {msg}", flush=True)

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

merged = pd.read_csv(snakemake.input.merged, sep="\t")
dosages = pd.read_csv(snakemake.input.dosages, sep="\t")

with open(snakemake.input.meta) as fh:
    meta = json.load(fh)

ancestry_cols          = snakemake.params.ancestry_cols
covariates_config      = snakemake.params.covariates_config
categorical_covariates = snakemake.params.categorical_covariates
analysis_type          = meta["analysis_type"]

# Merge dosages into merged TSV on eid
log(f"Merged TSV: {len(merged):,} rows")
log(f"Dosage TSV: {len(dosages):,} rows, SNP columns: {[c for c in dosages.columns if c != 'eid']}")

snp_cols = [c for c in dosages.columns if c != "eid"]
merged = merged.merge(dosages[["eid"] + snp_cols], on="eid", how="left")
log(f"After dosage merge: {len(merged):,} rows, {merged[snp_cols].notna().all(axis=1).sum():,} with all SNP dosages")

# ---------------------------------------------------------------------------
# Covariate names (same as analyze.py logic)
# ---------------------------------------------------------------------------

regular_cov_names = [k for k, v in covariates_config.items()
                     if str(v) != "lifestyle_pc"]
lifestyle_pc_cols = [c for c in merged.columns if c.startswith("lifestyle_PC")]
# Base covariates + lifestyle PCs + SNP dosages
covariate_names   = regular_cov_names + lifestyle_pc_cols + snp_cols

log(f"Covariates: {covariate_names}")

def formula_term(cov_name):
    if cov_name in categorical_covariates:
        return f"C({cov_name})"
    # SNP column names may contain special chars — wrap in Q()
    if cov_name in snp_cols:
        return f"Q('{cov_name}')"
    return cov_name

def fmt_pval(p):
    return float(f"{p:.6g}")

results = []

# ---------------------------------------------------------------------------
# Helpers (identical to analyze.py)
# ---------------------------------------------------------------------------

def run_logistic(df, outcome_col, anc, cov_names):
    terms    = [anc] + [formula_term(c) for c in cov_names]
    raw_cols = [anc] + cov_names + [outcome_col]
    formula  = f"{outcome_col} ~ " + " + ".join(terms)
    try:
        model = smf.logit(formula, data=df.dropna(subset=raw_cols)).fit(disp=0)
        return model
    except Exception as e:
        log(f"Logistic regression failed for {anc}: {e}")
        return None


def run_ols(df, outcome_col, anc, cov_names):
    terms    = [anc] + [formula_term(c) for c in cov_names]
    raw_cols = [anc] + cov_names + [outcome_col]
    formula  = "trait_value ~ " + " + ".join(terms)
    try:
        model = smf.ols(formula, data=df.dropna(subset=raw_cols)).fit()
        return model
    except Exception as e:
        log(f"OLS regression failed for {anc}: {e}")
        return None

# ---------------------------------------------------------------------------
# Case/control
# ---------------------------------------------------------------------------

if analysis_type == "case_control":
    outcome_col = "case_control"
    cases    = merged[merged[outcome_col] == 1]
    controls = merged[merged[outcome_col] == 0]

    for anc in ancestry_cols:
        case_vals = cases[anc].dropna().values
        ctrl_vals = controls[anc].dropna().values

        if len(case_vals) < 5 or len(ctrl_vals) < 5:
            log(f"Skipping {anc}: too few observations.")
            continue

        # Mann-Whitney (unadjusted, no SNP conditioning — for reference)
        u_stat, p_mw = stats.mannwhitneyu(case_vals, ctrl_vals, alternative="two-sided")
        results.append({
            "test":            "Mann-Whitney U",
            "ancestry":        anc,
            "cases_mean":      float(np.mean(case_vals)),
            "cases_median":    float(np.median(case_vals)),
            "controls_mean":   float(np.mean(ctrl_vals)),
            "controls_median": float(np.median(ctrl_vals)),
            "delta_median":    float(np.median(case_vals) - np.median(ctrl_vals)),
            "U_statistic":     float(u_stat),
            "p_value":         fmt_pval(p_mw),
            "n_cases":         len(case_vals),
            "n_controls":      len(ctrl_vals),
        })

        # Logistic regression + SNP covariates
        model = run_logistic(merged, outcome_col, anc, covariate_names)
        if model is not None and anc in model.params:
            ci = model.conf_int()
            results.append({
                "test":         "Logistic regression (adj.+SNP)",
                "ancestry":     anc,
                "beta_logodds": float(model.params[anc]),
                "OR":           float(np.exp(model.params[anc])),
                "OR_CI_low":    float(np.exp(ci.loc[anc, 0])),
                "OR_CI_high":   float(np.exp(ci.loc[anc, 1])),
                "SE":           float(model.bse[anc]),
                "p_value":      fmt_pval(model.pvalues[anc]),
                "n":            int(model.nobs),
                "AIC":          float(model.aic),
                "snp_cols":     ";".join(snp_cols),
            })

# ---------------------------------------------------------------------------
# Continuous
# ---------------------------------------------------------------------------

elif analysis_type in ("continuous", "categorical"):
    outcome_col = "trait_value"
    df_valid    = merged.dropna(subset=[outcome_col])

    for anc in ancestry_cols:
        valid = df_valid[[anc, outcome_col]].dropna()
        if len(valid) < 10:
            log(f"Skipping {anc}: too few observations.")
            continue

        # Pearson (unadjusted, no SNP conditioning — for reference)
        r, p_r = stats.pearsonr(valid[anc].values, valid[outcome_col].values)
        results.append({
            "test":      "Pearson r (unadjusted)",
            "ancestry":  anc,
            "r":         float(r),
            "r_squared": float(r ** 2),
            "p_value":   fmt_pval(p_r),
            "n":         len(valid),
        })

        # Linear regression + SNP covariates
        model = run_ols(df_valid, outcome_col, anc, covariate_names)
        if model is not None and anc in model.params:
            ci = model.conf_int()
            results.append({
                "test":      "Linear regression (adj.+SNP)",
                "ancestry":  anc,
                "beta":      float(model.params[anc]),
                "CI_low":    float(ci.loc[anc, 0]),
                "CI_high":   float(ci.loc[anc, 1]),
                "SE":        float(model.bse[anc]),
                "p_value":   fmt_pval(model.pvalues[anc]),
                "r_squared": float(model.rsquared),
                "n":         int(model.nobs),
                "snp_cols":  ";".join(snp_cols),
            })

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

stats_df = pd.DataFrame(results)
if stats_df.empty:
    log(f"WARNING: 0 result rows — writing empty stats file.")
    pd.DataFrame(columns=["test", "ancestry", "p_value"]).to_csv(
        snakemake.output.stats, sep="\t", index=False)
else:
    log(f"{len(results)} result rows written.")
    stats_df.to_csv(snakemake.output.stats, sep="\t", index=False)