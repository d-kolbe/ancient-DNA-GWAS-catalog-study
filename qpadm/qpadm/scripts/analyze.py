"""
scripts/analyze.py
==================
Called by the `analyze` Snakemake rule.

Each ancestry component is analysed independently — no joint model, no reference.
This sidesteps the sum-to-1 collinearity entirely and gives a clean per-component result.

case_control traits  (per ancestry component)
---------------------------------------------
  - Mann-Whitney U test (cases vs controls)
  - Logistic regression: outcome ~ <ancestry> + age + sex
      Reports: OR, 95 % CI, SE, p-value

continuous traits  (per ancestry component)
--------------------------------------------
  - Pearson correlation (unadjusted)
  - Linear regression:  trait ~ <ancestry> + age + sex
      Reports: β, 95 % CI, SE, p-value, R²
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

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

merged = pd.read_csv(snakemake.input.merged, sep="\t")

with open(snakemake.input.meta) as fh:
    meta = json.load(fh)

ancestry_cols          = snakemake.params.ancestry_cols
covariates_config      = snakemake.params.covariates_config
categorical_covariates = snakemake.params.categorical_covariates  # e.g. ["smoking", "alcohol"]
analysis_type          = meta["analysis_type"]
# Regular covariate names (exclude lifestyle_pc sentinels)
regular_cov_names = [k for k, v in covariates_config.items()
                     if str(v) != "lifestyle_pc"]

# Lifestyle PC column names come from merged.tsv columns directly
lifestyle_pc_cols = [c for c in merged.columns if c.startswith("lifestyle_PC")]

# All covariate names in the model
covariate_names = regular_cov_names + lifestyle_pc_cols

def formula_term(cov_name):
    if cov_name in categorical_covariates:
        return f"C({cov_name})"
    return cov_name

results = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_pval(p):
    return float(f"{p:.6g}")


def run_logistic(df, outcome_col, anc, cov_names):
    terms      = [anc] + [formula_term(c) for c in cov_names]
    raw_cols   = [anc] + cov_names + [outcome_col]
    formula    = f"{outcome_col} ~ " + " + ".join(terms)
    try:
        model = smf.logit(formula, data=df.dropna(subset=raw_cols)).fit(disp=0)
        return model
    except Exception as e:
        print(f"[analyze] Logistic regression failed for {anc}: {e}")
        return None


def run_ols(df, outcome_col, anc, cov_names):
    terms    = [anc] + [formula_term(c) for c in cov_names]
    raw_cols = [anc] + cov_names + [outcome_col]
    formula  = "trait_value ~ " + " + ".join(terms)
    try:
        model = smf.ols(formula, data=df.dropna(subset=raw_cols)).fit()
        return model
    except Exception as e:
        print(f"[analyze] OLS regression failed for {anc}: {e}")
        return None


# ---------------------------------------------------------------------------
# Case / Control — one model per ancestry component
# ---------------------------------------------------------------------------

if analysis_type == "case_control":
    outcome_col = "case_control"
    cases    = merged[merged[outcome_col] == 1]
    controls = merged[merged[outcome_col] == 0]

    for anc in ancestry_cols:
        case_vals = cases[anc].dropna().values
        ctrl_vals = controls[anc].dropna().values

        if len(case_vals) < 5 or len(ctrl_vals) < 5:
            print(f"[analyze] Skipping {anc}: too few observations.")
            continue

        # ── Mann-Whitney U ──
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

        # ── Logistic regression: outcome ~ <anc> + age + sex ──
        model = run_logistic(merged, outcome_col, anc, covariate_names)
        if model is not None and anc in model.params:
            ci    = model.conf_int()
            coef  = model.params[anc]
            se    = model.bse[anc]
            pval  = model.pvalues[anc]
            ci_lo = ci.loc[anc, 0]
            ci_hi = ci.loc[anc, 1]
            results.append({
                "test":         "Logistic regression (adj.)",
                "ancestry":     anc,
                "beta_logodds": float(coef),
                "OR":           float(np.exp(coef)),
                "OR_CI_low":    float(np.exp(ci_lo)),
                "OR_CI_high":   float(np.exp(ci_hi)),
                "SE":           float(se),
                "p_value":      fmt_pval(pval),
                "n":            int(model.nobs),
                "AIC":          float(model.aic),
            })


# ---------------------------------------------------------------------------
# Continuous — one model per ancestry component
# ---------------------------------------------------------------------------

elif analysis_type in ("continuous", "categorical"):
    outcome_col = "trait_value"
    df_valid    = merged.dropna(subset=[outcome_col])

    for anc in ancestry_cols:
        valid = df_valid[[anc, outcome_col]].dropna()

        if len(valid) < 10:
            print(f"[analyze] Skipping {anc}: too few observations.")
            continue

        # ── Pearson correlation (unadjusted) ──
        r, p_r = stats.pearsonr(valid[anc].values, valid[outcome_col].values)
        results.append({
            "test":      "Pearson r (unadjusted)",
            "ancestry":  anc,
            "r":         float(r),
            "r_squared": float(r ** 2),
            "p_value":   fmt_pval(p_r),
            "n":         len(valid),
        })

        # ── Linear regression: trait ~ <anc> + age + sex ──
        model = run_ols(df_valid, outcome_col, anc, covariate_names)
        if model is not None and anc in model.params:
            ci    = model.conf_int()
            coef  = model.params[anc]
            se    = model.bse[anc]
            pval  = model.pvalues[anc]
            ci_lo = ci.loc[anc, 0]
            ci_hi = ci.loc[anc, 1]
            results.append({
                "test":      "Linear regression (adj.)",
                "ancestry":  anc,
                "beta":      float(coef),
                "CI_low":    float(ci_lo),
                "CI_high":   float(ci_hi),
                "SE":        float(se),
                "p_value":   fmt_pval(pval),
                "r_squared": float(model.rsquared),
                "n":         int(model.nobs),
            })


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

stats_df = pd.DataFrame(results)
if stats_df.empty:
    print(f"[analyze] WARNING: FieldID {meta['field_id']} — 0 test rows produced. "
          f"analysis_type={analysis_type}, n_rows_in_merged={len(merged)}. "
          f"Likely too few observations or all regressions failed. "
          f"Writing empty stats file with header so downstream rules don't crash.")
    # Write a minimal placeholder with the core columns so plot.py can read it cleanly
    placeholder = pd.DataFrame(columns=["test", "ancestry", "p_value"])
    placeholder.to_csv(snakemake.output.stats, sep="\t", index=False)
else:
    print(f"[analyze] FieldID {meta['field_id']} — {len(results)} test rows written.")
    stats_df.to_csv(snakemake.output.stats, sep="\t", index=False)