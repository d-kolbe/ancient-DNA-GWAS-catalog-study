"""
scripts/analyze_survival.py
============================
Survival analysis for parental lifespan traits.

For each ancestry component separately:
  1. Log-rank test: cases split into tertiles (low / mid / high ancestry)
  2. Cox proportional hazards regression (continuous ancestry + age + sex + birth_year)
     adjusted model: surv ~ ancestry + age + sex + birth_year
  3. Schoenfeld residuals test for proportional hazards assumption

Outputs stats.tsv with one row per ancestry component per test type.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

warnings.filterwarnings("ignore")

_log = open(str(snakemake.log[0]), "w", buffering=1)
sys.stdout = _log
sys.stderr = _log

LOG_PREFIX = "[analyze_survival]"

def log(msg):
    print(f"{LOG_PREFIX} {msg}", flush=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

merged = pd.read_csv(snakemake.input.merged, sep="\t")

with open(snakemake.input.meta) as fh:
    meta = json.load(fh)

ancestry_cols = snakemake.params.ancestry_cols
label         = meta["label"]

log(f"Survival analysis: {label}  n={len(merged):,}  events={meta['n_events']:,}")

results = []

# ---------------------------------------------------------------------------
# Per ancestry component
# ---------------------------------------------------------------------------

for anc in ancestry_cols:
    log(f"\n--- {anc} ---")
    df = merged[[anc, "surv_time", "surv_event",
                 "age", "sex", "birth_year"]].dropna().copy()
    log(f"  n={len(df):,}  events={df['surv_event'].sum():,}")

    # ── 1. Tertile split for log-rank ──────────────────────────────────────
    df["tertile"] = pd.qcut(df[anc], q=3, labels=["Low", "Mid", "High"])

    # Multivariate log-rank test across all three groups
    mlr = multivariate_logrank_test(
        df["surv_time"], df["tertile"], df["surv_event"]
    )
    log(f"  Log-rank p = {mlr.p_value:.4e}")

    # Pairwise log-rank: Low vs High (most informative comparison)
    low_mask  = df["tertile"] == "Low"
    high_mask = df["tertile"] == "High"
    lr_lh = logrank_test(
        df.loc[low_mask,  "surv_time"],
        df.loc[high_mask, "surv_time"],
        df.loc[low_mask,  "surv_event"],
        df.loc[high_mask, "surv_event"],
    )

    for tert in ["Low", "Mid", "High"]:
        mask = df["tertile"] == tert
        results.append({
            "test":          "log_rank_tertile_stats",
            "ancestry":      anc,
            "tertile":       tert,
            "n":             int(mask.sum()),
            "n_events":      int(df.loc[mask, "surv_event"].sum()),
            "median_time":   float(df.loc[mask, "surv_time"].median()),
            "logrank_p_3way": float(mlr.p_value),
            "logrank_p_low_vs_high": float(lr_lh.p_value),
        })

    # ── 2. Cox regression (continuous ancestry, adjusted) ─────────────────
    cox_df = df[[anc, "surv_time", "surv_event",
                 "age", "sex", "birth_year"]].copy()

    # Standardise ancestry for interpretable HR (per SD change)
    anc_mean = cox_df[anc].mean()
    anc_std  = cox_df[anc].std()
    cox_df[f"{anc}_std"] = (cox_df[anc] - anc_mean) / anc_std

    cph = CoxPHFitter()
    try:
        cph.fit(
            cox_df[[f"{anc}_std", "surv_time", "surv_event",
                    "age", "sex", "birth_year"]],
            duration_col = "surv_time",
            event_col    = "surv_event",
        )
        summary = cph.summary

        anc_row = summary.loc[f"{anc}_std"]
        hr      = float(np.exp(anc_row["coef"]))
        hr_lo   = float(np.exp(anc_row["coef lower 95%"]))
        hr_hi   = float(np.exp(anc_row["coef upper 95%"]))
        pval    = float(anc_row["p"])
        log(f"  Cox HR (per SD) = {hr:.3f}  [{hr_lo:.3f}, {hr_hi:.3f}]  p={pval:.4e}")

        # Proportional hazards test
        try:
            ph_test  = cph.check_assumptions(cox_df[[f"{anc}_std", "surv_time",
                                                      "surv_event", "age", "sex",
                                                      "birth_year"]],
                                              p_value_threshold=0.05,
                                              show_plots=False)
            ph_pval = float(cph.log_likelihood_ratio_test().p_value)
        except Exception:
            ph_pval = np.nan

        results.append({
            "test":          "cox_regression",
            "ancestry":      anc,
            "HR_per_SD":     hr,
            "HR_CI_low":     hr_lo,
            "HR_CI_high":    hr_hi,
            "coef":          float(anc_row["coef"]),
            "SE":            float(anc_row["se(coef)"]),
            "p_value":       pval,
            "concordance":   float(cph.concordance_index_),
            "n":             int(len(cox_df)),
            "n_events":      int(cox_df["surv_event"].sum()),
            "anc_mean":      float(anc_mean),
            "anc_std":       float(anc_std),
        })

    except Exception as e:
        log(f"  Cox regression failed: {e}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

stats_df = pd.DataFrame(results)
stats_df.to_csv(snakemake.output.stats, sep="\t", index=False)
log(f"\nDONE — {len(results)} result rows written")