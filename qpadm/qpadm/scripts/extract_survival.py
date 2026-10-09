"""
scripts/extract_survival.py
============================
Builds survival outcome datasets for parental lifespan analysis.

For each parent (father / mother):
  - alive == 1  → time = current age,    event = 0  (censored, still alive)
  - alive == 0  → time = age at death,   event = 1  (event occurred)
  - alive == -1/-3 → excluded (unknown status)

For combined traits (mean / max):
  - Requires BOTH parents to have a valid (known) survival record
  - mean: average of father_time and mother_time
  - max:  maximum of father_time and mother_time
  - event for combined: 1 if the selected parent (mean→any died, max→the longer-lived died)

Covariates loaded: age at recruitment (21022), sex (31), birth year (34)
All from popchar_data.tsv.

Output: merged.tsv + trait_meta.json
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log = open(str(snakemake.log[0]), "w", buffering=1)
sys.stdout = _log
sys.stderr = _log

LOG_PREFIX = "[extract_survival]"

def log(msg):
    print(f"{LOG_PREFIX} {msg}", flush=True)

def log_count(label, df, extra=""):
    msg = f"  {label}: {len(df):,} rows"
    if extra:
        msg += f"  ({extra})"
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

trait_info      = snakemake.params.trait_info
ancestry_rename = snakemake.params.ancestry_rename
covariates      = snakemake.params.covariates   # {name: {field_id, data_file}}

survival_type   = trait_info["survival_type"]   # father | mother | mean | max
field_id        = trait_info["field_id"]

UKB_MISSING = {-1, -3, -7}

ALIVE_YES = 1
ALIVE_NO  = 0

log(f"Survival trait: {field_id} ({trait_info['label']}) — type={survival_type}")

# ---------------------------------------------------------------------------
# 1. Admixture
# ---------------------------------------------------------------------------

admix = pd.read_csv(snakemake.input.admixture, sep="\t")
admix = admix.rename(columns=ancestry_rename)
ancestry_cols = list(ancestry_rename.values())

n_before = len(admix)
valid_mask = pd.Series(True, index=admix.index)
for anc in ancestry_cols:
    valid_mask &= admix[anc].between(0, 1)
admix = admix[valid_mask].reset_index(drop=True)
log(f"  Admixture: {n_before:,} → {len(admix):,} after [0,1] filter")

# ---------------------------------------------------------------------------
# 2. Load parental survival fields from touchscreen
# ---------------------------------------------------------------------------

ts_file = snakemake.input.touchscreen

def load_parent(ts_file, alive_fid, death_fid, current_age_fid, label):
    """
    Returns DataFrame with columns: eid, {label}_time, {label}_event
    Excludes samples with unknown alive status.
    """
    header = pd.read_csv(ts_file, sep="\t", nrows=0)

    def find_col(fid):
        for c in [f"f_{fid}_0_0", f"f_{fid}_0"]:
            if c in header.columns:
                return c
        matches = [c for c in header.columns if c.startswith(f"f_{fid}_")]
        if matches:
            return matches[0]
        raise KeyError(f"FieldID {fid} not found in touchscreen")

    alive_col      = find_col(alive_fid)
    death_col      = find_col(death_fid)
    cur_age_col    = find_col(current_age_fid)

    log(f"  {label}: alive={alive_col}, death={death_col}, current_age={cur_age_col}")

    df = pd.read_csv(ts_file, sep="\t",
                     usecols=["eid", alive_col, death_col, cur_age_col],
                     low_memory=False)
    df = df.rename(columns={
        alive_col:    "alive",
        death_col:    "age_death",
        cur_age_col:  "age_current",
    })

    df["alive"]       = pd.to_numeric(df["alive"],       errors="coerce")
    df["age_death"]   = pd.to_numeric(df["age_death"],   errors="coerce")
    df["age_current"] = pd.to_numeric(df["age_current"], errors="coerce")

    # Exclude unknown alive status
    n_before = len(df)
    df = df[df["alive"].isin([ALIVE_YES, ALIVE_NO])].copy()
    log(f"  {label}: {n_before:,} → {len(df):,} after removing unknown alive status")

    # Build time and event
    df[f"{label}_event"] = (df["alive"] == ALIVE_NO).astype(int)
    df[f"{label}_time"]  = np.where(
        df["alive"] == ALIVE_NO,
        df["age_death"],
        df["age_current"]
    )

    # Remove rows where time is missing or implausible
    df = df[
        df[f"{label}_time"].notna() &
        (df[f"{label}_time"] > 0) &
        (df[f"{label}_time"] < 130)
    ].copy()
    log(f"  {label}: {len(df):,} rows with valid survival time")

    return df[["eid", f"{label}_time", f"{label}_event"]]


father_df = load_parent(ts_file,
    alive_fid=1797, death_fid=1807, current_age_fid=2946, label="father")
mother_df = load_parent(ts_file,
    alive_fid=1835, death_fid=3526, current_age_fid=1845, label="mother")

# ---------------------------------------------------------------------------
# 3. Build survival outcome based on trait type
# ---------------------------------------------------------------------------

if survival_type == "father":
    surv_df = father_df.rename(columns={
        "father_time":  "surv_time",
        "father_event": "surv_event",
    })

elif survival_type == "mother":
    surv_df = mother_df.rename(columns={
        "mother_time":  "surv_time",
        "mother_event": "surv_event",
    })

elif survival_type in ("mean", "max"):
    # Need both parents
    both = father_df.merge(mother_df, on="eid", how="inner")
    log_count(f"Both parents available", both)

    if survival_type == "mean":
        both["surv_time"]  = (both["father_time"] + both["mother_time"]) / 2
        # Event = 1 if either parent has died (at least one event observed)
        both["surv_event"] = ((both["father_event"] == 1) | (both["mother_event"] == 1)).astype(int)
    else:  # max
        # Max lifespan = the longer-lived parent
        # Time = max of the two times
        # Event = event of whichever parent lived longest
        #   If father_time > mother_time: use father's event
        #   Else: use mother's event
        both["surv_time"]  = both[["father_time", "mother_time"]].max(axis=1)
        both["surv_event"] = np.where(
            both["father_time"] >= both["mother_time"],
            both["father_event"],
            both["mother_event"]
        )

    surv_df = both[["eid", "surv_time", "surv_event"]]

log_count(f"Survival outcome built ({survival_type})", surv_df,
          f"events={surv_df['surv_event'].sum():,}, "
          f"censored={(surv_df['surv_event']==0).sum():,}")

# ---------------------------------------------------------------------------
# 4. Load covariates (age, sex, birth_year — all from popchar)
# ---------------------------------------------------------------------------

popchar_file = snakemake.input.popchar

def find_col_in(filepath, fid, instance=0):
    header = pd.read_csv(filepath, sep="\t", nrows=0)
    for c in [f"f_{fid}_{instance}_0", f"f_{fid}_{instance}", f"f_{fid}_0_0"]:
        if c in header.columns:
            return c
    matches = [c for c in header.columns if c.startswith(f"f_{fid}_")]
    if matches:
        return matches[0]
    # Also check plain 'eid' style
    raise KeyError(f"FieldID {fid} not found in {filepath}")

age_col        = find_col_in(popchar_file, 21022)
sex_col        = find_col_in(popchar_file, 31)
birthyear_col  = find_col_in(popchar_file, 34)

log(f"  Covariates: age={age_col}, sex={sex_col}, birth_year={birthyear_col}")

cov_df = pd.read_csv(popchar_file, sep="\t",
                     usecols=["eid", age_col, sex_col, birthyear_col],
                     low_memory=False)
cov_df = cov_df.rename(columns={
    age_col:       "age",
    sex_col:       "sex",
    birthyear_col: "birth_year",
})

# Recode sex: already 0/1 in popchar
unique_sex = set(cov_df["sex"].dropna().unique())
if unique_sex <= {1, 2}:
    cov_df["sex"] = cov_df["sex"].map({1: 0, 2: 1})
    log("  Sex recoded 1/2 → 0/1")
else:
    log("  Sex already 0/1")

# ---------------------------------------------------------------------------
# 5. Merge everything
# ---------------------------------------------------------------------------

merged = admix[["eid"] + ancestry_cols].copy()
log_count("Start (admixture)", merged)

merged = merged.merge(surv_df, on="eid", how="inner")
log_count("After merge with survival outcome", merged,
          f"lost {len(admix) - len(merged):,} eids without parental data")

merged = merged.merge(cov_df, on="eid", how="left")
log_count("After merge with covariates", merged)

# Drop rows missing any required column
required = ["eid"] + ancestry_cols + ["surv_time", "surv_event", "age", "sex", "birth_year"]
n_before = len(merged)
for col in required:
    n_miss = merged[col].isna().sum()
    if n_miss > 0:
        log(f"  Missing '{col}': {n_miss:,}")
merged = merged.dropna(subset=required)
log_count("After dropna on required columns", merged,
          f"dropped {n_before - len(merged):,}")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------

merged.to_csv(snakemake.output.merged, sep="\t", index=False)

meta = {
    "field_id":      field_id,
    "label":         trait_info["label"],
    "survival_type": survival_type,
    "analysis_type": "survival",
    "n_total":       int(len(merged)),
    "n_events":      int(merged["surv_event"].sum()),
    "n_censored":    int((merged["surv_event"] == 0).sum()),
    "median_time":   float(merged["surv_time"].median()),
}

with open(snakemake.output.meta, "w") as fh:
    json.dump(meta, fh, indent=2)

log(f"DONE — n={len(merged):,}, events={meta['n_events']:,}, "
    f"censored={meta['n_censored']:,}, median_time={meta['median_time']:.1f}")