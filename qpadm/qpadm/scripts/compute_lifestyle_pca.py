"""
scripts/compute_lifestyle_pca.py
=================================
Computes lifestyle principal components for use as covariates.

Steps
-----
1. Load each field from its source file (specified in config.lifestyle_pca.fields)
2. Ordinal-encode categorical variables using provided encoding maps
3. Standardise all variables to zero mean, unit variance
4. Iterative imputation (sklearn IterativeImputer — multivariate regression-based)
5. PCA on the imputed, standardised matrix
6. Write outputs:
     lifestyle_pcs.tsv        — eid + PC1..PCN scores (all samples with eid in admixture)
     lifestyle_pca_variance.png — scree plot
     lifestyle_pca_loadings.tsv — variable loadings per PC
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log = open(str(snakemake.log[0]), "w", buffering=1)
sys.stdout = _log
sys.stderr = _log

LOG_PREFIX = "[lifestyle_pca]"

def log(msg):
    print(f"{LOG_PREFIX} {msg}", flush=True)

def log_count(label, df):
    print(f"  {label}: {len(df):,} rows", flush=True)


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

pca_config      = snakemake.params.pca_config        # full lifestyle_pca config block
phenotype_files = snakemake.params.phenotype_files   # {group: {data, chars, cbkeys}}
n_pcs_keep      = pca_config.get("n_pcs_to_use", 10)
fields_config   = pca_config["fields"]               # {var_name: {field_id, source, encoding, ...}}

UKB_MISSING_CODES = {-1, -2, -3, -6, -7, -10, -11, -13}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_eid_col(columns):
    if "eid" in columns:
        return "eid"
    if "f_eid" in columns:
        return "f_eid"
    raise KeyError(f"No eid column found.")


def find_column(columns, field_id, instance=0):
    cols = list(columns)
    for candidate in [f"f_{field_id}_{instance}_0", f"f_{field_id}_{instance}"]:
        if candidate in cols:
            return candidate
    matches = [c for c in cols if c.startswith(f"f_{field_id}_")]
    if matches:
        return matches[0]
    raise KeyError(f"FieldID {field_id} not found. Tried f_{field_id}_{instance}_0 etc.")


# ---------------------------------------------------------------------------
# 1. Load admixture eids (defines our sample universe)
# ---------------------------------------------------------------------------

log("Loading admixture eids to define sample universe...")
admix_eids = pd.read_csv(snakemake.input.admixture, sep="\t", usecols=["eid"])["eid"]
log(f"  Admixture eids: {len(admix_eids):,}")

# ---------------------------------------------------------------------------
# 2. Load each field
# ---------------------------------------------------------------------------

log("Loading lifestyle fields...")

# Group fields by source file to minimise file reads
fields_by_source = {}
for var_name, spec in fields_config.items():
    source = spec["source"]
    if source not in fields_by_source:
        fields_by_source[source] = []
    fields_by_source[source].append((var_name, spec))

all_series = {}   # {var_name: pd.Series indexed by eid}

for source_group, field_list in fields_by_source.items():
    data_file = phenotype_files[source_group]["data"]
    header_df = pd.read_csv(data_file, sep="\t", nrows=0)
    eid_col   = get_eid_col(header_df.columns)

    cols_to_load = [eid_col]
    col_map = {}  # raw_col → var_name

    for var_name, spec in field_list:
        fid      = spec["field_id"]
        instance = spec.get("instance", 0)
        try:
            raw_col = find_column(header_df.columns, fid, instance)
            cols_to_load.append(raw_col)
            col_map[raw_col] = var_name
            log(f"  {var_name} (FieldID {fid}) → '{raw_col}' in {Path(data_file).name}")
        except KeyError as e:
            log(f"  WARNING: {e} — skipping {var_name}")

    df = pd.read_csv(data_file, sep="\t", usecols=cols_to_load, low_memory=False)
    df = df.rename(columns={eid_col: "eid"})

    for raw_col, var_name in col_map.items():
        spec   = fields_config[var_name]
        series = pd.to_numeric(df.set_index("eid")[raw_col], errors="coerce")

        # Replace UKB special missing codes
        series = series.where(~series.isin(UKB_MISSING_CODES), np.nan)

        # Apply ordinal encoding if specified
        encoding = spec.get("encoding")
        if encoding:
            # encoding is {original_value: new_value}
            series = series.map({float(k): float(v) for k, v in encoding.items()})

        all_series[var_name] = series

# ---------------------------------------------------------------------------
# 3. Assemble matrix
# ---------------------------------------------------------------------------

log("Assembling feature matrix...")
feat_df = pd.DataFrame(all_series)
feat_df.index.name = "eid"
feat_df = feat_df.reset_index()

# Keep only samples in admixture file
feat_df = feat_df[feat_df["eid"].isin(admix_eids)].copy()
log_count("After filtering to admixture eids", feat_df)

var_names = list(all_series.keys())
X_raw = feat_df[var_names].values.astype(float)

# Report missingness
log("Missingness per variable:")
for i, vn in enumerate(var_names):
    n_miss = np.isnan(X_raw[:, i]).sum()
    pct    = 100 * n_miss / len(X_raw)
    log(f"  {vn}: {n_miss:,} missing ({pct:.1f}%)")

# ---------------------------------------------------------------------------
# 4. Standardise (before imputation — imputer uses standardised scale)
# ---------------------------------------------------------------------------

log("Standardising variables...")
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_raw)
# Restore NaNs after scaling (StandardScaler propagates them)
nan_mask  = np.isnan(X_raw)
X_scaled[nan_mask] = np.nan

# ---------------------------------------------------------------------------
# 5. Iterative imputation
# ---------------------------------------------------------------------------

log("Running iterative imputation (IterativeImputer with BayesianRidge)...")
log("  Fitting on subsample of 50,000 then transforming full dataset to save memory...")

# Subsample for fitting — BayesianRidge is linear so this is statistically sound
rng         = np.random.default_rng(42)
n_fit       = min(50_000, len(X_scaled))
fit_idx     = rng.choice(len(X_scaled), size=n_fit, replace=False)
X_fit       = X_scaled[fit_idx]

imputer = IterativeImputer(
    estimator    = BayesianRidge(),
    max_iter     = 10,
    random_state = 42,
    verbose      = 0,
)
imputer.fit(X_fit)
log(f"  Imputer fitted on {n_fit:,} samples. Transforming full dataset...")

# Transform in chunks to avoid memory spikes
chunk_size = 50_000
chunks     = []
for start in range(0, len(X_scaled), chunk_size):
    chunks.append(imputer.transform(X_scaled[start:start + chunk_size]))
    log(f"  Transformed {min(start + chunk_size, len(X_scaled)):,} / {len(X_scaled):,}")
X_imputed = np.vstack(chunks)
log(f"  Imputation complete. Matrix shape: {X_imputed.shape}")

# ---------------------------------------------------------------------------
# 6. PCA
# ---------------------------------------------------------------------------

log(f"Running PCA (retaining all components, will use {n_pcs_keep})...")
pca    = PCA(random_state=42)
X_pcs  = pca.fit_transform(X_imputed)

var_explained     = pca.explained_variance_ratio_
cumvar_explained  = np.cumsum(var_explained)

log("Variance explained per PC:")
for i, (v, cv) in enumerate(zip(var_explained[:20], cumvar_explained[:20])):
    marker = " ← keeping" if i < n_pcs_keep else ""
    log(f"  PC{i+1:2d}: {v*100:.2f}%  (cumulative: {cv*100:.1f}%){marker}")

pcs_80 = int(np.searchsorted(cumvar_explained, 0.80)) + 1
log(f"  PCs needed for 80% variance: {pcs_80}")
log(f"  Using first {n_pcs_keep} PCs as covariates")

# ---------------------------------------------------------------------------
# 7. Write PC scores
# ---------------------------------------------------------------------------

log("Writing PC scores...")
pc_cols  = [f"lifestyle_PC{i+1}" for i in range(n_pcs_keep)]
pcs_df   = pd.DataFrame(X_pcs[:, :n_pcs_keep], columns=pc_cols)
pcs_df.insert(0, "eid", feat_df["eid"].values)
pcs_df.to_csv(snakemake.output.pcs, sep="\t", index=False)
log(f"  Written: {snakemake.output.pcs}  ({len(pcs_df):,} rows, {n_pcs_keep} PCs)")

# ---------------------------------------------------------------------------
# 8. Write loadings
# ---------------------------------------------------------------------------

log("Writing loadings...")
loadings_df = pd.DataFrame(
    pca.components_[:n_pcs_keep, :].T,
    index   = var_names,
    columns = pc_cols,
)
loadings_df.index.name = "variable"
loadings_df.to_csv(snakemake.output.loadings, sep="\t")

# Top loadings per PC
log("Top 5 loadings per PC:")
for pc in pc_cols:
    top = loadings_df[pc].abs().nlargest(5).index.tolist()
    log(f"  {pc}: {', '.join(top)}")

# ---------------------------------------------------------------------------
# 9. Scree plot
# ---------------------------------------------------------------------------

log("Generating scree plot...")

# theme
BG_COLOR   = "#ffffff"
TEXT_COLOR = "#1a1a1a"
GRID_COLOR = "#dddddd"
BAR_COLOR  = "#2a6bbf"
CUM_COLOR  = "#c0392b"

import matplotlib.pyplot as plt
plt.style.use("default")

n_show = min(30, len(var_explained))
fig, ax1 = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
ax1.set_facecolor("#f7f7f7")
fig.patch.set_facecolor(BG_COLOR)

xvals = np.arange(1, n_show + 1)
ax1.bar(xvals, var_explained[:n_show] * 100,
        color=BAR_COLOR, alpha=0.8, label="% variance per PC")
ax1.axvline(n_pcs_keep + 0.5, color=CUM_COLOR, lw=1.5,
            linestyle="--", label=f"Retained ({n_pcs_keep} PCs)")

ax2 = ax1.twinx()
ax2.plot(xvals, cumvar_explained[:n_show] * 100,
         color=CUM_COLOR, lw=2, marker="o", ms=4, label="Cumulative %")
ax2.axhline(80, color="#2e7d50", lw=1, linestyle=":", alpha=0.7, label="80% threshold")
ax2.set_ylabel("Cumulative variance explained (%)", color=TEXT_COLOR, fontsize=10)
ax2.tick_params(colors=TEXT_COLOR)
ax2.set_ylim(0, 105)

ax1.set_xlabel("Principal Component", color=TEXT_COLOR, fontsize=10)
ax1.set_ylabel("Variance explained (%)", color=TEXT_COLOR, fontsize=10)
ax1.set_title("Lifestyle PCA — Scree Plot", color=TEXT_COLOR, fontsize=12, fontweight="bold")
ax1.tick_params(colors=TEXT_COLOR)
ax1.set_xticks(xvals)

for spine in ax1.spines.values():
    spine.set_edgecolor(GRID_COLOR)
for spine in ax2.spines.values():
    spine.set_edgecolor(GRID_COLOR)
ax1.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6, axis="y")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           fontsize=8, facecolor="#f0f0f0", edgecolor="#bbbbbb",
           labelcolor=TEXT_COLOR, loc="center right")

plt.tight_layout()
plt.savefig(snakemake.output.variance_plot, dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
log(f"  Scree plot: {snakemake.output.variance_plot}")

# ---------------------------------------------------------------------------
# 10. PC1 vs PC2 scatter plot
# ---------------------------------------------------------------------------

log("Generating PC1 vs PC2 scatter plot...")

# Subsample for plotting — 50k points is plenty, full 490k would be slow to render
plot_n   = min(50_000, len(X_pcs))
plot_idx = np.random.default_rng(99).choice(len(X_pcs), size=plot_n, replace=False)
pc1_vals = X_pcs[plot_idx, 0]
pc2_vals = X_pcs[plot_idx, 1]

# Colour by smoking status (first variable, index 0) — gives biological meaning to the plot
# Clip to reasonable percentile range for colour scaling
smoking_raw = X_raw[plot_idx, var_names.index("smoking_status")]
valid_mask  = ~np.isnan(smoking_raw)

fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG_COLOR)
ax.set_facecolor("#f7f7f7")
fig.patch.set_facecolor(BG_COLOR)

# Background: all points in neutral grey
ax.scatter(pc1_vals, pc2_vals,
           c="#cccccc", s=1.5, alpha=0.4, rasterized=True, linewidths=0)

# Overlay: colour by smoking status where available
scatter = ax.scatter(
    pc1_vals[valid_mask], pc2_vals[valid_mask],
    c=smoking_raw[valid_mask], cmap="RdYlGn_r",
    s=1.5, alpha=0.5, rasterized=True, linewidths=0,
    vmin=0, vmax=2,
)
cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(["Never", "Previous", "Current"])
cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
cbar.outline.set_edgecolor(GRID_COLOR)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR, fontsize=8)
cbar.set_label("Smoking status", color=TEXT_COLOR, fontsize=9)

pct1 = var_explained[0] * 100
pct2 = var_explained[1] * 100
ax.set_xlabel(f"PC1  ({pct1:.1f}% variance)", color=TEXT_COLOR, fontsize=11)
ax.set_ylabel(f"PC2  ({pct2:.1f}% variance)", color=TEXT_COLOR, fontsize=11)
ax.set_title(f"Lifestyle PCA  ·  PC1 vs PC2  ·  n={plot_n:,} (subsampled, coloured by smoking)",
             color=TEXT_COLOR, fontsize=12, fontweight="bold")
ax.tick_params(colors=TEXT_COLOR, labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COLOR)
ax.grid(True, color=GRID_COLOR, linewidth=0.4, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(snakemake.output.pca_plot, dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
log(f"  PCA scatter plot: {snakemake.output.pca_plot}")

log("DONE.")