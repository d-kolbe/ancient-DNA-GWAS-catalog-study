#!/usr/bin/env python3
"""
plot_ars_forest.py
==================
ARS forest plot — individual bootstrap CI only.

Same category/trait selection as the directionality panel plot:
  - Top 6 categories from category_enrichment.txt (by enrichment p-value)
  - Top 10 traits per category from trait_directionality_summary.txt (by ttest_pval)
  - Same 2x3 panel arrangement

One point marker per trait, at the midpoint of the bootstrap 95% CI
(the CI bar itself is no longer drawn — with n_boot_samples this large the
bootstrap distribution is close to Gaussian, so the bar's length is ~constant
across every trait and carries no information; only its position does).
Colour: EF>WHG (warm red), WHG>EF (steel blue), ns (grey).
Stars to the right.
"""

import os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Palette — matches directionality bootstrap figure
COLOR_EF  = '#E8957E'
COLOR_WHG = '#6B9EC8'
COLOR_NS  = '#CCCCCC'
BG        = 'white'
MUTED     = '#888888'
GRID_CLR  = '#ebebeb'
SPINE_CLR = '#d0d0d0'

_SEMANTIC = {
    'neuropsychiatric': '#7B68C8', 'neurological': '#7B68C8',
    'psychiatric':      '#9480D4', 'sleep':        '#9B88D8',
    'lipids':           '#C8A820', 'metabolic':    '#C8A820',
    'gastrointestinal': '#DD7030', 'inflammatory': '#C05A18',
    'anthropometry':    '#4C72B0', 'body':         '#4C72B0',
    'sensory':          '#2A9EA8', 'hearing':      '#2A9EA8',
    'behavioral':       '#8172B2', 'substance':    '#9482C0',
    'respiratory':      '#5898C8', 'cancer':       '#505050',
}
_FALLBACK = ['#4C72B0', '#DD8452', '#55A868', '#8172B2', '#937860', '#CCB974']


def _norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def _stars(p):
    if pd.isna(p): return ''
    # The bootstrap p-value floor (1/n_bootstrap, e.g. 0.001 for 1000 iterations)
    # is only ever produced when ALL bootstrap draws agree in sign (0 crossings),
    # i.e. the true p-value is <= the floor, not exactly at a boundary that should
    # be excluded. Real (non-floored) p-values are always multiples of 2/n_bootstrap
    # and can never land exactly on 0.001, so "<=" here only affects floored cases.
    if p <= 0.001: return '***'
    if p < 0.01:   return '**'
    if p < 0.05:   return '*'
    return ''


# =============================================================================
# Data loading — with full diagnostics
# =============================================================================

def get_top_categories(enrichment_file, n_cats):
    enr     = pd.read_csv(enrichment_file, sep='\t', comment='#')
    cat_col = next((c for c in ['category', 'Category', 'trait_category']
                    if c in enr.columns), None)
    if cat_col is None:
        raise ValueError(f"No category column in {enrichment_file}. "
                         f"Columns: {list(enr.columns)}")
    p_col = next((c for c in ['mannwhitneyu_pval', 'mw_p_value', 'p_value',
                               'kolmogorov_smirnov_pval', 'ttest_pval']
                  if c in enr.columns), None)
    if p_col is None:
        raise ValueError(f"No p-value column in {enrichment_file}.")
    cats = (enr.dropna(subset=[p_col])
               .sort_values(p_col)
               .head(n_cats)[cat_col]
               .tolist())
    print(f"[ars_forest] Top-{n_cats} categories: {cats}", file=sys.stderr)
    return cats


def get_ordered_traits(directionality_file, ordered_cats, n_traits):
    df          = pd.read_csv(directionality_file, sep='\t', comment='#')
    dir_cats    = df['category'].unique()
    norm_to_dir = {_norm(c): c for c in dir_cats}
    print(f"[ars_forest] Directionality categories: {list(dir_cats)}", file=sys.stderr)

    result = {}
    for cat in ordered_cats:
        dir_cat = cat if cat in dir_cats else norm_to_dir.get(_norm(cat))
        if dir_cat is None:
            print(f"[ars_forest] WARNING: '{cat}' not in directionality file", file=sys.stderr)
            result[cat] = []
            continue
        cat_df = df[df['category'] == dir_cat].copy()
        # Filter by minimum SNPs extracted
        if 'n_snps_extracted' in cat_df.columns:
            before = len(cat_df)
            cat_df = cat_df[pd.to_numeric(cat_df['n_snps_extracted'], errors='coerce').fillna(0) >= 20]
            print(f"[ars_forest]   n_snps_extracted>=20 filter: {before}->{len(cat_df)}", file=sys.stderr)
        top = (cat_df.dropna(subset=['ttest_pval'])
                     .sort_values('ttest_pval')
                     .head(n_traits)['trait']
                     .tolist())
        result[cat] = top
        print(f"[ars_forest] {cat}: {len(top)} traits: {top}", file=sys.stderr)
    return result


def load_contrasts(results_dir, trait_manifest_path, traits_per_cat,
                   ci_level, exclude_traits=None, show_loo=True,
                   denominator='std'):
    lo = (1 - ci_level) / 2 * 100
    hi = 100 - lo

    manifest = pd.read_csv(trait_manifest_path, sep='\t')
    if 'original_trait' not in manifest.columns and 'trait' in manifest.columns:
        manifest = manifest.rename(columns={'trait': 'original_trait'})

    norm_to_safe = {_norm(r['original_trait']): r['safe_trait']
                    for _, r in manifest.iterrows()}
    safe_to_nvar = dict(zip(manifest['safe_trait'],
                            manifest.get('n_variants', [None]*len(manifest))))

    exclude_set = set(exclude_traits or [])
    all_traits = list(dict.fromkeys(
        t for ts in traits_per_cat.values() for t in ts
        if t not in exclude_set
    ))
    if exclude_set:
        print(f"[ars_forest] Excluding traits: {exclude_set}", file=sys.stderr)
    contrasts = {}

    for trait in all_traits:
        norm_t = _norm(trait)
        safe   = norm_to_safe.get(norm_t)

        if safe is None:
            print(f"[ars_forest] MISSING MANIFEST: '{trait}' "
                  f"(norm='{norm_t}') — not in trait_manifest.tsv", file=sys.stderr)
            continue

        npz = os.path.join(results_dir, safe, 'bootstrap_distributions.npz')
        if not os.path.exists(npz):
            print(f"[ars_forest] MISSING NPZ: '{trait}' -> results/{safe}/",
                  file=sys.stderr)
            continue

        try:
            data = np.load(npz, allow_pickle=True)
        except Exception as e:
            print(f"[ars_forest] NPZ ERROR: '{trait}': {e}", file=sys.stderr)
            continue

        anc = [str(a) for a in data['ancestry_names']]

        # Read pre-computed standardised contrast and CI.
        # denominator='std'   -> std_bootstrap (now variant-resampled, fixed)
        # denominator='persd' -> person_sd (population SD, no resampling)
        if denominator == 'persd' and 'observed_contrast_persd' in data:
            contrast_mean = float(data['observed_contrast_persd'])
            ci_lo         = float(data['ci_lower_persd'])
            ci_hi         = float(data['ci_upper_persd'])
            std_boot      = float(data['person_sd'])
        elif 'observed_contrast_std' in data:
            contrast_mean = float(data['observed_contrast_std'])
            ci_lo         = float(data['ci_lower_std'])
            ci_hi         = float(data['ci_upper_std'])
            std_boot      = float(data['std_bootstrap'])
        else:
            # Legacy fallback: compute from z_scores
            if 'EF' not in anc or 'WHG' not in anc:
                continue
            ei, wi = anc.index('EF'), anc.index('WHG')
            z_ind = data['z_scores'][:, ei] - data['z_scores'][:, wi]
            v_ind = z_ind[~np.isnan(z_ind)]
            if len(v_ind) < 10:
                continue
            contrast_mean = float(np.mean(v_ind))
            ci_lo         = float(np.percentile(v_ind, lo))
            ci_hi         = float(np.percentile(v_ind, hi))
            std_boot      = 1.0

        if np.isnan(contrast_mean):
            print(f"[ars_forest] NaN contrast: '{trait}' — skipped", file=sys.stderr)
            continue

        # Empirical p-value from raw bootstrap contrast distribution
        if 'boot_raw_contrast' in data:
            raw_boot = data['boot_raw_contrast']
            valid    = raw_boot[~np.isnan(raw_boot)]
            p_val    = 2.0 * min(float(np.mean(valid <= 0)), float(np.mean(valid >= 0)))
            p_val    = max(p_val, 1.0 / len(valid)) if len(valid) > 0 else np.nan
        else:
            p_val = np.nan

        entry = {
            # Point plotted on the forest plot: the actual full-sample point
            # estimate (observed_contrast_std or observed_contrast_persd).
            # Previously this plotted the CI midpoint instead, to guard
            # against the point estimate disagreeing in sign with a buggy
            # (person-level-resampled) CI. That bug is now fixed — the CI is
            # variant-resampled and properly centred on the real point
            # estimate — so there is no more reason to plot anything other
            # than the actual point estimate, and doing so avoids a separate
            # problem: the CI midpoint can sit far from the true estimate
            # whenever the interval is wide/asymmetric (which happens for
            # traits with few contributing variants).
            'contrast_mean': contrast_mean,
            'ci_lo':         ci_lo,
            'ci_hi':         ci_hi,
            'std_bootstrap': std_boot,
            'p_value':       p_val,
            'n_variants':    safe_to_nvar.get(safe),
            'loo_contrasts': np.array([]),
            'loo_labels':    [],
        }

        # LOO contrasts, scaled to match whichever denominator was selected
        # above. Skipped entirely when show_loo=False: per-SNP removal shifts
        # can be on a very different scale than either denominator (a single
        # large-effect SNP's LOO shift can dwarf the CI), so they are not
        # drawn by default. See draw_panel().
        loo_key = 'loo_contrasts_persd' if denominator == 'persd' else 'loo_contrasts'
        if show_loo and loo_key in data:
            loo = data[loo_key].astype(np.float64)
            if len(loo) > 0 and not np.all(np.isnan(loo)):
                entry['loo_contrasts'] = loo

                # Build gene-rsID labels from weights file (sorted by CHROM:POS)
                labels = []
                try:
                    wpath = os.path.join(results_dir, '..', 'variant_lists',
                                         'weights', f'{safe}_weights.tsv')
                    wpath = os.path.normpath(wpath)
                    wdf   = (pd.read_csv(wpath, sep='\t')
                               .drop_duplicates(subset=['CHROM','POS'])
                               .sort_values(['CHROM','POS'])
                               .reset_index(drop=True))
                    # Filter to variants actually in npz (same join as bootstrap)
                    if 'loo_chroms' in data and 'loo_pos' in data:
                        npz_keys = set(zip(data['loo_chroms'].tolist(),
                                           data['loo_pos'].tolist()))
                        wdf = wdf[wdf.apply(
                            lambda r: (int(r['CHROM']),int(r['POS'])) in npz_keys,
                            axis=1)].sort_values(['CHROM','POS']).reset_index(drop=True)
                    rsids = wdf.get('rsID', pd.Series([''] * len(wdf))).fillna('').tolist()
                    genes = wdf.get('gene', pd.Series([''] * len(wdf))).fillna('').tolist()
                    for rsid, gene in zip(rsids, genes):
                        rsid = rsid.split(';')[0].strip()
                        gene = gene.split(',')[0].strip()
                        if gene and rsid:
                            labels.append(f'{gene}-{rsid}')
                        elif gene:
                            labels.append(gene)
                        elif rsid:
                            labels.append(rsid)
                        else:
                            labels.append('')
                except Exception as e:
                    print(f"[ars_forest] Could not load weights for {safe}: {e}",
                          file=sys.stderr)
                entry['loo_labels'] = labels

        contrasts[trait] = entry

    print(f"[ars_forest] Contrasts loaded: {len(contrasts)}/{len(all_traits)} traits",
          file=sys.stderr)
    return contrasts


# =============================================================================
# Draw one panel
# =============================================================================

def draw_panel(ax, cat_name, ordered_traits, contrasts,
               x_lo, x_hi, show_xlabels, max_label=22, show_loo=True,
               label_fontsize=6.5):

    plot_traits = [t for t in ordered_traits if t in contrasts]

    if not plot_traits:
        ax.text(0.5, 0.5, 'No ARS data\n(see stderr for details)',
                ha='center', va='center', transform=ax.transAxes,
                color=MUTED, fontsize=9, multialignment='center')
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(-0.75, 0.75)
        ax.set_yticks([])
        title = str(cat_name).replace('_', ' ')
        ax.set_title(title[:51], color='#1a1a1a', fontsize=8.5,
                     fontweight='normal', pad=4, loc='left')
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(SPINE_CLR); sp.set_linewidth(0.8)
        ax.set_facecolor(BG)
        return

    # Sort ascending by the plotted point (most WHG-enriched at top)
    plot_traits = sorted(plot_traits, key=lambda t: contrasts[t]['contrast_mean'])
    n = len(plot_traits)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.75, n - 0.25)
    ax.set_facecolor(BG)

    serif = 0.08

    for i, trait in enumerate(plot_traits):
        c = contrasts[trait]
        ci_lo = c['ci_lo']
        ci_hi = c['ci_hi']
        # Point plotted: the actual full-sample point estimate (see
        # load_contrasts note on why this replaced the CI-midpoint version).
        d     = c['contrast_mean']
        p_val = c['p_value']
        n_var = c.get('n_variants')

        sig   = (ci_lo > 0) or (ci_hi < 0)
        color = (COLOR_EF if d > 0 else COLOR_WHG) if sig else COLOR_NS

        # Alternating row shading
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f4f4f4', linewidth=0, zorder=0)

        # Label
        name  = str(trait)
        name  = (name[:max_label - 1] + '\u2026') if len(name) > max_label else name
        n_str = f' (n={int(n_var)})' if n_var is not None and pd.notna(n_var) else ''
        stars = _stars(p_val)
        lbl   = f'{name}{n_str}  {stars}'.rstrip() if stars else f'{name}{n_str}'
        ax.text(x_lo + (x_hi - x_lo) * 0.018, i + 0.30,
                lbl, ha='left', va='bottom',
                fontsize=label_fontsize, color='#363636', alpha=0.88,
                zorder=5, clip_on=True)

        # Point marker + CI whisker: centered on the row when the LOO strip
        # is hidden; offset upward (leaving room below for the dot strip)
        # when it's shown.
        y_pt = i + 0.15 if show_loo else i
        ax.plot([ci_lo, ci_hi], [y_pt, y_pt], color=color, lw=1.6,
                solid_capstyle='butt', alpha=0.75, zorder=3)
        for xe in [ci_lo, ci_hi]:
            ax.plot([xe, xe], [y_pt - serif, y_pt + serif],
                    color=color, lw=1.6, alpha=0.75, zorder=3)

        # LOO dot strip — lower position. Disabled by default (show_loo=False):
        # per-SNP leave-one-out shifts are on a different scale than the
        # person-resampling std_bootstrap denominator used for the CI, so a
        # single large-effect SNP can land dozens of "SD units" away and blow
        # up the figure canvas (its label text isn't clipped by the axes).
        if show_loo:
            loo  = c.get('loo_contrasts', np.array([]))
            lbls = c.get('loo_labels', [])
            y_loo = i - 0.15
            if len(loo) > 0:
                valid_loo = loo[~np.isnan(loo)]

                ax.scatter(valid_loo, np.full(len(valid_loo), y_loo),
                           s=6, color=color, alpha=0.35,
                           linewidths=0, zorder=3)

                if len(valid_loo) > 0:
                    # Always label most extreme LOO dot in opposite direction
                    if d > 0:
                        extreme_idx = int(np.nanargmin(loo))
                    else:
                        extreme_idx = int(np.nanargmax(loo))
                    extreme_val = loo[extreme_idx]

                    ax.scatter([extreme_val], [y_loo],
                               s=22, color=color, alpha=0.95,
                               linewidths=0.8,
                               edgecolors='white', zorder=5)
                    lbl_txt = (lbls[extreme_idx]
                               if lbls and extreme_idx < len(lbls) and lbls[extreme_idx]
                               else '')
                    if lbl_txt:
                        ax.text(extreme_val,
                                y_loo - serif * 1.8,
                                lbl_txt,
                                ha='center', va='top',
                                fontsize=5.0, color=color,
                                alpha=0.9, zorder=5,
                                style='italic')

    ax.axvline(0, color='#c8c8c8', lw=0.9, linestyle='--', zorder=1)
    ax.set_yticks([])

    title = str(cat_name).replace('_', ' ')
    if len(title) > 52: title = title[:51] + '\u2026'
    ax.set_title(title, color='#1a1a1a', fontsize=8.5,
                 fontweight='normal', pad=4, loc='left')

    ax.tick_params(axis='x', labelsize=7.5 if show_xlabels else 0,
                   colors=MUTED, length=3, width=0.7)
    if show_xlabels:
        ax.set_xlabel(r'$(ARS_{EF} - ARS_{WHG}) / SE_{bootstrap}$',
                      fontsize=9.5, color=MUTED, labelpad=4)

    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(SPINE_CLR); sp.set_linewidth(0.8)
    ax.grid(axis='x', linestyle=':', linewidth=0.4, color=GRID_CLR, zorder=0)


# =============================================================================
# Main
# =============================================================================

def plot_ars_forest(results_dir, trait_manifest, enrichment_file,
                    directionality_file, output,
                    n_cats=6, n_traits=10, ci_level=0.95, fig_w=13.0,
                    fig_h=None, row_height=0.52, max_label=40,
                    label_fontsize=6.5, exclude_traits=None,
                    show_loo=False, denominator='std'):

    ordered_cats   = get_top_categories(enrichment_file, n_cats)
    traits_per_cat = get_ordered_traits(directionality_file, ordered_cats, n_traits)
    contrasts      = load_contrasts(results_dir, trait_manifest,
                                    traits_per_cat, ci_level,
                                    exclude_traits=exclude_traits,
                                    show_loo=show_loo,
                                    denominator=denominator)

    if not contrasts:
        raise ValueError('[ars_forest] No ARS contrasts — check stderr.')

    # Count available traits per category
    cat_n = {
        cat: len([t for t in traits_per_cat.get(cat, []) if t in contrasts])
        for cat in ordered_cats
    }

    # Sort: most traits first → sparse panels land on bottom row
    # (matches directionality plot layout where IBD/Hearing are bottom-right)
    cats_sorted = sorted(ordered_cats, key=lambda c: cat_n[c], reverse=True)

    # Global x range — symmetric around zero. Based on the full CI extent
    # (not just the point), since the whisker is drawn again now that the CI
    # is a proper individual-resampled interval.
    all_pts = [v for c in contrasts.values() for v in (c['contrast_mean'], c['ci_lo'], c['ci_hi'])]
    g_abs   = max(abs(min(all_pts)), abs(max(all_pts)))
    g_pad   = g_abs * 0.18 + 0.15
    x_lo, x_hi = -(g_abs + g_pad), (g_abs + g_pad)

    n_cols = 2
    n_rows = (n_cats + 1) // 2

    # Height ratios proportional to max trait count per row
    row_max_n = []
    for r in range(n_rows):
        rcats = cats_sorted[r * n_cols:(r + 1) * n_cols]
        row_max_n.append(max((cat_n.get(c, 1) for c in rcats), default=1))

    # Auto-compute height from content unless the caller explicitly set one
    # (e.g. via config's forest_fig_height) — auto-sizing keeps rows readable
    # by default, but a fixed height is useful for consistent figure
    # dimensions across repeated runs/figures in a manuscript.
    # row_height is inches per trait row (config: forest_row_height) — the
    # main knob for "too much/too little space per trait"; the per-row fixed
    # overhead (title/margins) and flat padding are separate constants below.
    if fig_h is None:
        fig_h = sum(row_max_n) * row_height + n_rows * 1.0 + 0.85

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={'height_ratios': row_max_n},
        facecolor=BG
    )
    fig.patch.set_facecolor(BG)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, cat in enumerate(cats_sorted[:n_rows * n_cols]):
        r, c = idx // n_cols, idx % n_cols
        ax   = axes[r, c]
        draw_panel(ax, cat,
                   traits_per_cat.get(cat, []),
                   contrasts,
                   x_lo=x_lo, x_hi=x_hi,
                   show_xlabels=(r == n_rows - 1),
                   max_label=max_label,
                   show_loo=show_loo,
                   label_fontsize=label_fontsize)
        ax.set_ylim(-0.75, row_max_n[r] - 0.25)

    for idx in range(len(cats_sorted), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.tight_layout(h_pad=1.2, w_pad=1.4)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    for ext in ('.pdf', '.svg'):
        path = os.path.splitext(output)[0] + ext
        fig.savefig(path, bbox_inches='tight', dpi=180, facecolor=BG)
        print(f'[ars_forest] Saved: {path}', file=sys.stderr)
    plt.close(fig)


# =============================================================================
# Entry points
# =============================================================================

if 'snakemake' in dir():
    p = snakemake.params
    plot_ars_forest(
        results_dir         = str(p.results_dir),
        trait_manifest      = str(snakemake.input.trait_manifest),
        enrichment_file     = str(snakemake.input.enrichment_file),
        directionality_file = str(snakemake.input.directionality_file),
        output              = str(snakemake.output.plot),
        n_cats              = int(getattr(p, 'n_cats',       6)),
        n_traits            = int(getattr(p, 'n_traits',     10)),
        ci_level            = float(getattr(p, 'ci_level',   0.95)),
        fig_w               = float(getattr(p, 'fig_width',  13.0)),
        fig_h               = (float(p.fig_height)
                               if getattr(p, 'fig_height', None) is not None
                               else None),
        row_height          = float(getattr(p, 'row_height', 0.52)),
        max_label           = int(getattr(p, 'max_label',    40)),
        label_fontsize      = float(getattr(p, 'label_fontsize', 6.5)),
        exclude_traits      = list(getattr(p, 'exclude_traits', [])),
        show_loo            = bool(getattr(p, 'show_loo', False)),
        denominator         = str(getattr(p, 'denominator', 'std')),
    )
else:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir',         required=True)
    parser.add_argument('--trait-manifest',      required=True)
    parser.add_argument('--enrichment-file',     required=True)
    parser.add_argument('--directionality-file', required=True)
    parser.add_argument('--output',              required=True)
    parser.add_argument('--n-cats',          type=int,   default=6)
    parser.add_argument('--n-traits',        type=int,   default=10)
    parser.add_argument('--ci-level',        type=float, default=0.95)
    parser.add_argument('--fig-width',       type=float, default=13.0)
    parser.add_argument('--fig-height',      type=float, default=None,
                        help='Fixed figure height in inches. Omit to '
                             'auto-size from the number of rows/traits.')
    parser.add_argument('--row-height',      type=float, default=0.52,
                        help='Inches per trait row, used when auto-sizing '
                             '(ignored if --fig-height is set).')
    parser.add_argument('--label-fontsize',  type=float, default=6.5,
                        help='Font size (pt) for trait labels.')
    parser.add_argument('--max-label',       type=int,   default=40)
    parser.add_argument('--exclude-traits',  nargs='+',  default=[])
    parser.add_argument('--show-loo',        action='store_true', default=False,
                        help='Draw leave-one-out SNP dot strip below each CI bar '
                             '(off by default — see note in draw_panel).')
    parser.add_argument('--denominator',     choices=['std', 'persd'], default='std',
                        help="'std' = std_bootstrap (variant-resampled SE); "
                             "'persd' = person_sd (population SD of each "
                             "individual's own ancestry-weighted contrast).")
    args = parser.parse_args()
    plot_ars_forest(
        results_dir=args.results_dir, trait_manifest=args.trait_manifest,
        enrichment_file=args.enrichment_file,
        directionality_file=args.directionality_file,
        output=args.output, n_cats=args.n_cats, n_traits=args.n_traits,
        ci_level=args.ci_level, fig_w=args.fig_width, fig_h=args.fig_height,
        row_height=args.row_height, label_fontsize=args.label_fontsize,
        max_label=args.max_label, exclude_traits=args.exclude_traits,
        show_loo=args.show_loo, denominator=args.denominator,
    )