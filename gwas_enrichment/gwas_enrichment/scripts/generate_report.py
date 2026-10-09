"""
generate_report.py
==================
Generates a self-contained HTML report combining:
  1. Fst enrichment results (category level)
  2. Trait directionality results (PGS-based)

Inputs (via snakemake):
  - enrichment file:        {output_dir}/enrichment/category_enrichment.txt
  - directionality summary: {output_dir}/directionality/trait_directionality_summary.txt
  - all traits enrichment:  {output_dir}/trait_decomposition/all_traits_enrichment.txt

Output:
  - {output_dir}/report/analysis_report.html
"""

import sys
import os
import pandas as pd
import numpy as np
import json

# =========================================================================
# Load inputs
# =========================================================================

enrichment_file     = snakemake.input['category_enrichment']
directionality_file = snakemake.input['directionality_summary']
traits_file         = snakemake.input['all_traits_enrichment']
output_html         = snakemake.output['report']

os.makedirs(os.path.dirname(output_html), exist_ok=True)

enrich_df = pd.read_csv(enrichment_file,     sep='\t')
direct_df = pd.read_csv(directionality_file, sep='\t')
traits_df = pd.read_csv(traits_file,         sep='\t')

# =========================================================================
# Prepare enrichment data
# =========================================================================

enrich_df = enrich_df.sort_values('mannwhitneyu_fdr')
enrich_df['neg_log10_fdr']   = -np.log10(enrich_df['mannwhitneyu_fdr'].clip(lower=1e-300))
enrich_df['neg_log10_pval']  = -np.log10(enrich_df['mannwhitneyu_pval'].clip(lower=1e-300))
enrich_df['significant']     = enrich_df['mannwhitneyu_fdr'] < 0.05
enrich_df['sig_label']       = enrich_df['significant'].map({True: 'FDR < 0.05', False: 'ns'})
enrich_df['category_short']  = enrich_df['category'].str.replace('_', ' ').str.replace(' - ', '\n')

# =========================================================================
# Prepare directionality data
# =========================================================================

direct_df['neg_log10_pval']  = -np.log10(direct_df['ttest_pval'].clip(lower=1e-300))
direct_df['direction_color'] = direct_df['direction'].map({
    'EF_higher':      '#e05c5c',
    'WHG_higher':     '#5b8dd9',
    'No_difference':  '#aaaaaa'
})
direct_df['sig_marker'] = direct_df['ttest_pval'].apply(
    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
)
direct_df['trait_label'] = direct_df['trait'] + ' ' + direct_df['sig_marker']
direct_df = direct_df.sort_values('ttest_pval')

# =========================================================================
# Build Plotly figure specs as JSON (embedded in HTML)
# =========================================================================

def safe_json(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, pd.Series):     return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")

# -------------------------------------------------------------------------
# Chart 1: Category Fst enrichment — horizontal bar chart ranked by FDR
# -------------------------------------------------------------------------
enrich_sorted = enrich_df.sort_values('neg_log10_fdr', ascending=True)

cat_bar = {
    'data': [{
        'type': 'bar',
        'orientation': 'h',
        'x': enrich_sorted['neg_log10_fdr'].tolist(),
        'y': enrich_sorted['category'].str.replace('_', ' ').tolist(),
        'marker': {
            'color': enrich_sorted['significant'].map({True: '#e05c5c', False: '#6b7fa3'}).tolist(),
            'line': {'width': 0}
        },
        'customdata': list(zip(
            enrich_sorted['mannwhitneyu_fdr'].round(4).tolist(),
            enrich_sorted['fold_enrichment'].round(3).tolist(),
            enrich_sorted['mean_fst'].round(5).tolist(),
            enrich_sorted['n_blocks'].tolist(),
            enrich_sorted['cohens_d'].round(3).tolist(),
        )),
        'hovertemplate': (
            '<b>%{y}</b><br>'
            'FDR: %{customdata[0]}<br>'
            'Fold enrichment: %{customdata[1]}x<br>'
            'Mean Fst: %{customdata[2]}<br>'
            'LD blocks: %{customdata[3]}<br>'
            "Cohen's d: %{customdata[4]}"
            '<extra></extra>'
        ),
    }],
    'layout': {
        'xaxis': {'title': '-log₁₀(FDR)', 'zeroline': False},
        'yaxis': {'tickfont': {'size': 11}, 'automargin': True},
        'shapes': [{'type': 'line', 'x0': -np.log10(0.05), 'x1': -np.log10(0.05),
                    'y0': -0.5, 'y1': len(enrich_sorted) - 0.5,
                    'line': {'color': '#ffcc00', 'width': 1.5, 'dash': 'dash'}}],
        'annotations': [{'x': -np.log10(0.05), 'y': len(enrich_sorted) - 1,
                         'text': 'FDR=0.05', 'showarrow': False,
                         'font': {'color': '#ffcc00', 'size': 10}, 'xshift': 30}],
        'height': max(500, len(enrich_sorted) * 24),
        'margin': {'l': 20, 'r': 40, 't': 20, 'b': 60},
        'plot_bgcolor': '#1a1e2e',
        'paper_bgcolor': '#1a1e2e',
        'font': {'color': '#c8cfe0'},
    }
}

# -------------------------------------------------------------------------
# Chart 2: Fold enrichment vs -log10(FDR) scatter (Manhattan-style)
# -------------------------------------------------------------------------
scatter_enrich = {
    'data': [{
        'type': 'scatter',
        'mode': 'markers+text',
        'x': enrich_df['fold_enrichment'].tolist(),
        'y': enrich_df['neg_log10_fdr'].tolist(),
        'text': enrich_df.apply(
            lambda r: r['category'].replace('_', ' ') if r['significant'] else '', axis=1
        ).tolist(),
        'textposition': 'top center',
        'textfont': {'size': 9, 'color': '#ffcc00'},
        'marker': {
            'size': (enrich_df['n_blocks'] / enrich_df['n_blocks'].max() * 25 + 6).tolist(),
            'color': enrich_df['neg_log10_fdr'].tolist(),
            'colorscale': 'Reds',
            'showscale': True,
            'colorbar': {'title': '-log₁₀(FDR)', 'tickfont': {'color': '#c8cfe0'},
                         'titlefont': {'color': '#c8cfe0'}},
            'line': {'width': 0.5, 'color': '#ffffff44'}
        },
        'customdata': list(zip(
            enrich_df['category'].str.replace('_', ' ').tolist(),
            enrich_df['mannwhitneyu_fdr'].round(4).tolist(),
            enrich_df['n_blocks'].tolist(),
            enrich_df['mean_fst'].round(5).tolist(),
            enrich_df['cohens_d'].round(3).tolist(),
        )),
        'hovertemplate': (
            '<b>%{customdata[0]}</b><br>'
            'Fold enrichment: %{x:.3f}x<br>'
            'FDR: %{customdata[1]}<br>'
            'LD blocks: %{customdata[2]}<br>'
            'Mean Fst: %{customdata[3]}<br>'
            "Cohen's d: %{customdata[4]}"
            '<extra></extra>'
        ),
    }],
    'layout': {
        'xaxis': {'title': 'Fold Enrichment (vs genome-wide)', 'zeroline': False,
                  'gridcolor': '#2a3050'},
        'yaxis': {'title': '-log₁₀(FDR)', 'zeroline': False, 'gridcolor': '#2a3050'},
        'shapes': [{'type': 'line', 'x0': 0, 'x1': enrich_df['fold_enrichment'].max() * 1.1,
                    'y0': -np.log10(0.05), 'y1': -np.log10(0.05),
                    'line': {'color': '#ffcc00', 'width': 1, 'dash': 'dash'}}],
        'height': 480,
        'margin': {'l': 60, 'r': 20, 't': 20, 'b': 60},
        'plot_bgcolor': '#1a1e2e',
        'paper_bgcolor': '#1a1e2e',
        'font': {'color': '#c8cfe0'},
    }
}

# -------------------------------------------------------------------------
# Chart 3: Directionality — delta_PGS lollipop chart coloured by direction
# -------------------------------------------------------------------------
direct_sorted = direct_df.sort_values('delta_PGS')
lollipop_shapes = []
for _, row in direct_sorted.iterrows():
    lollipop_shapes.append({
        'type': 'line',
        'x0': 0, 'x1': row['delta_PGS'],
        'y0': row['trait_label'], 'y1': row['trait_label'],
        'line': {'color': row['direction_color'], 'width': 1.5}
    })

direct_chart = {
    'data': [{
        'type': 'scatter',
        'mode': 'markers',
        'x': direct_sorted['delta_PGS'].tolist(),
        'y': direct_sorted['trait_label'].tolist(),
        'marker': {
            'color': direct_sorted['direction_color'].tolist(),
            'size': 9,
            'line': {'width': 1, 'color': '#ffffff44'}
        },
        'customdata': list(zip(
            direct_sorted['trait'].tolist(),
            direct_sorted['category'].tolist(),
            direct_sorted['direction'].tolist(),
            direct_sorted['ttest_pval'].apply(lambda p: f'{p:.3e}').tolist(),
            direct_sorted['cohens_d'].round(3).tolist(),
            direct_sorted['PGS_WHG_mean'].round(4).tolist(),
            direct_sorted['PGS_EF_mean'].round(4).tolist(),
            direct_sorted['n_snps_extracted'].tolist(),
        )),
        'hovertemplate': (
            '<b>%{customdata[0]}</b><br>'
            'Category: %{customdata[1]}<br>'
            'Direction: %{customdata[2]}<br>'
            'p-value: %{customdata[3]}<br>'
            "Cohen's d: %{customdata[4]}<br>"
            'PGS WHG: %{customdata[5]}<br>'
            'PGS EF: %{customdata[6]}<br>'
            'SNPs used: %{customdata[7]}'
            '<extra></extra>'
        ),
    }],
    'layout': {
        'xaxis': {'title': 'ΔPGS (EF − WHG)', 'zeroline': True,
                  'zerolinecolor': '#ffffff44', 'gridcolor': '#2a3050'},
        'yaxis': {'tickfont': {'size': 10}, 'automargin': True},
        'shapes': lollipop_shapes,
        'height': max(500, len(direct_sorted) * 22),
        'margin': {'l': 20, 'r': 40, 't': 20, 'b': 60},
        'plot_bgcolor': '#1a1e2e',
        'paper_bgcolor': '#1a1e2e',
        'font': {'color': '#c8cfe0'},
    }
}

# -------------------------------------------------------------------------
# Chart 4: Volcano — -log10(p) vs delta_PGS, coloured by direction
# -------------------------------------------------------------------------
volcano = {
    'data': [{
        'type': 'scatter',
        'mode': 'markers+text',
        'x': direct_df['delta_PGS'].tolist(),
        'y': direct_df['neg_log10_pval'].tolist(),
        'text': direct_df.apply(
            lambda r: r['trait'] if r['ttest_pval'] < 0.05 else '', axis=1
        ).tolist(),
        'textposition': 'top center',
        'textfont': {'size': 8, 'color': '#ffcc00'},
        'marker': {
            'color': direct_df['direction_color'].tolist(),
            'size': (direct_df['n_snps_extracted'] / direct_df['n_snps_extracted'].max() * 18 + 5).tolist(),
            'opacity': 0.85,
            'line': {'width': 0.5, 'color': '#ffffff33'}
        },
        'customdata': list(zip(
            direct_df['trait'].tolist(),
            direct_df['category'].tolist(),
            direct_df['direction'].tolist(),
            direct_df['ttest_pval'].apply(lambda p: f'{p:.3e}').tolist(),
            direct_df['cohens_d'].round(3).tolist(),
            direct_df['n_snps_extracted'].tolist(),
        )),
        'hovertemplate': (
            '<b>%{customdata[0]}</b><br>'
            'Category: %{customdata[1]}<br>'
            'Direction: %{customdata[2]}<br>'
            'p-value: %{customdata[3]}<br>'
            "Cohen's d: %{customdata[4]}<br>"
            'SNPs: %{customdata[5]}'
            '<extra></extra>'
        ),
    }],
    'layout': {
        'xaxis': {'title': 'ΔPGS (EF − WHG)', 'zeroline': True,
                  'zerolinecolor': '#ffffff44', 'gridcolor': '#2a3050'},
        'yaxis': {'title': '-log₁₀(p-value)', 'zeroline': False, 'gridcolor': '#2a3050'},
        'shapes': [{'type': 'line', 'x0': direct_df['delta_PGS'].min() * 1.1,
                    'x1': direct_df['delta_PGS'].max() * 1.1,
                    'y0': -np.log10(0.05), 'y1': -np.log10(0.05),
                    'line': {'color': '#ffcc00', 'width': 1, 'dash': 'dash'}}],
        'height': 500,
        'margin': {'l': 60, 'r': 20, 't': 20, 'b': 60},
        'plot_bgcolor': '#1a1e2e',
        'paper_bgcolor': '#1a1e2e',
        'font': {'color': '#c8cfe0'},
        'legend': {'bgcolor': '#1a1e2e'},
    }
}

# =========================================================================
# Summary stats
# =========================================================================
n_cat_sig    = int(enrich_df['significant'].sum())
n_cat_total  = int(len(enrich_df))
n_trait_sig  = int((direct_df['ttest_pval'] < 0.05).sum())
n_trait_total = int(len(direct_df))
n_ef_higher  = int((direct_df['direction'] == 'EF_higher').sum())
n_whg_higher = int((direct_df['direction'] == 'WHG_higher').sum())
top_cat      = enrich_df.iloc[0]['category'].replace('_', ' ') if len(enrich_df) else 'N/A'
top_fe       = round(float(enrich_df.iloc[0]['fold_enrichment']), 2) if len(enrich_df) else 0

# Serialize chart specs
cat_bar_json     = json.dumps(cat_bar,     default=safe_json)
scatter_json     = json.dumps(scatter_enrich, default=safe_json)
direct_json      = json.dumps(direct_chart, default=safe_json)
volcano_json     = json.dumps(volcano,     default=safe_json)

# =========================================================================
# HTML template
# =========================================================================

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WHG vs EF — Population Genetics Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:        #0f1117;
    --surface:   #1a1e2e;
    --surface2:  #212640;
    --border:    #2e3455;
    --text:      #c8cfe0;
    --muted:     #6b7fa3;
    --accent:    #ffcc00;
    --ef:        #e05c5c;
    --whg:       #5b8dd9;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
  }}

  html {{ scroll-behavior: smooth; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-weight: 300;
    line-height: 1.6;
    min-height: 100vh;
  }}

  /* ---- Header ---- */
  header {{
    padding: 48px 64px 32px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, #0f1117 0%, #1a1e2e 100%);
    position: relative;
    overflow: hidden;
  }}
  header::before {{
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, #ffcc0012 0%, transparent 70%);
    pointer-events: none;
  }}
  .header-eyebrow {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.18em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 12px;
  }}
  header h1 {{
    font-size: clamp(22px, 3vw, 36px);
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #eef1f8;
    margin-bottom: 8px;
  }}
  header p.subtitle {{
    color: var(--muted);
    font-size: 14px;
    max-width: 600px;
  }}

  /* ---- Nav ---- */
  nav {{
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    padding: 0 64px;
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  nav a {{
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    text-decoration: none;
    padding: 14px 20px;
    border-bottom: 2px solid transparent;
    transition: color 0.2s, border-color 0.2s;
  }}
  nav a:hover {{ color: var(--accent); border-bottom-color: var(--accent); }}

  /* ---- Stats bar ---- */
  .stats-bar {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1px;
    background: var(--border);
    border-bottom: 1px solid var(--border);
  }}
  .stat-cell {{
    background: var(--surface);
    padding: 20px 28px;
  }}
  .stat-value {{
    font-family: var(--mono);
    font-size: 28px;
    font-weight: 600;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 4px;
  }}
  .stat-label {{
    font-size: 11px;
    letter-spacing: 0.08em;
    color: var(--muted);
    text-transform: uppercase;
  }}

  /* ---- Main layout ---- */
  main {{
    padding: 48px 64px;
    max-width: 1400px;
    margin: 0 auto;
  }}

  section {{
    margin-bottom: 72px;
  }}

  .section-header {{
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 32px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }}
  .section-number {{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 0.1em;
  }}
  .section-header h2 {{
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #eef1f8;
  }}
  .section-header p {{
    font-size: 13px;
    color: var(--muted);
    margin-left: auto;
    max-width: 380px;
    text-align: right;
  }}

  /* ---- Chart cards ---- */
  .chart-grid {{
    display: grid;
    gap: 24px;
  }}
  .chart-grid.two-col {{
    grid-template-columns: 1fr 1fr;
  }}
  @media (max-width: 900px) {{
    .chart-grid.two-col {{ grid-template-columns: 1fr; }}
    main {{ padding: 32px 24px; }}
    nav {{ padding: 0 24px; }}
    header {{ padding: 32px 24px; }}
  }}

  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }}
  .card-header {{
    padding: 16px 20px 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 12px;
  }}
  .card-title {{
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #eef1f8;
    text-transform: uppercase;
  }}
  .card-subtitle {{
    font-size: 12px;
    color: var(--muted);
  }}
  .chart-wrap {{
    padding: 8px;
  }}

  /* ---- Legend pill ---- */
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    padding: 12px 20px;
    border-top: 1px solid var(--border);
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--muted);
  }}
  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }}

  /* ---- Footer ---- */
  footer {{
    border-top: 1px solid var(--border);
    padding: 24px 64px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
  }}
</style>
</head>
<body>

<header>
  <div class="header-eyebrow">Ancient DNA · Population Genetics · GWAS Enrichment</div>
  <h1>WHG vs Early Farmers — Genetic Differentiation Report</h1>
  <p class="subtitle">
    Fst-based GWAS enrichment analysis and polygenic score directionality
    between Western Hunter-Gatherers and Early European Farmers.
  </p>
</header>

<nav>
  <a href="#fst">Fst Enrichment</a>
  <a href="#directionality">Directionality</a>
</nav>

<div class="stats-bar">
  <div class="stat-cell">
    <div class="stat-value">{n_cat_sig}</div>
    <div class="stat-label">Significant categories</div>
  </div>
  <div class="stat-cell">
    <div class="stat-value">{n_cat_total}</div>
    <div class="stat-label">Total categories tested</div>
  </div>
  <div class="stat-cell">
    <div class="stat-value">{n_trait_sig}</div>
    <div class="stat-label">Significant traits (p&lt;0.05)</div>
  </div>
  <div class="stat-cell">
    <div class="stat-value">{n_ef_higher}</div>
    <div class="stat-label">Traits EF-higher</div>
  </div>
  <div class="stat-cell">
    <div class="stat-value">{n_whg_higher}</div>
    <div class="stat-label">Traits WHG-higher</div>
  </div>
  <div class="stat-cell">
    <div class="stat-value">{top_fe}x</div>
    <div class="stat-label">Top fold enrichment</div>
  </div>
</div>

<main>

  <!-- ================================================================ -->
  <!-- SECTION 1: FST ENRICHMENT                                        -->
  <!-- ================================================================ -->
  <section id="fst">
    <div class="section-header">
      <span class="section-number">01</span>
      <h2>Fst Category Enrichment</h2>
      <p>Mann-Whitney U test of category LD-block Fst vs genome-wide background. FDR corrected.</p>
    </div>

    <div class="chart-grid two-col">

      <div class="card">
        <div class="card-header">
          <span class="card-title">Enrichment Ranking</span>
          <span class="card-subtitle">categories ranked by FDR</span>
        </div>
        <div class="chart-wrap">
          <div id="cat-bar"></div>
        </div>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#e05c5c"></div>FDR &lt; 0.05</div>
          <div class="legend-item"><div class="legend-dot" style="background:#6b7fa3"></div>Not significant</div>
          <div class="legend-item"><div class="legend-dot" style="background:#ffcc00;border-radius:0;width:18px;height:2px"></div>FDR = 0.05 threshold</div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">Enrichment Landscape</span>
          <span class="card-subtitle">fold enrichment vs significance · bubble size = n blocks</span>
        </div>
        <div class="chart-wrap">
          <div id="scatter-enrich"></div>
        </div>
      </div>

    </div>
  </section>

  <!-- ================================================================ -->
  <!-- SECTION 2: DIRECTIONALITY                                        -->
  <!-- ================================================================ -->
  <section id="directionality">
    <div class="section-header">
      <span class="section-number">02</span>
      <h2>PGS Directionality</h2>
      <p>Polygenic score difference (EF − WHG). Positive = EF genetically predisposed higher.</p>
    </div>

    <div class="chart-grid">

      <div class="card">
        <div class="card-header">
          <span class="card-title">Δ PGS per Trait</span>
          <span class="card-subtitle">sorted by delta PGS · * p&lt;0.05  ** p&lt;0.01  *** p&lt;0.001</span>
        </div>
        <div class="chart-wrap">
          <div id="direct-chart"></div>
        </div>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#e05c5c"></div>EF higher</div>
          <div class="legend-item"><div class="legend-dot" style="background:#5b8dd9"></div>WHG higher</div>
          <div class="legend-item"><div class="legend-dot" style="background:#aaaaaa"></div>No difference</div>
        </div>
      </div>

    </div>

    <div class="chart-grid two-col" style="margin-top:24px">

      <div class="card">
        <div class="card-header">
          <span class="card-title">Volcano Plot</span>
          <span class="card-subtitle">ΔPGS vs significance · bubble size = n SNPs</span>
        </div>
        <div class="chart-wrap">
          <div id="volcano"></div>
        </div>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#e05c5c"></div>EF higher</div>
          <div class="legend-item"><div class="legend-dot" style="background:#5b8dd9"></div>WHG higher</div>
          <div class="legend-item"><div class="legend-dot" style="background:#ffcc00;border-radius:0;width:18px;height:2px"></div>p = 0.05</div>
        </div>
      </div>

    </div>
  </section>

</main>

<footer>
  <span>WHG vs Early Farmers · Ancient DNA GWAS Enrichment Pipeline</span>
  <span>Weir &amp; Cockerham Fst · Mann-Whitney U · PGS with mean imputation</span>
</footer>

<script>
const cfg = {{responsive: true, displayModeBar: true,
              modeBarButtonsToRemove: ['lasso2d','select2d'],
              toImageButtonOptions: {{format:'svg', filename:'plot'}}}};

Plotly.newPlot('cat-bar',        {cat_bar_json}['data'],     {cat_bar_json}['layout'],     cfg);
Plotly.newPlot('scatter-enrich', {scatter_json}['data'],     {scatter_json}['layout'],     cfg);
Plotly.newPlot('direct-chart',   {direct_json}['data'],      {direct_json}['layout'],      cfg);
Plotly.newPlot('volcano',        {volcano_json}['data'],     {volcano_json}['layout'],     cfg);
</script>
</body>
</html>"""

with open(output_html, 'w') as f:
    f.write(html)

print(f"Report written to: {output_html}")