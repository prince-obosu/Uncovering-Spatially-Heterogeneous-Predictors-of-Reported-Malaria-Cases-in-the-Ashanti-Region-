"""
=============================================================================
Malaria Risk Analysis -- Ashanti Region, Ghana (2023)
Spearman's Rank Correlation + Variance Inflation Factor (VIF) Screening
=============================================================================



Usage:
  python spearman_vif_analysis.py

Requirements:
  pip install pandas numpy scipy matplotlib

Output:
  Figures and results saved to ./output/
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = 'Malariadistrictcases_biophysical_variables.csv'
TARGET     = 'malaria cases'
PREDICTORS = [
    'Rainfall', 'Distance to Roads', 'Distance to Waterbodies',
    'Dry Season Rainfall', 'Elevation', 'EVI', 'NDWI',
    'Slope', 'Temperature', 'Wet Season Rainfall', 'NDBI'
]

PAIRWISE_SCREEN_THRESHOLD = 0.70   
VIF_THRESHOLD = 10.0               # conventional multicollinearity threshold
VIF_CANDIDATE_FOR_EXCLUSION = 'Rainfall'  # Annual Rainfall -- checked first

RED, DRED  = '#C0392B', '#922B21'
BLUE, LBLUE = '#1A5276', '#2E86C1'
GRAY, LGRAY = '#1B2631', '#7F8C8D'
PANEL = '#FDFEFE'

pdp_labels = {
    'EVI': 'EVI (Enhanced Vegetation Index)',
    'Temperature': 'Temperature (degC)',
    'NDWI': 'NDWI (Water Index)',
    'NDBI': 'NDBI (Normalized Diff. Built-up Index)',
    'Distance to Waterbodies': 'Distance to Water Bodies (m)',
    'Rainfall': 'Annual Rainfall (mm)',
    'Dry Season Rainfall': 'Dry Season Rainfall (mm)',
    'Wet Season Rainfall': 'Wet Season Rainfall (mm)',
    'Elevation': 'Elevation (m)',
    'Slope': 'Slope (deg)',
    'Distance to Roads': 'Distance to Roads (m)',
}
lbl_short = {
    TARGET: 'Malaria Cases',
    'Rainfall': 'Rainfall', 'Distance to Roads': 'Dist. Roads',
    'Distance to Waterbodies': 'Dist. Water', 'Dry Season Rainfall': 'Dry Rainfall',
    'Elevation': 'Elevation', 'EVI': 'EVI', 'NDWI': 'NDWI', 'Slope': 'Slope',
    'Temperature': 'Temperature', 'Wet Season Rainfall': 'Wet Rainfall', 'NDBI': 'NDBI',
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 -- Load and clean data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SPEARMAN'S RANK CORRELATION + VIF SCREENING")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
ndwi_median = df['NDWI'].replace(-2541, np.nan).median()
n_outliers = (df['NDWI'] == -2541).sum()
if n_outliers:
    df['NDWI'] = df['NDWI'].replace(-2541, ndwi_median)
    print(f"\n[Data] NDWI outlier (-2541) replaced with median "
          f"({ndwi_median:.4f}) in {n_outliers} row(s).")

n = len(df)
print(f"[Data] {n} districts x {len(PREDICTORS)} predictors loaded.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- Spearman's rank correlation: predictor vs. outcome (Table 2)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("1. SPEARMAN'S RANK CORRELATION (predictor vs. malaria case count)")
print("=" * 65)

spearman = {}
for pred in PREDICTORS:
    rho, pval = stats.spearmanr(df[pred], df[TARGET])
    sig = ('***' if pval < 0.001 else '**' if pval < 0.01 else
           '*' if pval < 0.05 else 'n.s.')
    spearman[pred] = {'rho': float(rho), 'pval': float(pval), 'sig_unadjusted': sig}
    print(f"  {pred:<28} rho={rho:+.4f}  p={pval:.4f}  {sig}")

sig_count = sum(1 for v in spearman.values() if v['sig_unadjusted'] != 'n.s.')
print(f"\n  {sig_count} of {len(PREDICTORS)} predictors significant at "
      f"unadjusted p < 0.05 (see bonferroni_correction.py for the "
      f"Bonferroni-adjusted columns of Table 2).")

with open('output/spearman_raw.json', 'w') as f:
    json.dump({'n': n, 'predictors': PREDICTORS, 'target': TARGET,
               'spearman': spearman}, f, indent=2)

# ── Fig 1: full correlation matrix (outcome + all predictors) ──────────────
all_vars = [TARGET] + PREDICTORS
nv = len(all_vars)
rho_mat = np.zeros((nv, nv))
pval_mat = np.zeros((nv, nv))
for i, v1 in enumerate(all_vars):
    for j, v2 in enumerate(all_vars):
        r, p = stats.spearmanr(df[v1], df[v2])
        rho_mat[i, j] = r
        pval_mat[i, j] = p

cmap_div = LinearSegmentedColormap.from_list(
    'bwr2', ['#1A5276', '#85C1E9', '#FDFEFE', '#F1948A', '#C0392B'])
fig1, ax1 = plt.subplots(figsize=(11, 9), facecolor=PANEL)
ax1.set_facecolor(PANEL)
im = ax1.imshow(rho_mat, cmap=cmap_div, vmin=-1, vmax=1, aspect='auto')
for x in np.arange(-0.5, nv, 1):
    ax1.axhline(x, color='white', lw=0.8)
    ax1.axvline(x, color='white', lw=0.8)
for i in range(nv):
    for j in range(nv):
        rv, pv = rho_mat[i, j], pval_mat[i, j]
        star = ('' if i == j else '***' if pv < 0.001 else '**' if pv < 0.01
                 else '*' if pv < 0.05 else '')
        txt = '1.00' if i == j else f'{rv:.2f}'
        fc = 'white' if abs(rv) > 0.45 else '#2C3E50'
        ax1.text(j, i, f'{txt}\n{star}', ha='center', va='center',
                  fontsize=7.2, color=fc, fontweight='bold' if star else 'normal')
labels = [lbl_short[v] for v in all_vars]
ax1.set_xticks(range(nv)); ax1.set_yticks(range(nv))
ax1.set_xticklabels(labels, rotation=40, ha='right', fontsize=9, color=GRAY)
ax1.set_yticklabels(labels, fontsize=9, color=GRAY)
cb = fig1.colorbar(im, ax=ax1, fraction=0.035, pad=0.02)
cb.set_label("Spearman's rho", fontsize=10)
cb.ax.tick_params(labelsize=8)
ax1.set_title("Spearman Rank Correlation Matrix\nMalaria Cases & Biophysical/"
              "Anthropogenic Variables -- Ashanti Region, Ghana (2023)",
              fontsize=11, fontweight='bold', pad=14, color='#1B2631')
plt.tight_layout()
fig1.savefig('output/fig1_spearman_matrix.png', dpi=180, bbox_inches='tight', facecolor=PANEL)
plt.close()

# ── Fig 2: scatter plots ─────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(3, 4, figsize=(18, 13), facecolor=PANEL)
fig2.suptitle("Malaria Cases vs Biophysical/Anthropogenic Variables -- "
              "Ashanti Region, Ghana (2023)\nSpearman rho with significance annotated",
              fontsize=13, fontweight='bold', color='#1B2631', y=1.01)
for i, pred in enumerate(PREDICTORS):
    ax = axes2.flatten()[i]
    x, yv = df[pred], df[TARGET]
    rv, pv = stats.spearmanr(x, yv)
    sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'n.s.'
    col = RED if rv < 0 else BLUE
    ax.scatter(x, yv, color=col, alpha=0.6, s=40, edgecolors='white', lw=0.5)
    z = np.polyfit(x, yv, 1)
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, np.polyval(z, xr), color=DRED if rv < 0 else LBLUE, lw=1.8, ls='--')
    ax.set_xlabel(pdp_labels.get(pred, pred), fontsize=8.5, color=GRAY)
    ax.set_ylabel('Malaria Cases' if i % 4 == 0 else '', fontsize=8.5, color=GRAY)
    ax.set_title(pred, fontsize=9.5, fontweight='bold', color='#1B2631')
    ax.tick_params(labelsize=7.5, colors=GRAY)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(LGRAY)
    ax.set_facecolor(PANEL)
    ax.annotate(f'rho = {rv:.3f} {sig}', xy=(0.97, 0.94), xycoords='axes fraction',
                ha='right', va='top', fontsize=8.5, color=DRED if rv < 0 else BLUE,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=LGRAY, alpha=0.85))
axes2.flatten()[-1].set_visible(False)
plt.tight_layout()
fig2.savefig('output/fig2_scatter_plots.png', dpi=180, bbox_inches='tight', facecolor=PANEL)
plt.close()
print("\n  Figures saved: output/fig1_spearman_matrix.png, output/fig2_scatter_plots.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 -- Pairwise predictor-predictor screen (|rho| > 0.70)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"2. PAIRWISE PREDICTOR SCREEN (|rho| > {PAIRWISE_SCREEN_THRESHOLD})")
print("=" * 65)

pred_pairs_flagged = []
for i in range(len(PREDICTORS)):
    for j in range(i + 1, len(PREDICTORS)):
        p1, p2 = PREDICTORS[i], PREDICTORS[j]
        r, p = stats.spearmanr(df[p1], df[p2])
        if abs(r) > PAIRWISE_SCREEN_THRESHOLD:
            pred_pairs_flagged.append({'pair': (p1, p2), 'rho': float(r), 'pval': float(p)})
            print(f"  FLAGGED: {p1} <-> {p2}   rho = {r:+.3f}")

if not pred_pairs_flagged:
    print("  No predictor pairs exceeded the threshold.")
print(f"\n  Note: this pairwise screen does not by itself exclude any "
      f"predictor; it motivates the more comprehensive VIF assessment below.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- Variance Inflation Factors (VIF)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("3. VARIANCE INFLATION FACTORS (VIF)")
print("=" * 65)

def compute_vif(df, predictors):
    """VIF_j = 1 / (1 - R_j^2), where R_j^2 is from an OLS regression of
    predictor j on all the other predictors in `predictors` (closed-form
    via numpy, no external dependency on statsmodels)."""
    X = df[predictors].values.astype(float)
    vifs = {}
    for j, pred in enumerate(predictors):
        y_j = X[:, j]
        others = np.delete(X, j, axis=1)
        others_aug = np.column_stack([np.ones(len(y_j)), others])
        beta, *_ = np.linalg.lstsq(others_aug, y_j, rcond=None)
        y_hat = others_aug @ beta
        ss_res = np.sum((y_j - y_hat) ** 2)
        ss_tot = np.sum((y_j - y_j.mean()) ** 2)
        r2_j = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        vif_j = 1.0 / (1.0 - r2_j) if r2_j < 1 - 1e-12 else np.inf
        vifs[pred] = vif_j
    return vifs

print(f"\n  VIF with all {len(PREDICTORS)} predictors:")
vif_full = compute_vif(df, PREDICTORS)
for pred, v in sorted(vif_full.items(), key=lambda kv: -kv[1]):
    flag = '  <-- VIF > 10' if v > VIF_THRESHOLD else ''
    print(f"    {pred:<28} VIF = {v:>8.2f}{flag}")

excluded = []
retained = list(PREDICTORS)
if vif_full.get(VIF_CANDIDATE_FOR_EXCLUSION, 0) > VIF_THRESHOLD:
    excluded.append(VIF_CANDIDATE_FOR_EXCLUSION)
    retained = [p for p in PREDICTORS if p != VIF_CANDIDATE_FOR_EXCLUSION]
    print(f"\n  {VIF_CANDIDATE_FOR_EXCLUSION} exceeds VIF > {VIF_THRESHOLD} and is "
          f"excluded; recomputing VIF on the remaining {len(retained)} predictors...")
    vif_reduced = compute_vif(df, retained)
    print(f"\n  VIF with {len(retained)} predictors "
          f"({VIF_CANDIDATE_FOR_EXCLUSION} excluded):")
    for pred, v in sorted(vif_reduced.items(), key=lambda kv: -kv[1]):
        flag = '  <-- still VIF > 10' if v > VIF_THRESHOLD else ''
        print(f"    {pred:<28} VIF = {v:>8.2f}{flag}")
    still_high = [p for p, v in vif_reduced.items() if v > VIF_THRESHOLD]
    if still_high:
        print(f"\n  WARNING: predictor(s) still exceed VIF > {VIF_THRESHOLD} "
              f"after excluding {VIF_CANDIDATE_FOR_EXCLUSION}: {still_high}. "
              f"The manuscript's Methods text assumes excluding "
              f"{VIF_CANDIDATE_FOR_EXCLUSION} alone is sufficient -- update "
              f"the Methods/Table 4 text if that assumption does not hold on "
              f"your actual data.")
else:
    vif_reduced = vif_full
    print(f"\n  {VIF_CANDIDATE_FOR_EXCLUSION} did not exceed VIF > {VIF_THRESHOLD} "
          f"on this data -- no predictor excluded. (This differs from what "
          f"the manuscript reports; re-check VIF_CANDIDATE_FOR_EXCLUSION / "
          f"VIF_THRESHOLD against your actual results.)")

print(f"\n  Predictors retained for MGWR: {retained}")
print(f"  (Pass this same exclusion list as EXCLUDED_FROM_MGWR in "
      f"mgwr_rf_analysis.py -- currently set there to {excluded or '[]'}.)")

vif_table = pd.DataFrame({
    'predictor': PREDICTORS,
    'VIF_all_11': [vif_full[p] for p in PREDICTORS],
    'VIF_after_exclusion': [vif_reduced.get(p, np.nan) for p in PREDICTORS],
})
vif_table.to_csv('output/vif_table.csv', index=False)

with open('output/vif_results.json', 'w') as f:
    json.dump({
        'threshold': VIF_THRESHOLD,
        'candidate_checked_first': VIF_CANDIDATE_FOR_EXCLUSION,
        'excluded_predictors': excluded,
        'retained_predictors': retained,
        'vif_all_predictors': vif_full,
        'vif_after_exclusion': vif_reduced,
        'pairwise_screen_threshold': PAIRWISE_SCREEN_THRESHOLD,
        'pairwise_flagged_pairs': pred_pairs_flagged,
    }, f, indent=2)

print("\nAll outputs saved to: ./output/")
print("Done.")
