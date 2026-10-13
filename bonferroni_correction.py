"""
=============================================================================
Malaria Risk Analysis -- Ashanti Region, Ghana (2023)
Bonferroni Correction for the Spearman Rank Correlations
=============================================================================

Produces the Bonferroni-adjusted columns of Table 2:
  - adjusted significance threshold alpha' = 0.05 / 11 (eleven predictor
    tests) ~= 0.0045
  - Bonferroni-adjusted p-value for each predictor (p_raw * 11, capped at 1)
  - Bonferroni significance flag (significant iff p_raw < alpha')



Usage:
  python bonferroni_correction.py

Requirements:
  pip install pandas numpy scipy

Output:
  output/table2_bonferroni.csv, output/bonferroni_results.json
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)

DATA_PATH  = 'Malariadistrictcases_biophysical_variables.csv'
TARGET     = 'malaria cases'
PREDICTORS = [
    'Rainfall', 'Distance to Roads', 'Distance to Waterbodies',
    'Dry Season Rainfall', 'Elevation', 'EVI', 'NDWI',
    'Slope', 'Temperature', 'Wet Season Rainfall', 'NDBI'
]

print("=" * 65)
print("BONFERRONI CORRECTION -- SPEARMAN RANK CORRELATIONS")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
ndwi_median = df['NDWI'].replace(-2541, np.nan).median()
n_outliers = (df['NDWI'] == -2541).sum()
if n_outliers:
    df['NDWI'] = df['NDWI'].replace(-2541, ndwi_median)
    print(f"\n[Data] NDWI outlier (-2541) replaced with median "
          f"({ndwi_median:.4f}) in {n_outliers} row(s).")

n_tests = len(PREDICTORS)
alpha = 0.05
alpha_adj = alpha / n_tests
print(f"\n[Setup] {n_tests} predictor-outcome correlation tests.")
print(f"        Unadjusted alpha = {alpha}")
print(f"        Bonferroni-adjusted alpha' = {alpha}/{n_tests} = {alpha_adj:.4f}")

# -- recompute raw Spearman correlations directly from the data --
spearman_recomputed = {}
for pred in PREDICTORS:
    rho, pval = stats.spearmanr(df[pred], df[TARGET])
    spearman_recomputed[pred] = {'rho': float(rho), 'pval': float(pval)}

# -- cross-check against spearman_vif_analysis.py's saved output, if present --
spearman_source = 'recomputed directly from data'
raw_path = 'output/spearman_raw.json'
if os.path.exists(raw_path):
    with open(raw_path) as f:
        saved = json.load(f)
    mismatches = []
    for pred in PREDICTORS:
        r_saved = saved['spearman'][pred]['rho']
        r_new = spearman_recomputed[pred]['rho']
        if abs(r_saved - r_new) > 1e-9:
            mismatches.append(pred)
    if mismatches:
        print(f"\n  WARNING: {raw_path} exists but disagrees with the "
              f"correlations recomputed here for: {mismatches}. Using the "
              f"values recomputed directly from {DATA_PATH} in this script; "
              f"re-run spearman_vif_analysis.py if that file is stale.")
    else:
        spearman_source = f'recomputed from data; matches {raw_path}'
        print(f"\n  Cross-check OK: matches {raw_path} from "
              f"spearman_vif_analysis.py.")

# -- apply Bonferroni correction --
rows = []
for pred in PREDICTORS:
    rho = spearman_recomputed[pred]['rho']
    p_raw = spearman_recomputed[pred]['pval']
    p_bonf = min(p_raw * n_tests, 1.0)
    sig_unadj = ('***' if p_raw < 0.001 else '**' if p_raw < 0.01 else
                 '*' if p_raw < 0.05 else 'n.s.')
    sig_bonf = 'sig.' if p_raw < alpha_adj else 'n.s.'
    rows.append({
        'predictor': pred, 'rho': rho,
        'p_unadjusted': p_raw, 'sig_unadjusted': sig_unadj,
        'p_bonferroni': p_bonf, 'sig_bonferroni_(alpha=0.05/11)': sig_bonf,
    })

table2 = pd.DataFrame(rows).sort_values('p_unadjusted')

print(f"\nTable 2 (Spearman correlations, with Bonferroni correction):")
print(f"  {'Predictor':<24} {'rho':>8} {'p (raw)':>9} {'unadj.':>7} "
      f"{'p (Bonf.)':>10} {'Bonf. sig.':>11}")
print("  " + "-" * 78)
for _, r in table2.iterrows():
    print(f"  {r['predictor']:<24} {r['rho']:>+8.3f} {r['p_unadjusted']:>9.4f} "
          f"{r['sig_unadjusted']:>7} {r['p_bonferroni']:>10.4f} "
          f"{r['sig_bonferroni_(alpha=0.05/11)']:>11}")

n_sig_unadj = sum(1 for r in rows if r['sig_unadjusted'] != 'n.s.')
n_sig_bonf  = sum(1 for r in rows if r['sig_bonferroni_(alpha=0.05/11)'] == 'sig.')
print(f"\n  Summary: {n_sig_unadj} of {n_tests} unadjusted (p < 0.05); "
      f"{n_sig_bonf} of {n_tests} after Bonferroni correction "
      f"(p < {alpha_adj:.4f}).")

table2.to_csv('output/table2_bonferroni.csv', index=False)
with open('output/bonferroni_results.json', 'w') as f:
    json.dump({
        'n_tests': n_tests, 'alpha_unadjusted': alpha, 'alpha_adjusted': alpha_adj,
        'spearman_source': spearman_source,
        'table2': rows,
        'n_significant_unadjusted': n_sig_unadj,
        'n_significant_bonferroni': n_sig_bonf,
    }, f, indent=2)

print("\nAll outputs saved to: ./output/")
print("Done.")
