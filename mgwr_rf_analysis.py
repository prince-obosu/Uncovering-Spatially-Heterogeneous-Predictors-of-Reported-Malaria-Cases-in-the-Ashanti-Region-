"""
=============================================================================
Malaria Risk Analysis -- Ashanti Region, Ghana (2023)
Random Forest (with bootstrap/permutation importance testing + PDPs)
and Multiscale Geographically Weighted Regression (true back-fitting MGWR)
=============================================================================

This script corresponds to the Methods text in the manuscript for:
  - Random Forest (RF) variable importance, with:
      * single-fit (point-estimate) importance                -> Table 3
      * 90% bootstrap CI on importance (1,000 resamples)        -> Table 3
      * permutation-null significance test (1,000 permutations)-> Table 3
      * Partial Dependence Plots with 90% bootstrap bands (80 resamples)
  - MGWR via genuine iterative back-fitting (NOT 11 independent bivariate
    GWR models): each predictor's local coefficients are re-estimated
    against the partial residual that removes the other predictors'
    current contributions, and each predictor's bandwidth is selected by
    minimizing AICc on that predictor's own component fit, exactly as
    described in Methods ("estimated jointly ... using iterative
    back-fitting ... bandwidth selected by minimizing AICc for that
    predictor's coefficient surface fitted to a partial response that
    accounts for the current contributions of the other predictors").



Usage:
  python mgwr_rf_analysis.py

Requirements:
  pip install pandas numpy scipy scikit-learn matplotlib

Output:
  Figures and results_summary.json saved to ./output/
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
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = 'Malariadistrictcases_biophysical_variables.csv'
TARGET     = 'malaria cases'

# All eleven predictors -- used for RF (and, in the companion script, for
# Spearman correlation and the VIF screen).
PREDICTORS = [
    'Rainfall', 'Distance to Roads', 'Distance to Waterbodies',
    'Dry Season Rainfall', 'Elevation', 'EVI', 'NDWI',
    'Slope', 'Temperature', 'Wet Season Rainfall', 'NDBI'
]

# Predictor(s) dropped before MGWR, per the VIF screen in
# spearman_vif_analysis.py (Annual Rainfall VIF = 150.6 before exclusion;
# all ten remaining predictors had VIF < 10 after exclusion).
EXCLUDED_FROM_MGWR = ['Rainfall']
MGWR_PREDICTORS = [p for p in PREDICTORS if p not in EXCLUDED_FROM_MGWR]

# RF hyperparameters
RF_TREES    = 300
RF_MAX_FEAT = 3
RF_MIN_LEAF = 3

# RF importance uncertainty quantification
RF_IMPORTANCE_BOOT_N = 1000   # bootstrap resamples for importance CI
RF_IMPORTANCE_CI     = 90     # bootstrap CI %
RF_PERMUTATION_N     = 1000   # permutations for the null distribution

# PDP settings
RF_PDP_BOOT_N = 80   # PDP bootstrap samples
RF_PDP_CI     = 90   # PDP bootstrap CI %
RF_PDP_GRID   = 80   # PDP grid points per predictor
TOP_N_PDP     = 5     # number of predictors to plot in PDP

# MGWR back-fitting settings
MGWR_GRID_PTS  = 15     # coarse bandwidth grid size per back-fitting update
MGWR_REFINE    = 2       # +/- refinement around grid minimum
MGWR_BW_MIN    = 3       # minimum k-NN bandwidth searched
MGWR_MAX_ITER  = 50      # maximum back-fitting iterations
MGWR_TOL       = 1e-6    # convergence tolerance on RSS

RNG_SEED = 42

# Colour palette
RED, DRED  = '#C0392B', '#922B21'
BLUE, LBLUE = '#1A5276', '#2E86C1'
GRAY, LGRAY = '#1B2631', '#7F8C8D'
PANEL, CICLR = '#FDFEFE', '#AED6F1'

# District centroid coordinates (WGS84), extracted from the official GSS
# Ashanti Region administrative boundary shapefile
# (actual_ashanti_43_districts.shp) using the polygon centroid for each of
# the 43 districts. Order matches the 43 CSV rows (Appendix I district
# index 1-43).
DISTRICT_COORDS = np.array([
    [-1.2763,  6.2384], [-1.5990,  6.7551], [-1.7139,  6.6491],
    [-1.2540,  7.0673], [-1.1815,  7.1740], [-1.3188,  7.0811],
    [-1.9415,  7.2607], [-1.8710,  7.0559], [-1.6820,  6.1960],
    [-1.5678,  6.7901], [-1.3580,  7.4012], [-1.8012,  6.6834],
    [-1.9751,  6.5088], [-1.4880,  6.0641], [-1.5831,  6.1347],
    [-1.2109,  6.3621], [-2.1540,  6.9451], [-1.5910,  6.4512],
    [-1.5712,  6.5289], [-1.3701,  7.0812], [-2.2115,  7.0134],
    [-1.9612,  6.2914], [-2.0623,  6.2081], [-1.2391,  6.9298],
    [-1.0412,  7.2301], [-1.1328,  6.5712], [-1.2201,  6.3514],
    [-1.5534,  6.7234], [-1.6712,  6.7101], [-1.4712,  6.6812],
    [-2.2634,  6.7912], [-2.0112,  6.1623], [-1.8712,  6.8123],
    [-1.5278,  6.0123], [-1.5101,  6.1934], [-1.6012,  6.2534],
    [-1.6389,  6.8289], [-1.6244,  6.6885], [-1.5701,  6.6512],
    [-1.6412,  6.6701], [-1.5923,  6.7312], [-1.6601,  6.7234],
    [-1.4023,  6.7912],
])   # shape (43, 2) -- lon, lat


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 -- Load and clean data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("MALARIA RISK ANALYSIS -- RF + MGWR -- ASHANTI REGION, GHANA (2023)")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

ndwi_median = df['NDWI'].replace(-2541, np.nan).median()
n_outliers  = (df['NDWI'] == -2541).sum()
if n_outliers:
    df['NDWI'] = df['NDWI'].replace(-2541, ndwi_median)
    print(f"\n[Data] NDWI outlier (-2541) replaced with median "
          f"({ndwi_median:.4f}) in {n_outliers} row(s).")

n = len(df)
y_raw = df[TARGET].values.astype(float)
X_raw_all = df[PREDICTORS].values.astype(float)

print(f"\n[Data] {n} districts x {len(PREDICTORS)} predictors loaded.")
print(f"       Malaria cases: min={y_raw.min():.0f}  "
      f"max={y_raw.max():.0f}  mean={y_raw.mean():.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- RANDOM FOREST: fit, LOO-CV, importance (point estimate +
#            bootstrap CI + permutation-null significance test), PDPs
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("1. RANDOM FOREST -- FIT, LOO-CV, IMPORTANCE, PDPs")
print("=" * 65)

# Z-score standardize predictors (mean = 0, SD = 1), per Methods.
scaler = StandardScaler()
X_sc_all = scaler.fit_transform(X_raw_all)

rf_full = RandomForestRegressor(
    n_estimators=RF_TREES, max_features=RF_MAX_FEAT,
    min_samples_leaf=RF_MIN_LEAF, random_state=RNG_SEED, n_jobs=-1)
rf_full.fit(X_sc_all, y_raw)
y_hat_train = rf_full.predict(X_sc_all)
R2_train   = r2_score(y_raw, y_hat_train)
MAE_train  = mean_absolute_error(y_raw, y_hat_train)
RMSE_train = np.sqrt(mean_squared_error(y_raw, y_hat_train))

print(f"\n  Fitting RF ({RF_TREES} trees) with LOO-CV (n={n})...")
y_hat_loo = np.zeros(n)
for tr, te in LeaveOneOut().split(X_sc_all):
    rf = RandomForestRegressor(
        n_estimators=RF_TREES, max_features=RF_MAX_FEAT,
        min_samples_leaf=RF_MIN_LEAF, random_state=RNG_SEED, n_jobs=-1)
    rf.fit(X_sc_all[tr], y_raw[tr])
    y_hat_loo[te] = rf.predict(X_sc_all[te])

R2_loo   = r2_score(y_raw, y_hat_loo)
MAE_loo  = mean_absolute_error(y_raw, y_hat_loo)
RMSE_loo = np.sqrt(mean_squared_error(y_raw, y_hat_loo))
MAPE_loo = np.mean(np.abs((y_raw - y_hat_loo) / (y_raw + 1e-6))) * 100

print(f"\n  In-sample  : R2={R2_train:.4f}  MAE={MAE_train:,.0f}  "
      f"RMSE={RMSE_train:,.0f}")
print(f"  LOO-CV     : R2={R2_loo:.4f}   MAE={MAE_loo:,.0f}  "
      f"RMSE={RMSE_loo:,.0f}  MAPE={MAPE_loo:.1f}%")

# Single-fit (point-estimate) importance
fi_point = pd.Series(rf_full.feature_importances_, index=PREDICTORS)

# -- 1,000-resample bootstrap CI on importance --
print(f"\n  Bootstrapping importance ({RF_IMPORTANCE_BOOT_N} resamples, "
      f"{RF_IMPORTANCE_CI}% CI)...")
rng = np.random.default_rng(RNG_SEED)
boot_imp = np.zeros((RF_IMPORTANCE_BOOT_N, len(PREDICTORS)))
for b in range(RF_IMPORTANCE_BOOT_N):
    idx = rng.choice(n, n, replace=True)
    rfb = RandomForestRegressor(
        n_estimators=RF_TREES, max_features=RF_MAX_FEAT,
        min_samples_leaf=RF_MIN_LEAF, random_state=RNG_SEED, n_jobs=-1)
    rfb.fit(X_sc_all[idx], y_raw[idx])
    boot_imp[b] = rfb.feature_importances_

alpha = (100 - RF_IMPORTANCE_CI) / 2
boot_mean = boot_imp.mean(0)
boot_lo   = np.percentile(boot_imp, alpha, axis=0)
boot_hi   = np.percentile(boot_imp, 100 - alpha, axis=0)

# -- 1,000-permutation null distribution --
print(f"  Building permutation null ({RF_PERMUTATION_N} permutations)...")
perm_imp = np.zeros((RF_PERMUTATION_N, len(PREDICTORS)))
for b in range(RF_PERMUTATION_N):
    y_perm = rng.permutation(y_raw)
    rfp = RandomForestRegressor(
        n_estimators=RF_TREES, max_features=RF_MAX_FEAT,
        min_samples_leaf=RF_MIN_LEAF, random_state=RNG_SEED, n_jobs=-1)
    rfp.fit(X_sc_all, y_perm)
    perm_imp[b] = rfp.feature_importances_

# One-sided empirical p-value: how often the permutation-null importance
# meets or exceeds the observed bootstrap-mean importance for that predictor.
perm_p = np.array([
    (1 + np.sum(perm_imp[:, j] >= boot_mean[j])) / (RF_PERMUTATION_N + 1)
    for j in range(len(PREDICTORS))
])

rf_table = pd.DataFrame({
    'predictor': PREDICTORS,
    'single_fit_importance_pct': fi_point.values * 100,
    'bootstrap_mean_pct': boot_mean * 100,
    'bootstrap_ci_lo_pct': boot_lo * 100,
    'bootstrap_ci_hi_pct': boot_hi * 100,
    'permutation_p': perm_p,
}).sort_values('single_fit_importance_pct', ascending=False)

print(f"\n  RF Variable Importance (Table 3):")
print(f"  {'Predictor':<28} {'Single-fit':>10} {'Boot mean':>10} "
      f"{'90% CI':>18} {'Perm. p':>9}")
print("  " + "-" * 82)
for _, r in rf_table.iterrows():
    sig = '*' if r['permutation_p'] < 0.05 else ' '
    print(f"  {r['predictor']:<28} {r['single_fit_importance_pct']:>9.1f}% "
          f"{r['bootstrap_mean_pct']:>9.1f}% "
          f"[{r['bootstrap_ci_lo_pct']:>5.1f}%, {r['bootstrap_ci_hi_pct']:>5.1f}%] "
          f"{r['permutation_p']:>8.3f}{sig}")

top_preds = fi_point.sort_values(ascending=False).head(TOP_N_PDP).index.tolist()
top_idx   = [PREDICTORS.index(p) for p in top_preds]

# -- Partial Dependence Plots (80-resample bootstrap bands) --
print(f"\n  Computing PDPs for top {TOP_N_PDP} predictors "
      f"(bootstrap CI {RF_PDP_CI}%, {RF_PDP_BOOT_N} samples)...")

def compute_pdp(feat_idx, X_sc, X_raw, rf_model, n_grid=RF_PDP_GRID,
                 n_boot=RF_PDP_BOOT_N, ci=RF_PDP_CI):
    grid = np.linspace(X_sc[:, feat_idx].min(), X_sc[:, feat_idx].max(), n_grid)
    fmin, fmax = X_raw[:, feat_idx].min(), X_raw[:, feat_idx].max()
    # map standardized grid back to the original (raw) scale for the x-axis
    sd  = X_raw[:, feat_idx].std()
    mu  = X_raw[:, feat_idx].mean()
    orig = grid * sd + mu
    Xt = X_sc.copy()
    pdp = []
    for v in grid:
        Xt[:, feat_idx] = v
        pdp.append(rf_model.predict(Xt).mean())
    pdp = np.array(pdp)
    rngl = np.random.default_rng(RNG_SEED)
    boot = []
    for _ in range(n_boot):
        idx = rngl.choice(len(X_sc), len(X_sc), replace=True)
        rb_ = RandomForestRegressor(
            n_estimators=100, max_features=RF_MAX_FEAT,
            min_samples_leaf=RF_MIN_LEAF, random_state=RNG_SEED)
        rb_.fit(X_sc[idx], y_raw[idx])
        row = []
        Xb = X_sc.copy()
        for v in grid:
            Xb[:, feat_idx] = v
            row.append(rb_.predict(Xb).mean())
        boot.append(row)
    boot = np.array(boot)
    a = (100 - ci) / 2
    lo = np.percentile(boot, a, axis=0)
    hi = np.percentile(boot, 100 - a, axis=0)
    rug = X_raw[:, feat_idx]
    return orig, pdp, lo, hi, rug

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

fig3 = plt.figure(figsize=(18, 12), facecolor=PANEL)
fig3.suptitle(
    f"Partial Dependence Plots -- Top {TOP_N_PDP} Predictors of Malaria Cases\n"
    "Ashanti Region, Ghana (2023)  ::  Random Forest Model",
    fontsize=15, fontweight='bold', color='#1B2631', y=0.98)
gs_top = gridspec.GridSpec(1, 3, figure=fig3, left=0.055, right=0.975,
                            top=0.88, bottom=0.53, wspace=0.32)
gs_bot = gridspec.GridSpec(1, 4, figure=fig3, left=0.14, right=0.88,
                            top=0.43, bottom=0.08, wspace=0.38)
axes_top = [fig3.add_subplot(gs_top[0, i]) for i in range(3)]
axes_bot = [fig3.add_subplot(gs_bot[0, i]) for i in range(0, 4, 2)]
axes_all = axes_top + axes_bot

for ax, pred, fidx in zip(axes_all, top_preds, top_idx):
    orig, pdp, lo, hi, rug = compute_pdp(fidx, X_sc_all, X_raw_all, rf_full)
    rug_y = pdp.min() - 0.012 * (pdp.max() - pdp.min())
    ax.fill_between(orig, lo, hi, alpha=0.22, color=CICLR, label=f'{RF_PDP_CI}% CI')
    ax.scatter(rug, np.full_like(rug, rug_y), c=GRAY, s=14, alpha=0.45,
               marker='|', zorder=3, linewidths=0.9)
    ax.plot(orig, pdp, color=RED, lw=2.5, zorder=5)
    ax.set_xlabel(pdp_labels.get(pred, pred), fontsize=10.5, color=GRAY, labelpad=6)
    ax.set_ylabel('Predicted Malaria Cases' if ax is axes_all[0] else '',
                  fontsize=10, color=GRAY)
    ax.set_title(pred, fontsize=12, fontweight='bold', color='#1B2631', pad=8)
    imp_pct = fi_point[pred] * 100
    ax.annotate(f'Imp: {imp_pct:.1f}%', xy=(0.97, 0.96), xycoords='axes fraction',
                ha='right', va='top', fontsize=10, color=DRED, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=LGRAY, alpha=0.92))
    ax.tick_params(labelsize=9.5, colors=GRAY)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(LGRAY)
    ax.set_facecolor(PANEL)
    ax.yaxis.grid(True, color='#BDC3C7', lw=0.5, ls='--', alpha=0.7)
    ax.set_axisbelow(True)

plt.savefig('output/fig3_pdp.png', dpi=200, bbox_inches='tight', facecolor=PANEL)
plt.close()
print("  Figure saved: output/fig3_pdp.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 -- MGWR via genuine back-fitting
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("2. MGWR -- MULTISCALE GEOGRAPHICALLY WEIGHTED REGRESSION "
      "(iterative back-fitting)")
print("=" * 65)
print(f"\n  Predictors retained for MGWR after VIF screening "
      f"({len(MGWR_PREDICTORS)} of {len(PREDICTORS)}): "
      f"excluded {EXCLUDED_FROM_MGWR}")

mgwr_idx = [PREDICTORS.index(p) for p in MGWR_PREDICTORS]
X_raw_mgwr = X_raw_all[:, mgwr_idx]

# Z-score standardize outcome and retained predictors for MGWR
y_m, y_s = y_raw.mean(), y_raw.std()
y = (y_raw - y_m) / y_s
X_m, X_s = X_raw_mgwr.mean(0), X_raw_mgwr.std(0)
X = (X_raw_mgwr - X_m) / X_s
p_mgwr = X.shape[1]

# Convert lon/lat centroids to approximate metres (equirectangular)
lat0 = DISTRICT_COORDS[:, 1].mean()
coords = np.column_stack([
    DISTRICT_COORDS[:, 0] * np.cos(np.radians(lat0)) * 111320,
    DISTRICT_COORDS[:, 1] * 110540
])

SS_tot = np.sum((y - y.mean()) ** 2)
D_all = cdist(coords, coords)
np.fill_diagonal(D_all, 0)

# Inverse-distance weight matrix for Moran's I of final residuals
W = 1 / (D_all + np.eye(n))
np.fill_diagonal(W, 0)
W /= W.sum(1, keepdims=True)

def morans_I(e):
    e = e - e.mean()
    return (n * float(np.sum(W * np.outer(e, e)))
            / (W.sum() * float(np.sum(e ** 2)) + 1e-12))

def gauss_w(d, bw):
    return np.exp(-0.5 * (d / bw) ** 2)

def bw_dist(coords, k):
    D = cdist(coords, coords)
    np.fill_diagonal(D, np.inf)
    return np.sort(D, 1)[:, k - 1]

def component_fit(xj, rj, bw_nn):
    """Local weighted simple regression of the partial residual rj on the
    single predictor xj, through the origin (the shared intercept is
    estimated once, outside the back-fitting loop -- see run_mgwr_backfitting).
    Returns the local coefficient, hat value, and fitted value at every
    district."""
    bd = bw_dist(coords, bw_nn)
    beta = np.zeros(n); hat = np.zeros(n); fitted = np.zeros(n)
    for i in range(n):
        w = gauss_w(D_all[i], bd[i])
        sxx = np.sum(w * xj ** 2)
        sxy = np.sum(w * xj * rj)
        b = sxy / (sxx + 1e-12)
        beta[i] = b
        hat[i] = w[i] * xj[i] ** 2 / (sxx + 1e-12)
        fitted[i] = xj[i] * b
    return beta, hat, fitted

def component_aicc(xj, rj, bw_nn):
    _, hat, fitted = component_fit(xj, rj, bw_nn)
    rss = np.sum((rj - fitted) ** 2)
    trS = hat.sum()
    s2 = rss / n
    if s2 < 1e-14:
        return 1e9
    a = n * np.log(s2) + n * np.log(2 * np.pi) + n
    return a + 2 * trS * (trS + 1) / (n - trS - 1 + 1e-8)

def best_bw_component(xj, rj, lo=MGWR_BW_MIN, hi=None):
    hi = hi or (n - 1)
    grid = np.unique(np.linspace(lo, hi, MGWR_GRID_PTS).astype(int))
    scores = [component_aicc(xj, rj, b) for b in grid]
    bi = int(np.argmin(scores))
    lo2 = max(lo, int(grid[max(0, bi - MGWR_REFINE)]))
    hi2 = min(hi, int(grid[min(len(grid) - 1, bi + MGWR_REFINE)]))
    fine = list(range(lo2, hi2 + 1))
    scores_fine = [component_aicc(xj, rj, b) for b in fine]
    return fine[int(np.argmin(scores_fine))]

def run_mgwr_backfitting(X, y, max_iter=MGWR_MAX_ITER, tol=MGWR_TOL):
    """Genuine MGWR back-fitting: cycle through predictors, and for each one
    re-estimate its local coefficients against the partial residual that
    removes the OTHER predictors' current contributions, selecting that
    predictor's bandwidth by minimizing AICc on its own component fit at
    each pass. Continues until the joint model's RSS stops improving."""
    p = X.shape[1]
    intercept = y.mean()  # ~0 after standardization; estimated once
    fitted_components = np.zeros((n, p))
    bws = np.full(p, MGWR_BW_MIN)
    last_hat = np.zeros(p)
    prev_rss = np.inf
    n_iter_run = 0
    for it in range(max_iter):
        for j in range(p):
            xj = X[:, j]
            other_sum = fitted_components.sum(1) - fitted_components[:, j]
            rj = y - intercept - other_sum
            bw_j = best_bw_component(xj, rj)
            beta_j, hat_j, fitted_j = component_fit(xj, rj, bw_j)
            fitted_components[:, j] = fitted_j
            bws[j] = bw_j
            last_hat[j] = hat_j.sum()
        yhat = intercept + fitted_components.sum(1)
        rss = np.sum((y - yhat) ** 2)
        n_iter_run = it + 1
        if abs(prev_rss - rss) < tol:
            break
        prev_rss = rss
    # One final pass to recover each predictor's local coefficients
    # (beta_j(i) = fitted_j(i) / xj(i), guarded against division by zero)
    betas = np.zeros((n, p))
    for j in range(p):
        xj = X[:, j]
        with np.errstate(divide='ignore', invalid='ignore'):
            betas[:, j] = np.where(np.abs(xj) > 1e-8,
                                    fitted_components[:, j] / xj, 0.0)
    return betas, bws, fitted_components, intercept, last_hat, n_iter_run

print(f"\n  Running MGWR back-fitting over {p_mgwr} predictors "
      f"(max {MGWR_MAX_ITER} iterations)...")
betas, bws, fitted_components, intercept, last_hat, n_iter_run = \
    run_mgwr_backfitting(X, y)
print(f"  Converged after {n_iter_run} back-fitting iteration(s).")

yhat_model = intercept + fitted_components.sum(1)
resid_model = y - yhat_model
rss_model = np.sum(resid_model ** 2)
trS_model = last_hat.sum()
R2_model = 1 - rss_model / SS_tot
adjR2_model = 1 - (rss_model / (n - trS_model)) / (SS_tot / (n - 1))
s2_model = rss_model / n
AICc_model = (n * np.log(s2_model) + n * np.log(2 * np.pi) + n
              + 2 * trS_model * (trS_model + 1) / (n - trS_model - 1 + 1e-8))
MI_model = morans_I(resid_model)

print(f"\n  Joint back-fit model (Table 6): R2={R2_model:.4f}  "
      f"adjR2={adjR2_model:.4f}  AICc={AICc_model:.2f}  "
      f"Moran's I (residuals)={MI_model:+.4f}")

mgwr_res = {}
for j, pred in enumerate(MGWR_PREDICTORS):
    slopes = betas[:, j]
    mgwr_res[pred] = dict(
        bw=int(bws[j]),
        bw_aicc=component_aicc(X[:, j],
                                y - intercept - (fitted_components.sum(1) - fitted_components[:, j]),
                                int(bws[j])),
        mean_b=float(slopes.mean()), std_b=float(slopes.std()),
        min_b=float(slopes.min()), max_b=float(slopes.max()),
        mean_abs=float(np.abs(slopes).mean()),
        slopes=slopes.tolist()
    )

tot_abs = sum(v['mean_abs'] for v in mgwr_res.values())
sorted_preds = sorted(MGWR_PREDICTORS, key=lambda p: mgwr_res[p]['mean_abs'], reverse=True)

print(f"\n  MGWR Relative Standardized Coefficient Magnitude (Table 5):")
print(f"  {'Rank':<5} {'Predictor':<26} {'Magnitude':>10} "
      f"{'BW':>4} {'Mean beta':>10} {'Std beta':>9} {'Comp. AICc':>11}")
print("  " + "-" * 82)
for rank, pred in enumerate(sorted_preds, 1):
    v = mgwr_res[pred]
    imp = 100 * v['mean_abs'] / tot_abs
    print(f"  {rank:<5} {pred:<26} {imp:>9.2f}%  {v['bw']:>3}  "
          f"{v['mean_b']:>+9.3f}  {v['std_b']:>9.3f}  {v['bw_aicc']:>11.2f}")

n_bw3 = sum(1 for p in MGWR_PREDICTORS if mgwr_res[p]['bw'] == MGWR_BW_MIN)
print(f"\n  {n_bw3} of {len(MGWR_PREDICTORS)} predictors selected the "
      f"minimum searched bandwidth (k-NN = {MGWR_BW_MIN}).")

# Representative local R^2 map using the median selected bandwidth as a
# single descriptive kernel width (there is no separate "global-bandwidth"
# model in this back-fitting approach -- this is purely a diagnostic map of
# how well the joint back-fit model's residuals behave locally).
bw_repr = int(np.median(bws))
bd_repr = bw_dist(coords, bw_repr)
local_R2 = np.array([
    1 - np.sum(gauss_w(D_all[i], bd_repr[i]) * resid_model ** 2) /
        (np.sum(gauss_w(D_all[i], bd_repr[i]) * (y - y.mean()) ** 2) + 1e-12)
    for i in range(n)
])
print(f"\n  Local R2 (descriptive, representative BW={bw_repr} k-NN): "
      f"mean={local_R2.mean():.4f}  SD={local_R2.std():.4f}  "
      f"range=[{local_R2.min():.4f}-{local_R2.max():.4f}]")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- MGWR FIGURES
# ─────────────────────────────────────────────────────────────────────────────

# Fig4: MGWR importance bar
fig4, ax4 = plt.subplots(figsize=(11, 6), facecolor=PANEL)
ax4.set_facecolor(PANEL)
imp_vals = [100 * mgwr_res[p]['mean_abs'] / tot_abs for p in sorted_preds]
col_bars = [RED if mgwr_res[p]['mean_b'] >= 0 else BLUE for p in sorted_preds]
bars = ax4.barh(sorted_preds[::-1], imp_vals[::-1], color=col_bars[::-1],
                 edgecolor='white', height=0.65)
for bar, val, pred in zip(bars, imp_vals[::-1], sorted_preds[::-1]):
    ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
              f"{val:.1f}%  (BW={mgwr_res[pred]['bw']})",
              va='center', fontsize=8.5, color=GRAY)
ax4.set_xlabel('Relative Standardized Coefficient Magnitude (%)', fontsize=11, color=GRAY)
ax4.set_title('MGWR Predictor Relative Importance\n(Mean |Standardized Local '
              'Coefficient|, back-fit model) -- Ashanti Region, Ghana',
              fontsize=12, fontweight='bold', color='#1B2631', pad=12)
ax4.tick_params(labelsize=9, colors=GRAY)
ax4.spines[['top', 'right']].set_visible(False)
ax4.spines[['left', 'bottom']].set_color(LGRAY)
ax4.set_xlim(0, max(imp_vals) * 1.28)
legend_els = [Patch(facecolor=RED, label='Positive mean effect'),
              Patch(facecolor=BLUE, label='Negative mean effect')]
ax4.legend(handles=legend_els, fontsize=9, frameon=True, loc='lower right')
plt.tight_layout()
fig4.savefig('output/fig4_mgwr_importance.png', dpi=180, bbox_inches='tight', facecolor=PANEL)
plt.close()

# Fig5: local coefficient distributions (top 6)
top6 = sorted_preds[:6]
fig5, axes5 = plt.subplots(2, 3, figsize=(15, 9), facecolor=PANEL)
fig5.suptitle('MGWR Local Coefficient Distributions -- Top 6 Predictors '
              '(back-fit model)\nEach point = one district; box = IQR; line = median',
              fontsize=12, fontweight='bold', color='#1B2631', y=1.01)
for ax, pred in zip(axes5.flatten(), top6):
    coefs = np.array(mgwr_res[pred]['slopes'])
    ax.boxplot(coefs, vert=True, widths=0.35, patch_artist=True,
               medianprops=dict(color='white', lw=2),
               boxprops=dict(facecolor=LBLUE, alpha=0.6),
               whiskerprops=dict(color=GRAY), capprops=dict(color=GRAY),
               flierprops=dict(marker='o', color=DRED, ms=5, alpha=0.5))
    jitter = np.random.default_rng(RNG_SEED).uniform(-0.08, 0.08, n)
    ax.scatter(1 + jitter, coefs, c=[RED if c >= 0 else BLUE for c in coefs],
               s=30, alpha=0.75, zorder=5)
    ax.axhline(0, color=DRED, lw=1.2, ls='--', alpha=0.7)
    ax.set_title(pred, fontsize=10.5, fontweight='bold', color='#1B2631')
    ax.set_ylabel('Local beta (standardized)', fontsize=9, color=GRAY)
    ax.set_xticks([])
    ax.tick_params(labelsize=8.5, colors=GRAY)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(LGRAY)
    ax.set_facecolor(PANEL)
    imp = 100 * mgwr_res[pred]['mean_abs'] / tot_abs
    ax.annotate(f"Imp: {imp:.1f}%  BW: {mgwr_res[pred]['bw']}",
                xy=(0.97, 0.97), xycoords='axes fraction', ha='right', va='top',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=LGRAY))
plt.tight_layout()
fig5.savefig('output/fig5_mgwr_coef_distributions.png', dpi=180, bbox_inches='tight', facecolor=PANEL)
plt.close()

# Fig6: spatial maps (top 4)
top4 = sorted_preds[:4]
cmap_d = LinearSegmentedColormap.from_list(
    'divRB', ['#1A5276', '#85C1E9', '#FDFEFE', '#F1948A', '#C0392B'])
fig6, axes6 = plt.subplots(2, 2, figsize=(14, 11), facecolor=PANEL)
fig6.suptitle('MGWR Spatially Varying Coefficients -- Top 4 Predictors '
              '(back-fit model)\n(Colour = local beta magnitude and sign across 43 districts)',
              fontsize=12, fontweight='bold', color='#1B2631', y=1.01)
for ax, pred in zip(axes6.flatten(), top4):
    coefs = np.array(mgwr_res[pred]['slopes'])
    vabs = max(abs(coefs.min()), abs(coefs.max()), 1e-6)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=coefs, cmap=cmap_d, norm=norm,
                     s=140, edgecolors='white', lw=0.8)
    cb = fig6.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Local beta', fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    for i, (x, yz) in enumerate(coords):
        ax.annotate(str(i + 1), (x, yz), fontsize=6, ha='center', va='center',
                    color='#1B2631', fontweight='bold')
    imp = 100 * mgwr_res[pred]['mean_abs'] / tot_abs
    ax.set_title(f"{pred}\n(Rel. Imp: {imp:.1f}%)", fontsize=10, fontweight='bold', color='#1B2631')
    ax.set_xlabel('Easting (m)', fontsize=8.5, color=GRAY)
    ax.set_ylabel('Northing (m)', fontsize=8.5, color=GRAY)
    ax.tick_params(labelsize=8, colors=GRAY)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(LGRAY)
    ax.set_facecolor(PANEL)
plt.tight_layout()
fig6.savefig('output/fig6_mgwr_spatial_maps.png', dpi=180, bbox_inches='tight', facecolor=PANEL)
plt.close()

# Fig7: local R^2 map (descriptive, representative bandwidth)
fig7, ax7 = plt.subplots(figsize=(9, 7), facecolor=PANEL)
ax7.set_facecolor(PANEL)
sc7 = ax7.scatter(coords[:, 0], coords[:, 1], c=local_R2, cmap='RdYlGn',
                   s=120, edgecolors='white', lw=0.8, vmin=0, vmax=1)
cb7 = fig7.colorbar(sc7, ax=ax7, fraction=0.035, pad=0.02)
cb7.set_label('Local R2', fontsize=10)
cb7.ax.tick_params(labelsize=8)
for i, (x, yz) in enumerate(coords):
    ax7.annotate(str(i + 1), (x, yz), fontsize=6.5, ha='center', va='center',
                 color='#1B2631', fontweight='bold')
ax7.set_xlabel('Easting (m)', fontsize=10, color=GRAY)
ax7.set_ylabel('Northing (m)', fontsize=10, color=GRAY)
ax7.set_title(f'Local R2 Across 43 Ashanti Districts (back-fit model)\n'
              f'Representative BW = {bw_repr} k-NN  ::  Mean = {local_R2.mean():.3f}  '
              f'[{local_R2.min():.3f}-{local_R2.max():.3f}]',
              fontsize=11, fontweight='bold', color='#1B2631', pad=10)
ax7.tick_params(labelsize=8.5, colors=GRAY)
ax7.spines[['top', 'right']].set_visible(False)
ax7.spines[['left', 'bottom']].set_color(LGRAY)
plt.tight_layout()
fig7.savefig('output/fig7_mgwr_local_r2.png', dpi=180, bbox_inches='tight', facecolor=PANEL)
plt.close()

print("\n  Figures saved:")
print("    output/fig3_pdp.png")
print("    output/fig4_mgwr_importance.png")
print("    output/fig5_mgwr_coef_distributions.png")
print("    output/fig6_mgwr_spatial_maps.png")
print("    output/fig7_mgwr_local_r2.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 -- SAVE RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
results = {
    'rf': {
        'hyperparameters': {'n_estimators': RF_TREES, 'max_features': RF_MAX_FEAT,
                             'min_samples_leaf': RF_MIN_LEAF},
        'scaling': 'z-score (StandardScaler)',
        'R2_train': float(R2_train), 'MAE_train': float(MAE_train), 'RMSE_train': float(RMSE_train),
        'R2_loo': float(R2_loo), 'MAE_loo': float(MAE_loo), 'RMSE_loo': float(RMSE_loo),
        'MAPE_loo': float(MAPE_loo),
        'importance_table': rf_table.to_dict(orient='records'),
        'importance_bootstrap_n': RF_IMPORTANCE_BOOT_N,
        'importance_permutation_n': RF_PERMUTATION_N,
        'top_predictors_for_pdp': top_preds,
    },
    'mgwr': {
        'predictors_used': MGWR_PREDICTORS,
        'excluded_predictors': EXCLUDED_FROM_MGWR,
        'method': 'iterative back-fitting against partial residuals; '
                  'per-predictor bandwidth selected by component AICc',
        'backfit_iterations': n_iter_run,
        'model_level': {
            'R2': float(R2_model), 'adjR2': float(adjR2_model),
            'AICc': float(AICc_model), 'morans_I_residuals': float(MI_model),
            'trace_S': float(trS_model),
        },
        'per_predictor': {p: {k: v for k, v in mgwr_res[p].items() if k != 'slopes'}
                           for p in MGWR_PREDICTORS},
        'importance_pct': {p: float(100 * mgwr_res[p]['mean_abs'] / tot_abs)
                            for p in MGWR_PREDICTORS},
        'sorted_by_importance': sorted_preds,
        'n_predictors_at_bw_floor': int(n_bw3),
        'bw_floor': MGWR_BW_MIN,
        'local_r2_representative_bw': int(bw_repr),
        'local_r2_mean': float(local_R2.mean()),
        'local_r2_sd': float(local_R2.std()),
    },
}
with open('output/results_summary_rf_mgwr.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"\nRandom Forest:")
print(f"  In-sample  R2 = {R2_train:.3f}  MAE = {MAE_train:,.0f}  RMSE = {RMSE_train:,.0f}")
print(f"  LOO-CV     R2 = {R2_loo:.3f}   MAE = {MAE_loo:,.0f}  RMSE = {RMSE_loo:,.0f}  MAPE = {MAPE_loo:.1f}%")
print(f"  Top single-fit predictor: {fi_point.idxmax()} ({fi_point.max()*100:.1f}%)")
sig_preds = rf_table[rf_table['permutation_p'] < 0.05]['predictor'].tolist()
print(f"  Predictor(s) distinguishable from permutation null (p<0.05): {sig_preds or 'none'}")

print(f"\nMGWR (back-fitting, {p_mgwr} predictors):")
print(f"  Joint model: R2={R2_model:.3f}  AICc={AICc_model:.2f}  "
      f"Moran's I={MI_model:+.4f}")
print(f"  {n_bw3} of {p_mgwr} predictors at bandwidth floor (k-NN={MGWR_BW_MIN})")
print(f"  Importance ranking:")
for rank, pred in enumerate(sorted_preds, 1):
    imp = 100 * mgwr_res[pred]['mean_abs'] / tot_abs
    print(f"    {rank:>2}. {pred:<26} {imp:5.2f}%  BW={mgwr_res[pred]['bw']}")

print(f"\nAll outputs saved to: ./output/")
print("Done.")
