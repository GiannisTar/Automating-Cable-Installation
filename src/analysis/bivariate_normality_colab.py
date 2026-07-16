# BIVARIATE NORMALITY ANALYSIS (Google Colab-ready)
# -------------------------------------------------
# Paste the entire file into a Colab cell (or save as a .py and import it).
# At the top of your Colab notebook run the pip install cell once if packages
# are missing:
#
#   !pip install -q pingouin scikit-learn statsmodels
#
# Then run the code below. It expects a pandas DataFrame named `df_bill` with
# the following columns (example):
#   pos_num  (e.g. 'pos_10' as in your sample)
#   end1_x, end1_y, end2_x, end2_y, pick_pos1_x, pick_pos1_y, pick_pos2_x, pick_pos2_y
#
# The main entrypoint is `run_all_gathers(df_bill, outdir='results')` which will
# produce diagnostic plots (PNG) and an aggregate CSV with p-values and verdicts.

# -----------------------
# Dependencies (safe import + install if needed)
# -----------------------
import importlib, subprocess, sys
packages = {'pingouin':'pingouin', 'sklearn':'scikit-learn', 'statsmodels':'statsmodels'}
for mod,pip_name in packages.items():
    try:
        importlib.import_module(mod)
    except Exception:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])

# standard imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from pingouin import multivariate_normality
from statsmodels.stats.multitest import multipletests

# -----------------------
# Helper functions
# -----------------------

def cov_ellipse_params(cov, n_std=1.0):
    """Return width, height, angle (degrees) for an ellipse representing
    n_std standard deviations for covariance matrix `cov` (2x2).
    """
    vals, vecs = np.linalg.eigh(cov)
    # sort descending
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width, height = 2 * np.sqrt(vals) * n_std
    # angle of largest eigenvector
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    return width, height, angle


def mardia_stats(X):
    """Compute Mardia's multivariate skewness (b1p) and kurtosis (b2p).
    X: (n, p) numpy array
    Returns (b1p, b2p)
    """
    X = np.asarray(X)
    n, p = X.shape
    mu = X.mean(axis=0)
    S = np.cov(X, rowvar=False)
    S_inv = np.linalg.inv(S)
    Xc = X - mu
    # A_ij = (Xc[i] @ S_inv @ Xc[j])
    A = Xc @ S_inv @ Xc.T
    b1p = np.sum(A**3) / (n**2)
    d2 = np.einsum('ij,ij->i', Xc @ S_inv, Xc)
    b2p = np.sum(d2**2) / n
    return float(b1p), float(b2p)


def mahalanobis_squared(X, mean=None, cov=None):
    X = np.asarray(X)
    if mean is None:
        mean = X.mean(axis=0)
    if cov is None:
        cov = np.cov(X, rowvar=False)
    cov_inv = np.linalg.inv(cov)
    Xc = X - mean
    d2 = np.einsum('ij,ij->i', Xc @ cov_inv, Xc)
    return d2


def ellipse_patch_from_cov(mean, cov, n_std=1.0, **kwargs):
    width, height, angle = cov_ellipse_params(cov, n_std=n_std)
    return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)

# -----------------------
# Per-position analysis function
# -----------------------

def analyze_position(df, pos_num, xcol, ycol, outdir='results', save_plot=True, show_plot=False):
    """Analyze one 2-D position for a given gather (pos_num).
    Returns a dict with summary statistics and optionally saves a 4-panel PNG plot.
    """
    sub = df[df['pos_num'] == pos_num]
    X = sub[[xcol, ycol]].dropna().values
    n = X.shape[0]
    if n < 5:
        raise ValueError('Too few observations for analysis (need at least 5)')

    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)

    # Marginal stats
    x_vals = X[:,0]
    y_vals = X[:,1]
    shapiro_x = stats.shapiro(x_vals)
    shapiro_y = stats.shapiro(y_vals)
    normaltest_x = stats.normaltest(x_vals)
    normaltest_y = stats.normaltest(y_vals)
    skew_x = float(stats.skew(x_vals))
    skew_y = float(stats.skew(y_vals))
    kurt_x_excess = float(stats.kurtosis(x_vals, fisher=True))
    kurt_y_excess = float(stats.kurtosis(y_vals, fisher=True))

    # Multivariate tests (Henze-Zirkler via pingouin)
    hz_res = multivariate_normality(pd.DataFrame(X, columns=[xcol, ycol]), alpha=0.05)
    # hz_res is an object like HZResults(hz=..., pval=..., normal=...)
    try:
        hz_stat = float(hz_res.hz)
        hz_pval = float(hz_res.pval)
        hz_normal = bool(hz_res.normal)
    except Exception:
        # fallback if pingouin returns tuple
        hz_stat, hz_pval = float(hz_res[0]), float(hz_res[1])
        hz_normal = bool(hz_res[2]) if len(hz_res) > 2 else (hz_pval > 0.05)

    # Mardia
    mardia_skew, mardia_kurt = mardia_stats(X)

    # Mahalanobis distances (classical)
    d2 = mahalanobis_squared(X, mean=mu, cov=cov)
    # robust center/cov via MCD
    mcd = MinCovDet().fit(X)
    mu_rob = mcd.location_
    cov_rob = mcd.covariance_
    d2_rob = mahalanobis_squared(X, mean=mu_rob, cov=cov_rob)

    # outlier flags
    cutoff_95 = chi2.ppf(0.95, df=2)
    cutoff_975 = chi2.ppf(0.975, df=2)
    cutoff_99 = chi2.ppf(0.99, df=2)
    n_above_95 = int((d2 > cutoff_95).sum())
    n_above_975 = int((d2 > cutoff_975).sum())
    n_above_99 = int((d2 > cutoff_99).sum())

    # Prepare plot (4-panel)
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    ax_scatter = axes[0,0]
    ax_marg = axes[0,1]
    ax_qq = axes[1,0]
    ax_chiqq = axes[1,1]

    # scatter + ellipses
    ax_scatter.scatter(X[:,0], X[:,1], s=16, alpha=0.7, label='points')
    ax_scatter.scatter(mu[0], mu[1], c='red', marker='x', label='mean')
    e1 = ellipse_patch_from_cov(mu, cov, n_std=1.0, edgecolor='C3', facecolor='none', linestyle='-')
    e2 = ellipse_patch_from_cov(mu, cov, n_std=2.0, edgecolor='C3', facecolor='none', linestyle='--')
    ax_scatter.add_patch(e1)
    ax_scatter.add_patch(e2)
    ax_scatter.set_title(f'Scatter: {pos_num} - {xcol}/{ycol}')
    ax_scatter.legend()

    # marginal histograms
    ax_marg.hist(x_vals, bins=18, alpha=0.6)
    ax_marg.axvline(mu[0], color='k', linestyle='--')
    ax_marg.set_xlabel(f'{xcol}\nskew={skew_x:.3f}  kurt_excess={kurt_x_excess:.3f}\nShapiro p={shapiro_x.pvalue:.3g}')
    ax2 = ax_marg.twinx()
    ax_marg.hist(y_vals, bins=18, alpha=0.4)
    ax_marg.set_ylabel('count')

    # QQ plots for marginals
    stats.probplot(x_vals, dist='norm', plot=ax_qq)
    ax_qq.set_title('Q-Q plot (X)')
    # create a small inset Q-Q for Y
    axin = fig.add_axes([0.6, 0.35, 0.25, 0.2])
    stats.probplot(y_vals, dist='norm', plot=axin)
    axin.set_title('Q-Q (Y)')

    # Mahalanobis chi2 QQ plot
    ord_d2 = np.sort(d2)
    q = np.linspace(1/(n+1), n/(n+1), n)
    theor = chi2.ppf(q, df=2)
    ax_chiqq.scatter(theor, ord_d2, s=16)
    lim = max(theor.max(), ord_d2.max())
    ax_chiqq.plot([0, lim], [0, lim], '--', color='gray')
    ax_chiqq.axhline(cutoff_975, color='red', linestyle=':', label='97.5% chi2(2)')
    ax_chiqq.set_xlabel('Theoretical chi2 quantiles (df=2)')
    ax_chiqq.set_ylabel('Ordered Mahalanobis d^2')
    ax_chiqq.set_title('Chi2 Q-Q (Mahalanobis d^2)')
    ax_chiqq.legend()

    plt.tight_layout()
    if save_plot:
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.join(outdir, f'{pos_num}__{xcol}_{ycol}.png')
        fig.savefig(fname, dpi=200)
        if not show_plot:
            plt.close(fig)
    elif show_plot:
        plt.show()
    else:
        plt.close(fig)

    # return summary dict
    summary = {
        'gather': pos_num,
        'position': f'{xcol}_{ycol}',
        'n': n,
        'mean_x': float(mu[0]), 'mean_y': float(mu[1]),
        'cov_00': float(cov[0,0]), 'cov_01': float(cov[0,1]), 'cov_11': float(cov[1,1]),
        'shapiro_x_stat': float(shapiro_x.statistic), 'shapiro_x_pval': float(shapiro_x.pvalue),
        'shapiro_y_stat': float(shapiro_y.statistic), 'shapiro_y_pval': float(shapiro_y.pvalue),
        'normaltest_x_stat': float(normaltest_x.statistic), 'normaltest_x_pval': float(normaltest_x.pvalue),
        'normaltest_y_stat': float(normaltest_y.statistic), 'normaltest_y_pval': float(normaltest_y.pvalue),
        'skew_x': skew_x, 'skew_y': skew_y,
        'kurt_x_excess': kurt_x_excess, 'kurt_y_excess': kurt_y_excess,
        'hz_stat': hz_stat, 'hz_pval': hz_pval, 'hz_normal': hz_normal,
        'mardia_skew': mardia_skew, 'mardia_kurt': mardia_kurt,
        'n_above_95': n_above_95, 'n_above_975': n_above_975, 'n_above_99': n_above_99,
        'mu_x': float(mu[0]), 'mu_y': float(mu[1]),
        'mu_rob_x': float(mu_rob[0]), 'mu_rob_y': float(mu_rob[1])
    }
    return summary

# -----------------------
# Orchestrator: run across all gathers & positions
# -----------------------

def run_all_gathers(df_bill, relevant_cols=None, outdir='results', alpha=0.05, save_plots=True):
    if relevant_cols is None:
        relevant_cols = ['end1_x', 'end1_y', 'end2_x', 'end2_y', 'pick_pos1_x', 'pick_pos1_y', 'pick_pos2_x', 'pick_pos2_y']

    pos_names = ['end1', 'end2', 'pick_pos1', 'pick_pos2']
    results = []

    unique_gathers = df_bill['pos_num'].unique()
    print(f'Found {len(unique_gathers)} gathers. Processing {len(unique_gathers)*4} position-sets...')

    for pos_num in unique_gathers:
        for i, pos in enumerate(pos_names):
            xcol = relevant_cols[2*i]
            ycol = relevant_cols[2*i + 1]
            try:
                summary = analyze_position(df_bill, pos_num, xcol, ycol, outdir=outdir, save_plot=save_plots, show_plot=False)
                results.append(summary)
            except Exception as e:
                print(f'Error analyzing {pos_num} {xcol}/{ycol}: {e}')

    df_res = pd.DataFrame(results)

    # Multiple testing correction: apply BH separately to HZ and to marginal Shapiro p-values
    df_res = df_res.sort_values(['gather', 'position']).reset_index(drop=True)
    # HZ
    hz_adj = multipletests(df_res['hz_pval'].values, alpha=alpha, method='fdr_bh')
    df_res['hz_pval_adj'] = hz_adj[1]
    df_res['hz_reject_fdr'] = hz_adj[0]
    # Shapiro X
    shx_adj = multipletests(df_res['shapiro_x_pval'].values, alpha=alpha, method='fdr_bh')
    df_res['shap_x_pval_adj'] = shx_adj[1]
    df_res['shap_x_reject_fdr'] = shx_adj[0]
    # Shapiro Y
    shy_adj = multipletests(df_res['shapiro_y_pval'].values, alpha=alpha, method='fdr_bh')
    df_res['shap_y_pval_adj'] = shy_adj[1]
    df_res['shap_y_reject_fdr'] = shy_adj[0]

    # Final verdict rule (example): reject if HZ rejects after FDR OR both marginals reject after FDR
    def verdict_row(r):
        if r['hz_reject_fdr']:
            return 'reject (HZ fdr)'
        if r['shap_x_reject_fdr'] and r['shap_y_reject_fdr']:
            return 'reject (marginals fdr)'
        # borderline: raw significant but not adjusted
        if (r['hz_pval'] <= alpha and not r['hz_reject_fdr']) or ((r['shapiro_x_pval'] <= alpha or r['shapiro_y_pval'] <= alpha) and not (r['shap_x_reject_fdr'] and r['shap_y_reject_fdr'])):
            return 'borderline'
        return 'consistent'

    df_res['verdict'] = df_res.apply(verdict_row, axis=1)

    # Save aggregate CSV
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, 'bivariate_normality_summary.csv')
    df_res.to_csv(csv_path, index=False)
    print(f'Saved summary CSV to: {csv_path}')
    print(f'Plots (if saved) are in: {os.path.abspath(outdir)}')
    return df_res

# -----------------------
# End of file
# -----------------------
