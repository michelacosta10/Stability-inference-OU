"""
Diagnostics of the fitted VAR(1) innovations (Ljung-Box tests) and Gaussianity of the soft mode
(linear-regime criterion r >> r*).

For each dataset / coarsening factor k / segment (trading day, EEG pre-ictal, plankton full series):
  1. fit Q_hat on the whole segment, innovations eps = X_{t+1} - Q_hat X_t;
  2. Ljung-Box per variable on eps (residual linear correlation) and on eps^2 (volatility clustering, i.e. the
     time-independence of the innovations), at cumulative lags fixed in physical time;
     reported as the fraction of variables rejected at 5% after Benjamini-Hochberg across variables;
  3. the same tests on the soft-mode projection ell^T eps (ell = left soft eigenvector of Q_hat);
  4. skewness, excess kurtosis and Jarque-Bera on ell^T eps and on the soft-mode coordinate ell^T X.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ou_core as ou

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--datasets", nargs="+", default=["fin", "eeg", "plankton"], choices=["fin", "eeg", "plankton"])
parser.add_argument("--fin-transforms", nargs="+", default=["abs_sub", "logrv"], choices=["abs_sub", "logrv"])
parser.add_argument("--fin-factors", type=int, nargs="+", default=[1, 2, 3, 5, 10])
parser.add_argument("--fin-lags-sec", type=float, nargs="+", default=[10, 60, 300, 1800])
parser.add_argument("--eeg-file", type=str, default="Sz1.mat")
parser.add_argument("--eeg-fs", type=float, default=512.0)
parser.add_argument("--eeg-segment-sec", type=float, nargs=2, default=[0.0, 180.0])
parser.add_argument("--eeg-factors", type=int, nargs="+", default=[1, 8, 64, 128])
parser.add_argument("--eeg-lags-sec", type=float, nargs="+", default=[0.05, 0.5, 2, 10])
parser.add_argument("--plank-lags-days", type=float, nargs="+", default=[1, 3, 10])
parser.add_argument("--alpha", type=float, default=0.05)
parser.add_argument("--tag", type=str, default="")
args = parser.parse_args()


def lags_in_samples(lags_phys, delta_t, M):
    """Physical lags -> sample lags; drop lags < 1 sample or > M / 4."""
    out = []
    for L in lags_phys:
        h = int(round(L / delta_t))
        if 1 <= h <= M // 4:
            out.append((L, h))
    return out


def diagnose(X, delta_t, lags_phys):
    """X: (T, N) standardized segment."""

    Xt = X.T
    r, Q = ou.r_hat(Xt, delta_t)
    eps = ou.innovations(Xt, Q)
    ell, q_soft = ou.soft_left_vector(Q)
    soft_eps = ell @ eps
    soft_x = ell @ Xt

    lags = lags_in_samples(lags_phys, delta_t, eps.shape[1])
    out = {"r_hat": r, "q_soft": q_soft, "M": Xt.shape[1], "lags_phys": [L for L, _ in lags]}

    if len(lags) == 0:
        return out

    H = [h for _, h in lags]
    p_lin = ou.ljung_box(eps, H)          # (N, n_lags)
    p_sq = ou.ljung_box(eps ** 2, H)
    out["frac_rej_lin"] = np.array([np.mean(ou.bh_reject(p_lin[:, j], args.alpha)) for j in range(len(H))])
    out["frac_rej_sq"] = np.array([np.mean(ou.bh_reject(p_sq[:, j], args.alpha)) for j in range(len(H))])
    out["soft_p_lin"] = ou.ljung_box(soft_eps, H)[0]
    out["soft_p_sq"] = ou.ljung_box(soft_eps ** 2, H)[0]
    out["gauss_soft_eps"] = ou.gaussianity(soft_eps)
    out["gauss_soft_x"] = ou.gaussianity(soft_x)
    out["ex_kurt_eps_median"] = float(np.median([ou.gaussianity(e)["ex_kurt"] for e in eps]))

    return out


def report(label, d):
    if "frac_rej_lin" not in d:
        print(f"{label}: no admissible lag")
        return
    lin = " ".join(f"{L:g}:{f:.2f}" for L, f in zip(d["lags_phys"], d["frac_rej_lin"]))
    sq = " ".join(f"{L:g}:{f:.2f}" for L, f in zip(d["lags_phys"], d["frac_rej_sq"]))
    g = d["gauss_soft_eps"]
    print(f"{label}: r_hat = {d['r_hat']:.4g}  LB eps [{lin}]  LB eps^2 [{sq}]  "
          f"soft eps: kurt = {g['ex_kurt']:.2f}, JB p = {g['jb_p']:.1e}, LB^2 p(max lag) = {d['soft_p_sq'][-1]:.1e}  "
          f"soft x kurt = {d['gauss_soft_x']['ex_kurt']:.2f}", flush=True)


if __name__ == "__main__":

    t0 = time.time()
    out = {"config": vars(args)}

    if "fin" in args.datasets:
        days = ou.fin_days(ou.load_fin_returns())
        out["fin"] = {}
        for tr in args.fin_transforms:
            for k in args.fin_factors:
                # same series construction as robustness_sampling.py (no deseasoning, offset 0)
                if tr == "logrv":
                    rv = []
                    for _, R in days:
                        n = R.shape[0] // k
                        rv.append((R[:n * k] ** 2).reshape(n, k, -1).sum(axis=1))
                    allrv = np.concatenate(rv)
                    floor = np.array([allrv[:, j][allrv[:, j] > 0].min() / 2 for j in range(allrv.shape[1])])
                    series = [np.log(np.maximum(x, floor)) for x in rv]
                else:
                    series = [np.abs(R[::k]) for _, R in days]
                allX = np.concatenate(series)
                mu = allX.mean(axis=0, keepdims=True)
                sd = (allX - mu).std()
                for (day, _), x in zip(days, series):
                    d = diagnose((x - mu) / sd, float(k), args.fin_lags_sec)
                    out["fin"][(tr, k, str(day))] = d
                    report(f"fin {tr:7s} k={k:3d} {day}", d)

    if "eeg" in args.datasets:
        raw = ou.load_eeg(args.eeg_file)
        a, b = (int(s * args.eeg_fs) for s in args.eeg_segment_sec)
        signal = ou.standardize(raw[a:b], "per-variable")
        out["eeg"] = {}
        for k in args.eeg_factors:
            d = diagnose(signal[::k], k / args.eeg_fs, args.eeg_lags_sec)
            out["eeg"][k] = d
            report(f"eeg k={k:4d}", d)

    if "plankton" in args.datasets:
        X = ou.standardize(ou.load_plankton(), "per-variable")
        d = diagnose(X, 1 / 3, args.plank_lags_days)
        out["plankton"] = d
        report("plankton", d)

    path = os.path.join(ou.RESULTS_DIR, f"innovation_tests{args.tag}.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved: {path}  ({time.time() - t0:.1f} s)")
