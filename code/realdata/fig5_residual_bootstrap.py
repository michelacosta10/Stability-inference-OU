"""
Residual bootstrap of sigma_r for the Fig. 5 points, compared with the moving-block bootstrap of
estimate_realdata_optimal.py and with the theory of fig5_boundaries.py.

The windows are rebuilt with the same preprocessing as results/realdata_fig5.pkl (per-variable centering, one
global std): EEG pre-ictal subsampled every k samples, financial |returns| at 1 s per day, plankton log-abundances.
For each window that has a moving-block sigma_r, ou_core.residual_bootstrap gives a second sigma_r with the stored
Q_hat; the dataset values are averaged as in estimate_realdata_optimal.summarize (EEG: mean over windows;
financial: mean over windows of each day, then over days; plankton: one window).
Output: results/fig5_residual_bootstrap.pkl. Run from code/realdata/.
"""

import argparse
import os
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ou_core as oc

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default=os.path.join(oc.RESULTS_DIR, "realdata_fig5.pkl"))
parser.add_argument("--n-boot", type=int, default=200)
parser.add_argument("--seed", type=int, default=20261025)
parser.add_argument("--n-jobs", type=int, default=-1)
parser.add_argument("--out", type=str, default=os.path.join(oc.RESULTS_DIR, "fig5_residual_bootstrap.pkl"))
args = parser.parse_args()


def windows(name, d, cfg):
    """List of (group, X (N, M), Q, sigma_block, r_hat) for the windows with a moving-block sigma_r."""

    if name == "eeg":
        a, b = (int(round(t * cfg["eeg_fs"])) for t in cfg["eeg_segment_sec"])
        sig = oc.standardize(oc.load_eeg()[a:b], "per-variable")[::d["k"]]
        groups = {"eeg": (sig, d)}
    elif name == "fin":
        df = oc.load_fin_returns()
        vals = oc.standardize(np.abs(df.values), "per-variable")
        dates = np.array(df.index.date)
        groups = {day: (vals[dates == day], dd) for day, dd in d["days"].items()}
    else:
        X = oc.standardize(oc.load_plankton(), "per-variable")
        return [("plankton", X.T, d["Q"], d["sigma_r"], d["lambda_min"])]

    W = d["window_size"]
    out = []
    for g, (v, dd) in groups.items():
        for i in dd["bootstrap_idx"]:
            s = dd["window_starts"][i]
            out.append((g, v[s:s + W].T, dd["Q"][i], dd["sigma_r"][i], dd["lambda_min"][i]))
    return out


if __name__ == "__main__":

    with open(args.data, "rb") as f:
        data = pickle.load(f)

    result = {}
    for name in ("eeg", "fin", "plankton"):
        d = data["datasets"][name]
        W = windows(name, d, data["config"])
        seeds = np.random.SeedSequence(args.seed).spawn(len(W))
        boots = Parallel(n_jobs=args.n_jobs)(
            delayed(oc.residual_bootstrap)(X, d["delta_t"], Q, args.n_boot, sd) for (_, X, Q, _, _), sd in zip(W, seeds))
        sig_res = np.array([np.nanstd(b) for b in boots])
        sig_blk = np.array([w[3] for w in W])
        r = np.array([w[4] for w in W])
        grp = np.array([str(w[0]) for w in W])

        # dataset averages as in estimate_realdata_optimal.summarize
        if name == "fin":
            G = sorted(set(grp))
            r_m = np.mean([np.mean(r[grp == g]) for g in G])
            res_m = np.mean([np.mean(sig_res[grp == g]) for g in G])
            blk_m = np.mean([np.mean(sig_blk[grp == g]) for g in G])
        else:
            r_m, res_m, blk_m = r.mean(), sig_res.mean(), sig_blk.mean()

        result[name] = {"n_windows": len(W), "r_hat": r_m, "rel_err_residual": res_m / r_m, "rel_err_block": blk_m / r_m,
                        "sigma_residual": sig_res, "sigma_block": sig_blk, "r_hat_windows": r, "group": grp,
                        "r_boot": boots}
        print(f"{name:9s}: windows = {len(W)}, sigma_r/r block = {blk_m / r_m:.3f}, residual = {res_m / r_m:.3f}")

    with open(args.out, "wb") as f:
        pickle.dump({"config": vars(args), "results": result}, f)
    print(f"Saved: {args.out}")
