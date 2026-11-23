"""
Robustness of r_hat and sigma_r/r to the sampling interval (SI Figs. 5-6).

For a coarsening factor k (sampling interval k * dt0) the series is built in one of three ways:
  fin  / logrv   : log realised volatility, log(sum of squared 1s returns over blocks of k seconds)
  fin  / abs_sub : |1s log-return| taken every k seconds (plain subsampling: exact for an OU process, Q -> Q^k)
  eeg  / sub     : EEG taken every k samples (plain subsampling), pre-ictal segment only

Two protocols:
  fixedM : windows of M = q * N samples at every k (the window lasts longer for larger k)
           -> tests the optimal interval dt* ~ 0.80 / r at fixed number of samples
  fixedT : windows of fixed duration T_sec (M = T_sec / (k dt0) decreases with k)
           -> information limit: sigma_r / r should NOT improve with coarser sampling

For each window: r_hat in 1/s (stable across k if the OU model holds), moving-block bootstrap sigma_r,
and the theoretical sigma_r/r with kappa_r and kappa_cW evaluated on the eigenvalues of Q_hat.
Windows never cross days (fin) and stay inside the pre-ictal segment (eeg).
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ou_core as ou

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", choices=["fin", "eeg"], required=True)
parser.add_argument("--transform", choices=["logrv", "abs_sub", "sub"], required=True)
parser.add_argument("--protocol", choices=["fixedM", "fixedT"], required=True)
parser.add_argument("--factors", type=int, nargs="+", default=None, help="coarsening factors k (in base samples)")
parser.add_argument("--q", type=float, default=20, help="fixedM: M = q * N")
parser.add_argument("--T-sec", type=float, default=None, help="fixedT: window duration in seconds")
parser.add_argument("--max-phases", type=int, default=1,
                    help="subsampling offsets used as extra windows (abs_sub, sub); 1 = offset 0 only")
parser.add_argument("--deseason", action="store_true",
                    help="fin: subtract the mean intraday profile (per asset, across days) after the transform")
parser.add_argument("--n-assets", type=int, default=47)
parser.add_argument("--eeg-file", type=str, default="Sz1.mat")
parser.add_argument("--eeg-fs", type=float, default=512.0)
parser.add_argument("--eeg-segment-sec", type=float, nargs=2, default=[0.0, 180.0], help="pre-ictal by default")
parser.add_argument("--n-boot", type=int, default=50)
parser.add_argument("--c0", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--n-jobs", type=int, default=-1)
parser.add_argument("--tag", type=str, default="")
args = parser.parse_args()

if args.dataset == "fin" and args.transform == "sub":
    parser.error("fin uses --transform logrv or abs_sub")
if args.dataset == "eeg" and args.transform != "sub":
    parser.error("eeg uses --transform sub")
if args.protocol == "fixedT" and args.T_sec is None:
    parser.error("fixedT needs --T-sec")

DEFAULT_FACTORS = {"fin": [1, 2, 3, 4, 5, 6, 8, 10, 15, 20], "eeg": [1, 2, 4, 8, 16, 32, 64, 96, 128, 192]}
factors = args.factors or DEFAULT_FACTORS[args.dataset]


######################################################################################################################
# SERIES AT COARSENING FACTOR k: list of (segment_id, X (T_seg x N))
######################################################################################################################

def fin_segments(days, k):

    segs, zero_frac = [], np.nan

    if args.transform == "logrv":
        rv = []
        for _, R in days:
            n = R.shape[0] // k
            rv.append((R[:n * k] ** 2).reshape(n, k, -1).sum(axis=1))
        allrv = np.concatenate(rv)
        zero_frac = float(np.mean(allrv == 0))
        # zeros -> half of the smallest positive value of each asset (same rule as the plankton zeros)
        floor = np.array([allrv[:, j][allrv[:, j] > 0].min() / 2 for j in range(allrv.shape[1])])
        series = [np.log(np.maximum(x, floor)) for x in rv]
        phases = [0]
        series = [[s] for s in series]
    else:
        phases = list(range(min(k, args.max_phases)))
        series = [[np.abs(R[p::k]) for p in phases] for _, R in days]
        zero_frac = float(np.mean(np.concatenate([s[0] for s in series]) == 0))

    if args.deseason:
        # mean intraday profile per asset over days (bins aligned within each phase)
        for ip in range(len(phases)):
            n = min(s[ip].shape[0] for s in series)
            prof = np.mean([s[ip][:n] for s in series], axis=0)
            for s in series:
                s[ip] = s[ip][:n] - prof

    # per-variable centering + global std over the whole panel at this k (paper Methods)
    allX = np.concatenate([x for s in series for x in s])
    mu = allX.mean(axis=0, keepdims=True)
    sd = (allX - mu).std()

    for (day, _), s in zip(days, series):
        for p, x in zip(phases, s):
            segs.append((f"{day}_p{p}", (x - mu) / sd))

    return segs, zero_frac


def eeg_segments(signal, k):
    phases = range(min(k, args.max_phases))
    return [(f"p{p}", signal[p::k]) for p in phases], 0.0


######################################################################################################################
# ONE WINDOW
######################################################################################################################

def analyse_window(X, delta_t, seed):
    """X: (N, M). Returns r_hat (1/s), bootstrap sigma_r and theoretical sigma_r/r."""

    N, M = X.shape
    out = {"M": M}

    try:
        r, Q = ou.r_hat(X, delta_t)
    except Exception:
        return None

    q_eigs = np.linalg.eigvals(Q)
    r_boot = ou.moving_block_bootstrap(X, delta_t, 1, args.n_boot, args.c0, Q, seed)

    out.update({
        "r_hat": r,
        "sigma_boot": np.nanstd(r_boot),
        "n_boot_ok": int(np.sum(np.isfinite(r_boot))),
        "kappa_r": ou.kappa_r(M - 1, q_eigs),
        "kappa_cW": ou.kappa_cW(M, N, q_eigs),
        "kappa_W": ou.kappa_W(M, N),
    })
    for name in ["kappa_r", "kappa_cW", "kappa_W"]:
        out["rel_theory_" + name[6:]] = ou.rel_err_theory(r, M - 1, delta_t, out[name])

    return out


######################################################################################################################
# MAIN
######################################################################################################################

if __name__ == "__main__":

    t0 = time.time()

    if args.dataset == "fin":
        df = ou.load_fin_returns(args.n_assets)
        days = ou.fin_days(df)
        dt0 = 1.0
        N = df.shape[1]
        print(f"fin: {len(days)} days, N = {N}, dt0 = 1 s")
    else:
        raw = ou.load_eeg(args.eeg_file)
        a, b = (int(s * args.eeg_fs) for s in args.eeg_segment_sec)
        signal = ou.standardize(raw[a:b], "per-variable")
        dt0 = 1.0 / args.eeg_fs
        N = signal.shape[1]
        print(f"eeg: {args.eeg_file}, segment {args.eeg_segment_sec} s, T = {signal.shape[0]}, N = {N}")

    print(f"transform = {args.transform}, protocol = {args.protocol}, factors = {factors}, "
          f"q = {args.q}, T_sec = {args.T_sec}, deseason = {args.deseason}, B = {args.n_boot}")

    root = np.random.SeedSequence(args.seed)
    results = {}

    for k, k_seed in zip(factors, root.spawn(len(factors))):

        delta_t = k * dt0
        segs, zero_frac = fin_segments(days, k) if args.dataset == "fin" else eeg_segments(signal, k)

        if args.protocol == "fixedM":
            M = int(round(args.q * N))
        else:
            M = int(round(args.T_sec / delta_t))

        if M <= N + 2:
            print(f"k = {k}: M = {M} <= N + 2, skipped")
            continue

        windows = []
        for seg_id, X in segs:
            for s in range(0, X.shape[0] - M + 1, M):    # non-overlapping
                windows.append((seg_id, s, X[s:s + M].T))

        if len(windows) == 0:
            print(f"k = {k}: no window of M = {M} fits in a segment, skipped")
            continue

        seeds = k_seed.spawn(len(windows))
        out = Parallel(n_jobs=args.n_jobs)(
            delayed(analyse_window)(W, delta_t, sd) for (_, _, W), sd in zip(windows, seeds)
        )
        keep = [i for i, o in enumerate(out) if o is not None]
        rows = [out[i] for i in keep]

        res = {key: np.array([r[key] for r in rows]) for key in rows[0]}
        res.update({
            "segment": np.array([windows[i][0] for i in keep]),
            "start": np.array([windows[i][1] for i in keep]),
            "delta_t": delta_t, "M": M, "N": N, "n_windows": len(rows), "zero_frac": zero_frac,
        })

        r = res["r_hat"]
        stable = r > 0
        res["summary"] = {
            "r_mean": np.mean(r),
            "r_median": np.median(r),
            "r_std_windows": np.std(r),
            "frac_stable": np.mean(stable),
            # paper convention (summarize in estimate_realdata_MOU.py): mean sigma_boot / |mean r_hat|
            "rel_boot": np.nanmean(res["sigma_boot"]) / abs(np.mean(r)),
            "rel_boot_median": np.nanmedian(res["sigma_boot"][stable] / r[stable]) if stable.any() else np.nan,
            "rel_theory_r": np.nanmedian(res["rel_theory_r"]),
            "rel_theory_cW": np.nanmedian(res["rel_theory_cW"]),
            "kappa_r": np.nanmedian(res["kappa_r"]),
            "kappa_cW": np.nanmedian(res["kappa_cW"]),
        }
        results[k] = res

        s = res["summary"]
        print(f"k = {k:4d}  dt = {delta_t:8.4f} s  M = {M:6d}  windows = {len(rows):4d}  "
              f"r_hat = {s['r_median']:.4f} 1/s (mean {s['r_mean']:.4f}, sd {s['r_std_windows']:.4f})  "
              f"stable = {s['frac_stable']:.2f}  sigma/r boot = {s['rel_boot']:.3f}  "
              f"theory(k_r) = {s['rel_theory_r']:.3f}  theory(k_cW) = {s['rel_theory_cW']:.3f}  "
              f"zeros = {zero_frac:.3f}", flush=True)

    name = f"robust_{args.dataset}_{args.transform}_{args.protocol}" + ("_deseason" if args.deseason else "") + args.tag
    out_path = os.path.join(ou.RESULTS_DIR, name + ".pkl")
    os.makedirs(ou.RESULTS_DIR, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump({"config": vars(args), "factors": factors, "dt0": dt0, "results": results}, f)

    print(f"\nSaved: {out_path}  ({time.time() - t0:.1f} s)")
