"""
Theoretical sigma_r/r with kappa_cW of the SI (Proof B: tau_bar = arithmetic mean of tau_a over all modes,
n_eff = n / tau_bar, kappa_cW = n_eff / (n_eff - N - 1)), for the robustness runs and the pooled financial run.

The windows of robustness_sampling.py / pooled_days.py are rebuilt here (same series, same non-overlapping windows)
and only the estimator is recomputed: no bootstrap. The bootstrap sigma_r is read from their pickles, and r_hat is
checked against the stored one.

Output: results/<run>_kcw.pkl (one per robustness run) and results/pooled_days_fin_kcw.pkl.
Run from code/realdata/: python3 theory_kcw.py
"""

import argparse
import os
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ou_core as ou

RUNS = ["robust_fin_logrv_fixedM", "robust_fin_abs_sub_fixedM", "robust_fin_logrv_fixedM_deseason",
        "robust_fin_logrv_fixedT", "robust_fin_abs_sub_fixedT", "robust_eeg_sub_fixedM", "robust_eeg_sub_fixedT"]

cli = argparse.ArgumentParser()
cli.add_argument("--n-jobs", type=int, default=-1)
cli.add_argument("--runs", nargs="+", default=RUNS)
cli.add_argument("--skip-pooled", action="store_true")
cli_args = cli.parse_args()

# robustness_sampling.py and pooled_days.py parse their arguments at import: give them a valid dummy command line,
# then replace their 'args' with the stored configuration of each run.
_argv = sys.argv
sys.argv = ["x", "--dataset", "fin", "--transform", "logrv", "--protocol", "fixedM"]
import robustness_sampling as rs
sys.argv = ["x"]
import pooled_days as pdays
sys.argv = _argv


def window_theory(W, delta_t):
    """W: (N, M). r_hat, Re lambda(A_hat) and the kappa_cW theory, n = M - 1 transitions."""
    N, M = W.shape
    try:
        r, Q = ou.r_hat(W, delta_t)
        re = ou.eig_A_re(Q, delta_t)
    except Exception:
        return None
    t = ou.kappa_cW_si(M - 1, N, re, delta_t)
    t.update({"r_hat": r, "rel_theory": ou.rel_err_theory_nan(r, M - 1, delta_t, t["kappa_cW"])})
    return t


def run_robust(name, data_cache):

    with open(os.path.join(ou.RESULTS_DIR, name + ".pkl"), "rb") as f:
        old = pickle.load(f)
    cfg = argparse.Namespace(**old["config"])
    rs.args = cfg

    if cfg.dataset == "fin":
        if "fin" not in data_cache:
            data_cache["fin"] = ou.fin_days(ou.load_fin_returns(cfg.n_assets))
        days = data_cache["fin"]
    else:
        if "eeg" not in data_cache:
            raw = ou.load_eeg(cfg.eeg_file)
            a, b = (int(s * cfg.eeg_fs) for s in cfg.eeg_segment_sec)
            data_cache["eeg"] = ou.standardize(raw[a:b], "per-variable")
        signal = data_cache["eeg"]

    out = {}
    for k, res in sorted(old["results"].items()):
        delta_t, M = res["delta_t"], res["M"]
        segs, _ = rs.fin_segments(days, k) if cfg.dataset == "fin" else rs.eeg_segments(signal, k)
        segs = dict(segs)
        Ws = [segs[sid][s:s + M].T for sid, s in zip(res["segment"], res["start"])]
        rows = Parallel(n_jobs=cli_args.n_jobs)(delayed(window_theory)(W, delta_t) for W in Ws)
        if any(r is None for r in rows):
            raise RuntimeError(f"{name} k={k}: estimator failed on a window that succeeded before")
        new = {key: np.array([r[key] for r in rows]) for key in rows[0]}

        dr = np.max(np.abs(new["r_hat"] - res["r_hat"]))
        if dr > 1e-8 * max(1.0, np.max(np.abs(res["r_hat"]))):
            raise RuntimeError(f"{name} k={k}: r_hat differs from the stored one (max |diff| = {dr:.3e})")

        r = res["r_hat"]
        rel_boot = np.where(r > 0, res["sigma_boot"] / np.where(r > 0, r, 1), np.nan)
        ratio = new["rel_theory"] / rel_boot
        out[k] = {
            "delta_t": delta_t, "M": M, "n": M - 1, "N": res["N"], "n_windows": len(r),
            "per_window": new, "rel_boot": rel_boot, "rel_theory_r_old": res["rel_theory_r"],
            "summary": {
                "r_median": np.median(r),
                "tau_bar_median": np.nanmedian(new["tau_bar"]),
                "q_eff_median": np.nanmedian(new["q_eff"]),
                "kappa_cW_median": np.nanmedian(new["kappa_cW"]),
                "frac_nan": np.mean(~np.isfinite(new["kappa_cW"])),
                "frac_unstable": np.mean(new["unstable"]),
                "rel_theory_median": np.nanmedian(new["rel_theory"]) if np.isfinite(new["rel_theory"]).any()
                else np.nan,
                "rel_theory_r_old_median": res["summary"]["rel_theory_r"],
                "rel_boot_median": res["summary"]["rel_boot_median"],
                "ratio_median": np.nanmedian(ratio) if np.isfinite(ratio).any() else np.nan,
            },
            "max_abs_diff_r": dr,
        }
        s = out[k]["summary"]
        print(f"{name:34s} k = {k:4d} dt = {delta_t:8.4f}  tau_bar = {s['tau_bar_median']:8.3f}  "
              f"q_eff = {s['q_eff_median']:7.3f}  kcW = {s['kappa_cW_median']:7.3f}  NaN = {s['frac_nan']:.2f}  "
              f"theory/boot = {s['ratio_median']:.3f}  (|dr| = {dr:.1e})", flush=True)

    with open(os.path.join(ou.RESULTS_DIR, name + "_kcw.pkl"), "wb") as f:
        pickle.dump({"config": old["config"], "results": out}, f)


def run_pooled(data_cache):

    with open(os.path.join(ou.RESULTS_DIR, "pooled_days_fin.pkl"), "rb") as f:
        old = pickle.load(f)
    if "fin" not in data_cache:
        data_cache["fin"] = ou.fin_days(ou.load_fin_returns())
    days = data_cache["fin"]

    out = []
    for o in old["results"]:
        segs = pdays.build(days, o["k"], o["transform"], o["centering"])
        r, Q, M = pdays.pooled_r(segs, o["delta_t"])
        if abs(r - o["r_hat"]) > 1e-8 * max(1.0, abs(o["r_hat"])):
            raise RuntimeError(f"pooled {o['transform']} {o['centering']} k={o['k']}: r_hat differs")
        N = segs[0].shape[1]
        t = ou.kappa_cW_si(M, N, ou.eig_A_re(Q, o["delta_t"]), o["delta_t"])   # n = M: M counts transitions
        t["rel_theory"] = ou.rel_err_theory_nan(r, M, o["delta_t"], t["kappa_cW"])
        t.update({k: o[k] for k in ["transform", "centering", "k", "delta_t", "M", "r_hat", "rel_boot",
                                    "rel_theory_r"]})
        t["ratio"] = t["rel_theory"] / o["rel_boot"] if np.isfinite(o["rel_boot"]) else np.nan
        out.append(t)
        print(f"pooled {o['transform']:7s} {o['centering']:8s} dt = {o['delta_t']:4.0f}  r = {r:.5f}  "
              f"tau_bar = {t['tau_bar']:.3f}  q_eff = {t['q_eff']:.1f}  kcW = {t['kappa_cW']:.4f}  "
              f"theory/boot = {t['ratio']:.3f}", flush=True)

    with open(os.path.join(ou.RESULTS_DIR, "pooled_days_fin_kcw.pkl"), "wb") as f:
        pickle.dump({"results": out}, f)


if __name__ == "__main__":
    cache = {}
    for name in cli_args.runs:
        run_robust(name, cache)
    if not cli_args.skip_pooled:
        run_pooled(cache)
    print("done")
