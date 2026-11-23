"""
Paper Fig. 5: EEG at the near-optimal sampling interval dt* ~ 0.80 / r_hat, pre-ictal segment only.

Same pipeline as estimate_realdata_MOU.py with per-variable centering (paper Methods): sliding windows, Bayes I
estimator, Hessian traces, moving-block bootstrap of lambda_min(A), with plain subsampling
(every k-th sample, offset 0), which is exact for an OU process (Q -> Q^k):
  eeg : k = 64  -> dt = 0.125 s  (dt* = 0.80 / 3.5 s^-1 ~ 0.23 s lies just past the end of the r_hat plateau of
                                   results/summary_robustness.csv; 0.125 s is the largest dt inside it)
  fin : k = 2   -> dt = 2 s      (dt* = 0.80 / 0.53 s^-1 ~ 1.5 s, per-variable run at 1 s)
The window length q*N cannot stay at the paper's q = 100 for the EEG (the 180 s pre-ictal segment gives 1440 samples
at 0.125 s), so q = 10 (M = 470) as in the fixed-M robustness test. The financial panel stays at dt = 1 s as in the paper
and plankton cannot be resampled more finely: both entries are copied unchanged from the per-variable run
(--fin-from; pass --fin-from '' to recompute the financial panel at --fin-k).

Mapping onto the GOE model (Fig. 5): units where the centre of the spectrum is 1, lam_bar = Tr(A_hat) / N (mean over
windows as for r_hat): r_mod = r_hat / lam_bar, dt_mod = lam_bar * dt, c_hat from r_bar(c, N) = r_mod.

The output pickle has the format of estimate_realdata_MOU.py and is read by plots_realdata_revised.py.
Run from code/realdata/: python3 estimate_realdata_optimal.py --n-jobs 8
"""

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import ou_core as oc

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--eeg-k", type=int, default=64, help="EEG subsampling factor (512 Hz / k)")
parser.add_argument("--fin-k", type=int, default=2, help="financial subsampling factor (1 s * k)")
parser.add_argument("--eeg-q", type=int, default=10, help="window length = q * N samples")
parser.add_argument("--fin-q", type=int, default=50, help="window length = q * N samples")
parser.add_argument("--step-div", type=int, default=4)
parser.add_argument("--eeg-bootstrap-stride", type=int, default=1, help="few windows at dt*: bootstrap all of them")
parser.add_argument("--fin-bootstrap-stride", type=int, default=5, help="as in estimate_realdata_MOU.py")
parser.add_argument("--eeg-fs", type=float, default=512.0)
parser.add_argument("--eeg-seizure-start-sec", type=float, default=180.0)
parser.add_argument("--eeg-segment-sec", type=float, nargs=2, default=[0.0, 180.0], help="pre-ictal by default")
parser.add_argument("--lag", type=int, default=1)
parser.add_argument("--n-boot", type=int, default=50)
parser.add_argument("--c0", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--n-jobs", type=int, default=-1)
parser.add_argument("--plankton-from", type=str,
                    default=os.path.join(oc.RESULTS_DIR, "reconcile", "real_processed",
                                         "realdata_per-variable_seed12345.pkl"))
parser.add_argument("--fin-from", type=str,
                    default=os.path.join(oc.RESULTS_DIR, "reconcile", "real_processed",
                                         "realdata_per-variable_seed12345.pkl"))
parser.add_argument("--out", type=str, default=os.path.join(oc.RESULTS_DIR, "realdata_fig5.pkl"))
args = parser.parse_args()


######################################################################################################################
# ONE WINDOW, HESSIAN TRACES (as in estimate_realdata_MOU.py)
######################################################################################################################

def estimate_window(X, delta_t, lag):

    N = X.shape[0]

    try:
        A, Q, S_dt, S_inf, _ = oc.estimate_bayes_I(X, delta_t, lag)
        eig_A = np.linalg.eigvals(A)
        return {"A": A, "Q": Q, "eig_A": eig_A,
                "eig_Sigma_dt": np.linalg.eigvals(S_dt), "eig_Sigma_inf": np.linalg.eigvals(S_inf),
                "lambda_min": np.min(np.real(eig_A))}
    except Exception as e:
        print(f"  estimation failed: {e}")
        nan_mat, nan_vec = np.full((N, N), np.nan), np.full(N, np.nan + 0j)
        return {"A": nan_mat, "Q": nan_mat, "eig_A": nan_vec, "eig_Sigma_dt": nan_vec, "eig_Sigma_inf": nan_vec,
                "lambda_min": np.nan}


def stack_windows(results):
    return {key: np.array([r[key] for r in results]) for key in results[0]}


def hessian_traces_from_eigs(eig_Sdt, eig_Sinf):

    eig_Sdt = np.real(eig_Sdt[np.isfinite(eig_Sdt)])
    eig_Sinf = np.real(eig_Sinf[np.isfinite(eig_Sinf)])

    if len(eig_Sdt) == 0 or len(eig_Sinf) == 0 or np.any(eig_Sdt == 0):
        return np.nan, np.nan, np.nan

    tr_JJ = np.mean(1.0 / np.outer(eig_Sdt, eig_Sdt).reshape(-1))
    tr_QQ = np.mean(1.0 / (2.0 * np.outer(1.0 / eig_Sdt, eig_Sinf).reshape(-1)))

    return tr_JJ, tr_QQ, tr_JJ + tr_QQ


def hessian_traces_all(eig_Sdt_all, eig_Sinf_all):
    out = np.array([hessian_traces_from_eigs(a, b) for a, b in zip(eig_Sdt_all, eig_Sinf_all)])
    return {"trace_HJJ_inv": out[:, 0], "trace_HQQ_inv": out[:, 1], "trace_H_inv": out[:, 2]}


def sliding_estimates(values, delta_t, window_size, step_size, stride, seed_seq):
    """values: (T, N). Window estimates plus bootstrap sigma_r on every stride-th window."""

    starts = np.arange(0, values.shape[0] - window_size + 1, step_size)

    est = stack_windows(Parallel(n_jobs=args.n_jobs)(
        delayed(estimate_window)(values[s:s + window_size].T, delta_t, args.lag) for s in starts))
    est.update(hessian_traces_all(est["eig_Sigma_dt"], est["eig_Sigma_inf"]))

    boot_idx = np.arange(0, len(starts), stride)
    seeds = seed_seq.spawn(len(boot_idx))

    boot = Parallel(n_jobs=args.n_jobs)(
        delayed(oc.moving_block_bootstrap)(values[starts[i]:starts[i] + window_size].T, delta_t, args.lag,
                                           args.n_boot, args.c0, est["Q"][i], seeds[j])
        for j, i in enumerate(boot_idx))

    sigma_r = np.full(len(starts), np.nan)
    for j, i in enumerate(boot_idx):
        sigma_r[i] = np.nanstd(boot[j])

    r_boot = np.concatenate(boot) if boot else np.array([])

    est.update({"window_starts": starts, "sigma_r": sigma_r, "bootstrap_idx": boot_idx,
                "r_boot": r_boot[np.isfinite(r_boot)]})
    return est


KEEP = ["lambda_min", "eig_A", "A", "Q", "trace_HJJ_inv", "trace_HQQ_inv", "trace_H_inv", "sigma_r",
        "bootstrap_idx", "window_starts"]


######################################################################################################################
# DATASETS
######################################################################################################################

def run_eeg(seed_seq):

    k = args.eeg_k
    a, b = (int(round(t * args.eeg_fs)) for t in args.eeg_segment_sec)
    signal = oc.standardize(oc.load_eeg()[a:b], "per-variable")[::k]   # standardize at 512 Hz, then subsample
    n_samples, N = signal.shape
    delta_t = k / args.eeg_fs
    window_size = args.eeg_q * N
    step_size = window_size // args.step_div

    print(f"EEG: k = {k}, dt = {delta_t:.4f} s, T = {n_samples}, window = {window_size} "
          f"({window_size * delta_t:.1f} s), step = {step_size}")

    est = sliding_estimates(signal, delta_t, window_size, step_size, args.eeg_bootstrap_stride, seed_seq)
    starts = est["window_starts"]

    out = {key: est[key] for key in KEEP}
    out.update({
        "N": N, "n_samples": n_samples, "delta_t": delta_t, "k": k, "q": args.eeg_q,
        "window_size": window_size, "step_size": step_size,
        "time_sec": args.eeg_segment_sec[0] + starts * delta_t, "seizure_start_sec": args.eeg_seizure_start_sec,
        "r_boot_all": est["r_boot"],
        "dt_label": rf"$\Delta t={k}/512$ s",
    })
    return out


def run_fin(seed_seq):

    k = args.fin_k
    df = oc.load_fin_returns()
    df = pd.DataFrame(oc.standardize(np.abs(df.values), "per-variable"), index=df.index, columns=df.columns)

    N = df.shape[1]
    delta_t = float(k)
    window_size = args.fin_q * N
    step_size = window_size // args.step_div

    print(f"FIN: k = {k}, dt = {delta_t} s, window = {window_size} ({window_size * delta_t / 60:.0f} min), "
          f"step = {step_size}")

    days = {}
    day_groups = list(df.groupby(df.index.date))

    for (day, df_day), day_seed in zip(day_groups, seed_seq.spawn(len(day_groups))):

        df_day = df_day.iloc[::k]
        values = df_day.values

        if values.shape[0] < window_size:
            print(f"{day}: no valid window (T_day = {values.shape[0]})")
            continue

        est = sliding_estimates(values, delta_t, window_size, step_size, args.fin_bootstrap_stride, day_seed)

        t_center = pd.to_datetime(df_day.index[est["window_starts"] + window_size // 2])
        days[day] = {key: est[key] for key in KEEP}
        days[day].update({"T_day": values.shape[0], "r_boot": est["r_boot"],
                          "time_sec": np.asarray((t_center - t_center[0].normalize()).total_seconds())})

        print(f"{day}: T_day = {values.shape[0]}, windows = {len(est['window_starts'])}, "
              f"r_hat*dt mean = {np.nanmean(est['lambda_min']) * delta_t:.3f}")

    return {
        "N": N, "n_samples": df.shape[0] // k, "delta_t": delta_t, "k": k, "q": args.fin_q,
        "window_size": window_size, "step_size": step_size, "days": days,
        "r_boot_all": np.concatenate([d["r_boot"] for d in days.values()]),
        "dt_label": rf"$\Delta t={k}$ s",
    }


def summarize(name, d):
    """Same as estimate_realdata_MOU.py."""

    lam_bar = lambda eig: np.mean(np.real(eig), axis=-1)     # Tr(A_hat) / N

    if name == "eeg":
        r_hat, sr, lb = np.nanmean(d["lambda_min"]), np.nanmean(d["sigma_r"]), np.nanmean(lam_bar(d["eig_A"]))
    elif name == "fin":
        r_list, sr_list, lb_list = [], [], []
        for day in d["days"].values():
            valid = np.isfinite(day["lambda_min"]) & np.isfinite(day["sigma_r"])
            if np.any(valid):
                r_list.append(np.mean(day["lambda_min"][valid]))
                sr_list.append(np.mean(day["sigma_r"][valid]))
                lb_list.append(np.mean(lam_bar(day["eig_A"][valid])))
        r_hat, sr, lb = np.nanmean(r_list), np.nanmean(sr_list), np.nanmean(lb_list)
    else:
        r_hat, sr, lb = d["lambda_min"], d["sigma_r"], lam_bar(d["eig_A"])

    dt = d["delta_t"]

    return {"r_hat": r_hat, "sigma_r": sr, "r_dt": r_hat * dt, "sigma_r_dt": sr * dt,
            "c_hat": (1 - r_hat * dt) / np.sqrt(2), "rel_err": sr / abs(r_hat) if r_hat != 0 else np.inf,
            "q": d["q"],
            "lam_bar": lb, "r_mod": r_hat / lb, "dt_mod": lb * dt, "c_hat_TW": oc.c_from_r(r_hat / lb, d["N"]),
            "sigma_c": sr / lb / np.sqrt(2)}


if __name__ == "__main__":

    t0 = time.time()
    seeds = dict(zip(["eeg", "fin"], np.random.SeedSequence(args.seed).spawn(2)))

    with open(args.plankton_from, "rb") as f:
        plankton = pickle.load(f)["datasets"]["plankton"]

    output = {"config": dict(vars(args), center="per-variable"), "datasets": {}, "summary": {}}
    output["datasets"]["eeg"] = run_eeg(seeds["eeg"])
    if args.fin_from:
        with open(args.fin_from, "rb") as f:
            output["datasets"]["fin"] = pickle.load(f)["datasets"]["fin"]
    else:
        output["datasets"]["fin"] = run_fin(seeds["fin"])
    output["datasets"]["plankton"] = plankton

    for name, d in output["datasets"].items():
        output["summary"][name] = summarize(name, d)

    print("\nSUMMARY")
    for name, s in output["summary"].items():
        print(f"{name:9s}: dt = {output['datasets'][name]['delta_t']:.4g}, r*dt = {s['r_dt']:.5f}, "
              f"c_hat = {s['c_hat']:.5f}, sigma_r/r = {s['rel_err']:.3f}, q = {s['q']:.1f}, "
              f"lam_bar = {s['lam_bar']:.4g}, dt_mod = {s['dt_mod']:.3f}, c_hat_TW = {s['c_hat_TW']:.4f}")

    with open(args.out, "wb") as f:
        pickle.dump(output, f)

    print(f"\nSaved: {args.out} ({time.time() - t0:.0f} s)")
