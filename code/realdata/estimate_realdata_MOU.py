import argparse
import glob
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import logm, solve_discrete_lyapunov
from joblib import Parallel, delayed

import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning
)


######################################################################################################################
# ARGUMENTS
######################################################################################################################

parser = argparse.ArgumentParser(
    description="OU estimation on the empirical datasets (EEG, financial, plankton): sliding-window Bayes I "
                "estimates, Hessian traces and moving-block bootstrap of lambda_min(A). Saves everything needed by plots_realdata_revised.py into a single pickle."
)
parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
parser.add_argument("--datasets", type=str, nargs="+", default=["eeg", "fin", "plankton"],
                    choices=["eeg", "fin", "plankton"])
parser.add_argument("--eeg-file", type=str, default=None,
                    help="EEG .mat file (variable 'EEG', shape T x N). Default: first Sz*.mat in <data-dir>/real/EEG_data")
parser.add_argument("--fin-file", type=str, default=None,
                    help="Wide 1s log-return panel, .csv or .parquet (index=timestamp, columns=ticker). "
                         "Default: <data-dir>/real/FINANCIAL_data/Combined_processed/Combined_processed/returns_wide_1s.csv")
parser.add_argument("--plankton-file", type=str, default=None,
                    help="Default: <data-dir>/real/PLANKTON_data/41467_2017_2571_MOESM6_ESM.xlsx")
parser.add_argument(
    "--center", type=str, default="global", choices=["global", "per-variable"],
    help="'global': subtract the global mean, divide by the global std. "
         "'per-variable' (paper Methods): subtract each variable's temporal mean, divide by the global std. "
         "For the financial data, per-variable centering is applied to |log-returns|."
)
parser.add_argument("--eeg-q", type=int, default=100, help="window length = q * N samples")
parser.add_argument("--fin-q", type=int, default=50, help="window length = q * N samples")
parser.add_argument("--step-div", type=int, default=4, help="window step = window_size // step_div")
parser.add_argument("--fin-n-assets", type=int, default=47)
parser.add_argument("--plank-n-species", type=int, default=47)
parser.add_argument("--plank-min-presence", type=float, default=0.90)
parser.add_argument("--eeg-fs", type=float, default=512.0)
parser.add_argument("--eeg-seizure-start-sec", type=float, default=180.0)
parser.add_argument("--plank-delta-t", type=float, default=1 / 3, help="days")
parser.add_argument("--lag", type=int, default=1)
parser.add_argument("--n-boot", type=int, default=50, help="bootstrap replicates B")
parser.add_argument("--c0", type=float, default=5.0, help="block length L = c0 * tau")
parser.add_argument("--bootstrap-stride", type=int, default=5, help="bootstrap every k-th window")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--n-jobs", type=int, default=-1)
parser.add_argument("--tag", type=str, default="REAL")
args = parser.parse_args()

real_dir = os.path.join(args.data_dir, "real")
out_dir = os.path.join(args.data_dir, "real_processed")
os.makedirs(out_dir, exist_ok=True)


######################################################################################################################
# ESTIMATOR: BAYES I
######################################################################################################################

def estimate_bayes_I(X, delta_t, lag):

    X = X.astype(np.float64)
    N, T = X.shape

    T1 = (X[:, lag:] @ X[:, lag:].T) / (T - lag)
    T2 = (X[:, lag:] @ X[:, :-lag].T) / (T - lag)
    T3 = (X[:, :-lag] @ X[:, :-lag].T) / (T - lag)

    T3_inv = np.linalg.inv(T3)

    Q_est = np.real(T2 @ T3_inv)

    Sigma_dt_est = T1 - T2 @ T3_inv @ T2.T

    A_est = np.real(-logm(Q_est) / (lag * delta_t))

    Sigma_inf_est = solve_discrete_lyapunov(Q_est, Sigma_dt_est)

    B_est = 0.5 * (A_est @ Sigma_inf_est + Sigma_inf_est @ A_est.T)

    return A_est, Q_est, Sigma_dt_est, Sigma_inf_est, B_est


def estimate_window(X, delta_t, lag):
    """X: (N, T). Returns spectra and matrices for one window; NaNs if the estimation fails."""

    N = X.shape[0]

    try:
        A, Q, S_dt, S_inf, _ = estimate_bayes_I(X, delta_t, lag)
        eig_A = np.linalg.eigvals(A)
        return {
            "A": A,
            "Q": Q,
            "eig_A": eig_A,
            "eig_Sigma_dt": np.linalg.eigvals(S_dt),
            "eig_Sigma_inf": np.linalg.eigvals(S_inf),
            "lambda_min": np.min(np.real(eig_A)),
        }
    except Exception as e:
        print(f"  estimation failed: {e}")
        nan_mat = np.full((N, N), np.nan)
        nan_vec = np.full(N, np.nan + 0j)
        return {
            "A": nan_mat, "Q": nan_mat,
            "eig_A": nan_vec, "eig_Sigma_dt": nan_vec, "eig_Sigma_inf": nan_vec,
            "lambda_min": np.nan,
        }


def stack_windows(results):
    return {key: np.array([r[key] for r in results]) for key in results[0]}


######################################################################################################################
# HESSIAN TRACES: Tr(H^{-1}) / N^2 from the eigenvalues of Sigma_dt and Sigma_inf
######################################################################################################################

def hessian_traces_from_eigs(eig_Sdt, eig_Sinf):

    eig_Sdt = np.real(eig_Sdt[np.isfinite(eig_Sdt)])
    eig_Sinf = np.real(eig_Sinf[np.isfinite(eig_Sinf)])

    nan3 = (np.nan, np.nan, np.nan)

    if len(eig_Sdt) == 0 or len(eig_Sinf) == 0 or np.any(eig_Sdt == 0):
        return nan3

    eig_HJJ = np.outer(eig_Sdt, eig_Sdt).reshape(-1)
    eig_HQQ = 2.0 * np.outer(1.0 / eig_Sdt, eig_Sinf).reshape(-1)

    if np.any(eig_HJJ == 0) or np.any(eig_HQQ == 0):
        return nan3

    eig_HJJ_inv = 1.0 / eig_HJJ
    eig_HQQ_inv = 1.0 / eig_HQQ

    if not (np.isfinite(eig_HJJ_inv).all() and np.isfinite(eig_HQQ_inv).all()):
        return nan3

    tr_JJ = np.mean(eig_HJJ_inv)
    tr_QQ = np.mean(eig_HQQ_inv)

    return tr_JJ, tr_QQ, tr_JJ + tr_QQ


def hessian_traces_all(eig_Sdt_all, eig_Sinf_all):
    out = np.array([hessian_traces_from_eigs(s_dt, s_inf) for s_dt, s_inf in zip(eig_Sdt_all, eig_Sinf_all)])
    return {"trace_HJJ_inv": out[:, 0], "trace_HQQ_inv": out[:, 1], "trace_H_inv": out[:, 2]}


######################################################################################################################
# MOVING-BLOCK BOOTSTRAP
######################################################################################################################

def estimate_tau(Q_est):

    eigvals_Q = np.abs(np.linalg.eigvals(Q_est))
    eigvals_Q = eigvals_Q[np.isfinite(eigvals_Q)]
    eigvals_Q = eigvals_Q[(eigvals_Q > 0) & (eigvals_Q < 1)]

    if len(eigvals_Q) == 0:
        return np.nan

    x = -np.log(np.max(eigvals_Q))

    return (1 + np.exp(-2 * x)) / (1 - np.exp(-2 * x))


def moving_block_bootstrap(X, delta_t, lag, n_boot, c0, Q_est, seed):

    rng = np.random.default_rng(seed)

    N, T = X.shape

    if not np.all(np.isfinite(Q_est)):
        return np.full(n_boot, np.nan), np.full((n_boot, N), np.nan + 0j)

    tau = estimate_tau(Q_est)

    L = lag + 1 if not np.isfinite(tau) else max(int(c0 * tau), lag + 1)
    L = max(min(L, T // 2), lag + 1)

    n_blocks = T - L + 1
    n_needed = int(np.ceil(T / L))

    r_boot = np.full(n_boot, np.nan)
    eig_A_boot = np.full((n_boot, N), np.nan + 0j, dtype=np.complex128)

    for b in range(n_boot):

        starts = rng.integers(0, n_blocks, size=n_needed)

        X_boot = np.concatenate([X[:, s:s + L] for s in starts], axis=1)[:, :T]

        try:
            A_b = estimate_bayes_I(X_boot, delta_t, lag)[0]
            eig_b = np.linalg.eigvals(A_b)
            eig_A_boot[b] = eig_b
            r_boot[b] = np.min(np.real(eig_b))
        except Exception:
            pass

    return r_boot, eig_A_boot


######################################################################################################################
# PREPROCESSING
######################################################################################################################

def standardize(X):
    """X: (T, N). Global or per-variable centering, then division by one global std."""

    if args.center == "global":
        X = X - np.mean(X)
    else:
        X = X - np.mean(X, axis=0, keepdims=True)

    std = np.std(X)
    if std == 0:
        std = 1.0

    return X / std


def window_starts(n_samples, window_size, step_size):
    return np.arange(0, n_samples - window_size + 1, step_size)


######################################################################################################################
# DATASET: EEG
######################################################################################################################

def run_eeg(seed_seq):

    print("\n" + "=" * 60)
    print("EEG")
    print("=" * 60)

    eeg_file = args.eeg_file
    if eeg_file is None:
        candidates = sorted(glob.glob(os.path.join(real_dir, "EEG_data", "Sz*.mat")))
        if len(candidates) == 0:
            raise FileNotFoundError(f"No Sz*.mat found in {os.path.join(real_dir, 'EEG_data')}")
        eeg_file = candidates[0]

    print(f"File EEG: {eeg_file}")

    signal = loadmat(eeg_file)["EEG"].astype(np.float64)
    n_samples, N = signal.shape

    signal = standardize(signal)

    delta_t = 1.0 / args.eeg_fs
    window_size = args.eeg_q * N
    step_size = window_size // args.step_div
    starts = window_starts(n_samples, window_size, step_size)

    print(f"T = {n_samples}, N = {N}, delta_t = {delta_t:.6f} s")
    print(f"q = {args.eeg_q}, window = {window_size}, step = {step_size}, n_windows = {len(starts)}")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(estimate_window)(signal[s:s + window_size].T, delta_t, args.lag) for s in starts
    )
    est = stack_windows(results)
    est.update(hessian_traces_all(est["eig_Sigma_dt"], est["eig_Sigma_inf"]))

    # bootstrap
    boot_idx = np.arange(0, len(starts), args.bootstrap_stride)
    seeds = seed_seq.spawn(len(boot_idx))

    print(f"Bootstrap: B = {args.n_boot}, windows = {len(boot_idx)}/{len(starts)}")

    boot = Parallel(n_jobs=args.n_jobs)(
        delayed(moving_block_bootstrap)(
            signal[starts[i]:starts[i] + window_size].T, delta_t, args.lag,
            args.n_boot, args.c0, est["Q"][i], seeds[k]
        )
        for k, i in enumerate(boot_idx)
    )

    sigma_r = np.full(len(starts), np.nan)
    for k, i in enumerate(boot_idx):
        sigma_r[i] = np.nanstd(boot[k][0])

    r_boot_all = np.concatenate([b[0] for b in boot])

    return {
        "file": eeg_file,
        "N": N,
        "n_samples": n_samples,
        "delta_t": delta_t,
        "q": args.eeg_q,
        "window_size": window_size,
        "step_size": step_size,
        "window_starts": starts,
        "time_sec": starts * delta_t,
        "seizure_start_sec": args.eeg_seizure_start_sec,
        "lambda_min": est["lambda_min"],
        "eig_A": est["eig_A"],
        "A": est["A"],
        "Q": est["Q"],
        "trace_HJJ_inv": est["trace_HJJ_inv"],
        "trace_HQQ_inv": est["trace_HQQ_inv"],
        "trace_H_inv": est["trace_H_inv"],
        "sigma_r": sigma_r,
        "bootstrap_idx": boot_idx,
        "r_boot_all": r_boot_all[np.isfinite(r_boot_all)],
        "eig_A_boot": np.array([b[1] for b in boot]),
    }


######################################################################################################################
# DATASET: FINANCIAL
######################################################################################################################

def run_fin(seed_seq):

    print("\n" + "=" * 60)
    print("FINANCIAL")
    print("=" * 60)

    fin_file = args.fin_file or os.path.join(
        real_dir, "FINANCIAL_data", "Combined_processed", "Combined_processed", "returns_wide_1s.csv"
    )
    print(f"File financial: {fin_file}")

    # .csv or .parquet (parquet needs pyarrow)
    if fin_file.endswith(".parquet"):
        fin_df = pd.read_parquet(fin_file)
    else:
        fin_df = pd.read_csv(fin_file, index_col=0, parse_dates=True)

    fin_df = fin_df.iloc[:, :args.fin_n_assets]
    print(f"Assets: {list(fin_df.columns)}")

    data = fin_df.values.astype(np.float64)

    # global: global standardization of the log-returns, then absolute value (volatility proxy).
    # per-variable: absolute log-returns, then per-asset centering and global scaling.
    if args.center == "global":
        data = np.abs(standardize(data))
    else:
        data = standardize(np.abs(data))

    fin_df = pd.DataFrame(data, columns=fin_df.columns, index=fin_df.index)

    n_samples, N = fin_df.shape
    delta_t = 1.0
    window_size = args.fin_q * N
    step_size = window_size // args.step_div

    print(f"T = {n_samples}, N = {N}, delta_t = {delta_t} s")
    print(f"q = {args.fin_q}, window = {window_size}, step = {step_size}")

    days = {}
    day_groups = list(fin_df.groupby(fin_df.index.date))
    day_seeds = seed_seq.spawn(len(day_groups))

    for (day, df_day), day_seed in zip(day_groups, day_seeds):

        values = df_day.values
        starts = window_starts(values.shape[0], window_size, step_size)

        if len(starts) == 0:
            print(f"{day}: no valid window (T_day = {values.shape[0]})")
            continue

        t_center = pd.to_datetime(df_day.index[starts + window_size // 2])
        time_sec = np.asarray((t_center - t_center[0].normalize()).total_seconds())

        results = Parallel(n_jobs=args.n_jobs)(
            delayed(estimate_window)(values[s:s + window_size].T, delta_t, args.lag) for s in starts
        )
        est = stack_windows(results)
        est.update(hessian_traces_all(est["eig_Sigma_dt"], est["eig_Sigma_inf"]))

        boot_idx = np.arange(0, len(starts), args.bootstrap_stride)
        seeds = day_seed.spawn(len(boot_idx))

        boot = Parallel(n_jobs=args.n_jobs)(
            delayed(moving_block_bootstrap)(
                values[starts[i]:starts[i] + window_size].T, delta_t, args.lag,
                args.n_boot, args.c0, est["Q"][i], seeds[k]
            )
            for k, i in enumerate(boot_idx)
        )

        sigma_r = np.full(len(starts), np.nan)
        for k, i in enumerate(boot_idx):
            sigma_r[i] = np.nanstd(boot[k][0])

        r_boot = np.concatenate([b[0] for b in boot])

        days[day] = {
            "T_day": values.shape[0],
            "window_starts": starts,
            "time_sec": time_sec,
            "lambda_min": est["lambda_min"],
            "eig_A": est["eig_A"],
            "A": est["A"],
            "Q": est["Q"],
            "trace_HJJ_inv": est["trace_HJJ_inv"],
            "trace_HQQ_inv": est["trace_HQQ_inv"],
            "trace_H_inv": est["trace_H_inv"],
            "sigma_r": sigma_r,
            "bootstrap_idx": boot_idx,
            "r_boot": r_boot[np.isfinite(r_boot)],
            "eig_A_boot": np.array([b[1] for b in boot]),
        }

        print(f"{day}: T_day = {values.shape[0]}, windows = {len(starts)}, "
              f"bootstrapped = {len(boot_idx)}, sigma_r mean = {np.nanmean(sigma_r):.6f}")

    return {
        "file": fin_file,
        "N": N,
        "n_samples": n_samples,
        "delta_t": delta_t,
        "q": args.fin_q,
        "window_size": window_size,
        "step_size": step_size,
        "days": days,
        "r_boot_all": np.concatenate([d["r_boot"] for d in days.values()]),
    }


######################################################################################################################
# DATASET: PLANKTON
######################################################################################################################

def run_plankton(seed_seq):

    print("\n" + "=" * 60)
    print("PLANKTON")
    print("=" * 60)

    plank_file = args.plankton_file or os.path.join(
        real_dir, "PLANKTON_data", "41467_2017_2571_MOESM6_ESM.xlsx"
    )
    print(f"File plankton: {plank_file}")

    df_raw = pd.read_excel(plank_file, index_col=0).drop(columns=["ConsensusLineage"])
    print(f"Shape raw (OTU x time): {df_raw.shape}")

    threshold = int(df_raw.shape[1] * args.plank_min_presence)
    df_filtered = df_raw[(df_raw > 0).sum(axis=1) >= threshold]
    print(f"OTU with presence >= {args.plank_min_presence:.0%}: {df_filtered.shape[0]}")

    top = df_filtered.mean(axis=1).nlargest(args.plank_n_species).index
    data = df_filtered.loc[top].values.T.astype(np.float64)     # T x N

    n_samples, N = data.shape

    # zeros -> half of the smallest non-zero abundance of each species
    for j in range(N):
        col = data[:, j]
        col[col == 0] = col[col > 0].min() / 2

    data = standardize(np.log(data))

    delta_t = args.plank_delta_t

    print(f"T = {n_samples}, N = {N}, delta_t = {delta_t:.4f} d")

    est = estimate_window(data.T, delta_t, args.lag)
    traces = hessian_traces_from_eigs(est["eig_Sigma_dt"], est["eig_Sigma_inf"])

    r_boot, eig_A_boot = moving_block_bootstrap(
        data.T, delta_t, args.lag, args.n_boot, args.c0, est["Q"], seed_seq
    )

    return {
        "file": plank_file,
        "N": N,
        "n_samples": n_samples,
        "delta_t": delta_t,
        "q": n_samples / N,
        "otu_ids": np.asarray(top),
        "lambda_min": est["lambda_min"],
        "eig_A": est["eig_A"],
        "A": est["A"],
        "Q": est["Q"],
        "trace_HJJ_inv": traces[0],
        "trace_HQQ_inv": traces[1],
        "trace_H_inv": traces[2],
        "sigma_r": np.nanstd(r_boot),
        "r_boot_all": r_boot[np.isfinite(r_boot)],
        "eig_A_boot": eig_A_boot,
    }


######################################################################################################################
# SUMMARY (values placed on the empirical phase diagram)
######################################################################################################################

def summarize(name, d):

    if name == "eeg":
        r_hat = np.nanmean(d["lambda_min"])
        sr = np.nanmean(d["sigma_r"])
    elif name == "fin":
        # per day: mean over windows with a bootstrap estimate, then mean over days
        r_list, sr_list = [], []
        for day in d["days"].values():
            valid = np.isfinite(day["lambda_min"]) & np.isfinite(day["sigma_r"])
            if np.any(valid):
                r_list.append(np.mean(day["lambda_min"][valid]))
                sr_list.append(np.mean(day["sigma_r"][valid]))
        r_hat = np.nanmean(r_list)
        sr = np.nanmean(sr_list)
    else:
        r_hat = d["lambda_min"]
        sr = d["sigma_r"]

    dt = d["delta_t"]

    return {
        "r_hat": r_hat,
        "sigma_r": sr,
        "r_dt": r_hat * dt,
        "sigma_r_dt": sr * dt,
        "c_hat": (1 - r_hat * dt) / np.sqrt(2),
        "rel_err": sr / abs(r_hat) if r_hat != 0 else np.inf,
        "q": d["q"],
    }


######################################################################################################################
# MAIN
######################################################################################################################

RUNNERS = {"eeg": run_eeg, "fin": run_fin, "plankton": run_plankton}

if __name__ == "__main__":

    total_start = time.time()

    print(f"center = {args.center}, lag = {args.lag}, B = {args.n_boot}, c0 = {args.c0}, "
          f"stride = {args.bootstrap_stride}, seed = {args.seed}")

    root_seed = np.random.SeedSequence(args.seed)
    dataset_seeds = dict(zip(RUNNERS, root_seed.spawn(len(RUNNERS))))

    output = {"config": vars(args), "datasets": {}, "summary": {}}

    for name in args.datasets:
        t0 = time.time()
        output["datasets"][name] = RUNNERS[name](dataset_seeds[name])
        output["summary"][name] = summarize(name, output["datasets"][name])
        print(f"{name} done in {time.time() - t0:.1f} s")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, s in output["summary"].items():
        print(f"{name:9s}: r*dt = {s['r_dt']:.5f}, c_hat = {s['c_hat']:.5f}, "
              f"sigma_r/r = {s['rel_err']:.3f}, q = {s['q']:.1f}")

    out_path = os.path.join(out_dir, f"realdata_{args.center}_{args.tag}.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time() - total_start:.1f} s")
