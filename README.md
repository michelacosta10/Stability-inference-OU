# Code to reproduce the figures of the manuscript

## System requirements

- **Operating system:** tested on Linux (CentOS 7). The code is plain Python and has no OS-specific parts, so it
  is expected to run also on macOS and Windows.
- **Python:** tested with Python 3.9.13 (Anaconda 2022.10).
- **Dependencies** (versions used for the paper, also in `requirements.txt`):
  numpy 1.21.5, scipy 1.9.1, pandas 1.4.4, matplotlib 3.5.2, joblib 1.1.0, openpyxl 3.0.10.
- **Hardware:** no non-standard hardware required. Any standard computer is enough for the demo and for redrawing the figures. The full recomputation of the
  results is heavy (see below).

## Installation guide

```bash
cd stability-inference-OU
python3 -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt
```

No compilation is needed. Typical installation time: a few minutes on a standard computer (time to download the
packages).

## Demo (simulated data)

A small run of the simulation and estimation pipeline on a GOE drift matrix (N = 50 variables, M = 500 samples,
two values of c, 4 matrices x 5 trajectories each). Run from the `stability-inference-OU` folder:

```bash
python3 code/synthetic/generate_data.py --ensemble GOE --N 50 --q 10 --n-A 4 --n-sim 5 --c-values 0.2 0.5 \
        --tag demo --n-jobs 4 --data-dir demo_output
python3 code/synthetic/estimate.py --ensemble GOE --q 10 --tag demo --n-jobs 4 --data-dir demo_output
```

Expected output (timings will differ slightly):

```
GOE  N=50  M=500  dt=1.0  n_A=4  n_sim=5  c=[0.2 0.5]
c = 0.2: 2.6 s, rejected A = 0  ->  demo_output/sim_trajectories/DATASIM_GOE_N50_q10_Dt1_nA4_nsim5_c0.2_demo.npz
c = 0.5: 0.3 s, rejected A = 0  ->  demo_output/sim_trajectories/DATASIM_GOE_N50_q10_Dt1_nA4_nsim5_c0.5_demo.npz

2 files, symmetrize=False
c = 0.2: 2.8 s, failures = 0  ->  demo_output/estimates/EST_GOE_N50_q10_Dt1_nA4_nsim5_c0.2_demo.npz
c = 0.5: 0.1 s, failures = 0  ->  demo_output/estimates/EST_GOE_N50_q10_Dt1_nA4_nsim5_c0.5_demo.npz
```

The first command writes the simulated trajectories, the second the estimated A, Q, Sigma_dt and Sigma_inf for
each trajectory (`.npz` files, about 4 MB in total). Run time: about 5 s per command (10 s in total) with 4
processes. The same two scripts, with the parameters in `run.sh`, produce the results behind
the synthetic figures of the paper.

## Redrawing the figures

```bash
bash run.sh
```

This redraws the synthetic figures from the saved results in `results/`, in about one minute (50 s on our
machine). The figures appear in `figures/` with the names used in the manuscript; side outputs are in
`figures/other/`.

The results derived from the financial data cannot be redistributed (see *Data*), so the empirical figures are not
redrawn by `bash run.sh` alone: they need the raw data and `bash run.sh realdata` first. The empirical figures
that do not involve financial data are included as images in `figures/`.

## Which script produces which figure

| figure | script (in `code/`) | saved input |
|---|---|---|
| Main Fig. 1 | `synthetic/fig1_spectra.py` | `results/synthetic/plotdata/` |
| Main Fig. 2 | `synthetic/fig2_soft_mode.py` | `results/synthetic/plotdata/` |
| Main Fig. 3 | `synthetic/fig3_phase_diagram.py` (theory: `synthetic/phase_diagram_kcw.py`) | `results/synthetic/phase_diagram/` |
| Main Fig. 4 | `realdata/plots_realdata_revised.py --figures rhat_physical` (estimates: `realdata/estimate_realdata_optimal.py`) | needs `run.sh realdata` |
| Main Fig. 5 | `realdata/plots_realdata_revised.py --figures phase_diagram` (also `realdata/fig5_boundaries.py`, `realdata/fig5_residual_bootstrap.py`) | needs `run.sh realdata` |
| Main Fig. 6 | `synthetic/fig_ginibre_sigma_r.py` | `results/synthetic/plotdata/` |
| SI Fig. 1 | `synthetic/fig_kcW_vs_mc.py` (Monte Carlo: `synthetic/kappa_r_mc.py`) | `results/synthetic/kappa_r_mc/` |
| SI Fig. 2 | `synthetic/fig2_soft_mode.py --tau-w` | `results/synthetic/plotdata/` |
| SI Fig. 3 | `realdata/plots_realdata_revised.py --figures hessian_traces` (estimates: `realdata/estimate_realdata_MOU.py`) | needs `run.sh realdata` |
| SI Fig. 4 | `realdata/si_fig4_var_q.py` | needs `run.sh realdata` |
| SI Figs. 5-6 | `realdata/plots_revised.py` (estimates: `realdata/robustness_sampling.py`, `realdata/theory_kcw.py`) | needs `run.sh realdata` |
| Supplementary tables (Ljung-Box) | `realdata/innovation_tests.py` | needs `run.sh realdata` |

Shared functions are in `code/synthetic/mou_core.py` and `code/realdata/ou_core.py`; the figure style is in
`code/style/`. Every script prints its options with `python3 <script> --help`; the exact command behind each
figure is in `run.sh`.

## Recomputing the results from scratch

```bash
N_JOBS=16 bash run.sh synthetic    # simulations and estimates (many hours, tens of GB on disk)
N_JOBS=16 bash run.sh realdata     # empirical analyses (needs the raw data in data/, see data/README.md)
bash run.sh                        # then draw all the figures
```

These steps are plain Python calls with fixed random seeds, so they give the same numbers on any machine. `N_JOBS` sets the
number of parallel processes.

## Data

- **EEG:** intracranial EEG from the SWEC-ETHZ iEEG database, short-term dataset, http://ieeg-swez.ethz.ch
  first patient (ID1), first seizure (`Sz1.mat`); not included, download it from there.
- **Plankton:** Supplementary Data of Martin-Platero et al., Nat. Commun. 9, 266 (2018) (public, included in
  `data/PLANKTON_data/`).
- **Financial:** order-book data from LOBSTER and Databento. They are **not included** for licensing reasons,
  and neither are the results derived from them.

Where to put the files is described in `data/README.md`.

## License

The code is released under the MIT License (see `LICENSE`). The plankton data in `data/PLANKTON_data/` come from
Martin-Platero et al., Nat. Commun. 9, 266 (2018), and keep their original terms.
