#!/bin/bash
# One script for everything.
#
#   bash run.sh                 redraw the paper figures from the saved results (about 1 minute, any laptop);
#                               the empirical figures need "bash run.sh realdata" first
#   bash run.sh synthetic       recompute the synthetic results from scratch (heavy: many hours, tens of GB)
#   bash run.sh realdata        recompute the empirical results from scratch (heavy; needs the raw data in data/)
#
# After "synthetic" or "realdata", run "bash run.sh" again to draw the figures.
# N_JOBS sets the number of parallel processes for the heavy steps (default: all cores), e.g. N_JOBS=16 bash run.sh synthetic
set -euo pipefail
R=$(cd "$(dirname "$0")" && pwd)
SYN="python3 $R/code/synthetic"
REAL="python3 $R/code/realdata"
RES=$R/results/realdata
OUT=$R/figures
J=${N_JOBS:--1}
MODE=${1:-figures}

figures () {
  mkdir -p "$OUT"
  O=$OUT/other
  # synthetic figures (main 1, 2, 3, 6; SI 1, 2)
  $SYN/fig1_spectra.py --fisher-block QQ --replot
  $SYN/fig2_soft_mode.py --panels --q 15 --line-colors purple_magenta --debias --replot
  $SYN/fig2_soft_mode.py --panels --q 15 --line-colors purple_magenta --debias --tau-w --replot
  $SYN/fig3_phase_diagram.py > /dev/null
  $SYN/fig_ginibre_sigma_r.py --panels --replot
  $SYN/fig_kcW_vs_mc.py --tau-w --output-dir $O/fig_kcW_vs_mc > /dev/null
  cp_both () { for e in png pdf; do [ -f "$1.$e" ] && cp "$1.$e" "$OUT/$2.$e"; done; return 0; }
  cp_both $O/fig1_spectra/fig1_spectra_QQ                            MAIN_FIG1_spectra_QQ
  cp_both $O/fig2_soft_mode/fig2_soft_mode_panels_q15_pm_debias      MAIN_FIG2_soft_mode
  cp_both $O/fig3_phase_diagram/fig3_phase_diagram_kcw               MAIN_FIG3_phase_diagram
  cp_both $O/fig_ginibre_sigma_r/fig_ginibre_sigma_r_panels          MAIN_FIG6_ginibre_sigma_r
  cp_both $O/fig_kcW_vs_mc/fig_kcW_vs_mc_qeff_tauw_main              SI_FIG1_kappa_test
  cp_both $O/fig2_soft_mode/fig2_soft_mode_panels_q15_pm_tauw_debias SI_FIG2_soft_mode_tauw

  # empirical figures (main 4, 5; SI 3, 4, 5, 6): the results derived from the financial data cannot be
  # redistributed, so these figures are drawn only after "bash run.sh realdata" (needs all raw data in data/)
  if [ ! -f "$RES/realdata_fig4.pkl" ]; then
    echo "Synthetic figures written to $OUT. Empirical figures skipped: run 'bash run.sh realdata' first (see data/README.md)."
    return 0
  fi
  $REAL/plots_realdata_revised.py --dataset $RES/realdata_fig4.pkl --output-dir $O/realdata --figures rhat_physical
  $REAL/plots_realdata_revised.py --dataset $RES/realdata_fig5.pkl --output-dir $O/realdata --figures phase_diagram \
      --residual-bootstrap $RES/fig5_residual_bootstrap.pkl
  for FMT in png pdf; do
    $REAL/plots_realdata_revised.py --dataset $RES/realdata_global_REAL.pkl --output-dir $O/realdata \
        --figures hessian_traces --format $FMT --dpi 600
  done
  $REAL/plots_revised.py > /dev/null     # writes figures/robust_fin.pdf and figures/robust_eeg.pdf
  $REAL/si_fig4_var_q.py --replot > /dev/null

  # final empirical figures with the names used in the manuscript
  cp $O/realdata/realdata_rhat_physical.pdf                          $OUT/MAIN_FIG4_rhat_realdata.pdf
  cp $O/realdata/realdata_phase_diagram.pdf                          $OUT/MAIN_FIG5_phase_diagram_realdata.pdf
  cp_both $O/realdata/realdata_hessian_trace_eeg                     SI_FIG3_hessian_trace_EEG
  cp_both $O/realdata/SI_FIG4a_est_var_EEG                           SI_FIG4a_est_var_EEG
  cp_both $O/realdata/SI_FIG4b_est_var_FIN                           SI_FIG4b_est_var_FIN
  cp_both $O/realdata/SI_FIG4c_est_var_PLANK                         SI_FIG4c_est_var_PLANK
  mv $OUT/robust_fin.pdf $OUT/SI_FIG5_robust_fin.pdf
  mv $OUT/robust_eeg.pdf $OUT/SI_FIG6_robust_eeg.pdf
  echo "Paper figures written to $OUT (side outputs in $O)"
}

synthetic () {
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  # Main Fig. 1: GOE and Ginibre, q = 10, dt = 1 (tag main, seed 20261024)
  for ENS in GOE GINIBRE; do
    $SYN/generate_data.py --ensemble $ENS --q 10 --n-jobs $J
    $SYN/estimate.py      --ensemble $ENS --q 10 --n-jobs $J
  done
  $SYN/fig1_spectra.py --fisher-block QQ                      # also writes results/synthetic/plotdata/
  # Main Fig. 2 and SI Fig. 2: GOE, q = 15, dt = 1, 0.5, 0.2
  for spec in "1.0 q15dt1 20261119" "0.5 q15dt05 20261120" "0.2 q15dt02 20261121"; do
    set -- $spec
    $SYN/generate_data.py --ensemble GOE --q 15 --delta-t $1 --seed $3 --tag $2 --n-jobs $J
    $SYN/estimate.py      --ensemble GOE --q 15 --tag $2 --n-jobs $J
  done
  $SYN/fig2_soft_mode.py --panels --q 15 --line-colors purple_magenta --debias
  $SYN/fig2_soft_mode.py --panels --q 15 --line-colors purple_magenta --debias --tau-w
  # Main Fig. 6: Ginibre, q = 15, dt = 1, 0.5, 0.2
  for spec in "1.0 ginq15dt1 20261122" "0.5 ginq15dt05 20261123" "0.2 ginq15dt02 20261124"; do
    set -- $spec
    $SYN/generate_data.py --ensemble GINIBRE --q 15 --delta-t $1 --seed $3 --tag $2 --n-jobs $J
    $SYN/estimate.py      --ensemble GINIBRE --q 15 --tag $2 --n-jobs $J
  done
  $SYN/fig_ginibre_sigma_r.py --panels
  # Main Fig. 3: phase diagrams from the kappa_cW theory (fast)
  $SYN/phase_diagram_kcw.py
  # SI Fig. 1: Monte Carlo kappa_r
  $SYN/kappa_r_mc.py --n-jobs $J
}

realdata () {
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  # 1. sliding-window pipeline: per-variable centering (financial and plankton inputs of Figs. 4-5)
  #    and global centering (SI Fig. 3)
  mkdir -p $RES/reconcile/real_processed
  ln -sfn ../../../data $RES/reconcile/real
  $REAL/estimate_realdata_MOU.py --data-dir $RES/reconcile --center per-variable --seed 12345 --tag seed12345 --n-jobs $J
  $REAL/estimate_realdata_MOU.py --data-dir $RES/reconcile --center global --seed 12345 --tag seed12345 --n-jobs $J
  cp $RES/reconcile/real_processed/realdata_global_seed12345.pkl $RES/realdata_global_REAL.pkl
  # 2. Main Fig. 4: EEG full recording at dt = 1/64 s, financial 1 s
  $REAL/estimate_realdata_optimal.py --n-jobs $J --eeg-k 8 --eeg-segment-sec 0 100000 --out $RES/realdata_fig4.pkl
  # 3. Main Fig. 5: EEG pre-ictal at dt = 0.125 s; theory boundaries; plankton residual bootstrap
  $REAL/estimate_realdata_optimal.py --n-jobs $J
  $REAL/fig5_boundaries.py --n-jobs $J
  $REAL/fig5_residual_bootstrap.py --n-jobs $J
  # 4. SI Figs. 5-6: robustness to the sampling interval
  PY="$REAL/robustness_sampling.py --n-jobs $J --n-boot 50"
  $PY --dataset fin --transform logrv   --protocol fixedM --q 20
  $PY --dataset fin --transform abs_sub --protocol fixedM --q 20
  $PY --dataset fin --transform logrv   --protocol fixedT --T-sec 7800
  $PY --dataset fin --transform abs_sub --protocol fixedT --T-sec 7800
  $PY --dataset fin --transform logrv   --protocol fixedM --q 20 --deseason
  $PY --dataset eeg --transform sub --protocol fixedM --q 10 --max-phases 8
  $PY --dataset eeg --transform sub --protocol fixedT --T-sec 90 --max-phases 8
  for TR in logrv abs_sub; do
    $PY --dataset fin --transform $TR --protocol fixedT --T-sec 23400 --factors 1 2 5 10 30 60 --tag _fullday
  done
  $REAL/pooled_days.py --n-jobs $J --n-boot 200
  # SI Fig. 4: estimated variance of the entries of Q (fast)
  $REAL/si_fig4_var_q.py
  $REAL/theory_kcw.py --n-jobs $J
  # 5. Ljung-Box tests on the innovations (Supplementary tables)
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 $REAL/innovation_tests.py
}

case $MODE in
  figures)   figures ;;
  synthetic) synthetic; echo "Done. Now: bash run.sh" ;;
  realdata)  realdata;  echo "Done. Now: bash run.sh" ;;
  *) echo "usage: bash run.sh [figures|synthetic|realdata]"; exit 1 ;;
esac
