#!/bin/bash
#SBATCH --job-name      win_compare
#SBATCH --partition     short
#SBATCH --mem           4G
#SBATCH --time          00:20:00
#SBATCH --cpus-per-task 1
#SBATCH --output        slogs/win_compare.%j.out
#SBATCH --exclude       compg009,compg010,compg011,compg013
# Compare the two halves of sweep_window_check.sl: figures/window_check_win against
# figures/window_check_dense.
#
# MODE=exact      the flag must not have touched the dynamics: energies to rounding,
#                 diagnostics to the dropped tail. Right for stage 1 (diagnostics only).
# MODE=statistics the dynamics run on windows, so trajectories part company at the
#                 1e-6 level and, in a fluid tissue, then differ in when each
#                 rearrangement happens. The observables are compared instead, in units
#                 of their spread over seeds taken from SPREAD_FROM (a sweep with several
#                 seeds per point at the same parameters), and the wall-time speedup is
#                 reported per point. Right from stage 2 step three onward.
#
#   sbatch --dependency=afterany:<WIN_CHECK_JOBID> compare_window.sl

REPO=${REPO:-/users/kir-fritzsche/aif490/devel/migration_theory}   # override:  sbatch --export=ALL,REPO=/other/checkout ...
LABEL=${LABEL:-N50_stage3b}         # as in sweep_window_check.sl. Override without editing:
                                    #   sbatch --export=ALL,LABEL=N50_stage3 compare_window.sl
REFERENCE=${REFERENCE:-${REPO}/figures/window_check_${LABEL}_dense}   # override with --export=ALL,REFERENCE=...
VARIANT=${VARIANT:-${REPO}/figures/window_check_${LABEL}_win}          # a full path, or a folder name under figures/
MODE=${MODE:-statistics}                # exact | statistics; override with --export=ALL,MODE=exact
# A bare folder name is taken to live under figures/, so
#   REFERENCE=window_check_N50_stage3b_win VARIANT=window_check_N50_step2_win MODE=exact sbatch compare_window.sl
# compares two windowed runs of the same points, e.g. before and after a refactor.
[ -d "${REFERENCE}" ] || [ ! -d "${REPO}/figures/${REFERENCE}" ] || REFERENCE=${REPO}/figures/${REFERENCE}
[ -d "${VARIANT}" ]   || [ ! -d "${REPO}/figures/${VARIANT}" ]   || VARIANT=${REPO}/figures/${VARIANT}
SPREAD_FROM=""                      # a multi-seed sweep at the same points, for spreads; e.g.
                                    # ${REPO}/figures/activity_eps40 for the 16-cell check. Empty:
                                    # differences are reported relative to the dense value instead.

module purge
source /well/kir/config/modules.sh
module load Python/3.11.3-GCCcore-12.3.0
source ~/devel/venv/Python-3.11.3-GCCcore-12.3.0/migration_env/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SPREAD=""
[ -n "${SPREAD_FROM}" ] && SPREAD="--spread-from ${SPREAD_FROM}"

python3 -u ${REPO}/scripts/compare_runs.py ${REFERENCE} ${VARIANT} --token win --transient 0.3 \
    --mode ${MODE} ${SPREAD}
