#!/bin/bash
#SBATCH --job-name      win_compare
#SBATCH --partition     short
#SBATCH --mem           4G
#SBATCH --time          00:20:00
#SBATCH --cpus-per-task 1
#SBATCH --output        slogs/win_compare.%j.out
#SBATCH --exclude       compg009,compg010,compg011,compg013
# Compare the two halves of sweep_window_check.sl file by file: figures/window_check_win
# against figures/window_check_dense. Energies and contacts must agree to rounding, areas
# and perimeters to the tail the window drops, centres to the dense circular mean's
# small bias. The log ends with the worst difference per quantity.
#
#   sbatch --dependency=afterany:<WIN_CHECK_JOBID> compare_window.sl

REPO=/users/kir-fritzsche/aif490/devel/migration_theory
REFERENCE=${REPO}/figures/window_check_dense
VARIANT=${REPO}/figures/window_check_win

module purge
source /well/kir/config/modules.sh
module load Python/3.11.3-GCCcore-12.3.0
source ~/devel/venv/Python-3.11.3-GCCcore-12.3.0/migration_env/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python3 -u ${REPO}/scripts/compare_runs.py ${REFERENCE} ${VARIANT} --token win --transient 0.2
