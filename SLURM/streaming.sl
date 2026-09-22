#!/bin/bash
#SBATCH --job-name      streaming
#SBATCH --partition     short
#SBATCH --mem           4G
#SBATCH --time          00:20:00
#SBATCH --cpus-per-task 1
#SBATCH --output        slogs/streaming.%j.out
#SBATCH --exclude       compg009,compg010,compg011,compg013
# Dissect the streaming in saved runs: the full velocity-correlation curve at every lag,
# its longitudinal and transverse parts, a map of the drift-subtracted velocity field,
# and the size of groups of neighbours moving together. Nothing is simulated: it reads
# the .npz files, one row of the figure per file, and prints a table.
#
# Two questions the two blocks below ask. Comment one set of FILES out.
#   1. Is the streaming persistence-driven?  Same activity (a = 4), persistence times
#      1000, 333 and 100 s from the persistence sweep, plus the 2 s animation run when
#      it has finished (its .npz lands in figures/activity_eps40/).
#   2. Is it incompressibility-driven?  Same activity (a = 2), packings 0.6 to 1.2 from
#      the packing sweep: a tissue with free space need not close its streams into swirls.
#
#   sbatch streaming.sl

# ── Configuration ─────────────────────────────────────────────────────────────
REPO=/users/kir-fritzsche/aif490/devel/migration_theory
LAG=20                              # sample intervals for the split correlation, arrows and clusters (20 = 500 s)
TRANSIENT=0.3

# 1. persistence, at a = 4 (E_a = 11.3137), seed 0
#LABEL_BY=rotational_diffusion
#OUT=${REPO}/figures/activity_persistence_eps40_N75/streaming_persistence_a4.png
#FILES=(
#    ${REPO}/figures/activity_persistence_eps40_N75/*_Ea11.3137_*Dr0.001_*seed0.npz
#    ${REPO}/figures/activity_persistence_eps40_N75/*_Ea11.3137_*Dr0.003_*seed0.npz
#    ${REPO}/figures/activity_persistence_eps40_N75/*_Ea11.3137_*Dr0.01_*seed0.npz
#   ${REPO}/figures/activity_eps40/*_N75_*_Ea28.4_*Dr0.5_*seed0.npz     # a = 10, 2 s: uncomment once the animate job has written it
#)

# 2. packing, at a = 2 (E_a = 5.65685), seed 0
# LABEL_BY=packing
# OUT=${REPO}/figures/activity_packing_eps40_N100/streaming_packing_a2.png
# FILES=(
#     ${REPO}/figures/activity_packing_eps40_N100/*_pack0.6_*_Ea5.65685_*seed0.npz
#     ${REPO}/figures/activity_packing_eps40_N100/*_pack0.8_*_Ea5.65685_*seed0.npz
#     ${REPO}/figures/activity_packing_eps40_N100/*_pack1_*_Ea5.65685_*seed0.npz
#     ${REPO}/figures/activity_packing_eps40_N100/*_pack1.2_*_Ea5.65685_*seed0.npz
# )

# 4. Is the swirl the box?  Same activity (a = 4) and packing 1 at 50, 75 and 100 cells.
#    C(r) crosses zero at ~2.3 spacings and stays negative to the half box, and the
#    arrow maps show one swirl per box. If the zero crossing moves with the box size,
#    the pattern is the largest mode the box allows, not a length of the tissue.
 LABEL_BY=n_cells
 OUT=${REPO}/figures/activity_eps40/streaming_boxsize_a4.png
 FILES=(
     ${REPO}/figures/window_check_N50_stage3b_win/*_Ea12_*seed0.npz
     ${REPO}/figures/activity_persistence_eps40_N75/*_Ea11.3137_*Dr0.001_*seed0.npz
     ${REPO}/figures/activity_packing_eps40_N100/*_pack1_*_Ea11.3137_*seed0.npz
)

# 3. Is it force transmission?  Same free speed (0.05 um/s) at cell friction xi = 3, 10
#    and 30, from the speed-friction sweep. If neighbours share velocity because the
#    packing transmits force, a cell that is harder to drag through its surroundings
#    (higher xi) is pushed along less by them, and the correlation length shrinks.
# LABEL_BY=cell_friction
# OUT=${REPO}/figures/speed_friction_eps40_N100/streaming_friction_v0.05.png
# FILES=(
#     ${REPO}/figures/speed_friction_eps40_N100/*_Ea1.8_xi3_*seed0.npz
#     ${REPO}/figures/speed_friction_eps40_N100/*_Ea6_xi10_*seed0.npz
#     ${REPO}/figures/speed_friction_eps40_N100/*_Ea18_xi30_*seed0.npz
# )

module purge
source /well/kir/config/modules.sh
module load Python/3.11.3-GCCcore-12.3.0
source ~/devel/venv/Python-3.11.3-GCCcore-12.3.0/migration_env/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

python3 -u ${REPO}/scripts/streaming.py "${FILES[@]}" --label-by ${LABEL_BY} --lag ${LAG} \
    --transient ${TRANSIENT} --out ${OUT}
