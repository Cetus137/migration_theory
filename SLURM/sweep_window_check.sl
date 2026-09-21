#!/bin/bash
#SBATCH --job-name      win_check
#SBATCH --partition     short
#SBATCH --mem           2G
#SBATCH --time          01:00:00
#SBATCH --cpus-per-task 1
#SBATCH --array         0-49
#SBATCH --output        slogs/win_check.%A_%a.out
#SBATCH --exclude       compg009,compg010,compg011,compg013
# Run the 16-cell activity sweep twice from the same code -- tasks 0-24 dense into
# figures/window_check_dense, tasks 25-49 with --window into figures/window_check_win --
# so the two sets differ in nothing but the flag and can be compared file by file with
# compare_window.sl. The dynamics are untouched by the flag at this stage, so the
# energies must agree to rounding and the diagnostics to the tail the window drops.
#
# Both sets are run fresh rather than reusing figures/activity_eps40: that sweep was
# made before the repulsion rewrite, and over 133k steps a rounding difference in an
# active tissue can move a rearrangement, which would look like a failure and is not.
#
# 50 points at ~20 min each. Then:   sbatch --dependency=afterany:<JOBID> compare_window.sl

# ── Configuration ─────────────────────────────────────────────────────────────
REPO=/users/kir-fritzsche/aif490/devel/migration_theory

OVER=active-energy
VALUES=$(seq -s ' ' 0 1.5 36)       # 25 values, as the activity_eps40 sweep
SEEDS="0"                           # one seed is enough for a file-by-file comparison

ALPHA=0.5
K=2.0
EPSILON=40
ADHESION=0.0
AREA_LAMBDA=6000
CELL_RADIUS=12
PACKING=1.0
N_CELLS=16
SEEDING=tessellated
FRICTION=10
PROPULSION=force
SPEED=0.0
ACTIVE_ENERGY=0                     # swept
CELL_FRICTION=10
ROTATIONAL_DIFFUSION=1e-3
TIME_UNIT=s
GRID_SPACING=1.0
SAFETY=0.4

DURATION=10000
WARMUP=25
SNAPSHOTS=200
TRANSIENT=0.2

# the two halves of the array: same point, with and without the flag
N_POINTS=25
POINT=$(( SLURM_ARRAY_TASK_ID % N_POINTS ))
if [ ${SLURM_ARRAY_TASK_ID} -lt ${N_POINTS} ]; then
    WINDOW=""
    TAG=window_check_dense
else
    WINDOW="--window"
    TAG=window_check_win
fi

module purge
source /well/kir/config/modules.sh
module load Python/3.11.3-GCCcore-12.3.0
source ~/devel/venv/Python-3.11.3-GCCcore-12.3.0/migration_env/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

mkdir -p ${REPO}/figures/${TAG}

python3 -u ${REPO}/scripts/sweep.py \
    --over                 ${OVER} \
    --values               ${VALUES} \
    --seeds                ${SEEDS} \
    --alpha                ${ALPHA} \
    --K                    ${K} \
    --epsilon              ${EPSILON} \
    --adhesion             ${ADHESION} \
    --area-lambda          ${AREA_LAMBDA} \
    --cell-radius          ${CELL_RADIUS} \
    --packing              ${PACKING} \
    --n-cells              ${N_CELLS} \
    --seeding              ${SEEDING} \
    --friction             ${FRICTION} \
    --propulsion           ${PROPULSION} \
    --speed                ${SPEED} \
    --active-energy        ${ACTIVE_ENERGY} \
    --cell-friction        ${CELL_FRICTION} \
    --rotational-diffusion ${ROTATIONAL_DIFFUSION} \
    --time-unit            ${TIME_UNIT} \
    --grid-spacing         ${GRID_SPACING} \
    --safety               ${SAFETY} \
    ${WINDOW} \
    --duration             ${DURATION} \
    --warmup               ${WARMUP} \
    --snapshots            ${SNAPSHOTS} \
    --transient            ${TRANSIENT} \
    --outdir               ${REPO}/figures/${TAG} \
    --task-index           ${POINT}
