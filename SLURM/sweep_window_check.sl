#!/bin/bash
#SBATCH --job-name      win_check
#SBATCH --partition     short
#SBATCH --mem           4G
#SBATCH --time          04:00:00
#SBATCH --cpus-per-task 1
#SBATCH --array         0-5
#SBATCH --output        slogs/win_check.%A_%a.out
#SBATCH --exclude       compg009,compg010,compg011,compg013
# Run the same points twice from the same code -- the first N_POINTS tasks dense into
# figures/window_check_<LABEL>_dense, the next N_POINTS with --window into
# figures/window_check_<LABEL>_win -- so the two sets differ in nothing but the flag
# and can be compared with compare_window.sl. With the dynamics on windows the two
# trajectories agree to the dropped tail and then, in the fluid phase, may part company
# on when rearrangements happen; the comparison is of the observables, and of the wall
# time, which is the point of the windows.
#
# Measured 2026-09-21 at 16 cells (LABEL=N16, 25 points, DURATION=10000): identical
# energies to 1e-8 for the whole run at every activity up to 21, observables within
# noise everywhere, speedup 1.17x median -- modest because at 16 cells the window is
# two thirds of the box. The saving grows with the box: this configuration is the 50-cell
# measurement, three activities, where the window is about a fifth of the box.
#
# --array must be 0 to (2 * N_POINTS - 1). Then:
#   sbatch --dependency=afterany:<JOBID> compare_window.sl     (with the same LABEL)

# ── Configuration ─────────────────────────────────────────────────────────────
REPO=/users/kir-fritzsche/aif490/devel/migration_theory
LABEL=${LABEL:-N50_stage3b}         # names the pair of output folders; use it in compare_window.sl too.
                                    # Override without editing:  sbatch --export=ALL,LABEL=xyz sweep_window_check.sl
                                    # N50 = stage 2 windows; N50_stage3 = shared sum from windows (loop);
                                    # N50_stage3b = one gather per step + bincount accumulation

OVER=active-energy
VALUES="0 12 24"                    # solid, transition, fluid; 3 points -> N_POINTS=3, --array 0-5
N_POINTS=3
SEEDS="0"                           # one seed is enough for a file-by-file comparison

ALPHA=0.5
K=2.0
EPSILON=40
ADHESION=0.0
AREA_LAMBDA=6000
CELL_RADIUS=12
PACKING=1.0
N_CELLS=50                          # box 150 um = 150^2 points; a window is ~69 points, a fifth of the area
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

DURATION=2500                       # dense at 50 cells is ~50 ms/step: ~35 min for 38k steps on a quiet node
WARMUP=25
SNAPSHOTS=200
TRANSIENT=0.3

# the two halves of the array: same point, with and without the flag
POINT=$(( SLURM_ARRAY_TASK_ID % N_POINTS ))
if [ ${SLURM_ARRAY_TASK_ID} -lt ${N_POINTS} ]; then
    WINDOW=""
    TAG=window_check_${LABEL}_dense
else
    WINDOW="--window"
    TAG=window_check_${LABEL}_win
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
