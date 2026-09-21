#!/bin/bash
#SBATCH --job-name      sweep
#SBATCH --partition     short
#SBATCH --mem           4G
#SBATCH --time          03:00:00
#SBATCH --cpus-per-task 1
#SBATCH --array         0-101
#SBATCH --output        slogs/sweep.%A_%a.out
#SBATCH --exclude       compg009,compg010,compg011,compg013
# One point of a parameter sweep per array task. Each task writes one
# figures/<TAG>/<name>.npz; submit sweep_aggregate.sl afterwards to tabulate them.
#
# The sweep is over OVER, or over the grid OVER x OVER2 when OVER2 is set. Points are
# numbered with the seed varying fastest, then OVER2, then OVER: with n2 values of
# OVER2 and n_s seeds, point k is OVER index k // (n2 * n_s), OVER2 index
# (k // n_s) % n2, seed k % n_s. --array must run 0 to (n1 * n2 * n_s - 1).
#
# Every model parameter is listed below at its value. The swept ones' own lines in
# the model block are ignored, since each point replaces them. Everything else is
# held fixed at the value written here, so nothing is left to a default that could
# change.
#
# numpy is single-threaded here, so 1 CPU is right whatever the size. Measured
# 2026-09-20: 16 cells on 85^2 points take ~5 ms per step (a DURATION=10000 point in
# ~20 min, 60 MB); 100 cells on 213^2 points take ~0.45 s per step (a DURATION=5000
# point, ~77k steps, in ~10 h, 310 MB). --time and --mem are set for the latter.
#
# Edit the block below (keep it identical to sweep_aggregate.sl), set --array, then:
#   mkdir -p slogs && sbatch sweep_point.sl

# ── Configuration ─────────────────────────────────────────────────────────────
REPO=/users/kir-fritzsche/aif490/devel/migration_theory
TAG=speed_friction_eps40_N100      # all outputs go to figures/${TAG}/, the figure as ${TAG}.png

# the sweep
OVER=free-speed                     # um/s of an unobstructed cell; the same on every friction line
VALUES=$(seq -s ' ' 0 0.0125 0.2)  # 0 to 0.2 um/s in 17 values; at xi = 10 that is activity 0 to 8, threshold near 0.055
OVER2=cell-friction                 # xi: at fixed speed the force E_a/R = v xi rises with it; one line per value
VALUES2="3 10 30"                   # a decade of friction, so a decade of force at the same speed
SEEDS="0 1"                         # 17 x 3 x 2 = 102 tasks -> --array 0-101

# free energy (energy units are whatever alpha is quoted in; lengths in um)
ALPHA=0.5                           # double-well depth; with K sets width sqrt(K/alpha) = 2 um and tension 0.2357
K=2.0                               # gradient energy; see ALPHA
EPSILON=40                          # overlap repulsion; the model default 0.1 is too weak once active
ADHESION=0.0                        # omega
AREA_LAMBDA=6000                    # area constraint; sets the timestep at these values

# geometry
CELL_RADIUS=12                      # um; target area pi R^2
PACKING=1.0                         # total cell area / box area; 1 is confluent
N_CELLS=100                         # box 213 um = 213^2 points; windows are a seventh of it
SEEDING=tessellated                 # tessellated | circles

# dynamics (time in seconds; time_unit is a label only)
FRICTION=10                         # gamma: resists the field deforming
PROPULSION=force                    # force | velocity
SPEED=0.0                           # v0; velocity mode only
ACTIVE_ENERGY=0                     # E_a -- ignored here, set from the swept speed and friction as v R xi
CELL_FRICTION=10                    # xi -- ignored here, it is swept
ROTATIONAL_DIFFUSION=1e-3           # D_r; persistence time 1000 s
TIME_UNIT=s

# numerics
GRID_SPACING=1.0                    # dx in um; pure resolution, appears in no physical quantity
SAFETY=0.4                          # fraction of the stability limit the timestep takes
WINDOW="--window"                   # each cell on its own patch: same physics, ~6x faster at this size (measured 2026-09-21)

# run
DURATION=5000                       # 5 persistence times
WARMUP=25                           # passive relaxation before recording
SNAPSHOTS=200                       # samples per run; sets the sampling interval of the analysis
TRANSIENT=0.3                       # fraction of the run discarded before measuring; 0.3 covers the 720 s shape relaxation

module purge
source /well/kir/config/modules.sh
module load Python/3.11.3-GCCcore-12.3.0
source ~/devel/venv/Python-3.11.3-GCCcore-12.3.0/migration_env/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

SECOND=""
[ -n "${OVER2}" ] && SECOND="--over ${OVER2} --values ${VALUES2}"

mkdir -p ${REPO}/figures/${TAG}

python3 -u ${REPO}/scripts/sweep.py \
    --over                 ${OVER} \
    --values               ${VALUES} \
    ${SECOND} \
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
    --task-index           ${SLURM_ARRAY_TASK_ID}
