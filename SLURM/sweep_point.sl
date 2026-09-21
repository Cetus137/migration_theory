#!/bin/bash
#SBATCH --job-name      sweep
#SBATCH --partition     short
#SBATCH --mem           8G
#SBATCH --time          12:00:00
#SBATCH --cpus-per-task 1
#SBATCH --array         0-599
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
TAG=activity_adhesion_eps40_N100    # all outputs go to figures/${TAG}/, the figure as ${TAG}.png

# the sweep
OVER=active-energy                  # first swept parameter: the x axis of the figure
VALUES=$(seq 0 15 | awk '{printf "%g ", $1 * 36 / 19}')   # 0 to 36 in 20 values; grid Peclet E_a/24 must stay below 2, so ~45 is the ceiling at dx = 1
OVER2=adhesion                      # second swept parameter, one line per value; empty for a one-parameter sweep
VALUES2=$(seq -s ' ' 0 0.06 0.54)   # 0 to 0.54 in 10 values; must stay below the ceiling, measured between 0.7 and 0.9 at these K and alpha
SEEDS="0 1 2"                       # 20 x 10 x 3 = 600 tasks -> --array 0-599

# free energy (energy units are whatever alpha is quoted in; lengths in um)
ALPHA=0.5                           # double-well depth; with K sets width sqrt(K/alpha) = 2 um
K=2.0                               # gradient energy; with alpha sets tension sqrt(2 K alpha)/6
EPSILON=40                          # overlap repulsion; the model default 0.1 is too weak once active
ADHESION=0.0                        # omega -- ignored here, it is swept
AREA_LAMBDA=6000                    # area constraint; sets the timestep at these values

# geometry
CELL_RADIUS=12                      # um; target area pi R^2
PACKING=1.0                         # total cell area / box area; 1 is confluent
N_CELLS=75                         # box side scales as sqrt(N): 85 um at 16, 213 um at 100
SEEDING=tessellated                 # tessellated | circles

# dynamics (time in seconds; time_unit is a label only)
FRICTION=10                         # gamma: resists the field deforming
PROPULSION=force                    # force | velocity
SPEED=0.0                           # v0; velocity mode only
ACTIVE_ENERGY=0                     # E_a; force mode only -- ignored here, it is swept
CELL_FRICTION=10                    # xi: drag on a translating cell; force mode only
ROTATIONAL_DIFFUSION=1e-3           # D_r; persistence time 1/D_r = 1000 s
TIME_UNIT=s

# numerics
GRID_SPACING=1.0                    # dx in um; pure resolution, appears in no physical quantity
SAFETY=0.4                          # fraction of the stability limit the timestep takes

# run
DURATION=5000                       # physical time; 5 persistence times at D_r = 1e-3
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
    --duration             ${DURATION} \
    --warmup               ${WARMUP} \
    --snapshots            ${SNAPSHOTS} \
    --transient            ${TRANSIENT} \
    --outdir               ${REPO}/figures/${TAG} \
    --task-index           ${SLURM_ARRAY_TASK_ID}
