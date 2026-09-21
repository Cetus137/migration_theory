#!/bin/bash
#SBATCH --job-name      sweep_agg
#SBATCH --partition     short
#SBATCH --mem           4G
#SBATCH --time          00:30:00
#SBATCH --cpus-per-task 1
#SBATCH --output        slogs/sweep_agg.%j.out
#SBATCH --exclude       compg009,compg010,compg011,compg013
# Tabulate a sweep run by sweep_point.sl: loads every point's .npz from
# figures/<TAG>/, prints the observables with mean +/- spread over seeds, and writes
# figures/<TAG>/<TAG>.png -- the observables against OVER, one line per value of
# OVER2. Points whose file is missing are listed, not fatal, so a sweep with a failed
# or timed-out task still tabulates -- hence afterany, not afterok.
#
# The block below must match sweep_point.sl exactly: the filenames it looks for are
# built from these values, and a differing one means "file not found".
#
# Edit the block below, then, with the array's job id from sbatch:
#   sbatch --dependency=afterany:<ARRAY_JOBID> sweep_aggregate.sl

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
    --tag                  ${TAG} \
    --aggregate  --figure
