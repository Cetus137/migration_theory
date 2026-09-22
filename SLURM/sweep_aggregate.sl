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
TAG=friction_a3_eps40_N75           # all outputs go to figures/${TAG}/, the figure as ${TAG}.png

# the sweep
OVER=friction                       # gamma: how fast cells remodel; shape time gamma R^2 / K = 180 s at 2.5, 1730 s at 24
VALUES="0.5 1 1.4 2 2.5 3 3.5 4 5 6 7 8.5 10 12 14 17 20 24"   # 18 values, log-spaced, De = shape time / traversal time from 0.2 to 10. Peclet at a = 3 reached 2.18 for one seed at 28, so 24 is the safe top
OVER2=                              # single axis: a fluid tissue at the standard tension, friction alone varied
VALUES2=
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"         # 18 x 20 = 360 tasks -> --array 0-359; 0.5, 1, 1.4, 2 were added after the first run and are tasks 0-79

# free energy (energy units are whatever alpha is quoted in; lengths in um)
ALPHA=0.5                           # double-well depth; with K sets width sqrt(K/alpha) = 2 um and sigma = 0.2357
K=2.0                               # gradient energy; with alpha sets tension sqrt(2 K alpha)/6
EPSILON=40                          # overlap repulsion; the model default 0.1 is too weak once active
ADHESION=0.0                        # omega
AREA_LAMBDA=6000                    # area constraint; sets the timestep at these values

# geometry
CELL_RADIUS=12                      # um; target area pi R^2
PACKING=1.0                         # confluent
N_CELLS=75                          # box 184 um = 184^2 points; a point costs ~25 min x 10/gamma: 7 h at gamma = 0.5, 10 min at 24
SEEDING=tessellated                 # tessellated | circles

# dynamics (time in seconds; time_unit is a label only)
FRICTION=10                         # -- ignored here, it is swept
PROPULSION=force                    # force | velocity
SPEED=0.0                           # v0; velocity mode only
ACTIVE_ENERGY=0                     # -- ignored here: ACTIVITY below sets E_a from each point's tension
ACTIVITY=3                          # dimensionless E_a/(sigma R) held at every point; near the confluent unjamming threshold
CELL_FRICTION=10                    # xi: drag on a translating cell; force mode only
ROTATIONAL_DIFFUSION=1e-3           # D_r; persistence time 1000 s
TIME_UNIT=s

# numerics
GRID_SPACING=1.0                    # dx in um; pure resolution, appears in no physical quantity
SAFETY=0.4                          # fraction of the stability limit the timestep takes
WINDOW="--window"                   # each cell on its own patch: same physics, ~6x faster at this size (measured 2026-09-21)

# a minority population (0 cells = the uniform tissue, and nothing below matters)
MINORITY_COUNT=0                    # how many cells differ; they are cells 0..COUNT-1, ringed in the gif
MINORITY_SIZE=1.0                   # their radius / R  (a dendritic cell among T cells: ~2 to 2.5)
MINORITY_ACTIVITY=1.0               # their dimensionless activity a / the bulk's (0 = passive obstacle)
MINORITY_PERSISTENCE=1.0            # their persistence time / the bulk's
MINORITY_FRICTION=1.0               # their cell friction xi / the bulk's

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
    --activity             ${ACTIVITY} \
    --cell-friction        ${CELL_FRICTION} \
    --rotational-diffusion ${ROTATIONAL_DIFFUSION} \
    --time-unit            ${TIME_UNIT} \
    --grid-spacing         ${GRID_SPACING} \
    --safety               ${SAFETY} \
    ${WINDOW} \
    --minority-count       ${MINORITY_COUNT} \
    --minority-size        ${MINORITY_SIZE} \
    --minority-activity    ${MINORITY_ACTIVITY} \
    --minority-persistence ${MINORITY_PERSISTENCE} \
    --minority-friction    ${MINORITY_FRICTION} \
    --duration             ${DURATION} \
    --warmup               ${WARMUP} \
    --snapshots            ${SNAPSHOTS} \
    --transient            ${TRANSIENT} \
    --outdir               ${REPO}/figures/${TAG} \
    --tag                  ${TAG} \
    --aggregate  --figure
