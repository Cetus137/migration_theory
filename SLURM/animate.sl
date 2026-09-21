#!/bin/bash
#SBATCH --job-name      animate
#SBATCH --partition     short
#SBATCH --mem           4G
#SBATCH --time          10:00:00
#SBATCH --cpus-per-task 1
#SBATCH --output        slogs/animate.%j.out
#SBATCH --exclude       compg009,compg010,compg011,compg013
# Simulate one run and render it as a GIF. Everything lands in figures/<TAG>/: the
# .npz, <name>.gif and the free-energy figure <name>_energy.png. The name encodes
# every parameter, the same way the sweep files are named.
#
# Every model parameter is listed below at its value, so nothing is left to a default
# that could change. To see a point of a sweep, set the model block to match the
# sweep's fixed parameters, put the swept value in, and keep SEED, DURATION and
# WARMUP the same: the run is then the very same trajectory the sweep measured, not a
# new realisation. The GIF opens and plays in the VS Code image preview; an mp4 would
# have to be copied off the cluster.
#
# FRAMES x grid points x 8 bytes is held in memory for rendering -- ~12 MB at 200
# frames on an 85^2 grid, so 4G is generous. --time as for one sweep task, plus a
# few minutes to render.
#
# Edit the block below, then:   sbatch animate.sl

# ── Configuration ─────────────────────────────────────────────────────────────
REPO=/users/kir-fritzsche/aif490/devel/migration_theory
TAG=activity_eps40                  # all outputs go to figures/${TAG}/

# free energy (energy units are whatever alpha is quoted in; lengths in um)
ALPHA=0.5                           # double-well depth; with K sets width sqrt(K/alpha) = 2 um
K=2.0                               # gradient energy; with alpha sets tension sqrt(2 K alpha)/6
EPSILON=40                          # overlap repulsion; the model default 0.1 is too weak once active
ADHESION=0.0                        # omega; must stay below K, in practice below ~0.7
AREA_LAMBDA=6000                    # area constraint; sets the timestep at these values

# geometry
CELL_RADIUS=12                      # um; target area pi R^2
PACKING=1.0                         # total cell area / box area; 1 is confluent
N_CELLS=75
SEEDING=tessellated                 # tessellated | circles

# dynamics (time in seconds; time_unit is a label only)
FRICTION=10                         # gamma: resists the field deforming
PROPULSION=force                    # force | velocity
SPEED=0.0                           # v0; velocity mode only
ACTIVE_ENERGY=28.4                     # E_a; force mode only -- the swept value in the activity sweep
CELL_FRICTION=10                    # xi: drag on a translating cell; force mode only
ROTATIONAL_DIFFUSION=1e-3           # D_r; persistence time 1/D_r = 1000 s
TIME_UNIT=s

# numerics
GRID_SPACING=1.0                    # dx in um; pure resolution, appears in no physical quantity
SAFETY=0.4                          # fraction of the stability limit the timestep takes
WINDOW=""                           # "--window" measures each cell on its own patch of the grid; same numbers, stage 1 of windowed storage

# run
SEED=0
DURATION=5000                      # physical time; 10 persistence times at D_r = 1e-3
WARMUP=25                           # passive relaxation before recording
TRANSIENT=0.2                       # fraction of the run discarded before measuring

# output
FRAMES=200                          # frames in the gif; also the number of snapshots
FPS=20
FORMAT=gif                          # gif | mp4
DPI=80                              # frame resolution; gif size scales with FRAMES x DPI^2 (200 -> ~10 MB, 80 -> ~1.7 MB)
COLOUR=""                           # e.g. "--vmin 0.5 --vmax 1.2"; empty = automatic from percentiles
EXTRA="--energy --analyse"          # also write the energy figure and print observables

module purge
source /well/kir/config/modules.sh
module load Python/3.11.3-GCCcore-12.3.0
source ~/devel/venv/Python-3.11.3-GCCcore-12.3.0/migration_env/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

mkdir -p ${REPO}/figures/${TAG}

python3 -u ${REPO}/scripts/animate.py \
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
    --seed                 ${SEED} \
    --duration             ${DURATION} \
    --warmup               ${WARMUP} \
    --transient            ${TRANSIENT} \
    --frames               ${FRAMES} \
    --fps                  ${FPS} \
    --format               ${FORMAT} \
    --dpi                  ${DPI} \
    --outdir               ${REPO}/figures/${TAG} \
    ${COLOUR} \
    ${EXTRA}
