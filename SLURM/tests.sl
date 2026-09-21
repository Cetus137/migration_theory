#!/bin/bash
#SBATCH --job-name      tests
#SBATCH --partition     short
#SBATCH --mem           4G
#SBATCH --time          00:30:00
#SBATCH --cpus-per-task 1
#SBATCH --output        slogs/tests.%j.out
#SBATCH --exclude       compg009,compg010,compg011,compg013
# Run the test suite. Nothing is run on the login node, so this is how a code change
# gets checked: every free-energy term against its own finite difference, the
# linear-time repulsion against the literal pair loop, the stepper's identities, and
# the analysis on short simulations. Takes a few minutes; the slowest tests are the
# ones that simulate.
#
#   sbatch tests.sl

# ── Configuration ─────────────────────────────────────────────────────────────
REPO=/users/kir-fritzsche/aif490/devel/migration_theory
SELECT=""                           # e.g. "-k repulsion" to run a subset; empty = everything

module purge
source /well/kir/config/modules.sh
module load Python/3.11.3-GCCcore-12.3.0
source ~/devel/venv/Python-3.11.3-GCCcore-12.3.0/migration_env/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

cd ${REPO}
python3 -m pytest -q --durations=10 ${SELECT}
