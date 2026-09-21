#!/bin/bash
#SBATCH --job-name      structure
#SBATCH --partition     short
#SBATCH --mem           4G
#SBATCH --time          00:15:00
#SBATCH --cpus-per-task 1
#SBATCH --output        slogs/structure.%j.out
#SBATCH --exclude       compg009,compg010,compg011,compg013
# g(r) of the cell centres for several saved runs on one figure, with the hexagonal
# lattice shells marked, plus the hexatic order and neighbour-number statistics printed.
# Nothing is simulated: it reads the .npz files. Set FILES to the runs to compare.
#
#   sbatch structure.sl

# ── Configuration ─────────────────────────────────────────────────────────────
REPO=/users/kir-fritzsche/aif490/devel/migration_theory
DIR=${REPO}/figures/activity_adhesion_eps40_N100
LABEL_BY=active_energy              # the model field that names each curve
OUT=${DIR}/gofr_activity_om0.png
FILES=(                             # zero adhesion, seed 0, solid to fluid
    ${DIR}/*om0_*_Ea0_*seed0.npz
    ${DIR}/*om0_*_Ea3.78947_*seed0.npz
    ${DIR}/*om0_*_Ea11.3684_*seed0.npz
    ${DIR}/*om0_*_Ea18.9474_*seed0.npz
    ${DIR}/*om0_*_Ea28.4211_*seed0.npz
)

module purge
source /well/kir/config/modules.sh
module load Python/3.11.3-GCCcore-12.3.0
source ~/devel/venv/Python-3.11.3-GCCcore-12.3.0/migration_env/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

python3 -u ${REPO}/scripts/structure.py "${FILES[@]}" --label-by ${LABEL_BY} --transient 0.3 --out ${OUT}
