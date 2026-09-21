# migration_theory

A 2D multi-phase-field simulation of interacting, active cells in a tissue.

Each cell `i` is a smooth field `φ⁽ⁱ⁾(x)` on a shared periodic grid — near 1 inside the
cell, 0 outside, passing through an interface of finite width. Cell shape is an output,
not an imposed polygon, and the quantities a particle model needs a tessellation for
come out as integrals: `Aᵢ = ∫φᵢ²` is a cell's area and `∫φᵢ²φⱼ²` its contact with a
neighbour.

## The model

```
F = Σᵢ ∫ [ α φᵢ²(φᵢ−1)²  +  (K/2)|∇φᵢ|² ] dx        double well + gradient
  + ε  Σᵢ<ⱼ ∫ φᵢ²φⱼ² dx                              overlap repulsion
  + ω  Σᵢ<ⱼ ∫ ∇(φᵢ²)·∇(φⱼ²) dx                       cell–cell adhesion
  + λ  Σᵢ ( 1 − Aᵢ/A₀ )²,        A₀ = πR²            area constraint
```

evolved by

```
∂φᵢ/∂t  +  vᵢ·∇φᵢ  =  −(1/γ) δF/δφᵢ
```

with the polarity diffusing as `dθᵢ = √(2Dᵣ) dWᵢ`, `pᵢ = (cos θᵢ, sin θᵢ)`.

### Two propulsion modes

How `vᵢ` arises is a choice, set by `propulsion`.

**`"velocity"`** imposes it directly, `vᵢ = v₀ pᵢ`. Simple, and **wrong once activity is
strong**: an imposed velocity is not a force, so nothing can resist it. Measured at
`γv₀/σ = 21`, cells drive straight through their neighbours — occupancy reaches 2.0 and
the overlap integral sits at 7× its equilibrium value. Any observable from that regime is
measuring fields ghosting through each other, not a fluid tissue.

**`"force"`** balances an active force against the mechanical force from the free
energy:

```
F_i^active  = (E_a / R) p_i
F_i^passive = INT mu_i grad(phi_i) dx,     mu_i = dF/dphi_i
xi v_i      = F_i^active + F_i^passive
```

Because `Σᵢ Fᵢ^passive = 0` — translating every cell together cannot change `F` — a cell
pushed by a neighbour pushes back just as hard. A blocked cell stalls and its neighbour
gets moved. That force transmission is what the imposed-velocity form cannot do.

`E_a` is the mechanical work a cell can do over about one cell radius, so it is directly
comparable with the passive free energy; `v₀` never was. It is an **energy, not a rate** —
the power delivered is `E_a²/(R²ξ)`.

Measured at four times the activity, the imposed mode's overlap grows 60% while force
balance stays flat.

---

## Units

**Lengths are in microns.** The grid spacing is a *separate* parameter, so physics and
resolution never share a knob: `R = 12` is a cell of radius 12 µm whatever
`grid_spacing` happens to be, and halving `grid_spacing` refines the grid without
touching a single physical coefficient.

**Times** are in whatever unit `γ`, `ξ`, `v₀` and `Dᵣ` are quoted in, named by `time_unit`
(a label only — nothing computes differently). **Energies** are in whatever `α` is
quoted in; the energy scale cancels out of the dynamics, so only ratios matter.

A paper quoting lengths in grid spacings maps on directly by reading its `δx` as 1 µm.

---

## Parameters

Everything below is a field of `Model`. Nothing else is an input.

### Free energy

| name | symbol | dim | default | meaning |
|---|---|---|---|---|
| `alpha` | `α` | `E/L²` | 0.5 | Double-well depth. Penalises intermediate `φ`, giving a cell an inside and an outside. Sets the energy scale. |
| `K` | `K` | `E` | 2.0 | Gradient energy coefficient. Charges for spatial variation, so it resists the interface sharpening. |
| `epsilon` | `ε` | `E/L²` | 0.1 | Overlap repulsion. Cost per unit of shared area — excluded volume between cells. |
| `adhesion` | `ω` | `E` | 0.0 | Cell–cell adhesion. Refunds part of the interfacial cost at a junction. **Must stay below `K`** — see Ceilings. |
| `area_lambda` | `λ` | `E` | 6000 | Area constraint. Holds each cell near `A₀ = πR²`. Soft: area is held *near*, not at. |

`α` and `K` are a basis, not a description. Neither is separately observable. Prefer
`interface_terms(width, tension)` to set them, or read the derived
`interface_width` and `surface_tension` back.

### Geometry

| name | symbol | dim | default | meaning |
|---|---|---|---|---|
| `cell_radius` | `R` | `L` | 12.0 | Target cell radius in µm. Gives `A₀ = πR²`. |
| `packing` | — | — | 1.0 | Total target cell area as a fraction of the box. 1 is confluent; below leaves gaps, above starts the tissue compressed. Sets the box size, since `R` fixes cell *size* rather than density. |
| `n_cells` | `N` | — | 16 | Number of cells. |

### Dynamics

| name | symbol | dim | default | meaning |
|---|---|---|---|---|
| `friction` | `γ` | `ET/L²` | 10.0 | Resists the **field** deforming — shape relaxation. For a passive run it is only a time unit. |
| `propulsion` | — | — | `"velocity"` | `"velocity"` or `"force"`. See above. |
| `speed` | `v₀` | `L/T` | 0.0 | Self-propulsion speed. **`"velocity"` mode only.** |
| `active_energy` | `E_a` | `E` | 0.0 | Active work scale. **`"force"` mode only.** |
| `cell_friction` | `ξ` | `ET/L²` | `None` | Resists a **cell** translating — drag. `None` reuses `γ`. **`"force"` mode only.** |
| `rotational_diffusion` | `Dᵣ` | `1/T` | 1e-4 | How fast a cell forgets its direction. Persistence time is `1/Dᵣ`. |
| `time_unit` | — | — | `"s"` | Label for the time unit. Changes nothing; makes reported timescales legible. |

**`γ` and `ξ` are not the same thing**, despite sharing dimensions in 2D. `γ` is cortical
viscosity — how hard a cell is to *deform*. `ξ` is drag — how hard it is to *drag*. A cell
can be stiff to move but floppy to deform. In `"velocity"` mode `γ` does both jobs, which
conflates them.

**`E_a` and `ξ` are not redundant either.** `E_a/ξ` sets the free speed, but `1/ξ`
separately sets how strongly neighbours can push a cell aside. Holding the free speed
fixed while raising both leaves a measurably different tissue.

**Polarity is intent, not outcome.** `v₀` and `Dᵣ` govern the direction a cell *tries*
to crawl. What it achieves is that minus whatever the neighbours do — in a confluent
tissue, much less. There is no alignment of any kind: cells do not respond to their own
velocity, to neighbours, or to their shape.

### Numerics

| name | symbol | dim | default | meaning |
|---|---|---|---|---|
| `grid_spacing` | `dx` | `L` | 1.0 | Grid spacing in µm. Pure resolution — appears in no physical quantity. See `Model.refine`. |
| `timestep` | `dt` | `T` | `None` | `None` derives `dt` from the stability limit. Set it to override; it is still checked. |
| `safety` | — | — | 0.4 | Fraction of the stability limit `dt` takes when `timestep` is unset. |
| `window` | — | — | `False` | Measure each cell on its own patch of the grid rather than the whole box. Same numbers; first stage of windowed storage. `--window` on the command line. |

---

## Derived quantities

Not inputs. Computed from the above, and reported by `Model.summary()` and `print(model)`.

| property | formula | at defaults |
|---|---|---|
| `interface_width` | `w = √(K/α)` | 2.0 µm |
| `surface_tension` | `√(2Kα)/6` | 0.2357 |
| `cell_cell_tension` | `σ(2 − ω/K)` | 0.4714 |
| `adhesion_ceiling` | `K` | 2.0 |
| `target_area` | `πR²` | 452.4 µm² |
| `cell_spacing` | mean centre-to-centre | 22.83 µm |
| `width_to_radius` | `w/R` | 0.167 |
| `points_per_radius` | `R/dx` | 12 |
| `points_per_interface` | `w/dx` | 2 |
| `box_length` / `box_points` | from `N`, `A₀`, packing | 85.1 µm / 85² |
| `free_speed` | `v₀`, or `E_a/(Rξ)` | — |
| `activity` | `E_a/(σR)` | — |
| `effective_cell_friction` | `ξ`, else `γ` | 10.0 |
| `persistence_time` | `1/Dᵣ` | 10 000 s |
| `persistence_length` | `free_speed/Dᵣ` | — |
| `shape_relaxation_time` | `γR²/K` | 720 s |
| `interface_relaxation_time` | `γw²/K` | 20 s |
| `traversal_time` | `R/free_speed` | — |

`Model.concerns()` returns a list of warnings about a parameter choice — wide
interfaces, under-resolved grids, adhesion near its ceiling, extreme packing. It
returns them rather than raising, so the caller decides.

---

## Ceilings and constraints

**Adhesion must stay below `K`.** Not `2K`, where the sharp-interface tension
`σ_cc = σ(2 − ω/K)` would vanish — the field-level limit bites first. The gradient block
of the Hessian loses positive-definiteness once adhesion matches the gradient term,
after which neighbouring fields vary against each other for free, interface
proliferates, and the energy runs away.

With the `∇(φ²)` form the coupling carries a weight `4φᵢφⱼ`, so the realised ceiling is
`K` divided by that weight — about 1 at a clean interface, larger wherever cells
overlap. **Measured threshold at the defaults is between `ω = 0.7` and `0.9`**, against
a naive `K = 2`. Use `Adhesion.coupling_strength(fields)` for the realised weight, and
treat `adhesion_ceiling` as optimistic.

A run whose fields stop being finite raises `Diverged` rather than returning a
trajectory full of `NaN`.

**The timestep is computed, not guessed.** Each free-energy term reports its own
stability limit and the stepper takes the smallest. At the defaults `AreaConstraint`
binds at `dt = 0.188`; a too-large `dt` raises `UnstableTimestep` naming the term
responsible.

**Speed is limited by the grid, not just the physics.** Central-difference advection is
stabilised only by the diffusion the gradient term supplies, and needs a grid Péclet
number `v·dx·γ/K` below about 2. At `dx = 1 µm` that caps the speed near `0.4 µm/s`
whichever propulsion mode is used; `UnstableTimestep` says so by name. Halving `dx`
doubles the ceiling.

**Active runs need a passive warm-up.** The seeded circles overlap heavily — circles
cannot tile — so the mechanical forces at `t = 0` are far larger than the relaxed tissue
ever sees. Under force balance those forces *set the velocity*, so without a warm-up the
run opens with a velocity spike that is an artefact of seeding. Pass `warmup=` to
`simulate` or `--warmup` on the command line; a few hundred time units suffices.

**Interface resolution.** Below about 1.5 grid points across the interface it pins to
the grid. Measured, though, the discretisation is more forgiving than the usual rule of
thumb: surface tension stays accurate to 1.5% even at 1 point per interface.

---

## Notation warning

Papers in this area swap these symbols. Ours follows neither exactly; the code spells
every name out, but printed output uses the shorthand below.

| | interface width | field friction | cell friction | tension | area constraint |
|---|---|---|---|---|---|
| paper with the parameter table | `ξ` | `γ` | — | — | `λ` |
| paper with Eq. (4) | `λ` | `M⁻¹` | `ξ` | `γ` | `μ` |
| **this code** | `interface_width` (`w`) | `friction` (`γ`) | `cell_friction` (`ξ`) | `surface_tension` (`σ`) | `area_lambda` (`λ`) |

We follow Eq. (4) in using `ξ` for the **cell friction**, so the interface width is
written `w` throughout this document — it was `ξ` in an earlier draft, and in the other
paper. The code spells both out, so only prose and printed output are ambiguous.

If transcribing from a paper, also check two conventions: whether its sum over pairs is
ordered (`j ≠ i`, double counting) or unordered (`i < j`, as here), and whether its
"surface tension" is `σ` or a prefactor three times it.

---

## Observables

`analysis.tissue_state(trajectory)` returns all of these as one dict. They describe what
kind of tissue a run produced, as opposed to what was put into it.

| function | gives |
|---|---|
| `shape_statistics` | distribution of `q = P/√A` — mean, std, median, 5th/95th percentiles, area CV |
| `mean_squared_displacement` | time-averaged MSD vs lag |
| `diffusion_coefficient` | `D` from `MSD = 4Dt`, **and the MSD exponent** |
| `persistent_random_walk` | effective speed and persistence, fitted from the motion achieved |
| `neighbour_exchange_rate` | T1 rate per cell per unit time, plus mean coordination |
| `velocity_correlation` | spatial velocity correlation `C(r)` and the length over which it decays to `1/e` |
| `velocity_correlations` | that length, whether it is only a lower bound, and the correlation between cells in contact |
| `tracks` | cell paths with the periodic boundary unwrapped |

**The MSD exponent is the clearest single indicator**: ~2 ballistic, ~1 diffusive, below
1 subdiffusive — a caged cell in a jammed tissue. That plus an exchange rate of zero is
what a solid looks like.

`persistent_random_walk` returns `speed_ratio` and `persistence_ratio` — measured over
imposed. Both fall well below 1 in a crowded tissue, and how far below is a direct read
on confinement.

**Velocity correlations are computed after subtracting the tissue's drift.** Under
force balance the net active force on the tissue does not cancel, so the whole tissue
drifts at roughly `1/√N` of the free speed, which with 16 cells is a quarter of it.
That is motion of the box's contents as a body, and the velocity correlations remove
it. The MSD-based observables above do not, and are dominated by it at small `N`.
Separations only reach half the box, under two cell spacings at 16 cells, so a
correlation length reported at that bound with `velocity_correlation_censored = 1` is
a lower bound, not a measurement; the model has no alignment term, so any correlation
is force transmission between neighbours.

Two caveats. The shape index is biased high at finite interface width, because
`A = ∫φ²` undershoots the sharp-interface area — a perfect circle measures 3.79 rather
than 3.545 at `w/R = 0.1`. Compare between runs at the same width rather than against
literature thresholds. And the exchange rate is a **lower bound** when sampling is
coarse, so compare only runs sampled at the same interval.

---

## Running

```bash
# force balance, the mode to prefer for anything active
python scripts/animate.py --propulsion force --active-energy 3 \
                         --rotational-diffusion 1e-3 \
                         --duration 10000 --warmup 800 --no-video --analyse

python scripts/animate.py --energy                          # also the energy figure
python scripts/steady_states.py --sweep packing             # passive parameter sweep
python scripts/seed_random_cells.py                         # seeding diagnostics
```

`--no-video` skips rendering and drops the per-frame image data, which is most of the
memory and all of the render time — the right mode for a sweep. `--analyse` prints the
observables. Every run writes `figures/<name>.npz` regardless, with the name encoding
every parameter, so nothing is measured and then discarded.

Or directly:

```python
from migration_theory import Model, simulate, tissue_state

model = Model(propulsion="force", active_energy=3.0, adhesion=0.5,
              rotational_diffusion=1e-3)
print(model)                      # derived quantities and timescales
for note in model.concerns():     # warnings about this choice
    print(note)

trajectory = simulate(model, duration=50_000, warmup=800)
print(tissue_state(trajectory))
```

Runs are saved and reloaded rather than repeated:

```python
from migration_theory import save_trajectory, load_trajectory
save_trajectory(trajectory, "figures/myrun")     # ~30 kB per 40 snapshots
again = load_trajectory("figures/myrun")         # analysis without re-simulating
```

The `.npz` holds every `Model` field and every per-snapshot measurement — centres, areas,
perimeters and the full contact matrix — but not the fields, which is why it is kilobytes
rather than megabytes. `tissue` comes back as `None`.

`Model` is frozen, so a sweep is `model.replace(speed=0.01)` and a resolution check is
`model.refine(2)` — which changes `dx` only, leaving every physical coefficient
untouched.

---

## Layout

```
src/migration_theory/
  model.py        Model — parameters, and everything derived from them
  free_energy.py  DoubleWell, GradientEnergy, Repulsion, Adhesion, AreaConstraint
  dynamics.py     ExplicitEuler, run
  simulate.py     simulate() -> Trajectory
  analysis.py     tissue observables
  tissue.py       evolving state: fields, polarity, time
  fields.py       per-cell phase fields and seeding
  grid.py         periodic grid and its operators
  diagnostics.py  areas, perimeters, shape indices, contacts
  activity.py     polarity and advection
  initialise.py   cell-centre seeding (Lloyd, Poisson disc, lattice)
  box.py          periodic box
scripts/
  animate.py, steady_states.py, seed_random_cells.py
  render.py, plotstyle.py    plotting only
```

Nothing in `migration_theory` imports matplotlib, and `render.py` never simulates — so a
run can be done headless and rendered later from the `Trajectory`.
