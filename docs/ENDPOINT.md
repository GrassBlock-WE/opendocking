# End-point rescoring (MM-GBSA-style), as the approximation it is

Every score this project reports is a docking score: an empirical, rigid-receptor,
single-structure number. `odock.endpoint` adds the next cheapest thing a modeller
reaches for — an **end-point** estimate in the MM-GBSA family, evaluated over a
pose ensemble and reported **with its spread**:

```
dG  ~  <E_interaction>  +  dG_electrostatic  +  dG_nonpolar  [+ (-T dS)]  [+ strain]
```

```bash
python -c "
from odock import endpoint
models = endpoint.read_pose_models('poses.pdbqt')
result = endpoint.endpoint_ensemble(models, open('receptor.pdbqt').read(), label='benzamidine')
print(result.text()); print(result.as_dict()['interval'])"
python -m pytest tests/test_endpoint.py -q      # the same numbers as tests
```

## 1. What this is not

Read this before any number below.

1. **Not a free-energy calculation.** There is no alchemical path, no
   Boltzmann-weighted trajectory, and no relaxation of the complex. Every pose is
   scored where the docking left it.
2. **No explicit water.** Solvent is a SASA term plus a distance-dependent
   dielectric. The bridging waters that [`docs/WATERS.md`](WATERS.md) can *see* are
   invisible here, and so is the cost of displacing one.
3. **Entropy is omitted by default.** The torsional estimate that can be switched
   on (`with_entropy=True`) is a counting rule — 3 states per rotor at 298.15 K —
   not a normal-mode or quasi-harmonic entropy.
4. **The dielectric is a fudge with a named convention.** `eps = 4r` is the
   simplest defensible form; there is no Poisson–Boltzmann solver, no
   Generalised-Born radii, no ionic strength. The kernel's *own* AD4
   distance-dependent dielectric is a different documented convention
   (`dielectric = -0.1465`, AutoGrid 4.2, see [`odock.export`](../python/odock/export.py))
   and the two must not be mixed.
5. **A dG in kcal/mol from this is a ranking device.** It is not an experimental
   affinity. The bootstrap interval is the spread of the *pose ensemble*, not the
   method's error, and §5 shows the pose ensemble is often a single pose.

## 2. The terms, and what each one is

| term | what it is | how it is computed |
|---|---|---|
| `interaction` | the gas-phase interaction energy of the pose | `consensus.rescore_poses` → `inter`, the kernel's own number |
| `electrostatic` | ligand–receptor Coulomb sum, `332.0637 q_i q_j / (eps r)`, `eps = 4r` | `endpoint.coulomb_energy`, charges read from the PDBQT |
| `nonpolar` | `-gamma * dSASA_buried`, `gamma = 0.0072 kcal/mol/A^2` | `odock.sasa`, both sides of the interface |
| `entropy` | `-T dS` from a rotor count (optional) | `metrics.entropy_penalty` |
| `strain` | ligand conformational strain (optional) | `metrics.ligand_strain` |

**No double counting, and one explicit warning about it.** Vina and Vinardo have
*no* electrostatic term, so adding the Coulomb term is additive. AD4 *does* have
one, so `scoring="ad4"` makes electrostatics count twice; the result carries a note
saying exactly that (asserted in the tests).

`interaction` is read through the same call the consensus layer uses, so the module
cannot drift away from the pipeline: the tests assert
`endpoint.poses[i].interaction == consensus.rescore_poses(...)["vina"]["inter"][i]`
to the last bit, and that the affinity is the one the docking run reported for that
pose (to 1e-2, see §5 for the measured difference).

### Two measured numbers that define the terms

**The Coulomb term is closed-form.** Two unit charges 4 Å apart with `eps = 4r`:

```
332.0637 * (+1)(-1) / (4 * 4^2)  =  -5.188 kcal/mol
```

which the tests assert directly, together with the `1/r^2` falloff that the
distance-dependent dielectric produces and the factor of 4 that changing the
convention to `eps = 1` introduces.

**The nonpolar term is where this module was first wrong, and the mistake is worth
recording.** The first run reported a **-357 kcal/mol** nonpolar term, which is
impossible for a 16-heavy-atom ligand. The cause was measured: for a large atom
set, `sasa.burial(receptor, ligand).reference_total` is **58 377.1 Å²** where the
free receptor's own area is **9 274.9 Å²**, so a difference of the two is
meaningless. `sasa.interface_area` is internally consistent on the same input
(`free_area` 9 274.9, `complex_area` 9 096.9, `buried_area` 178.0), so the
receptor side uses that and the ligand side uses `burial` (free 426.3 Å², complex
104.1 Å² — consistent). The test `test_the_terms_are_physically_sized_endpoint`
pins the magnitude so the mistake cannot return silently.

## 3. Measured: benzamidine in the trypsin S1 site

`-e 2 -n 3 --seed 42`, the box from `demo/systems/3ptb/box.json`:

| pose | interaction | electrostatic | nonpolar | **total** | Vina affinity |
|---|---|---|---|---|---|
| 1 | −7.683 | +9.992 | −3.602 | **−1.292** | −5.945 |
| 2 | −7.582 | +9.996 | −3.692 | **−1.278** | −5.867 |
| 3 | −7.559 | +9.715 | −3.620 | **−1.464** | −5.849 |
| **mean** | | | | **−1.345** | −5.887 |

95 % bootstrap interval (3 poses, 2000 resamples, seeded): **[−1.464, −1.278]**,
width 0.186 kcal/mol, sd 0.085. Buried area **500.3 Å²** = 322.2 (ligand side) +
178.0 (receptor side). With the torsional entropy switched on, the `-T dS` term is
**+3.255 kcal/mol** for the 5 rotatable bonds, and the estimate becomes
**+1.91 kcal/mol**.

Two things to read from that table. The **electrostatic term is positive and
large** (+10 kcal/mol): an unscreened Coulomb sum between a partially charged
ligand and a charged pocket is not a solvation-corrected electrostatics, and it is
the weakest part of the estimate — that is exactly what "no PB solver" costs. And
**ΔG (−1.35) is far less favourable than the Vina score (−5.89)** for the same
poses, because the docking score is a fitted function and this is a physical
decomposition with no fitted hydrophobic term. They are not on the same scale and
must not be compared as if they were.

## 4. Uncertainty, and the ensemble that is often not there

The interval is a **percentile bootstrap over poses**, so it answers "how much does
this ligand's own pose ensemble move the estimate", and nothing else. Two
behaviours are enforced rather than documented:

* with **one pose** there is nothing to resample, so the interval collapses to the
  point and the report says so in words (`"one pose: no ensemble to resample, so
  the interval is the point estimate and must be read as such"`) — a single pose
  presented with an interval of zero would be the worst kind of decoration;
* with **no poses** the result is `nan`, not 0.

In the 17-ligand campaign of §5, **13 of the 17 ligands have a single pose**, so
their interval is degenerate. Where two poses exist the widths are **0.008
(benzoquinone), 0.024 (triphenylene), 0.375 (caffeine) and 0.646 kcal/mol
(warfarin)** — all far smaller than the between-ligand spread of the estimates
(−4.27 to +4.28 kcal/mol). So on this system **the pose spread does not explain the
disagreement with the docking ranking; the method does.**

## 5. The measurement that decides whether it was worth building

17 demo ligands docked into 3PTB (the campaign of
[`docs/WATERS.md`](WATERS.md) §4), each rescored end-to-end, compared with the Vina
ranking of the same poses:

| | end-point | Vina |
|---|---|---|
| best ligand | **fluorobenzamidine** (−4.27) | **warfarin** (−7.33) |
| worst ligand | triphenylene (+4.28) | benzoquinone (−4.86) |
| range | 8.6 kcal/mol | 2.5 kcal/mol |

**Spearman rho = +0.297, 95 % interval [−0.278, +0.857], n = 17.** The interval
spans zero: on this set the end-point estimate and the docking score are **not
significantly correlated**. **Kendall tau = +0.279, interval [−0.132, +0.662]**,
which excludes perfect agreement, so the two orderings **are distinguishable** at
this n (`distinguishable: True`): 15 of the 17 ranks change, and the top ligand is
different. With the torsional entropy included the picture is the same
(rho = +0.230 [−0.376, +0.823], tau = +0.235 [−0.235, +0.647], top-1 still
different, 15 ranks changed).

**That is the answer to whether this was worth building, and it is a negative one
about the *ranking*:** the end-point estimate is a physically decomposable number
with an honest spread, and on this system it does **not** reproduce the docking
ranking, is not significantly correlated with it, and is resolvably *different*
from it. Since neither ranking can be validated against experiment with the data
in this repository, the module's conclusion is that the end-point estimate is a
**second, independent opinion** — useful as a disagreement detector, not as a
better ranking. The honest use is to look at the ligands the two methods *disagree*
about, not to substitute one ordering for the other.

### What the comparison cannot say

* **n = 17 ligands, one receptor.** The bootstrap resamples these ligands; it says
  nothing about a new ligand or another target.
* **No experimental affinities are used**, because the repository has none for this
  system. "The rankings differ" is not "one of them is wrong".
* **The dielectric and the missing entropy are not resampled.** They are systematic
  choices, so the interval is narrower than the method's real uncertainty — which
  is stated here rather than implied by the width.

## 6. API

```python
from odock import endpoint

models = endpoint.read_pose_models("poses.pdbqt")        # list of model texts
result = endpoint.endpoint_ensemble(
    models, open("receptor.pdbqt").read(),
    box=box,                      # the grid the poses came from
    label="benzamidine", receptor="3PTB",
    with_entropy=True,            # off by default
    gamma=endpoint.DEFAULT_GAMMA, dielectric=endpoint.DEFAULT_DIELECTRIC,
)
result.mean, result.spread, result.n_poses
result.bootstrap()                # {'mean','low','high','width','n','samples', ...}
result.text(); result.as_dict()
best = result.best                # the single most favourable pose

comparison = endpoint.compare_rankings([result, ...], [vina_scores...])
# {'n','rho','rho_low','rho_high','tau','tau_low','tau_high','top1_agreement',
#  'n_rank_changes','distinguishable','note'}
```

`endpoint_score(...)` decomposes a single pose; `coulomb_energy(...)` and
`buried_area(...)` expose the two solvation terms on their own;
`kendall_tau(...)` is the tau-b used by the comparison, so no scipy is needed.
