# Residue coupling and pathways from an elastic network

[`docs/GENERATED_ENSEMBLES.md`](GENERATED_ENSEMBLES.md) §6 falsified one use of an
ANM with numbers: the low-frequency modes of 3ERT are essentially **orthogonal** to
the antagonist-to-agonist difference (site overlaps −0.002/−0.010/−0.010; best of
the first thirty +0.296). They are not a conformational-change predictor.

A mode set is also a **coupling** description — which residues move together — and
that is a different question, one the same eigenvectors can answer. This document
measures it, and measures what it is worth:

```bash
odock ensemble coupling -r 1HVR.pdb --site active=ASP25 --site flaps=ILE50,GLY51 \
    --route active=flaps --modes 10 --json-out coupling.json
odock ensemble coupling -r 3PTB.pdb --box-ligand BEN          # site -> far, derived
odock ensemble coupling -r 3ERT.pdb --box-ligand OHT \
    --site h12=LEU525,TYR526,MET528,LYS529,LEU536 --route site=h12
python -m pytest tests/test_coupling.py -q                    # the same numbers
```

## 1. What this is not

These four statements are a section, not a footnote, because every number below is
read against them:

1. **A harmonic coupling is not a signal-transduction mechanism.** The covariance
   comes from a Gaussian network around *one* structure. Nothing here is a
   message, an energy, or a rate.
2. **Correlated motion is not causality.** Two residues can move together because
   they share a hinge, because a spring joins them, or because they are the same
   rigid body. The analysis cannot tell those apart, and §4 measures how little it
   can: a strong coupling between neighbours is guaranteed by construction.
3. **A short path in a graph is not a physical channel.** The graph is built from
   one structure's contacts and the weights are correlations; a route through it
   is a route through a model.
4. **A coupling computed from one structure inherits every limitation already
   documented for one-structure ensembles**: harmonic, one connectivity, no
   populations, no anharmonic transitions, and no information about a motion the
   structure cannot make.

## 2. The method

* **Network**: the ANM of [`odock.modes`](GENERATED_ENSEMBLES.md) — Cα atoms as
  nodes, springs below `--cutoff` (13 Å), the six rigid-body modes removed.
* **Coupling**: `C_ij = <ΔR_i·ΔR_j> / sqrt(<ΔR_i²><ΔR_j²>)` with
  `<ΔR_i·ΔR_j> = Σ_k (T/λ_k)(v_k,i·v_k,j)` over the lowest `--modes` modes
  (default 10; 3N−6 available). The diagonal is 1, every entry is in [−1, 1], and
  `+1` means two residues move in the same direction with the same amplitude.
  Ranking uses **|C|**, so a hinge (anti-correlated halves) counts as coupled.
  Pairs closer than `--min-separation` (3) in sequence are skipped when ranking:
  neighbours move together because they are bonded.
* **Pathways**: two graph routes between two sites. The **lowest-cost** route
  minimises `Σ 1/|C|` over edges at or above a threshold (default: the 95th
  percentile of |C|, reported). The **bottleneck** route maximises its weakest
  link, computed exactly on the maximum spanning tree of |C|.
* **Sites**: `--site NAME=RES[,RES...]`, or derived from `--box`/`--box-ligand`
  as `site` (the pocket) plus `far` (the residue farthest from it). Each route's
  endpoint inside a site is that site's residue nearest the other site, and
  residues shared by both sites are excluded from both ends and reported.

## 3. Measured couplings (10 modes)

**3PTB (trypsin), 223 nodes, 663 modes available:**

| residue A | residue B | C | sequence separation |
|---|---|---|---|
| SER195 (catalytic) | VAL213 (S1 wall) | +0.989 | 14 |
| GLY196 (oxyanion hole) | VAL213 | +0.989 | 13 |
| GLY197 | VAL213 | +0.988 | 12 |
| GLY196 | ILE212 | +0.984 | 12 |
| **GLY43** | **GLY197** | **+0.976** | **152** |
| **GLY43** | **GLY196** | **+0.976** | **151** |
| GLY133 (autolysis loop) | ILE162 | +0.966 | 29 |

The catalytic serine couples to the S1 pocket wall at 0.989, and — the interesting
part — **GLY43 couples to the oxyanion-hole loop at 0.976 across 151 residues of
sequence**. Long-range couplings are not an artefact of sequence proximity.

**3ERT (ERα), 246 nodes, 732 modes:** ASP313↔VAL316 +0.987; ILE482↔THR485 +0.985;
**LEU308↔VAL478 +0.984 at a separation of 170**; ALA312↔THR485 +0.983 (173);
ALA307↔PRO365 +0.982 (58); ALA322↔ARG363 +0.981 (41). The helices of the ligand-
binding domain move as coupled pairs across the whole domain.

**1HVR (HIV-1 protease), 198 nodes, 588 modes** — and here the couplings land on
the functionally known pairs:

| residue A | residue B | C | separation |
|---|---|---|---|
| GLY27 A | GLY27 B | +0.963 | 99 |
| ASP25 A | GLY27 B | +0.958 | 101 |
| ASP25 B | ALA28 B | +0.957 | 3 |
| GLN2 A | ASN98 B | +0.956 | 195 |
| **GLY48 A** | **PHE53 A** | **+0.950** | 5 |
| ASP29 B | GLY86 B | +0.947 | 57 |
| GLU65 A | LYS70 A | +0.942 | 5 |

The **two catalytic aspartates couple to each other's catalytic loop (0.958)**, the
**flap residues GLY48–PHE53 couple (0.950)**, and the dimer-interface termini
(GLN2 A–ASN98 B) couple across 195 residues. The known coupled pairs are in the
list, which is the sanity signal.

## 4. The baseline that keeps §3 honest: coupling is mostly distance

An elastic network couples what is near it, because springs join neighbours. The
distance profile is the number every single coupling has to be read against:

| Cα–Cα distance | 3PTB mean \|C\| | 3ERT mean \|C\| | 1HVR mean \|C\| |
|---|---|---|---|
| 0–6 Å | **0.849** | **0.928** | **0.879** |
| 6–8 Å | 0.701 | 0.870 | 0.754 |
| 8–10 Å | 0.548 | 0.784 | 0.597 |
| 10–12 Å | 0.388 | 0.708 | 0.435 |
| 12–16 Å | 0.235 | 0.549 | 0.216 |
| 16–20 Å | 0.280 | 0.363 | 0.180 |
| 20–30 Å | 0.321 | 0.230 | 0.304 |
| > 30 Å | 0.253 | 0.209 | 0.216 |

So "0.98 between two residues" is unremarkable if they are 5 Å apart and notable if
they are 40 Å apart. In 3PTB the GLY43–GLY196 coupling (0.976 at ~20 Å) is well
above its distance baseline; the SER195–VAL213 coupling (0.989) is
indistinguishable from what any 6 Å pair scores.

## 5. Does the coupling list depend on how many modes are kept? Yes, largely

Item 1 asked for this explicitly, and the answer is that the **ranking** is
truncation-dependent while the **coarse structure** is not:

| modes kept | 3PTB top-20 Jaccard / Spearman | 3ERT | 1HVR |
|---|---|---|---|
| 3 | 0.11 / 0.728 | **0.03** / 0.695 | 0.11 / 0.667 |
| 5 | 0.74 / 0.896 | 0.21 / 0.890 | 0.38 / 0.815 |
| 10 (reference) | 1.00 / 1.000 | 1.00 / 1.000 | 1.00 / 1.000 |
| 20 | 0.54 / 0.772 | 0.67 / 0.963 | 0.60 / 0.954 |
| 50 | 0.43 / 0.575 | 0.33 / 0.911 | 0.29 / 0.876 |

Read it as two statements. The **top-20 list is unstable**: at three modes it
overlaps the ten-mode list by as little as **0.03 (3ERT)** and never exceeds 0.74
at any other count. The **matrix as a whole is much more stable**: the rank
correlation of all 19 110–29 646 off-diagonal |C| values stays between 0.58 and
0.96, rising to 0.95+ for the counts near the reference. So a *specific* strong pair
is a statement about the mode count; the *overall coupling pattern* is a statement
about the structure. **Any single pair quoted from this analysis must be quoted with
its mode count**, which is why the report always prints it.

## 6. Pathways

**HIV-1 protease: the catalytic aspartate to the flap tip.** The known coupled pair
is the flap tips and the catalytic aspartates, and the route found is exactly that:

```
lowest cost (edges >= 0.719), 5 steps, bottleneck |C| 0.831:
  ASP25 A -> ASP25 B -> ILE84 B -> VAL82 B -> THR80 B -> ILE50 A
```

It crosses to the **second monomer's catalytic aspartate**, then runs through
**THR80/VAL82/ILE84 — the flap hinge — out to ILE50 at the flap tip**. Not through
space: through the residues whose motion is tied to the flaps. The maximin route is
stronger at every step (bottleneck 0.917) but **40 steps long**, which is the
lesson about that definition: maximin maximises the weakest link, it does not
minimise the number of steps, so it wanders through the whole dimer.

**Trypsin: the S1 pocket to the C-terminal helix.** Auto-derived sites (the 22
residues around the benzamidine box; `far` = SER244):

```
lowest cost (edges >= 0.730), 8 steps, bottleneck |C| 0.739:
  HIS57 -> ILE103 -> VAL231 -> TYR234 -> ILE238 -> LYS239 -> ILE242 -> ALA243 -> SER244
```

From the catalytic histidine through the C-terminal helix (231–244). Maximin: 22
steps, 0.785.

**ERα: the ligand pocket to helix 12** — the honest test of item 3, next section.

## 7. The honest test: does the coupling flag helix 12 as coupled to the pocket?

**Yes, under both site definitions tried** — and it still does not rescue the mode
set, which is the interesting part.

| configuration | route | direct block (max / mean / pairs) |
|---|---|---|
| box-derived 14-residue pocket | THR347 → ASN348 → **ASP351** → LEU540 → LEU536 (4 steps, bottleneck 0.868) | **0.885 / 0.486** / 52 (LEU525 shared and excluded) |
| explicit 6-residue pocket | THR347 → MET343 → GLU419 → GLY420 → LEU525 (4 steps, 0.869) | **0.851 / 0.592** / 30 |

Both routes run from the pocket to the start of helix 12, and the box-derived one
passes through **ASP351** — the residue whose position distinguishes agonist from
antagonist and which moves 1.709 Å experimentally. That looks like a positive
result, and §4 is why it is not one: the pocket and helix 12 are **neighbours**
(they share LEU525). Against the distance baselines of §4 a mean of 0.486 or 0.592
is **below** what any pair scores at 8–10 Å (0.784 in 3ERT) and near what pairs
score at 12–16 Å (0.549). The flag is contact geometry.

So the two analyses say different things, and both are true:

* the modes **do** couple the pocket to helix 12 — they are adjacent and move
  together in the soft modes;
* the **direction** of that coupled motion is not the experimental one (site
  overlap 0.03), and the pocket-to-helix-12 coupling is what neighbours score.

**A coupling or pathway analysis is not a substitute for the direction test.** It
flags *which pairs move together*; it cannot say whether the motion is the one that
happens. Consistent with the earlier falsification, and reported as such rather
than as a rescue.

## 8. What would make this better

* A **mode-independent** coupling estimator (a covariance from several structures
  or from MD) so the answer is not a truncation of one harmonic model.
* A **statistical null** for "is this pair coupled *more than its distance
  predicts*" — the distance profile in §4 is the crude version of that; a proper
  null would randomise the modes and give a p-value per pair.
* An **anisotropic** pathway analysis (dynamic connectivity with flow, rather than
  a shortest path) would be the honest version of item 2's "dynamic connectivity".
* A **second structure** for every pathway, so that "the path is the same in both"
  can be tested rather than assumed. The repository has one pair (3ERT/1ERE_A) and
  the direction test there already answers the question.

## 9. Reproducing every number

```bash
odock ensemble coupling -r 3PTB.pdb --box-ligand BEN --modes 10 --json-out c3ptb.json
odock ensemble coupling -r 1HVR.pdb --site active=ASP25 --site flaps=ILE50,GLY51 \
    --route active=flaps --modes 10 --json-out chiv.json
odock ensemble coupling -r 3ERT.pdb --box-ligand OHT \
    --site h12=LEU525,TYR526,MET528,LYS529,LEU536 --route site=h12 \
    --modes 10 --json-out cer.json
python -m pytest tests/test_coupling.py -q
```

## 10. API

```python
from odock import coupling, ensemble, modes

conformation = ensemble.read_conformations(["1HVR.pdb"])[0]
network = modes.build_network(conformation, cutoff=13.0, modes=10)
correlation = coupling.cross_correlation(network, modes=10)      # (N, N) in [-1, 1]

pairs = coupling.strongest_couplings(correlation, network.labels, top=20)
profile = coupling.distance_profile(network, correlation)        # the baseline
report = coupling.mode_count_sensitivity(network, network.labels, reference=10)

cheap, wide = coupling.pathway(correlation, network.labels, source, target)
print(cheap.text())
```

`analyse_coupling(conformation, sites=..., routes=...)` returns a
`CouplingAnalysis` with `couplings`, `profile`, `sensitivity`, `sites`,
`site_labels`, `pathways`, `route_coupling`, `coupling_table()`,
`sensitivity_table()`, `site_table()`, `text()` and `as_dict()`.
