# OpenDocking scoring functions: derivation and validation

This is the **step 3 deliverable** of the project brief: the full derivation of
everything the kernel scores, from the published potentials down to the numbers
that appear in the output table.

All formulas below are transcribed from the implementation in
`crates/dock-core/src/scoring/`, `crates/dock-core/src/kinematics.rs` and
`crates/dock-core/src/math.rs`. Every numeric constant is quoted with the
upstream source it was ported from.

The project avoids a math renderer, so all mathematics is written in fenced code
blocks in plain ASCII.

---

## Scope and attribution

Three force fields are implemented, each matching its reference implementation
term for term:

| Choice | Terms | Reference |
|---|---|---|
| `vina` (default) | gauss1, gauss2, repulsion, hydrophobic, non-directional H-bond, linear attraction | Trott & Olson, *J. Comput. Chem.* **31**, 455 (2010); AutoDock Vina `potentials.h` |
| `vinardo` | gauss, repulsion, hydrophobic, non-directional H-bond, linear attraction | Quiroga & Villarreal, *PLoS ONE* **11**, e0155183 (2016); AutoDock Vina `potentials.h` |
| `ad4` | 12-6 van der Waals, 12-10 H-bond, distance-dependent-dielectric electrostatics, desolvation, linear attraction | Morris et al., *J. Comput. Chem.* **30**, 2785 (2009); AutoDock 4.2 as exposed by AutoDock Vina `potentials.h` |

**Attribution.** AutoDock Vina is Copyright (c) 2006–2010, The Scripps Research
Institute, distributed under the Apache License 2.0. It is the reference for the
Vina and Vinardo potentials, the analytic-derivative formulation, the trilinear
grid evaluation, the `slope_step`/`smoothen`/`curl` helpers, the iterated-local-
search protocol and the AD4 parameter table in `atom_constants.h`. AutoDock 4 is
Copyright (c) 1989–2007, The Scripps Research Institute, distributed under the
GPL; it is the reference for the AD4.2 force-field form and the AD4 atom typing.
**No AutoDockTools (ADT / MGLTools) source was read, borrowed or copied** — the
PDBQT handling and the preparation layer were written from the format
specification plus RDKit, with Meeko as the behavioural reference only.

---

## Notation and the shared definitions

```text
i, j          two atoms
x_i           Cartesian position of atom i (Å)
d_ij          |x_i - x_j|, the raw interatomic distance (Å)
t_i, t_j      X-Score types of the two atoms
R(t)          X-Score van der Waals radius of type t (Å)
opt_ij        optimal (minimum-energy) distance of the pair (Å):
                  opt_ij = R(t_i) + R(t_j),  or 0 when either type is a
                  macrocycle closure dummy (G0..G3)
r_ij          the REDUCED distance:  r_ij = d_ij - opt_ij
E_ij          pair energy (kcal/mol)
cutoff_k      interaction cutoff of term k (Å); the term is exactly 0 at and
              beyond it
w_k           weight of term k
```

`opt_ij = 0` for a glue pair means the reduced distance equals the raw distance,
which is what makes the linear attraction act on the true separation.

### `slope_step` — Vina's clamped linear ramp

```text
slope_step(x_bad, x_good, x):
    if x_bad < x_good:
        if x <= x_bad:  return 0
        if x >= x_good: return 1
    else:
        if x >= x_bad:  return 0
        if x <= x_good: return 1
    return (x - x_bad) / (x_good - x_bad)

d/dx slope_step(x_bad, x_good, x):
    (lo, hi) = (min(x_bad, x_good), max(x_bad, x_good))
    if x <= lo or x >= hi: return 0
    return 1 / (x_good - x_bad)
```

It is 1 on the "good" side, 0 on the "bad" side, and linear in between. Note the
*ordering* of the two bounds is handled on both ends, which is why the
hydrophobic term can be written with `(bad, good) = (1.5, 0.5)` and the H-bond
term with `(0.0, -0.7)`.

Endpoint behaviour is pinned by `slope_step_endpoints`:
`slope_step(1.5, 0.5, 2.0) == 0`, `slope_step(1.5, 0.5, 0.0) == 1`,
`slope_step(1.5, 0.5, 1.0) == 0.5`.

The ramp is continuous but **not** differentiable at `r = opt + good/bad`: those
are genuine derivative kinks (the potential has a discontinuous second
derivative there), and the finite-difference validation explicitly skips them —
see [Validation](#validation).

### `smoothen` — the AD4 shifted-radius operator

```text
smoothen(r, rij, smoothing):
    half = smoothing / 2
    if half <= 0:            return (r, 1)          # disabled
    if r > rij + half:       return (r - half, 1)
    if r < rij - half:       return (r + half, 1)
    return (rij, 0)                                 # inside the plateau

returns (r_shifted, dr_shifted/dr)
```

The operator pulls the radius *towards* the well position and freezes it inside
a plateau of full width `smoothing` centred on `rij`. Because
`d(r_shifted)/dr = 0` inside the plateau, the term's derivative vanishes there,
which is what removes the derivative discontinuity of the raw 12-6/12-10 form at
the well. Behaviour is pinned by `smoothen_plateau`:
`smoothen(3.0, 3.0, 0.5) = (3.0, 0.0)`,
`smoothen(4.0, 3.0, 0.5) = (3.75, 1.0)`,
`smoothen(2.0, 3.0, 0.5) = (2.25, 1.0)`.

### `curl` — the soft energy cap

```text
curl(e, deriv, v):
    if e > 0 and v < 0.1 * MAX_F:          # not_max(v)
        tmp = (v < EPSILON) ? 0 : v / (v + e)
        e     = e * tmp
        deriv = deriv * tmp * tmp
```

For a positive energy the map is `e -> v*e/(v+e)`, monotone and bounded by `v`;
its derivative is `d(e')/de = (v/(v+e))^2 = tmp^2`, which is exactly the factor
applied to the gradient. Energy and gradient therefore stay consistent after the
cap, which is what keeps the analytic gradient usable inside BFGS.

`curl_scalar(e, v)` is the scalar-only variant, used where no gradient is being
accumulated. `Caps::hunt()` supplies the tight caps used during the search
(`intra = 10`, `grid = 1.5`, `pairs = 10`, i.e. Vina's `hunt_cap`),
`Caps::authentic()` the loose caps used for reported energies (`1000` for all
three, i.e. Vina's `authentic_v`).

`math::smooth_div` (Vina's `smooth_div`) is also provided; no term in the current
force fields calls it.

### `not_max`

```text
not_max(x) = x < 0.1 * MAX_F
```

The guard that distinguishes a real cap from "no cap"
(`scoring::NO_CAP == MAX_F`).

---

## The Vina force field

### Numeric definitions

Term parameters (`XsScoringFunction::vina`):

| # | Term | Parameters | Cutoff (Å) |
|---|---|---|---|
| 0 | gauss1 | `offset = 0.0`, `width = 0.5` | 8.0 |
| 1 | gauss2 | `offset = 3.0`, `width = 2.0` | 8.0 |
| 2 | repulsion | `offset = 0.0` | 8.0 |
| 3 | hydrophobic | `good = 0.5`, `bad = 1.5` | 8.0 |
| 4 | H-bond | `good = -0.7`, `bad = 0.0` | 8.0 |
| 5 | linear attraction (glue) | — | 20.0 |

Weights (`Weights::vina_default()`, from `vina.h`'s `set_vina_weights`
defaults):

```text
w = [ w_gauss1, w_gauss2, w_repulsion, w_hydrophobic, w_hydrogen, w_glue ]
  = [ -0.035579, -0.005156,  +0.840245,  -0.035069,   -0.587439,  50.0    ]
w_rot = 0.05846
```

Radii (`XS_VDW_RADII`, Vina's `xs_vdw_radii[]`), by X-Score type:

| Type | CH | CP | NP | ND | NA | NDA | OP | OD | OA | ODA | SP | PP | FH | ClH | BrH | IH | Si | At | MetD | G0..G3 | W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Å | 1.9 | 1.9 | 1.8 | 1.8 | 1.8 | 1.8 | 1.7 | 1.7 | 1.7 | 1.7 | 2.0 | 2.1 | 1.5 | 1.8 | 2.0 | 2.2 | 2.2 | 2.3 | 1.2 | 1.9 | 0.0 |

### The pair energy

```text
For r = d_ij - opt_ij, with all terms zero for d_ij >= 8 (term 5: d_ij >= 20):

  gauss1(r)      = exp( -(r / 0.5)^2 )
  gauss2(r)      = exp( -( (r - 3.0) / 2.0 )^2 )
  repulsion(r)   = r^2                    if r < 0 else 0
  hydrophobic(r) = slope_step(1.5, 0.5, r)         if both types hydrophobic
  hbond(r)       = slope_step(0.0, -0.7, r)        if a donor/acceptor pair
  glue(r)        = r                               if the two types are glued

  E_ij = w1*gauss1(r) + w2*gauss2(r) + w3*repulsion(r)
       + w4*hydrophobic(r) + w5*hbond(r) + w6*glue(r)
```

The hydrophobic term requires `t_i.is_hydrophobic() && t_j.is_hydrophobic()`
(`CH`, `FH`, `ClH`, `BrH`, `IH`); the H-bond term requires
`t_i.h_bond_possible(t_j)`, i.e. one of the two is a donor (`ND`, `NDA`, `OD`,
`ODA`, `MetD`) and the other an acceptor (`NA`, `NDA`, `OA`, `ODA`); the linear
attraction requires `is_glued(t_i, t_j)`, i.e. a closure dummy paired with its
own closure carbon.

The two Gaussians contribute a single asymmetric well each; the raw shape of the
Vina potential is their weighted sum. `gauss1` peaks at `r = 0`, i.e. exactly at
the sum of the van der Waals radii, and is *negative* (`w1 < 0`); `gauss2` peaks
at `r = 3 Å` beyond that, also negative but ten times shallower.

**Known values**, pinned by the `gauss_terms_have_known_values`,
`hydrophobic_and_hbond_ramps` and `conf_independent_*` tests (CH/CH has
`opt = 3.8 Å`, ND/OA has `opt = 3.5 Å`):

```text
gauss1  (CH/CH, d = 3.8 -> r = 0.0)   = 1.0        -> w1 * 1.0    = -0.035579
gauss2  (CH/CH, d = 6.8 -> r = 3.0)   = 1.0        -> w2 * 1.0    = -0.005156
repul.  (CH/CH, d = 3.9 -> r = 0.1)   = 0.0
repul.  (CH/CH, d = 3.3 -> r = -0.5)  = 0.25       -> w3 * 0.25   = +0.210061
hydro   (CH/CH, d = 3.8 -> r = 0.0)   = 1.0
hydro   (CH/CH, d = 4.3 -> r = 0.5)   = 1.0
hydro   (CH/CH, d = 4.8 -> r = 1.0)   = 0.5        -> w4 * 0.5    = -0.017535
hydro   (CH/CH, d = 5.3 -> r = 1.5)   = 0.0
hydro   (CH/OA, any r)                = 0.0        (not both hydrophobic)
hbond   (ND/OA, d = 3.5 -> r = 0.0)   = 0.0
hbond   (ND/OA, d = 2.8 -> r = -0.7)  = 1.0        -> w5 * 1.0    = -0.587439
hbond   (ND/OA, d = 3.15 -> r = -0.35)= 0.5        -> w5 * 0.5    = -0.293720
all six terms at d = 8.5              = 0.0        (beyond every cutoff)
```

That the first Gaussian peaks at `r = 0` and the second 3 Å further out is what
gives the Vina potential its characteristic double-well character: a steep
repulsive wall below the contact distance, a narrow attractive well at the
contact distance, and a broad, shallow second attraction around 3 Å beyond it.

### Cutoff rules

* Each term independently returns `(0.0, 0.0)` for `d_ij >= cutoff_k`, tested
  on the **raw** distance, not the reduced one.
* The exact scorer's neighbour loop uses `ScoringFunction::cutoff()` = 8.0 Å for
  Vina and Vinardo, so no receptor atom beyond that is even considered.
* Macrocycle glue pairs use `ScoringFunction::max_cutoff()` = 20.0 Å, because
  the linear attraction is deliberately long-ranged; no other term survives past
  8 Å.
* The residual step at the cutoff is real but tiny: at `d = 8 Å` only `gauss2`
  is non-zero, with a magnitude `|w2| * exp(-((8 - opt - 3)/2)^2)`. For CH/CH
  (`opt = 3.8`) that is `|w2| * exp(-0.36) = 0.0036` kcal/mol. The module records
  the residual as `w2 * exp(-1/4) ~ -0.004` kcal/mol. Reference implementations
  work around this by tabulating the potential and differentiating the table;
  OpenDocking keeps the analytic form and accepts the step.

### Derivatives, step by step

**Term 0 and 1 — the Gaussians.** With `u = r - offset` (`width` is the Gaussian
width, `W` the term weight):

```text
E   = W * exp( -(u / width)^2 )
dE/dr = W * exp( -(u / width)^2 ) * d/dr[ -(u/width)^2 ]
      = W * exp( -(u / width)^2 ) * ( -2u / width^2 )
      = E * ( -2u / width^2 )
```

so `gauss1` (`u = r`, `width = 0.5`) and `gauss2` (`u = r - 3`, `width = 2.0`)
are both smooth and `C^inf`; there is no kink anywhere in their support. In code:

```rust
let u = r - (self.optimal(t1, t2, radii) + offset);
let e = (-sqr(u / width)).exp();
(e, e * (-2.0 * u / sqr(width)))
```

**Term 2 — the soft-sphere repulsion.** With `u = r - offset`:

```text
E    = u^2   if u < 0   else 0
dE/dr = 2u   if u < 0   else 0
```

The derivative is *negative* for `u < 0` (moving the atoms further apart lowers
the energy) and clamps to zero on the attractive side, so the term never
contributes attraction. In code:

```rust
let d = r - (self.optimal(t1, t2, radii) + offset);
if d > 0.0 { (0.0, 0.0) } else { (d * d, 2.0 * d) }
```

**Term 3 — hydrophobic contact.** With `x = r` (the reduced distance, no
offset) and `S = slope_step(1.5, 0.5, x)`:

```text
S(x)     = 1                for x <= 0.5
         = (1.5 - x)/1.0    for 0.5 < x < 1.5
         = 0                for x >= 1.5

dS/dx    = 0                outside (0.5, 1.5)
         = -1               inside
```

so the weighted derivative is `w4 * (-1) = +0.035069` inside the ramp: as `r`
decreases through the ramp, the energy becomes more negative, i.e. the term is
attractive. In code the ramp is evaluated as `slope_step(bad, good, x)` and its
derivative as `slope_step_deriv(bad, good, x) = 1 / (good - bad) = -1`.

**Term 4 — non-directional H-bond.** With `x = r` and
`S = slope_step(0.0, -0.7, x)`:

```text
S(x)     = 0                for x >= 0.0
         = -x / 0.7         for -0.7 < x < 0.0
         = 1                for x <= -0.7

dS/dx    = 0                outside (-0.7, 0.0)
         = 1 / (-0.7 - 0.0) = -1.428571...   inside
```

so the weighted derivative is `w5 * (-1/0.7) = +0.839199` inside the ramp, and
the well is a *plateau* of value `w5 = -0.587439` kcal/mol for all
`r <= -0.7 Å` (i.e. `d <= opt - 0.7`). The H-bond optimum therefore sits 0.7 Å
inside the van der Waals contact distance, and the term has no directional
dependence at all: it does not look at angles, only at the donor/acceptor flags
and the distance — exactly like Vina's `vina_hbond`.

**Term 5 — linear attraction (macrocycle glue).**

```text
E    = r        (only when the pair is glued and r < 20)
dE/dr = 1
```

with weight `w6 = 50.0`, so a glued pair is pulled together by a constant
50 kcal/mol per Å. This term exists to keep the two halves of a closed
macrocycle in contact while allowing the closure pseudo-atoms to be treated as
independent particles. It is evaluated over the *glue pair list* only, with the
longer 20 Å cutoff.

### Vinardo

Vinardo is the same machinery with a different parameterisation: one Gaussian, a
slightly different hydrophobic ramp and no `gauss2`.

| # | Term | Parameters | Weight |
|---|---|---|---|
| 0 | gauss | `offset = 0.0`, `width = 0.8` | `-0.045` |
| 1 | repulsion | `offset = 0.0` | `+0.8` |
| 2 | hydrophobic | `good = 0.0`, `bad = 2.5` | `-0.035` |
| 3 | H-bond | `good = -0.6`, `bad = 0.0` | `-0.600` |
| 4 | linear attraction | — | `50.0` |

with `w_rot = 0.05846` and the Vinardo radius table (`XS_VINARDO_VDW_RADII`):
CH 2.0, CP 2.0, NP/ND/NA/NDA 1.7, OP/OD/OA/ODA 1.6, SP 2.0, PP 2.1, FH 1.5,
ClH 1.8, BrH 2.0, IH 2.2, Si 2.2, At 2.3, MetD 1.2, closure types 2.0, W 0.0.

The derivatives are *identical* to the Vina ones with the corresponding
parameters substituted — the functional forms do not change. The only
structural difference is the absence of the second Gaussian.

---

## The AutoDock 4.2 force field

`Ad4ScoringFunction` refuses to be gridded (`is_grid_capable() == false`) and is
not X-Score typed (`is_xs_typed() == false`), so it is always evaluated from the
explicit atoms and its pair terms need the AD4 type and the partial charge.

### Numeric definitions

| # | Term | Parameters | Cutoff (Å) | Weight |
|---|---|---|---|---|
| 0 | `Ad4Vdw` | `smoothing = 0.5`, `cap = 1e5` | 8.0 | `0.1662` |
| 1 | `Ad4Hb` | `smoothing = 0.5`, `cap = 1e5` | 8.0 | `0.1209` |
| 2 | `Ad4Elec` | `cap = 100.0` | 20.48 | `0.1406` |
| 3 | `Ad4Solv` | `sigma = 3.6`, `solvation_q = 0.01097`, charge-dependent | 20.48 | `0.1322` |
| 4 | linear attraction | — | 20.0 | `50.0` |

`w_rot = 0.2983`, applied **additively** (see
[Torsional penalty](#torsional-penalty-and-the-reported-affinity)).
`ScoringFunction::cutoff()` for AD4 is 20.48 Å (the largest term cutoff) and
`max_cutoff()` is also 20.48 Å.

The per-pair classification comes from the AD4 table
(`ATOM_KIND_DATA`, Vina's `atom_kind_data[]`):

```text
pa, pb        = ad_type_property(a.ad), ad_type_property(b.ad)
hb_depth      = pa.hb_depth * pb.hb_depth
vdw_rij       = pa.radius + pb.radius
vdw_depth     = sqrt(pa.depth * pb.depth)
hb_rij        = pa.hb_radius + pb.hb_radius

a pair is an H-BOND  iff hb_depth < 0
a pair is a vdW pair iff hb_depth >= 0      (the two are mutually exclusive)
```

### 12-6 van der Waals with `smoothen`

```text
(r_s, dp) = smoothen(r, vdw_rij, smoothing = 0.5)

c12 = vdw_rij^12 * vdw_depth
c6  = vdw_rij^6  * vdw_depth * 2

E    = c12 / r_s^12 - c6 / r_s^6
dE/dr = ( -12*c12 / r_s^13  +  6*c6 / r_s^7 ) * dp
```

*Derivation of the coefficients.* The standard 12-6 form is
`E(r) = eps * ((R/r)^12 - 2*(R/r)^6)`. Expanding,

```text
E(r) = (eps * R^12) / r^12 - (2 * eps * R^6) / r^6
```

so with `c12 = R^12 * eps` and `c6 = 2 * R^6 * eps` the implementation is exactly
that form. The factors are chosen so that the well sits at `r = R` and is exactly
`-eps` deep:

```text
E(R)    = eps - 2*eps = -eps
dE/dr   = -12*eps/R + 12*eps/R = 0        at r = R
```

Here `R = vdw_rij = pa.radius + pb.radius` (the *sum* of the atomic radii, the
AD4 convention) and `eps = sqrt(pa.depth * pb.depth)` (the geometric mean of the
well depths, the AD4 combining rule). For a C/C pair: `R = 2.0 + 2.0 = 4.0 Å`,
`eps = sqrt(0.15 * 0.15) = 0.15` kcal/mol, so the well is −0.15 kcal/mol at
4.0 Å.

*Derivation of the derivative.* Term by term,

```text
d/dr [ c12 * r^-12 ] = -12 * c12 * r^-13
d/dr [ -c6 * r^-6  ] =  +6 * c6  * r^-7
```

and the chain rule through the shift contributes the factor `dp = dr_s/dr`,
which is 1 outside the plateau and 0 inside it. Inside the plateau the shifted
radius is pinned at `r = vdw_rij`, where the true derivative is zero anyway, so
setting it to zero there is consistent rather than merely convenient.

*Cap and degeneracy.* If `r_s <= EPSILON` (a numerically coincident pair) the
term returns `(cap, 0.0)`. Otherwise the energy is compared with `cap = 1e5`: if
`E >= cap`, the pair returns `(cap, 0.0)` — the energy is clamped and the
gradient is zeroed, so an atom cannot be pushed out of a clash by an infinite
force.

### 12-10 hydrogen bond

```text
(r_s, dp) = smoothen(r, hb_rij, smoothing = 0.5)

d     = -hb_depth                 (> 0, because hb_depth < 0)
c12   = hb_rij^12 * d * 10 / 2
c10   = hb_rij^10 * d * 12 / 2

E     = c12 / r_s^12 - c10 / r_s^10
dE/dr = ( -12*c12 / r_s^13  +  10*c10 / r_s^11 ) * dp
```

*Derivation of the coefficients.* Write `E(r) = A/r^12 - B/r^10`. Requiring the
minimum at `r = hb_rij` and the value there to be `hb_depth` (a negative number
for a real H-bond) gives two equations:

```text
dE/dr |_{r=R} = -12A/R^13 + 10B/R^11 = 0   =>   B = (12/10) * A / R^2
E(R)          = A/R^12 - B/R^10 = A/R^12 - (6/5)*A/R^12 = -(1/5)*A/R^12
                = hb_depth
             =>  A = -5 * hb_depth * R^12 = R^12 * d * 5      with d = -hb_depth
                 B = (6/5) * A / R^2     = R^10 * d * 6
```

which is exactly `c12 = R^12 * d * 10/2` and `c10 = R^10 * d * 12/2`. For the
`NA/HD` pair: `R = 1.9 + 0.0 = 1.9 Å`, `hb_depth = (-5.0) * (+1.0) = -5.0`,
`d = 5.0`, and at `r = 1.9` the energy is `25 - 30 = -5.0` kcal/mol, exactly
`hb_depth`. The AD4 H-bond term is therefore not a "well depth scaled by a
geometric mean" but the literal `hb_depth` of the pair.

As with the van der Waals term, `r_s <= EPSILON` and `E >= cap` both clamp to
`(cap, 0.0)`.

### Distance-dependent dielectric electrostatics

```text
B      = 78.4 + 8.5525 = 86.9525
LB     = -B * 0.003627 = -0.315389...

eps(r)     = -8.5525 + B / (1 + 7.7839 * exp(LB * r))
d eps/dr   = -B / (1 + e)^2 * e * LB          with e = 7.7839 * exp(LB*r)

q1q2   = q_a * q_b * 332.0
raw    = 1 / (r * eps(r))

E      = q1q2 * raw
dE/dr  = q1q2 * ( -raw / r  -  (d eps/dr) / (r * eps(r)^2) )

cap:   if raw >= 100:  E = q1q2 * 100, gradient 0
       if r < EPSILON: E = q1q2 * cap / eps(0)  (when eps(0) > 0), gradient 0
```

*Derivation of the derivative.* Differentiate `E = q1q2 * (r * eps(r))^-1`:

```text
d/dr [ 1 / (r * eps) ] = -(1 / (r * eps)^2) * d/dr[ r * eps ]
                       = -(1 / (r * eps)^2) * ( eps + r * eps' )
                       = -1 / (r^2 * eps)  -  eps' / (r * eps^2)
                       = -raw / r          -  eps' / (r * eps^2)
```

which is the expression in the code with `raw = 1/(r*eps)`. The 332.0 factor is
the standard conversion from `e^2/Å` to kcal/mol for charges in electron units.

*The dielectric function.* `eps(0) = 1.3465` (the test asserts 1.3465 within
1e-3) and `eps(inf) = -8.5525 + 86.9525 = 78.4` (the test asserts 78.4 at
`r = 1000`). It is a smooth, monotone ramp from a low-dielectric interior to the
bulk value of water; the AutoDock 4 form is a shifted exponential, and the three
literals (`78.4`, `8.5525`, `0.003627`) are the published AD4 constants. The
charge-dependent screening is what makes AD4's electrostatics a
*screened* Coulomb interaction rather than a vacuum one.

### Desolvation

```text
s1, s2 = solvation_parameter(a), solvation_parameter(b)
v1, v2 = atomic_volume(a), atomic_volume(b)
mq     = solvation_q if charge_dependent else 0.0

C      = (s1 + mq * |q_a|) * v2  +  (s2 + mq * |q_b|) * v1
x      = r / sigma                        (sigma = 3.6 Å)
E      = C * exp( -0.5 * x^2 )
dE/dr  = -E * r / sigma^2
```

*Derivation of the derivative.* With `g(r) = exp(-r^2 / (2*sigma^2))`,

```text
dg/dr  = exp( -r^2/(2 sigma^2) ) * (-r / sigma^2) = -g * r / sigma^2
dE/dr  = C * dg/dr = -C * g * r / sigma^2 = -E * r / sigma^2
```

a single `exp()` and two multiplications. The term is a Gaussian shell of width
`sigma` around each atom, weighted by the product of the two atoms' volumes and
a solvation parameter that becomes charge-dependent through
`solvation_q = 0.01097` — the AD4 desolvation model of the buried-surface
penalty on binding.

The two per-atom helpers:

```text
solvation_parameter(a) =
    ad_type_property(a.ad).solvation          if a.ad != Unknown
    METAL_SOLVATION_PARAMETER = -0.00110      else if a.xs == MetD
    0.0                                       otherwise

atomic_volume(a) =
    ad_type_property(a.ad).volume             if a.ad != Unknown
    4/3 * pi * a.xs.radius()^3                otherwise
```

so a metal the AD4 table does not cover still gets a physically sane volume (a
sphere of its X-Score radius) and a solvation parameter.

### AD4 ordering and exclusions

* The vdW and H-bond terms are mutually exclusive per pair, decided by the sign
  of `hb_depth`; there is no double counting.
* All four pairwise terms are zero at or beyond their own cutoff.
* The `Ad4Elec` and `Ad4Solv` terms have the long 20.48 Å cutoff, which is why
  `ScoringFunction::cutoff()` for AD4 is 20.48 Å and why the AD4 neighbour search
  visits substantially more receptor atoms than the Vina search does.

---

## From `dE/dr` to a Cartesian gradient

Every term returns its radial derivative. The Cartesian gradient follows from
`r_ij = |x_i - x_j|`:

```text
dr/dx_i = (x_i - x_j) / r_ij

dE_ij/dx_i = (dE_ij/dr_ij) * (x_i - x_j) / d_ij
dE_ij/dx_j = -dE_ij/dx_i
```

which is exactly what the exact scorer does:

```rust
let r_ba = adj - b.coords;              // x_a - x_b
let (pair_e, de) = sf.pair_energy_deriv(a, b, r);
deriv += r_ba * (de / r);               // dE/dx_a
```

and what `eval_pairs` does for an explicit pair list:

```rust
let mut grad = if r > 0.0 { r_ba * (de / r) } else { DVec3::ZERO };
curl(&mut e, &mut grad, v);
f[p.a] += grad;
f[p.b] -= grad;
```

The accumulated buffer `MovableModel::minus_forces` therefore holds
`F_i = dE/dx_i` for every movable atom (the name follows Vina's `m.minus_forces`
member; the sign convention is the *gradient*, not the force).

### Rigid-body projection

For an infinitesimal rotation `d(omega)` about a point `o`, an atom moves by
`dx_i = d(omega) x (x_i - o)`. Using the scalar triple product identity
`a . (b x c) = b . (c x a)`:

```text
dE = sum_i (dE/dx_i) . (d(omega) x (x_i - o))
   = d(omega) . sum_i (x_i - o) x (dE/dx_i)
```

and for a torsion about a unit axis `a` through `o`, with
`d(omega) = a * d(theta)`:

```text
dE/dtheta = a . sum_i (x_i - o) x (dE/dx_i)
```

so the projection the kernel implements is

```text
dE/dtrans     = sum_i  F_i                              (translation)
dE/domega     = sum_i  (x_i - o) x F_i                  (rotation vector)
dE/dtorsion   = axis . sum_i (x_i - o) x F_i            (torsion)
```

with `F_i = dE/dx_i` and `o` the rotation centre of the frame that owns atom
`i`. This is the standard rigid-body chain rule; it is also literally the text of
the `kinematics.rs` module documentation.

Two details make the recursion correct rather than merely plausible:

1. **Nested frames need a lever arm.** A child segment rotates about *its own*
   origin, not about the parent's. When the child's summed gradient `F_child` and
   torque `tau_child` are folded into the parent, the parent must add the lever
   arm `(o_child - o_parent) x F_child` before adding `tau_child`:

   ```rust
   for child in &self.children {
       let (cf, ct) = child.accumulate_gradient(coords, forces, g, None, torsion_base);
       force  += cf;
       torque += (child.origin - self.origin).cross(cf) + ct;
   }
   ```

   Without that term, rotating a torsion far down the tree would report a torque
   measured about the wrong point.

2. **The rigid-body centre is the root atom.** `LigandConf::position` *is*
   `Node.origin` for the root, and `accumulate_gradient` accumulates the torque
   as `sum (coords[i] - self.origin) x f`, i.e. about the root atom. This matches
   `Conf::increment_flat`, which adds the translation first and then applies the
   quaternion increment about the (translated) position — so the analytic
   gradient and the finite-difference perturbation use the same rotation centre.

For a flexible-residue tree the root is pinned: `KinematicTree::accumulate_gradient`
handles `TreeKind::Flex` separately and writes only the root's own torsion slot,
because a pinned segment has no translation or rotation degrees of freedom.

### The quaternion increment convention

Quaternions are stored as `(x, y, z, w)` with `w` the scalar part, and the
rotation matrix is

```text
      | w^2+x^2-y^2-z^2    2(xy - wz)      2(xz + wy)   |
  R = | 2(wz + xy)      w^2-x^2+y^2-z^2    2(yz - wx)   |
      | 2(xz - wy)        2(yz + wx)    w^2-x^2-y^2+z^2 |
```

which is `glam::DMat3::from_quat` and, bit for bit, Vina's `quaternion_to_r3`
(pinned by `rotation_matrix_matches_vina_convention`).

A rotation-vector increment `d(omega)` is turned into a quaternion and composed
on the **left**:

```text
rotation_vector_to_quaternion(dw):
    angle = |dw|
    if angle <= EPSILON: return identity
    return angle_to_quaternion(dw / angle, normalize_angle(angle))

quaternion_increment(q, dw):
    r = rotation_vector_to_quaternion(dw) * q        # left multiplication
    return r / |r|                                   # exact renormalisation
```

Left multiplication means the increment is expressed in the **parent / world
frame**: the new orientation is `q_inc * q_old`, not `q_old * q_inc`. That is the
convention the torque formula above assumes (the torque is computed in the
laboratory frame), and it is what makes the rotation-vector slots of the flat
gradient directly usable as BFGS directions.

Two conventions worth stating explicitly:

* `angle_to_quaternion` wraps its angle into `(-pi, pi]` via `normalize_angle`,
  and `normalize_angle` is total: `+pi` and `-pi` are the same angle, and a
  non-finite input yields `0.0` rather than a `NaN`.
* Vina's `quaternion_increment` uses an approximate renormalisation and warns
  that rounding errors grow slowly; OpenDocking renormalises exactly, which
  removes the caveat without changing the convention
  (pinned by `quaternion_increment_is_normalised_and_composes` and
  `quaternion_difference_round_trips`).

`LigandChange::orientation` is therefore a rotation *vector*, not a quaternion:
direction = axis, length = angle. `LigandChange::position` is an additive
translation, and `LigandChange::torsions` are additive angle increments. The
flat DOF vector that BFGS manipulates is

```text
[ t_x t_y t_z | w_x w_y w_z | theta_0 ... theta_n-1 | flex torsions ... ]
     0   1   2     3   4   5        6 ...
```

and `Conf::increment_flat` applies it in that order, wrapping every torsion into
`(-pi, pi]` both for the increment and for the result.

---

## Torsional penalty and the reported affinity

### Vina's torsion count

`scoring::num_tors` reproduces Vina's `num_tors`:

```text
for each rotatable bond (parent, attachment) from the PDBQT topology:
    if the attachment atom is heavy and has more than one heavy neighbour:
        n += 0.5
    if the parent is movable, heavy, and has more than one heavy neighbour:
        n += 0.5
```

so a rotor contributes 0.5 per *non-terminal* endpoint, `N_tors` is a multiple of
0.5, and a terminal group such as `-CH3` or `-OH` never contributes a full rotor
(pinned by `num_tors_matches_vina_semantics`: a butane-like chain with one rotor
between two CH2 groups gives exactly `1.0`).

### The rule

```text
Vina, Vinardo:
    base  = inter + intra - unbound
    total = base / (1 + w_rot * N_tors)          for N_tors >= EPSILON
    total = base                                 otherwise

AD4:
    total = inter + w_rot * N_tors               (additive)

reported affinity = total = ScoreComponents::total
```

Expressed through the trait method:

```rust
fn conf_independent(&self, e: f64, num_tors: f64) -> f64 {
    if num_tors.abs() < EPSILON { return e; }        // Vina / Vinardo
    e / (1.0 + self.weights.rot * num_tors)
}

fn conf_independent(&self, e: f64, num_tors: f64) -> f64 {
    e + self.weights.rot * num_tors                  // AD4
}
```

The Vina form is the algebra of Vina's `num_tors_div`: that evaluator computes
`conf_smooth_div(x, 1 + weight * num_tors / 5)` with `weight = 0.1 * (w + 1)`,
and `Vina::set_vina_weights` (and `set_vinardo_weights`) pushes
`w = 5 * weight_rot / 0.1 - 1` as the last weight. Substituting gives
`weight = 5 * rot` and therefore a divisor of exactly `1 + rot * N_tors`, which
is the form implemented here, with the published `w_rot = 0.05846`. Vina routes
the division through `conf_smooth_div` (a division that degrades gracefully near
zero); OpenDocking uses a plain division guarded by `|N_tors| < EPSILON`.
The AD4 form is Vina's `ad4_tors_add` (`x + weight * torsdof`) with
`w_rot = 0.2983`.

The test `conf_independent_reproduces_the_published_penalty` pins the published
value: `-10.0 / (1 + 0.05846 * 4) = -8.1048` kcal/mol.
`ad4_torsion_term_is_additive` pins `conf_independent(1.5, 4.0) = 1.5 + 0.2983*4`.

### Why the ligand's internal energy cancels

With a **rigid receptor**, the ligand's intramolecular energy depends only on the
ligand's own conformation:

```text
E_intra(conformation)   = sum over intra-ligand pairs of E_ij(conformation)
E_unbound(conformation) = the same sum, for the same ligand in isolation
```

Both are evaluated with the identical pair list (`Shared::intra_pairs` plus
`Shared::glue_pairs`), the identical force field and the identical conformation,
so `E_intra == E_unbound` **exactly, by construction** — the implementation does
not have to estimate or approximate the unbound state, it reuses the number:

```rust
let unbound = match self.shared.sf.choice() {
    SfChoice::Ad4 => 0.0,
    _             => intra,
};
let base = inter + intra - unbound;
```

Consequences:

* For Vina and Vinardo the ligand's internal strain is removed from the reported
  affinity: `base = inter`. The affinity is a pure intermolecular quantity,
  divided by the entropic torsional factor.
* For AD4 the reference implementation does *not* subtract the unbound term and
  instead adds the torsional term, so OpenDocking sets `unbound = 0.0` for AD4
  and reports `total = inter + w_rot * N_tors`. The `intra` value is still
  computed and reported in the pose's `INTRA` remark, but it is not part of the
  affinity.
* `ScoreComponents::conf_independent` is *defined* as `total - base`. For
  Vina/Vinardo that is the (negative) shift produced by the divisor; for AD4 it
  is `w_rot * N_tors - intra`.

### The objective the search minimises

The torsional factor depends only on the topology, so it is a positive constant
for a given ligand and cannot move the location of a minimum. The search
therefore minimises the undivided energy:

```text
Vina, Vinardo:  objective = inter + intra - unbound        (= inter here)
AD4:            objective = inter + conf_independent
```

and the divisor is applied only when the final numbers are reported. This mirrors
the reference implementation exactly, and it is why the energy a pose is
*optimised* on can differ from the affinity that is *printed* for it by the
divisor — for a benzene-like ligand with `N_tors = 1` and `w_rot = 0.05846` the
printed value is 5.5 % smaller in magnitude than the objective.

Worked example from the validation run (mode 1 of the 3PTB re-docking,
benzamidine: the single aryl–amidine rotor has two endpoints with more than one
heavy neighbour, so `N_tors = 1.0`):

```text
inter            = -8.458 kcal/mol
intra            = -0.043
unbound          = -0.043
base             = -8.458 + (-0.043) - (-0.043) = -8.458
N_tors           = 1.0
divisor          = 1 + 0.05846 * 1.0 = 1.05846
affinity (total) = -8.458 / 1.05846 = -7.9909  ->  prints as -7.991
```

and indeed the file records `INTER + INTRA: -8.501`, `INTER: -8.458`,
`INTRA: -0.043`, `UNBOUND: -0.043`, `CONF_INDEPENDENT: 0.467` and
`VINA RESULT: -7.991`. The last two numbers are the *reported* affinity and the
shift produced by the divisor: `-7.991 - (-8.458) = 0.467`.

---

## The affinity grid

### Why a grid, and what it changes

The search evaluates the scoring function millions of times. The exact
ligand-receptor evaluation costs one `exp()` per term per receptor pair inside
the cutoff; a pre-computed grid reduces that to eight array reads and one
trilinear blend per ligand atom, roughly two orders of magnitude faster. An
interpolated energy is *not* the true energy, so every reported pose is finally
relaxed on the exact surface with `refine_pose` (see
[`ARCHITECTURE.md`](ARCHITECTURE.md#data-flow-of-a-docking-run)). The grid is
built with the same `pair_energy_xs` code the exact scorer uses, so the two paths
share one implementation of the physics.

The grid is only used for X-Score-typed force fields: `is_grid_capable()` is
`false` for AD4, and `build_system` therefore sets `use_grid = false` for it.

### Trilinear interpolation

Let `f_ijk` be the eight corner values of the cell containing the probe, and
`(x, y, z)` the probe's *fractional cell coordinates* in `[0, 1]^3`:

```text
w_0(t) = 1 - t
w_1(t) = t

f(x, y, z) = sum_{i,j,k in {0,1}}  f_ijk * w_i(x) * w_j(y) * w_k(z)
```

Written out, this is the evaluation order the code uses:

```text
mx = 1 - x;  my = 1 - y;  mz = 1 - z

f = f000*mx*my*mz + f100*x*my*mz
  + f010*mx*y*mz  + f110*x*y*mz
  + f001*mx*my*z  + f101*x*my*z
  + f011*mx*y*z   + f111*x*y*z
```

### The analytic gradient of the interpolated energy

Differentiating the blend with respect to `x` (the other two weights are
constants for that derivative) collapses each pair of corners into a difference:

```text
df/dx = (f100 - f000)*my*mz + (f110 - f010)*y*mz
      + (f101 - f001)*my*z  + (f111 - f011)*y*z

df/dy = (f010 - f000)*mx*mz + (f110 - f100)*x*mz
      + (f011 - f001)*mx*z  + (f111 - f101)*x*z

df/dz = (f001 - f000)*mx*my + (f101 - f100)*x*my
      + (f011 - f010)*mx*y  + (f111 - f110)*x*y
```

These are the exact derivatives of the interpolated function, not a
finite-difference approximation of it, which is what keeps the BFGS step
consistent with the energy it is minimising. This is AutoDock Vina's
`grid::evaluate_aux` (`grid.cpp`, Apache-2.0), reproduced term for term.

The fractional cell coordinate is

```text
s_i       = (p_i - begin_i) * factor_i
factor_i  = (n_points_i - 1) / span_i = n_voxels_i / (spacing * n_voxels_i)
          = 1 / spacing
inv_factor_i = spacing
a_i       = floor(s_i)            (cell index; clamped for out-of-box probes)
frac_i    = s_i - a_i             (the x, y, z above)
```

and the conversion back to world units multiplies by `factor_i`:

```text
df/dp_i = df/ds_i * factor_i
```

which the code applies only when the probe is inside the box on that axis.

### The out-of-box penalty

For each axis, the probe's fractional coordinate is compared with the valid
range `[0, n_voxels]`:

```text
s_i < 0:            a_i = 0,           frac_i = 0,   miss_i = -s_i * spacing
s_i >= n_voxels:    a_i = n_voxels-1,  frac_i = 1,   miss_i = (s_i - n_voxels) * spacing
otherwise:          a_i = floor(s_i),  frac_i = s_i - a_i,  miss_i = 0

penalty   = slope * (miss_0 + miss_1 + miss_2)          slope = 1e6
gradient  = slope * region_i          with region_i = -1, 0 or +1
```

So the probe is *clamped* to the boundary cell for the purpose of the
interaction energy and a linear penalty of `slope x distance` is added on top.
The gradient on an escaped axis is exactly `slope * region_i` — a constant
million kcal/mol/Å pulling the probe back — which keeps the search inside the
user's box without adding a constraint to the optimiser. Note that the penalty is
added *after* the `curl` soft cap, and that an escaped axis replaces the
interpolated gradient component entirely.

The exact scorer applies the same penalty in `NonCache::clamp_to_box`, with the
gradient sign inverted to match (`deriv[j] = -1` when the atom is below the box,
`+1` when it is above), and adds the penalty *after* curling the interaction
energy. The `DEFAULT_SLOPE = 1e6` constant is Vina's.

Pinned behaviour:

* `grid_gradient_matches_finite_differences` — analytic vs central difference,
  tolerance `5e-3 * (1 + |numeric|)`, `h = 1e-6`.
* `grid_energy_tracks_the_exact_energy` — a single-carbon probe is within
  0.05 kcal/mol of the exact pair energy.
* `outside_the_box_is_penalised_linearly` (grid) — a probe 10 Å outside a 6 Å box
  gives `E > 1e6`, `is_in_grid(margin=0) == false`, `is_in_grid(margin=100) ==
  true`.
* `out_of_box_penalty_is_a_linear_ramp` (exact) — `E >= 3.0e6` for a probe 3 Å
  outside a box with `slope = 1e6`, and `forces[0].x == 1e6`.

### Grid type curation

```text
grid_type(t):
    G0 | G1 | G2 | G3 | W          -> None            (no map at all)
    CHCg0 | CHCg1 | CHCg2 | CHCg3  -> Some(CH)
    CPCg0 | CPCg1 | CPCg2 | CPCg3  -> Some(CP)
    otherwise                      -> Some(t)
```

so the closure carbons share their parent type's map and the closure dummies and
untyped hydrogens have none. `GRID_TYPES` lists the nineteen types that can be
stored, and `build_system` only builds the maps for the types the ligand actually
presents (typically 6–10). `AffinityGrid::has_types` treats a type with no map as
satisfied, because such an atom is skipped by the evaluator anyway.

---

## Hydrogens and X-Score typing

This is the curation rule that makes the Vina/Vinardo typing reproducible from a
PDBQT file that has already had its non-polar hydrogens merged away.

```text
XsType::from_element(el, ad, bonded_to_hd, bonded_to_heteroatom):
    ...
    Element::H => XsType::W
```

Every hydrogen is typed `XsType::W`, the sentinel that means "no X-Score type".
`W` has radius 0.0, is neither hydrophobic, donor nor acceptor, and is not a glue
type. The consequences are explicit at four places in the kernel:

1. `grid_type(XsType::W) == None`, so a hydrogen never gets a grid map.
2. `NonCache::eval` skips a movable atom with `xs == XsType::W` and writes a zero
   gradient for it.
3. `ligand_pairs` and `eval_pairs` skip a pair when either partner is `W`, so a
   ligand hydrogen contributes no intra-ligand term either.
4. `XsType::from_index` clamps out-of-range values to `W`, so the sentinel is
   also the safe default.

**Hydrogens still matter.** The donor flag of the heavy neighbour is derived from
the *bond graph*, not from the file:

```text
donor = (el == Element::Met) || bonded_to_hd
```

where `bonded_to_hd == true` iff the atom has a bond to an atom whose AD4 type is
`HD` (polar hydrogen). So a hydroxyl oxygen becomes `ODA` (donor **and**
acceptor), a carbonyl oxygen only ever `OA`, a backbone amide nitrogen `ND`, and
an aromatic carbon attached to nitrogen `CP` rather than `CH`. This is exactly
Vina's `model::assign_types`, and it is pinned by the
`donor_acceptor_typing_matches_vina` and `perceives_water_and_methane_bonds`
tests:

```text
from_element(O,  OA, bonded_to_hd = false, ...) = OA    (carbonyl: acceptor)
from_element(O,  OA, bonded_to_hd = true,  ...) = ODA   (hydroxyl: both)
from_element(N,  N,  bonded_to_hd = true,  ...) = ND    (amide: donor)
from_element(C,  A,  heteroatom   = true,  ...) = CP    (polar carbon)
from_element(C,  A,  heteroatom   = false, ...) = CH
from_element(Met, Zn, ...)                      = MetD
```

Because the heavy-atom typing depends on which hydrogens are present, the
preparation layer keeps **polar** hydrogens (bound to N, O or S) and merges
**non-polar** ones (bound to C) into their parent carbon, which is the AutoDock
united-atom convention. The Python side implements this in
`odock.pdbqt.strip_nonpolar_hydrogens` / `nonpolar_hydrogens` /
`polar_hydrogens`, and the AD4 typing marks a hydrogen `HD` exactly when its
heavy neighbour is N, O or S.

Receptor polar hydrogens are treated identically: they keep the `W` sentinel and
are only ever seen as the *immobile* partner of a pair, where the X-Score typing
contributes no hydrophobic and no H-bond term (a `W` atom is neither
hydrophobic, nor a donor, nor an acceptor), so only the two Gaussians and the
soft-sphere repulsion act through it.

---

## Validation

### Finite-difference tests in the Rust suite

Every analytic derivative in the kernel is verified against central finite
differences. The tests live next to the code they check:

| Test | Location | What it checks | Step / tolerance |
|---|---|---|---|
| `analytic_radial_derivatives_match_finite_differences` | `scoring/mod.rs` | every Vina and Vinardo `dE/dr`, over the X-Score pairs `CH/CH`, `CH/OA`, `ND/OA`, `CP/NA`, `SP/CH`, at `r = 0.5 + 0.035k` Å up to the cutoff | `h = 1e-6`; `|de - numeric| < 2e-5 * (1 + |numeric|)`; points with `|forward - backward| > 1e-3` are skipped because `slope_step` has genuine kinks there |
| `analytic_radial_derivatives_match_finite_differences` (AD4 branch) | `scoring/mod.rs` | the AD4 `dE/dr` for `C/C`, `C/OA (0.2, -0.4)`, `NA/HD (-0.3, 0.2)`, `A/SA (0.1, -0.1)`, at `r = 0.5 + 0.08k` Å | `h = 1e-6`; `|de - numeric| < 5e-3 * (1 + |numeric|)`; skipped when both sides are below 1e-3 |
| `analytic_gradient_matches_finite_differences` | `scoring/noncache.rs` | the exact scorer's Cartesian gradient, four receptor atoms against two movable atoms | `h = 1e-6`; `|force - numeric| < 1e-5 * (1 + |numeric|)` |
| `grid_gradient_matches_finite_differences` | `scoring/grid.rs` | the trilinear grid gradient including the interpolation factors, three receptor atoms, `spacing = 0.375 Å` | `h = 1e-6`; `< 5e-3 * (1 + |numeric|)` |
| `gradient_matches_finite_differences_for_the_rigid_body` | `kinematics.rs` | the full degree-of-freedom gradient through the kinematic tree (translation, rotation vector and torsion) against `Conf::increment_flat` | `eps = 1e-6`; `|g[k] - numeric[k]| < 1e-5` for all 7 slots of a 4-atom, 1-torsion system |
| `exact_eval_matches_a_brute_force_sum` | `scoring/noncache.rs` | the cell-list-accelerated energy against an explicit double loop that curls each atom's summed energy | `< 1e-10` |
| `grid_energy_tracks_the_exact_energy` | `scoring/grid.rs` | interpolated grid energy against the exact pair energy for a probe in a 0.25 Å grid | `< 0.05 kcal/mol` |
| `outside_the_box_is_penalised_linearly`, `out_of_box_penalty_is_a_linear_ramp` | `scoring/grid.rs`, `scoring/noncache.rs` | the out-of-box penalty value and gradient | `E > 1e6`, `force = 1e6` |
| `smoothen_plateau`, `dielectric_limits` | `scoring/mod.rs` | the AD4 helpers | exact triples; `eps(0) = 1.3465`, `eps(inf) = 78.4` |
| `gauss_terms_have_known_values`, `hydrophobic_and_hbond_ramps` | `scoring/mod.rs` | the term values quoted above | `< 1e-12` |
| `conf_independent_reproduces_the_published_penalty`, `ad4_torsion_term_is_additive` | `scoring/mod.rs` | the torsional rules | `< 1e-12` |
| `forward_kinematics_at_identity_reproduces_input`, `torsion_rotates_only_the_moving_side`, `rigid_body_transform_is_an_isometry` | `kinematics.rs` | forward kinematics: identity round trip, axis atoms immobile, all pairwise distances preserved | `< 1e-12` |
| `rotation_matrix_matches_vina_convention`, `quaternion_increment_is_normalised_and_composes`, `quaternion_difference_round_trips` | `math.rs` | the quaternion convention | `< 1e-14` |
| `bfgs_converges_on_a_quadratic` | `search/mod.rs` | the optimiser on an analytic surface | `< 1e-8` |

The two tolerances that look loose (`5e-3`) are the ones covering the AD4 terms:
inside the `smoothen` plateau the analytic derivative is exactly zero while a
central difference sees the plateau edge, and the electrostatics/desolvation
terms change by several kcal/mol over 1e-6 Å at short range, so the comparison
is a relative one. Every term is `C^1` on `(0, cutoff)` apart from the
`slope_step` kinks, which are excluded by the explicit forward/backward
disagreement test rather than by loosening the tolerance.

### Re-docking validation: PDB 3PTB

`tests/validate_3ptb.py` is the end-to-end acceptance test: it splits the
crystal structure of bovine trypsin with benzamidine, prepares both partners
through the OpenDocking pipeline, derives the box from the crystallographic
ligand, docks it back, and measures the RMSD of every pose against the
experimental position (heavy atoms, **without** superposition, so the number is
the crystallographic figure of merit).

```bash
python tests/validate_3ptb.py --exhaustiveness 16 --seed 42
```

Result of the recorded run:

```text
target                 PDB 3PTB (bovine trypsin + benzamidine)
force field            vina
independent MC runs    16               (--exhaustiveness 16)
seed                   42
top-pose RMSD          1.124 Å          (heavy atoms, no superposition)
affinity (mode 1)      -7.991 kcal/mol
top-pose RMSD (fitted) 0.354 Å          (after optimal superposition)
verdict                PASS (threshold 2.0 Å)
```

The pose file of that run records the full decomposition and the top four modes:

```text
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1    -7.991      0.000      0.000
   2    -7.797      0.202      1.608
   3    -7.307      2.614      3.602
   4    -7.007      2.276      3.267

mode 1 REMARK block:
  VINA RESULT:       -7.991      0.000      0.000
  INTER + INTRA:       -8.501
  INTER:               -8.458
  INTRA:               -0.043
  CONF_INDEPENDENT:     0.467
  UNBOUND:             -0.043
```

The affinity decomposes exactly as the formula above predicts:
`INTER + INTRA - UNBOUND = -8.458 + (-0.043) - (-0.043) = -8.458`, the torsional
divisor gives `-8.458 / (1 + 0.05846 * 1) = -7.991`, and the reported
`CONF_INDEPENDENT` is `-7.991 - (-8.458) = 0.467`. A 1.124 Å RMSD with no
superposition means the search found the crystallographic binding mode and only
the (rigid) benzamidine orientation inside it is slightly rotated — the ligand is
essentially a rigid body with one rotor, so the residual is a small in-plane
rotation rather than a wrong pose. After optimal superposition the same pose is
0.354 Å from the crystal structure.

The validation is also the practical check on the whole chain: preparation,
typing, the kinematic tree, the grid, the search protocol, refinement and
scoring all have to be right for that number to come out.

---

## Summary of every constant in the scoring path

```text
# Vina
w           = [-0.035579, -0.005156, 0.840245, -0.035069, -0.587439, 50.0]
w_rot       = 0.05846
gauss1      = offset 0.0,  width 0.5, cutoff 8.0
gauss2      = offset 3.0,  width 2.0, cutoff 8.0
repulsion   = offset 0.0,  cutoff 8.0
hydrophobic = good 0.5,    bad 1.5,  cutoff 8.0
hbond       = good -0.7,   bad 0.0,  cutoff 8.0
glue        = cutoff 20.0

# Vinardo
w           = [-0.045, 0.8, -0.035, -0.600, 50.0]
w_rot       = 0.05846
gauss       = offset 0.0, width 0.8, cutoff 8.0
repulsion   = offset 0.0, cutoff 8.0
hydrophobic = good 0.0,   bad 2.5,  cutoff 8.0
hbond       = good -0.6,  bad 0.0,  cutoff 8.0
glue        = cutoff 20.0

# AutoDock 4.2
w           = [0.1662, 0.1209, 0.1406, 0.1322, 50.0]
w_rot       = 0.2983      (additive)
vdw         = smoothing 0.5, cap 1.0e5, cutoff 8.0
hbond       = smoothing 0.5, cap 1.0e5, cutoff 8.0
electrostatics: cap 100.0, cutoff 20.48, factor 332.0
dielectric  : 78.4, 8.5525, 0.003627, 7.7839
desolvation : sigma 3.6, solvation_q 0.01097, cutoff 20.48,
              charge-dependent
metal solvation parameter = -0.00110

# Shared
out-of-box slope         = 1.0e6
hunt caps                = intra 10.0, grid 1.5, pairs 10.0
authentic caps           = 1000.0 for all three
default grid spacing     = 0.375 Å
default box              = 22.5 Å cube
Metropolis temperature   = 1.2 kcal/mol
mutation amplitude       = 2.0 Å
```

---

## Attribution

* **AutoDock Vina** — Copyright (c) 2006–2010, The Scripps Research Institute,
  Apache License 2.0. Reference for the Vina and Vinardo potentials and their
  derivatives, `slope_step` / `smoothen` / `curl`, the trilinear grid evaluation
  and its analytic gradient, the out-of-box penalty, the `num_tors` counting
  rule, the torsional divisor, and the AD4 parameter table.
* **AutoDock 4** — Copyright (c) 1989–2007, The Scripps Research Institute, GPL.
  Reference for the AD4.2 12-6 van der Waals form, the 12-10 H-bond form, the
  distance-dependent dielectric, the desolvation term and the AD4 atom typing.
* **Meeko** — Copyright (c) Forli Lab, Scripps Research, LGPL-2.1. Behavioural
  reference for ligand preparation only; no code was copied.
* **No AutoDockTools (ADT / MGLTools) code was used.**
