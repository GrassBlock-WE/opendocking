# Seeing the chemistry: surfaces, burial and interop

This document describes the visualisation layer added to the OpenDocking
workbench: the **molecular surface**, the two properties it is painted with,
the **SASA and burial measurements**, and the **interop exports** that carry a
session into PyMOL, ChimeraX or a plain PDB. It is written from the source in
this repository and from measurements taken on the bundled demo.

Everything here is deliberately specific about *which* approximation is in
play, because the whole point of a surface picture is that a chemist trusts it
enough to make a decision. Where a number is quoted it is either derived in
closed form, cross-checked against an independent implementation, or reported
by the code that produced it.

| module | what it owns |
|---|---|
| `python/odock/sasa.py` | Shrake–Rupley solvent-accessible surface area, per atom and per residue; burial against a reference; the ligand's buried contact area |
| `python/odock/gui/surface.py` | the surface field, marching-tetrahedra triangulation, the SAS→SES derivation, the property scales and the colour ramps |
| `python/odock/interop.py` | the PyMOL `.pml`, the ChimeraX `.cxc`, the pose PDB and the surface OBJ |
| `python/odock/gui/viewport.py` | the surface render pass, the world-space clipping plane |
| `python/odock/gui/app.py` | View ▸ Surface, the legend and scale bar in the viewport's HUD, Analysis ▸ SASA & burial, File ▸ Export ▸ PyMOL/ChimeraX/PDB/OBJ |
| `tools/render_surface_previews.py` | regenerates every figure in this document from the bundled demo |
| `tests/test_sasa.py`, `tests/test_surface.py`, `tests/test_interop.py` | 119 tests, every one of them either a closed-form check or a round trip |

![The S1 pocket of trypsin as a solvent-excluded surface, coloured by the Kyte–Doolittle hydropathy of the residue each patch belongs to](../out/surface/01_pocket_ses_hydrophobicity.png)

*PDB 3PTB (bovine trypsin, benzamidine). Orange is hydrophobic, blue is polar,
white is neutral. The legend on the left and the scale bar at the bottom are
drawn by the workbench itself, so a screenshot is a figure rather than a bare
render.*

---

## 1. The surface

### 1.1 Two surfaces from one field

Let every atom carry a Bondi van der Waals radius `r_i`, and let the solvent be
a probe sphere of radius 1.4 Å (a water molecule). Define the **probe-centre
field**

```
F(p) = min_i ( |p - c_i| - (r_i + probe) )
```

which is negative exactly where the centre of a water molecule would overlap an
atom. Its zero level set is the **solvent-accessible surface (SAS)** — the
surface the probe *centre* traces, which is the same surface
`odock.sasa` integrates analytically with the Shrake–Rupley point test in
§3. Triangulating `F = 0` therefore produces a mesh whose area can be checked
against a completely different numerical method, and `tests/test_surface.py`
does exactly that (agreement within 6 % at the default spacing, and the
difference closes as the grid is refined).

The **solvent-excluded surface (SES)**, the "molecular surface" every viewer
draws by default, is the boundary of the region a water molecule can never
occupy. A point is outside the SES when *some* legal probe position covers it,
so with `u` running over the unit sphere,

```
K(p) = max_{|u| = 1} F(p + probe · u)
```

and `K = 0` is the SES. This is exact as an erosion of the probe-centre
exclusion volume; the implementation samples `u` on a 42-point golden-spiral
lattice and reads the shifted field from the grid by trilinear interpolation,
so the **reentrant (concave) patches are resolved to roughly the grid
spacing**. The contact patches — everything a chemist looks at — are exact, and
the SES is always smaller than the SAS, which the tests pin.

Two closed-form checks make the derivation falsifiable rather than plausible:

| system | exact | mesh | error |
|---|---|---|---|
| one carbon, SAS: `4π(r+1.4)²` | 120.76 Å² | 119.96 Å² | −0.7 % |
| one carbon, SES: `4πr²` | 36.32 Å² | 35.93 Å² | −1.1 % |
| a sphere of radius 3.1 Å, `4πR²` at 0.32 Å spacing | 120.76 Å² | 120.43 Å² | −0.27 % |

and the triangulation error falls monotonically as the grid is refined
(−3.9 % at 1.2 Å, −1.7 % at 0.8 Å, −0.67 % at 0.5 Å, −0.27 % at 0.32 Å).

### 1.2 Triangulation: marching tetrahedra

Each grid cube is split into **six tetrahedra sharing the main diagonal** — a
partition with no gap and no overlap, chosen identically in every cube so two
neighbouring cubes agree on the face they share and the mesh has no cracks.
Each tetrahedron has 4 corners and therefore 16 sign cases; the case table is
*derived* at import time (`_tetrahedron_cases`) rather than transcribed, which
removes the one part of the algorithm that is normally copied by hand and
mis-copied. One vertex inside (or three) gives one triangle, two inside gives
the quad at the corners, four edges, two triangles.

Normals come from the **gradient of the field** the mesh was cut from, read
trilinearly at each vertex — the same construction
`skimage.measure.marching_cubes` uses, and the only one that is also right on
the reentrant patches, where "point away from the nearest atom" is visibly
wrong. A test asserts every normal points away from the atom the vertex
belongs to.

### 1.3 The default spacing, chosen from a convergence table

The mesh is a triangulated isosurface, so its area sits *under* the analytic
Shrake–Rupley integral and closes as the grid is refined. That gap is the honest
cost of the representation, so the default is chosen from the measurement rather
than from convenience. `tools/surface_convergence.py` builds the bundled 3PTB
receptor (**1 994 atoms**) at every setting:

| spacing Å | grid points | triangles | area Å² | error vs SR | wall time |
|---|---|---|---|---|---|
| 1.20 | 105 092 | 52 152 | 8 473.2 | −8.64 % | 2.21 s |
| 1.00 | 173 600 | 76 600 | 8 588.5 | −7.40 % | 3.49 s |
| 0.80 | 315 248 | 121 592 | 8 705.5 | −6.14 % | 5.49 s |
| **0.65 (default)** | 565 064 | 186 652 | 8 804.5 | **−5.07 %** | **9.21 s** |
| 0.50 | 1 188 260 | 319 624 | 8 902.0 | −4.02 % | 20.54 s |
| 0.40 | 2 237 742 | 503 848 | 8 974.2 | −3.24 % | 24.37 s |

(the reference is the analytic SASA of the same atoms, **9 274.9 Å²**; the
automatic setting, `spacing = 0`, picks 0.63 Å from the point budget for this
receptor and lands at −5.04 % in 9.5 s.)

**0.65 Å is the knee of that table**: the largest accuracy step short of the
0.50 Å setting, which costs 2.2× the time and 1.7× the triangles for one more
point of error. The default therefore changed from 0.8 Å to 0.65 Å, and the
price is stated: on this receptor a full-receptor build went from 5.5 s / 122 k
triangles / −6.1 % to **9.2 s / 187 k triangles / −5.1 %**. 0.8 Å remains the
quick-look setting, and 0.4 Å is available when the area itself is the point.

The residual several-per-cent deficit does not vanish quickly because it is not
faceting: a single sphere converges to −0.27 % at 0.4 Å, while a protein keeps
losing area in the *crevices* between atoms, where the field has ridges that a
linear interpolant rounds off. `docs` and the code both say so rather than
implying the mesh area is the SASA.

The same table for the **SES** reports a *ratio*, not an error — the SES has no
analytic reference in this project (MSMS is not available offline), and it is
genuinely smaller than the SAS by construction:

| spacing Å | grid points | triangles | area Å² | SES/SAS | wall time |
|---|---|---|---|---|---|
| 1.20 | 105 092 | 43 836 | 6 989.8 | 75.4 % | 7.4 s |
| 1.00 | 173 600 | 65 876 | 7 329.0 | 79.0 % | 7.0 s |
| 0.80 | 315 248 | 105 108 | 7 473.2 | 80.6 % | 13.7 s |
| 0.65 (default) | 565 064 | 164 640 | 7 765.6 | 83.7 % | 13.4 s |
| 0.50 | 1 188 260 | 287 176 | 8 030.8 | 86.6 % | 21.8 s |
| 0.40 | 2 237 742 | 462 560 | 8 303.2 | 89.5 % | 30.5 s |

The ratio is still moving at 0.4 Å, so an SES area should be read as a *rendered*
quantity, not a measured one — which is exactly why `Surface.stats` reports the
analytic SAS alongside it and why §3 measures areas with `odock.sasa` instead.

### 1.4 Cost at 2 000 atoms, and how the window stays alive

Measured on the same receptor, at the settings a user actually picks:

| build | grid spacing | grid points | triangles | area | wall time |
|---|---|---|---|---|---|
| whole receptor, SAS, default | 0.65 Å | 565 064 | 186 652 | 8 805 Å² | **9.2 s** |
| whole receptor, SAS, quick look | 0.80 Å | 315 248 | 121 592 | 8 705 Å² | **5.5 s** |
| whole receptor, SAS, automatic | 0.63 Å | 607 240 | 197 340 | 8 807 Å² | **9.5 s** |
| whole receptor, SES, default | 0.65 Å | 565 064 | 164 640 | 7 766 Å² | **13.4 s** |
| pocket (277 atoms within 11 Å), SAS, 0.45 Å | 0.45 Å | 248 292 | 104 016 | 2 351 Å² | **4.1 s** |
| pocket (277 atoms), SES, 0.45 Å | 0.45 Å | 248 292 | 84 416 | 1 928 Å² | **3.9 s** |
| pocket lining (126 atoms within 8 Å), SES, 0.4 Å | 0.40 Å | 162 792 | 56 772 | 1 020 Å² | **2.9 s** |

The numbers are printed by the workbench itself (View ▸ Surface ▸ Surface
statistics, and every build logs them): `Surface.stats` carries the atom count,
the grid shape and point count, the spacing, the triangle count, the area, the
enclosed volume and whether the mesh is closed, the analytic SAS reference for
an SES build, the lid triangles of a capped mesh, and the wall time.

Three decisions keep that affordable and the interface responsive:

* **The work is done on a worker thread.** `DockingWorkbench._rebuild_surface`
  reads everything that decides the mesh on the main thread, hands the worker
  plain data, and puts the finished surface on the scene from the callback
  (the same `_CallWorker` path the bond check and the filters already use).
  Orbiting, picking and the pose slider keep working during a build.
* **The grid is bounded.** `resolve_spacing` picks the coarsest spacing that
  keeps the grid inside `SurfaceSettings.max_points` (700 000 by default) and
  clamps it to `[0.32, 1.6] Å`, so no structure can allocate an unbounded
  array. "Pocket lining only" removes the rest of the protein.
* **Only cubes that straddle the level are touched.** The work scales with the
  *surface area*, not the volume: a 600 000-point grid triangulates in a
  fraction of a second because only a few tens of thousands of its cubes
  contain a crossing.

The neighbour search is a uniform cell list with a cell as wide as the search
radius, filtered by the exact distance before the pairs are handed on — the
27-cell box is about six times the volume of the sphere it must contain, so
returning unfiltered candidates would double the most expensive step. On a
regular grid the field is filled from a rectangular block per atom instead
(a slice, no index arithmetic); `tests/test_surface.py` asserts the two paths
agree to 1e-12, because having two implementations of one field is only safe
if something checks them against each other.

---

## 2. The properties, and why those properties

A colour map without a documented scale is decoration. The surface can be
painted with three things, each of which is defined in the code and tested
against a value from its source.

### Hydrophobicity — Kyte & Doolittle (1982)

`HYDROPATHY_KD` is the published scale (*J. Mol. Biol.* **157**, 105): Ile
+4.5 … Arg −4.5. It is normalised onto `[0, 1]` between those two published
extremes, so **Arg is exactly 0.0 and Ile is exactly 1.0** — a test asserts
both, which is what stops the ramp from being rescaled by accident. Every atom
of a residue takes its residue's value, which is why a residue's patch on the
surface is one flat colour.

An atom whose residue is not in the table (a ligand, a cofactor, an ion) gets
the fragment value of its AutoDock 4 atom type (`AD4_TYPE_HYDROPHOBICITY`: C/A
0.85, S 0.55, SA 0.35, P 0.35, halogens 0.45–0.55, N 0.20, NA 0.10, O 0.05),
which is the same apolar/polar split `crates/dock-core/src/atom.rs` implements.
With `scale="atom"` the **Vina hydrophobic rule** is applied on top of the
perceived bonds: `xs_is_hydrophobic` is true only for `CH`/`FH`/…, i.e. a
carbon *with no polar neighbour*, and the AD4 dictionary cannot express that (a
methyl carbon and a carboxyl carbon are both `C`). A test checks that the rule
moves the carboxyl carbon and leaves the methyl carbon alone.

### Electrostatic potential — sampled on the surface

The value on a surface vertex is the Coulomb potential of **every partial
charge** at that point:

```
φ(p) = 332.0637 · Σ_i q_i · exp(−κ r_ip) / ( ε(r_ip) · r_ip )     kcal/(mol·e)
```

The charges are the ones the project already computes — `odock.chem.charges`
(Gasteiger–Marsili by default, or the Kollman united-atom scheme) written into
the PDBQT, which the viewer reads back as `Atom.charge`; a test pins a single
`+1` charge against the closed form to twelve digits. `odock.gui.surface` also
exposes `charges_from_mol(mol)`, which calls
`odock.chem.charges.assign_charges` for a caller that holds an RDKit molecule,
but the *scene* deliberately uses the per-atom charges the file carries rather
than re-deriving them from `receptor_mol`. That is the same caution as §4's: a
PDBQT carries no bond orders and an atom order that survives editing is not
guaranteed, so a charge array computed for one graph and applied to another
would produce a plausible and completely wrong picture. The dielectric defaults
to **`ε = 4r`, the AutoDock 4 convention the scoring kernel uses**, which is
both consistent with the score and far better contrasted on a protein surface;
`"uniform"` (ε = 4) is available for a local-field picture. Debye–Hückel
screening (`κ = 0.104·√I` in 1/Å at 298 K) is optional.

This is a *picture* of the field, not the scoring function, and the legend says
so: it carries the unit (kcal/(mol·e)) and the exact range that was used.

**The dielectric is a control, not a hidden constant.** View ▸ Surface ▸
Dielectric switches between

* **`ε = 4r`** (default) — the AutoDock 4 distance-dependent convention, i.e.
  the screening the scoring kernel itself uses, and much the better contrasted
  picture on a protein surface;
* **`ε = 4`** — a constant dielectric, the unscreened local field.

Both are the same Coulomb sum with a different ε, and both are documented at
their definition. On the 3PTB S1 pocket the two differ by an order of magnitude
in absolute value (site-centre potential −24.5 vs −274.0 kcal/(mol·e)) while
ranking the same residues first, which is the honest way to read them: the
*ranking* is robust, the *number* is model-dependent.

### Which residues make the pocket electropositive

The potential at a point is a **sum over atoms**, so it splits exactly:
View ▸ Surface ▸ *Potential breakdown…* evaluates each residue's own
contribution — no approximation is involved in asking who is responsible.

`tools/potential_breakdown.py` on the 3PTB S1 pocket, `ε = 4r`, site centre at
(−1.96, 14.12, 16.31), 2 004 of 108 168 surface points sampled:

| residue | atoms | Σq | at the site | share | mean over the surface |
|---|---|---|---|---|---|
| A/ASP189 | 9 | −0.44 | **−1.61** | 6.6 % | −0.46 |
| A/SER190 | 8 | −0.19 | −1.15 | 4.7 % | −0.20 |
| A/GLY219 | 5 | −0.12 | −0.81 | 3.3 % | −0.12 |
| A/GLN192 | 12 | −0.20 | −0.76 | 3.1 % | −0.21 |
| A/TRP215 | 16 | −0.47 | −0.75 | 3.1 % | −0.45 |
| A/GLY216 | 5 | −0.12 | −0.72 | 2.9 % | −0.14 |
| A/TYR228 | 14 | −0.39 | −0.62 | 2.5 % | −0.42 |
| A/CYS191 | 7 | −0.15 | −0.55 | 2.3 % | −0.14 |

ASP189 — the residue that gives trypsin's S1 pocket its specificity for basic
ligands — is the largest single contributor at the site, and the rest of the
list is the S1 lining. The *mean over the whole surface* column is much less
discriminating (every residue looks similar), which is why the site value is the
headline number: a mean over a pocket dilutes exactly the residue you are
looking for.

**What this is not, and what it costs in interpretation.** It is a point-charge
Coulomb sum with a stated dielectric: no Poisson–Boltzmann solution, no
dielectric boundary, no ionic atmosphere beyond the optional Debye term, and no
polarisation. The decomposition is exact *for that model*. And the model is only
as good as the charge column it reads — the dialog prints the diagnostic:

> charge column: 1866 of 1994 atoms non-zero, sum **−48.5 e**

A 1 994-atom protein is not −48.5 electrons. The bundled demo receptor carries
the project's default hydrogen-suppressed Gasteiger column, which is
backbone-dominated and non-conserving, so **the map is relative, not
absolute** — it is good for "which part of this pocket is negative compared with
which other part", and a caller who needs a defensible ESP must supply a
properly parameterised charge set. The dialog says this in the same words.

### Element

CPK colours, for checking the surface against every other viewer.

### The colour range, and why it is reported back

A diverging potential has to keep 0 white or it stops meaning anything, so the
electrostatic range defaults to **symmetric**, with the magnitude set to the
**85th percentile of |φ|** rather than the maximum. One buried charge owns the
extreme; letting it set the scale turns a whole picture into a single flat
colour. A trypsin S1 pocket is mostly negative — the *variation within the
negative range* is the information — and the percentile range is what makes it
visible:

![The same pocket coloured by electrostatic potential; symmetric range, ±7.7 kcal/(mol·e)](../out/surface/03_pocket_electrostatic.png)

`Surface.value_range` is the range that was actually used, and the legend draws
that, not the range the caller asked for: a bar that shows something other than
the picture is worse than no bar. The user can fix the range from
View ▸ Surface ▸ Colour range (and `0` returns it to automatic).

---

## 3. SASA, burial and the ligand's contact area

`python/odock/sasa.py` implements Shrake & Rupley, *J. Mol. Biol.* **79**
(1973) 351, with the Bondi radii and a **golden-spiral** sphere sampling. The
spiral is used instead of a latitude/longitude grid because a lat/long grid
clusters points at the poles, and a *cap* — which is exactly what one atom cuts
out of another — then has an area error that grows with the polar angle. 92
points per atom is the count of the original paper.

### What is checked, and how

* **An isolated sphere** has every point accessible, so its area is *exactly*
  `4π(r+probe)²` — asserted to machine precision.
* **Two contacting spheres** have a closed form: the part of sphere *a* inside
  sphere *b* is a cap of height `h = R_a − (d² + R_a² − R_b²)/(2d)` and area
  `2πR_a·h`. The tests compare against that for four distances at 512 points
  (1 %) and at the default 92 points (3 %), which is what pins the point count
  to a measured error rather than to a feeling.
* **Monotonicity**: an occluding atom can only remove accessible points, so a
  neighbour walking in from 12 Å to 2.6 Å must never increase the area. Burial
  is then always in `[0, 1]` and needs no clamping.
* **Rigid-motion invariance**: the sphere lattice is fixed in space while the
  atoms move, so a rotation re-samples every cap boundary. The test therefore
  asserts the change is bounded by the area *one point* carries — an honest
  bound rather than "nothing happened".

### Per-residue burial, against a stated reference

```
report = odock.sasa.burial(receptor, ligand, reference="unbound")
```

| reference | meaning |
|---|---|
| `"free"` | every residue measured **on its own** — the classical folding burial |
| `"unbound"` | the same structure **without the ligand** — how much surface the ligand itself hides |

`BurialReport` carries the per-residue areas, `most_buried(n)`, a fixed-width
`table()` for the log and the dialog, and `as_dict()` for a script. The bundled
demo makes the point: benzamidine buries **101.9 Å²** of the trypsin surface
(1.1 % of 9 172.9 Å²), and the most buried residues it reports are

```
residue      area Å²  buried Å²  buried %
A/GLN192         77.9      19.7     20.2
A/TRP215         34.1      15.0     30.5
A/SER190          0.0      12.0    100.0
A/SER195          9.2      11.4     55.2
A/GLY216         27.1       9.2     25.3
A/CYS191          0.0       7.7    100.0
A/GLY219         40.5       7.0     14.7
A/SER214          6.0       6.0     50.0
```

— Gln192, Trp215, Ser190, Cys191, Gly216, Gly219: the S1 pocket of trypsin.
That list is not an input to the code; it is what the measurement returns, and
it is the kind of sanity check a new geometry pipeline has to pass.

### The ligand's buried contact area

```
record = odock.sasa.ligand_buried_contact_area(ligand, receptor)
```

Only the ligand's own sphere points are ever evaluated, so it is cheap enough
to run for every pose. For the top 3PTB pose: **282.8 Å² free, 15.8 Å² in the
complex, 267.1 Å² buried — 94.4 % of the ligand's surface is in contact with
the pocket.** That is the number that turns "it fits" into a measurement, and
`buried_contact_per_pose` returns one record per pose (Analysis ▸ Burial per
pose shows the table).

`interface_area` does the same for the receptor side, with an optional
`radius` shell. The shell is not an approximation: the buried surface of a
contact is local by construction, so a sphere that contains the contact gives
the same answer, and a test asserts the shell and the whole protein agree
exactly.

### 3.1 Is the pocket-lining surface open, and how big is it?

"Pocket lining only" builds from a subset of atoms, and the question is whether
that leaves an open cut sheet with a rim. It does not, and the reason is
structural rather than incidental: the zero level of the field over a subset of
atoms is the boundary of a **union of balls**, which is a closed set with a
closed boundary. `tools/pocket_closure.py` measures it rather than asserting it:

| build | atoms | triangles | area Å² | boundary loops | closed | volume Å³ |
|---|---|---|---|---|---|---|
| whole receptor SAS | 1 994 | 186 652 | 8 804.6 | **0** | yes | 15 420.3 |
| pocket lining SAS, r = 11 Å | 277 | 48 644 | 2 308.1 | **0** | yes | 3 125.7 |
| pocket lining SAS, r = 8 Å | 126 | 28 628 | 1 355.6 | **0** | yes | 1 790.0 |
| pocket lining SES, r = 8 Å | 126 | 21 156 | 1 009.1 | **0** | yes | 1 185.3 |

So for the same region the lining is 26 % of the whole receptor's area at
r = 11 Å and 15 % at r = 8 Å, and the closed form has a **volume**: 3 125.7 Å³
for the r = 11 Å lining against 15 420.3 Å³ for the whole receptor (all at the
default spacing). The lining of a pocket is a thin curved slab, so that volume
is the slab's — the *cavity* the ligand sits in is a different quantity and is
not claimed here.

What *does* need closing is a mesh that was **cut** — by the clip plane, or by
keeping only the triangles near a site — because a hole has no inside and
therefore no volume. `SurfaceSettings.close_rim` adds flat lid triangles over
every boundary loop (found by welding coincident vertices and counting edges
used by exactly one triangle), drawn in a neutral grey, counted separately in
`Surface.stats` as `cap_triangles`/`cap_area` and excluded from the reported
surface area, so a capped figure still reports the surface it is a figure *of*.
`Surface.volume` returns `None` for an open mesh rather than a number for a
shape that has no inside.

Volume is measured by the divergence theorem, which needs one consistent
outside, so the builder re-winds every triangle against the field-gradient
normals (`orient_mesh`). That is also what makes the exported OBJ a well-formed
surface for another program. The closed forms check it: a hand-built cube gives
exactly 8.0 Å³, a single atom's SAS sphere gives 123.2 Å³ against an exact
124.8 Å³, and two contacting spheres give 206.4 Å³ against an analytic union of
208.3 Å³.

---

## 4. Interop: the result into somebody else's tool

`odock.interop` writes three things from the live scene:

* **`<stem>.pml`** — PyMOL: loads the two structures, hides everything,
  applies the current styles with real PyMOL commands, draws **every
  interaction as a `distance` object** coloured per interaction type, turns
  each measurement into pseudoatoms and a `distance`, marks the search box,
  shows the surface, colours it by the property, and restores the camera with
  an 18-number `set_view`.
* **`<stem>.cxc`** — the same in ChimeraX's vocabulary (`cartoon`, `style
  stick`, `distance #1/A:189@OD1 #2/A:1@N1`, `surface #1`, `color
  byattribute bfactor palette bluered range …`).
* **`<stem>_pose.pdb`** — receptor and posed ligand, column-correct, serially
  renumbered, with the per-atom property in the **B-factor column**. That
  column is the whole trick: it is what makes `spectrum b` in PyMOL and
  `color byattribute bfactor` in ChimeraX reproduce the workbench's surface
  colouring from the coordinates alone.

`export_bundle(directory, state)` writes all of them plus a
`<stem>_surface.obj` (Wavefront OBJ, `v x y z r g b`, so the geometry carries
its colours) and returns a manifest of the paths. File ▸ Export ▸ *PyMOL
script…* / *ChimeraX script…* / *PDB with pose…* / *Surface OBJ…* pick a
directory and write the bundle.

Two places where the honest thing is a comment rather than a command:

* **ChimeraX has no `set_view`**, so the camera is not reproduced; the script
  says so and emits `zoom #2` onto the ligand instead.
* **PyMOL has no tube representation**, so the `tube` style falls back to a
  cartoon — with a comment in the script that says so.

The module has no Qt, ModernGL or RDKit dependency at import time (a test runs
`python -c "import odock.interop"` in a fresh interpreter and asserts neither
PyQt6 nor moderngl is in `sys.modules`), and the tables it shares with the GUI
are duplicated on purpose with a test that asserts the two copies agree — the
style name sets and the interaction colours.

A caution the module is built around: **a PDBQT carries no bond orders and an
atom order that survives editing is not guaranteed.** Everything that maps a
property back onto atoms goes through `Surface.selected` and
`Surface.source_atom()`; getting that mapping wrong paints the wrong atoms with
the right numbers, which is the one error a picture cannot show you.

---

## 5. Interaction dashes: the "many white lines" report

A user reported "many white lines after docking" they could not identify, and
asked whether bonds were being drawn to the wrong atoms. The answer, measured by
ablation (re-render with one draw pass replaced by a no-op and count the
changed pixels) is neither: the lines are the **contact annotations**, the
geometry is correct, and the defect was **legibility**. On a 900×700 frame of the
docked 3PTB pose:

| pass | pixels it owns |
|---|---|
| the whole interaction pass | 23 966 px |
| the hydrophobic contacts alone | 17 473 px |
| the receptor/ligand mesh (control) | 234 925 px |

The cause was the hydrophobic swatch: `(0.62, 0.64, 0.68)` — luminance 0.64
against a canvas of 0.09, and a chroma of 0.06, i.e. **grey**. The mesh shader
lights a fragment as `colour·(0.34 + 0.72·d) + d²⁴·0.28 + rim`, so an achromatic
pale swatch reaches the top of the range on its lit side: which is why the
contacts read as stray white strokes. Two changes, both in
`odock/gui/viewport.py`:

* the swatch is now a **muted olive** `(0.56, 0.58, 0.44, 0.90)` — luminance
  0.565, chroma 0.141, and legible against *both* viewport backgrounds (the dark
  canvas 0.086/0.094/0.125 and the light one 0.784/0.804/0.831);
* every dash now carries a **dark under-stroke**: a 1.9× fatter cylinder of
  `(0.05, 0.06, 0.09)` pushed away from the camera along its own view direction
  by exactly the radius difference (plus 4 mÅ), so the coloured core always wins
  the depth test and only the rim survives. That generalises the fix past this
  one colour: no contact can read as a bare white line on either background, and
  a pale contact on the light canvas gets the dark edge it needs to be visible.

Measured on the same frame, over exactly the pixels the swatch governs
(8 540 px), old swatch → new:

| quantity | before | after |
|---|---|---|
| pixels reading light (luminance ≥ 0.55) | 1 320 | **122** (10.8× fewer) |
| median / 99th-percentile luminance | 0.438 / 0.638 | 0.411 / 0.555 |
| contrast to the light canvas | 0.16 | **0.25** |
| swatch chroma | 0.059 | **0.141** |

`tools/measure_interactions.py` reproduces the table and writes the before/after
frames; `tests/test_viewport_interactions.py` pins the contract (both
backgrounds, mutual distinguishability of the six kinds, the geometry of the
under-stroke, and a rendered before/after pixel count), and
`tests/test_interop.py` asserts the exported scripts carry the *same* colours.

---

## 6. Using it

**View ▸ Surface**

| action | what it does |
|---|---|
| Show surface | builds one if there is none, otherwise shows/hides it |
| Rebuild surface | rebuilds from the current settings (worker thread) |
| Surface type ▸ SAS / SES | probe-centre surface, or the molecular surface |
| Colour by ▸ hydrophobicity / potential / element | rebuilds with that property |
| Dielectric ▸ Distance (ε=4r) / Uniform (ε=4) | which screening the potential uses; rebuilds |
| Potential breakdown… | each residue's own contribution to the potential at the site, plus the charge-column diagnostic |
| Colour range… | `low, high`, or `0` for automatic |
| Opacity… | 0.05–1.0; below 1.0 the surface blends without writing depth, so a ligand inside the pocket shows through the wall |
| Pocket lining only | builds from the atoms within 9 Å of the ligand (or the box) |
| Highlight the pocket | tints the lining residues on the surface *and* draws them as ball-and-stick through the interaction-focus machinery |
| Cut in front of site | a world-fixed clipping plane through the camera direction, placed 0.5 Å in front of the ligand's frontmost atom |
| Cut through the site | the same plane through the ligand's centroid |
| Remove the cut | clears it |
| Surface statistics | the measured cost of the last build, whether the mesh is closed, and the volume it encloses |
| Colour bar | shows/hides the legend, including the charge caveat when there is one |

The clipping plane is **fixed in the model**, unlike the camera-space front
clip: two poses of the same protein are cut identically, which is what makes two
pictures comparable. It is applied in the fragment shader to the surface, the
mesh passes and the atom spheres, so an opened pocket does not leave its atoms
floating inside the opening.

**Analysis** gains *SASA & burial…* (the per-residue table, the ligand's own
buried area and the receptor interface area), *Ligand burial* (one log line for
the pose on screen) and *Burial per pose* (the table for the whole pose set,
with a *Save figure…* button that writes the SVG bar chart of buried area per
pose). The figure is a real chart, not a screenshot: the bars are the buried
area in Å², the round-number gridlines are the axis, each pose's affinity is
printed above its bar, and the document is checked as XML by the test suite.

![Buried contact area per pose: a table is what a modeller checks, a figure is what they paste into a report](../out/surface/08_burial_per_pose.png)

For the bundled 3PTB run the chart reads: poses 1–3 bury 267–268 Å² (94–95 % of
the ligand's surface), and the three weaker poses drop to 234–250 Å² (82–87 %).
The buried area and the score agree on the *shape* of this pose set, which is
the kind of statement a single affinity cannot make.

**Screenshots** carry the legend and the scale bar: `ViewportWidget.snapshot`
renders the frame through the GL context and then paints the same overlay into
the image at the image's own scale, so a 2 400 px export is a figure, not a
bare render. The scale bar is derived from the projection the frame was drawn
with (`2·d·tan(fov/2)/height`), so a length in the picture is a measurement.

---

## 6. The figures in this document

`tools/render_surface_previews.py` regenerates every image in `out/surface/`
against the bundled 3PTB demo, offscreen, and prints the statistics for each
build; `tools/check_interop_bridge.py` walks the live scene through
`DockingWorkbench._interop_state()` into `export_bundle` and prints the
manifest, which is how the GUI→interop bridge (including the per-atom property
mapping) is checked by hand:

```
python -X utf8 tools/render_surface_previews.py
```

(Use the interpreter of the virtual environment you built the extension into —
`python` there, `.venv/Scripts/python.exe` on Windows, `.venv/bin/python`
elsewhere.)

| file | what it shows |
|---|---|
| `01_pocket_ses_hydrophobicity.png` | the S1 pocket as an SES, coloured by hydropathy |
| `02_pocket_sas_hydrophobicity.png` | the same region as an SAS, for comparison |
| `03_pocket_electrostatic.png` | the pocket coloured by the electrostatic potential |
| `04_pocket_lining_highlight.png` | a pocket-lining shell with the lining residues highlighted, ligand visible |
| `05_pocket_cut_open.png` | the cut plane: the pocket opened towards the camera |
| `06_whole_receptor_surface.png` | the whole receptor, SAS, hydropathy |
| `07_whole_receptor_translucent.png` | the same surface at 55 % opacity over the cartoon || `08_burial_per_pose.svg/.png` | the buried contact area of every pose, as a chart |

The translucent figure shows the cartoon *through* the surface, and the speckled
patches in it are where the cartoon tube geometrically pokes out of the
solvent-accessible surface — the ribbon is a smooth curve through the C-alpha
positions, not the atoms themselves, so it leaves the surface wherever the
backbone bulges. That is the geometry, not a depth-buffer artefact; an opaque
surface (figure 06) hides it.

![The whole 3PTB receptor as a solvent-accessible surface coloured by residue hydropathy](../out/surface/06_whole_receptor_surface.png)

---

## 7. What is *not* claimed

* The **SES reentrant patches are grid-resolved**. The erosion samples the
  probe sphere on 42 directions and interpolates trilinearly, so a concave
  patch is right to about the grid spacing; the contact patches are exact.
* A **pocket-lining surface is an open shell**. Colours, areas and the property
  are right; the mesh has a rim, and from an angle that looks through the
  opening the background is visible.
* The **electrostatic map is a Coulomb picture**, not a Poisson–Boltzmann
  solution. It is the potential of the point charges the project already
  computes, screened by the stated dielectric; the unit and the range are on
  the legend so the number can be read for what it is.
* The **SAS mesh area is a few per cent below the analytic Shrake–Rupley
  area** at the default spacing, because a triangulated isosurface inscribes
  the true surface. The gap is measured, printed in the statistics and falls as
  the grid is refined; it is not asserted away. §1.3 is the table.
* The **SES area is a rendered quantity**, not a measured one: its ratio to the
  SAS is still moving at 0.4 Å (75 % → 90 % across the table) and there is no
  analytic SES reference in this project. Use `odock.sasa` when an area is the
  point.
* **An electrostatic map is only as good as its charge column, and the column
  can lie in two detectable ways** — it may not carry a group's formal charge
  (Gasteiger gives benzamidine's formally cationic amidine nitrogens **−0.31 e
  each**), and it may not conserve charge at all (the bundled demo receptor sums
  to **−48.5 e** over 1 994 atoms). `odock.gui.surface.charge_caveat` runs
  `odock.protonation.detect_groups`/`salt_bridge_warnings` — imported, not
  re-implemented — and the workbench prints the caveat in the log and under the
  legend title. A map that may be wrong-signed is labelled as such rather than
  left looking like an answer.
* The **pocket-lining mode is a closed shell around the lining atoms**, not an
  open cut sheet (measured: zero boundary loops), so its volume is the thin
  slab's and *not* the cavity's. The cavity volume is a different quantity and
  is not claimed here.
* The **surface is built from the atoms it is handed**, so a structure whose
  atom order changed between preparation and display would be painted wrongly.
  Every path that maps vertices back to atoms goes through
  `Surface.selected`, and the display uses the same atom list the renderer draws.
