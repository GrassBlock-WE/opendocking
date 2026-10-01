# Protonation: the assumption under every electrostatic term

A charge model is not a chemical model.  Gasteiger, MMFF94 and a PDBQT written from a
neutral SMILES all answer *"how are the electrons distributed in this molecule"*, and
none of them asks whether the molecule is the one that binds.  This document states
the assumption, measures what it costs on the bundled data, and records where each
electrostatic consumer in the project stands.

## 1. The worked example, measured

Benzamidine in trypsin's S1 pocket.  The bound species is the **amidinium**, and its
salt bridge with Asp189 is the interaction that defines the pocket.  Three charge
assignments at the **crystallographic pose** (`demo/3ptb/ligand.pdbqt`, whose two
amidine nitrogens are **2.87 Å** from the nearest Asp189 oxygen):

| assignment | charge on the amidine N | Σ q·φ | ESP term | score |
|---|---|---|---|---|
| **(a) as prepared** — the file's own charges, neutral | −0.62 e total | **+22.09** | 0.000 (clamped) | 0.471 |
| **(b) Gasteiger on the amidinium** (`charged_copy`) | −0.11 e total | **+20.98** | 0.000 (clamped) | 0.471 |
| **(c) (b) with the amidinium's formal +1 restored** | **+1.00 e** | **−28.75** | 1.000 | **0.971** |
| (c′) the same correction applied to (a)'s charges | **+1.00 e** | **−13.95** | 1.000 | **0.971** |

Row (c′) is the same correction from a different starting distribution — the file's
neutral charges rather than Gasteiger-on-the-amidinium — and it is there because the
*magnitude* depends on where the correction starts while the *conclusion* does not:
the sign flips, the term is used, and the score goes from 0.471 to 0.971 either way.
That is the honest way to report a 15 kcal/mol spread.

Three things follow, and the second is the one that generalises furthest:

1. **The sign is wrong as prepared.**  +22.09 kcal/mol for the complex whose salt
   bridge is its defining feature.  A term that says that is not making a small error.
2. **Fixing the protonation state is not enough.**  Protonating the molecule moves the
   energy by only **1.11 kcal/mol** (+22.09 → +20.98) and changes nothing: Gasteiger is
   a sigma-electronegativity model, so a *formally cationic* nitrogen still comes out
   **negative (−0.11 e)** — the +1 is spread over the whole ion.  Only restoring the
   group's formal charge flips the sign.  **No protonation-state change without a
   charge model that can hold a formal charge.**
3. **It changes a ranking, not just a number.**  Ranking `demo/library.smi` against the
   3PTB pocket, the five ring-amidines take ranks **1, 2, 3, 10, 11** with the ESP term
   clamped for every molecule (so the "combined" score *is* the shape term), and
   **1, 2, 3, 5, 6** with the formal-charge correction — chloro_benzamidine moves from
   11 to 6 and benzamidine_methyl from 10 to 5.

The cost of the assumption is therefore **50.8 kcal/mol of interaction energy and
0.500 of score on one ligand**, and a materially different order for the series.

## 2. The check, at the source

`odock.protonation` perceives the formally charged groups and reports the state:

| family | perceived pattern | charge expected at pH 7.4 |
|---|---|---|
| amidinium / guanidinium | `[CX3] (=[NX2,NX3+])[NX3]` | +1 |
| carboxylate | `[CX3] (=O)[OX2H1,OX1-]` | −1 |
| phosphate | `[PX4] (=O)([OX2,OX1-])[OX2,OX1-]` | −1 |
| sulfonate / sulfate | `[SX4] (=O)(=O)[OX2,OX1-]` | −1 |
| primary/secondary/tertiary amine | `[NX3;H0..H2…]` or `[NX4+;H1..H3]` | +1 |
| imidazole | `c1cnc[nH1]1` | **0** |
| thiol | `[SX1-,SX2H1]` | **0** |

(The space before each branch is a documentation artefact: written without it, a SMARTS
is indistinguishable from a markdown link and the documentation link checker reads it
as one.  `odock/protonation.py` has the patterns without the space, which is where a
copy-paste should come from.)

**The zeros are deliberate and they are chemistry.**  Imidazole (pKa ≈ 6.0) and a thiol
(pKa ≈ 10) are neutral in the majority at physiological pH, so flagging a neutral
imidazole as "wrong" would be a false alarm; aromatic amines (aniline, pKa 4.6) are
excluded from the amine pattern for the same reason.  A missed group produces no
warning, which is the safe failure for a check whose output is a warning.

The patterns have to see **both** states, and two of them did not at first: `[NX2]`
misses an amidinium's imine nitrogen (it carries two hydrogens, so its connectivity is
3) and `[NX3]` misses an ammonium (connectivity 4).  A detector that cannot see the
*correct* state cannot compare a molecule against it — both were found by testing the
charged form, and both are pinned in `tests/test_protonation.py`.

`protonation_report` attaches the contradiction to the **preparation report** as a
warning, with the partial charges that were actually assigned:

```text
benzamidine: net charge +0 (at pH 7.4 the families would be +1) — amidinium/guanidinium neutral
WARNING: amidinium/guanidinium is NEUTRAL although pH 7.4 makes it +1: every
electrostatic term computed from this molecule is computed for the wrong species, and
if the pocket is lined by an oppositely charged residue the sign of the interaction is
wrong rather than merely inaccurate.
```

`salt_bridge_warnings` adds the geometric half: a neutral group that *should* be
charged, within 4.0 Å of an oppositely charged residue (ASP/GLU for a cation,
LYS/ARG/HIS for an anion), or the reverse.  That is the checkable condition, and it is
why `PocketAtom` carries residue names.  **The API enforces the atom order it needs**:
group indices index the molecule, so coordinates in another order are rejected rather
than silently measured — a mismatched order turned the real 2.87 Å salt bridge into
4.96 Å and the warning never fired, which is exactly the kind of silent wrong answer
this module exists to prevent.

## 3. Where the project's electrostatic terms stand

| consumer | where the charges come from | protonation-aware? |
|---|---|---|
| `odock.pocket_score` | the PDBQT file, or Gasteiger | **yes** — flags `esp_term_used`, warns, and offers the correction |
| `odock.consensus` (AD4 component) | the PDBQT the docking wrote | **now reported** — `ConsensusResult.charge` says what the charges can support; the AD4 number is unchanged |
| `odock.analysis` (interaction profiler) | **structure and hydrogen count**, not a charge model | **yes, and it is the pattern to copy** — see the audit below |
| `odock.lbvs` (shape/ESP overlay) | Gasteiger via `odock.overlay` | **no** — same model, ligand-versus-ligand |
| `odock.gui.surface` ESP colouring | `charges_from_mol(mol, model="gasteiger")` | **no** — a neutral amidine is coloured by its neutral charges |

**The intended entry point is `protonation.can_represent_formal_charge(mol, charges)`.**
It answers *"can this charge set hold the formal charge this chemistry implies?"* and
returns a `FormalChargeCapability` with the per-group sums and a quotable statement, so
the next consumer does not have to rediscover the Gasteiger limitation.  It is used by
`pocket_score.rank_library` (which notes how many molecules fail it) and by
`consensus.charge_context`.

**The `consensus` AD4 component** now carries a `ChargeContext`: whether the ligand's
chemistry was supplied at all, whether a formal-charge correction was possible, whether
one was applied, and notes naming the state.  `needs_attention` is True when the ligand
is unknown — a report that was not given the ligand cannot claim the check was made —
or when the charges cannot represent the formal charges and nothing was corrected.
**The AD4 number itself is not touched**: the kernel's score is what it is, and the fix
is about not letting the *report* imply more than the charges support.

**The `analysis` audit: it is already right, and here is why, so nobody changes it
back.**  It does *not* read formal charges from a charge model — it perceives the
charged centres from the **structure and the hydrogen count**:

* a carboxylate is a carbon with **two oxygens that carry no hydrogen**;
* a phosphate or sulfate has at least three oxygens, one of them free;
* a deprotonated oxygen has one heavy neighbour and no hydrogen;
* an ammonium is a nitrogen with **four connections**; a guanidinium is a carbon with
  three nitrogens, at least one hydrogen-bearing; a protonated aromatic nitrogen is a
  ring nitrogen with three connections including its hydrogen;
* explicit formal charges **always win** over those structural guesses, and a partial
  charge of |q| ≥ 0.75 is the last fallback.

That is better than Gasteiger and better than a pH assumption for this question: it
reads the protonation evidence actually present in the file — the polar hydrogens, which
PDBQT keeps — instead of inferring a state from a model or from a pH.  The one caveat to
record is inside the structural path: for a structure with **no hydrogens at all** the
donor test falls back to "any heteroatom", so the charged-centre inference weakens
exactly when a file was written without polar hydrogens.  This is the pattern the
remaining consumers should move towards, not away from.

## 4. What this does not establish

* **It is not a pKa predictor and it does not titrate.**  The expected charges are the
  majority species at pH 7.4 read off standard pKa values; a real calculation needs the
  local environment, and a buried ionisable group can shift by several units.
* **The receptor is not titrated either.**  A residue is assumed anionic because it is
  ASP or GLU; a protonated aspartate, a neutral lysine or the two tautomers of
  histidine are all real and all invisible here.  HIS is genuinely ambiguous and is
  treated as cationic only for the salt-bridge check.
* **The formal-charge correction is minimal, not a charge model — 0.971 is a
  corrected sign, not a validated energy.**  `correct_formal_charges` shifts a group's
  partial charges by a constant so they sum to the formal charge; it leaves their
  distribution — and every other atom — untouched.  It makes a declared formal charge
  visible to a Coulomb sum, which is what turns +22.09 kcal/mol into a favourable
  number; it is not a re-parameterisation, it does not re-run a charge model, and the
  0.971 it produces must not be read as a better *energy* than 0.471 was.  What changed
  is the sign of one term, and a genuine fix is a charge model that assigns formal
  charges correctly in the first place (see the `analysis` audit in §3 for a
  structural approach that needs no partial charges at all).
* **One ligand, one pocket, one charge model.**  The 50.8 kcal/mol above is benzene
  amidine in 3PTB with Gasteiger and a distance dielectric.  The *direction* of the
  finding generalises; the magnitude is that case.
* **A favourable Coulomb term is still not binding.**  Fixing the sign makes the
  electrostatics say the right thing about one interaction; it does not add
  desolvation, entropy or a moving receptor.
