<!--
  A pull request that fills this in is reviewed in minutes; one that says "fixes
  the bug" is not. Keep it short — the checklist is the point.
-->

## What this changes

<!-- One paragraph. What was wrong or missing, and what the change does about it. -->

Fixes #

## How it was verified

<!--
  The command you ran and what it printed. "make test" plus the specific test, or
  the benchmark number, or the before/after of the bug. If a check could not run,
  say which and why — do not delete it from the list below.
-->

```text
$ make test
...
```

- [ ] `make test` is green (or: not run, because ...)
- [ ] `make rust-test` is green (or: not run, because ...)
- [ ] A regression test covers the fix, and it fails on the code before it
- [ ] `make test-slow` / `make bench` run if this touches scoring, search or the grid
      (or: not run, because ...)
- [ ] `make lint` is green (ruff plus clippy)

## Notes for the reviewer

<!--
  Anything that is deliberately not done, an approximation that was chosen over an
  exact answer, a number that moved — and by how much. Scientific code is judged on
  its stated limits as much as on its results.
-->

## Checklist

- [ ] The change is scoped to one concern (a second, unrelated fix is a second PR)
- [ ] New behaviour is documented (docstring, `docs/`, or `--help` text)
- [ ] No `continue-on-error`, no skipped test, no `xfail` without an issue link
- [ ] Licence and attribution headers are intact (GPL-3.0-or-later)
