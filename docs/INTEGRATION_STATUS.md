# Course contribution integration — 24 November 2026

This branch consolidates the students' distinct code, tests, documentation, and examples. It is an **integration snapshot**, not a claim of successful end-to-end validation. Unresolved issues are intentionally retained and tracked in GitHub.

## Source pull requests

| Student | Topic 1 | Topic 2 | Topic 3 | Topic 4 | Topic 5 |
| --- | --- | --- | --- | --- | --- |
| Bracamonte | #1 | #3 | #14 | #18 | #19 |
| García | #4 | #5 | #10 | #12 | [fork PR #1](https://github.com/adrianjgm339/therisktheory/pull/1) |
| Monticelli | #2 | #6 | #20 | #21 | #30 |
| Passanisi | #7 | #13 | #26 | #27 | #29 |
| Pérez | #8 | #9 | #17 | #24 | #25 |
| Valera | #15 | #16 | #22 | #23 | #28 |

The work from #1 and #30 was already merged into `master`. All other unique contributions were collected from their current PR heads, without merging stale copies of other people's modules. The fork-only García Topic 5 contribution is also included.

## Resolution strategy

- Preserve one owner-maintained version of each shared module; do not overwrite it with an older duplicate included in a different PR.
- Keep separate modules for distinct actuarial models, and expose them through lazy package exports to limit import-time coupling.
- Move Pérez's Topic 1 retention example from the accidentally nested `examples/examples/` into `examples/`.
- Rename Pérez's Topic 3 transformation tests to `tests/test_transformations.py` so that Passanisi's distinct `tests/test_approximations.py` is not overwritten.
- Use the canonical `ruin_theory` module from #30; do not duplicate an older `ruin_teory` copy found in #2.
- Existing common configuration and foundational sources in `master` take precedence over stale copies contained in old PR diffs.

## Unresolved issues

1. **#24 Bayesian credibility:** the student's 24 November post-integration update supplies `bayesian_credibility_premium` and `posterior_predictive_mean`. The corrected source and expanded documentation are incorporated; numerical and CI validation remain pending.
2. **#2 Solvency:** capital requirements are a simplified teaching convention, not a regulatory formula. Cross-module dependency and numerical cases require integration testing.
3. **#21 Partial credibility:** review duplicated premium functions and reconcile with #18 and #12; remove the conditional fallback when an integrated API is established.
4. **#25 and #28 Lundberg and reinsurance:** confirm valid moment-generating-function domains and optimized retention boundary cases with independent tests.
5. **#19 and García fork #1:** differentiate discrete observed insolvency from continuous-time ruin; reproduce the Monte Carlo benchmark after integration.
6. **All PRs:** re-run targeted and full `pytest`, style checks, actuarial benchmarks, documentation checks, and peer reviews on the combined branch. No pass result is asserted here.

Integration is administrative. Mathematical and CI acceptance remain separate tasks.

Prof. Luis Pitta

## Conflict resolution and source history — 26 November 2026

The integration snapshot from PR #31 is the resolved version for overlapping or stale files. Original PR heads are included as merge parents to preserve authorship and history. The later #24 Gamma–Poisson implementation and expanded documentation supersede the incorrect Bayesian module; its prior misplaced retention code is preserved in a compatibility module. Existing foundational files and lazy exports are retained instead of accepting stale modifications or deletions. Original branches are unchanged. This is administrative integration, not mathematical or CI approval.
