# Contributing

TheRiskTheory is a mathematical software project. Passing software tests is necessary but not sufficient.

## Student workflow

Each contributor owns a development area but all changes are submitted through pull requests.

Recommended branch format:

```text
student/<area>/<function-name>
```

Example:

```text
student/frequency/negative-binomial-pmf
```

## Definition of Done

A mathematical implementation must include:

- [ ] clear mathematical definition;
- [ ] assumptions and parameter domain;
- [ ] NumPy/SciPy vectorized implementation;
- [ ] type hints;
- [ ] NumPy-style docstring;
- [ ] bibliographic reference;
- [ ] analytical unit test;
- [ ] vectorized-input test;
- [ ] edge/invalid-input test;
- [ ] example;
- [ ] validation state;
- [ ] review by another contributor.

## AI use

AI tools are allowed for:

- implementation assistance;
- refactoring;
- test generation;
- documentation drafting;
- code review assistance.

AI output is not a mathematical source.

The contributor is responsible for checking every formula, derivation, assumption, and numerical result.

## Coding requirements

- Python 3.11+
- NumPy for vectorized numerical computation
- SciPy when appropriate
- No DataFrame dependency in the mathematical core
- No unnecessary row-by-row loops
- Public functions require type hints and docstrings
- New behavior requires tests

## Pull requests

Keep one mathematical concept or tightly related set of concepts per pull request when practical.

A pull request should explain:

1. what mathematical object is implemented;
2. which source was used;
3. what assumptions are made;
4. how it was tested;
5. current mathematical validation status.

## Review rule

A contributor must not grant final mathematical validation to their own work.

## Merge authority

Anyone may contribute to TheRiskTheory by opening a pull request.

The `master` branch is protected. Contributors must not push directly to `master`.

All pull requests targeting `master` require final approval from the repository maintainer:

**@luispitta**

Peer reviews are encouraged and may be required for mathematical validation, but they do not replace maintainer approval.

A pull request may be merged only after:

- required CI checks pass;
- required mathematical review is complete;
- `@luispitta` has approved the pull request.

See [`GOVERNANCE.md`](GOVERNANCE.md) for the complete policy.
