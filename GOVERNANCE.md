# Repository Governance

## Open Contribution Model

TheRiskTheory is open to contributions.

Anyone may:

- fork the repository;
- create a development branch;
- implement new functionality;
- improve documentation;
- add or improve tests;
- report issues;
- open pull requests.

Contribution access does not imply merge authority.

## Protected `master` Branch

The `master` branch is the authoritative and stable branch of TheRiskTheory.

Changes must not be pushed directly to `master`.

Every change to `master` must be introduced through a pull request.

## Final Approval Authority

All pull requests targeting `master` require final approval from:

**Luis Pitta — GitHub: `@luispitta`**

Other contributors are encouraged to review code, mathematical derivations, tests, documentation, and numerical results. Their reviews are valuable, but they do not replace the maintainer's final approval.

Only the maintainer's approval authorizes a pull request to be merged into `master`.

## Mathematical Review

Mathematical validation and repository merge approval are separate concepts.

A contribution may receive peer mathematical review from another contributor before maintainer approval.

Recommended process:

```text
Contributor implementation
        ↓
Automated tests
        ↓
Peer mathematical review
        ↓
Maintainer review (@luispitta)
        ↓
Final approval
        ↓
Merge to master
```

A contributor must not mark their own work as `mathematically_validated`.

## Required GitHub Branch Protection

The following rules should be enabled for `master`:

1. Require a pull request before merging.
2. Require at least 1 approving review.
3. Require review from Code Owners.
4. Dismiss stale pull request approvals when new commits are pushed.
5. Require approval of the most recent reviewable push.
6. Require status checks to pass before merging.
7. Require the repository CI workflow to pass.
8. Block force pushes.
9. Block branch deletion.
10. Restrict direct pushes to `master`.
11. Apply the rules to administrators when supported.

The repository's `.github/CODEOWNERS` file assigns all repository paths to `@luispitta`.

With **Require review from Code Owners** enabled, a pull request changing any repository file requires review from `@luispitta`.

## Merge Policy

Pull requests should normally be merged only when:

- CI passes;
- required tests pass;
- documentation is complete;
- mathematical assumptions are documented;
- references are provided where required;
- relevant peer review has been completed;
- `@luispitta` has approved the pull request.

The maintainer may request changes or decline a pull request when it does not meet project standards.

## License

Contributions accepted into the repository are provided under the Apache License 2.0 unless explicitly stated otherwise.
