# Release process

This guide is for maintainers preparing a version of SolarCast. Building a package,
running checks or preparing files does not authorize publication. GitHub repository
updates, release tags and PyPI uploads are separate decisions requiring owner approval.

See [Python package validation and publishing](python-packages.md) for TestPyPI,
isolated registry installation, the proposed OIDC workflow and draft release notes.

## Prepare and verify

1. Review the changes, supported Python versions, dependency bounds, version,
   changelog, public API and user examples. Keep unreleased versions labelled as such.
2. Run the full tests, Ruff, formatting and mypy checks described in
   [contributing](../../CONTRIBUTING.md). Normal scientific tests stay offline;
   native TSFM inference requires explicit opt-in and is not routine release CI.
3. Run `python scripts/check_docs.py` and the canonical quickstart. Check model
   identifiers, installation instructions and all benchmark claims against their
   authenticated sources. Retain development/confirmation and availability caveats.
4. Review intended source files for credentials, private paths, large data, weights,
   caches and local working records. Ignore rules do not remove already tracked files;
   inspect both tracked files and new additions. Keep the preview of intended paths
   locally rather than including a one-time publication inventory in the package.
5. Build fresh wheel and sdist with `python -m build`. Inspect both using
   `python scripts/check_distributions.py`; validate metadata with `python -m twine
   check dist/*` after installing Twine in an appropriate development environment.
   Twine checks locally; this command does not upload anything.
6. Run `python scripts/check_wheel.py` to install the local wheel into a fresh
   temporary base-only environment and run the quickstart outside the repository.
   Package dependency installation uses an index; the benchmark itself blocks sockets.

## Review publication separately

Confirm the approved owner, visibility, repository URL and intentionally public
commit identity. Review the exact staged paths and full diff before committing.
Use real project URLs only once they exist, and verify hosted CI and rendered
documentation after an approved push. Configure private vulnerability reporting
as described in [security](../../SECURITY.md).

Before any PyPI release, confirm distribution-name registration/ownership, release
metadata, README asset URLs, package notices and clean-install behavior. Configure
publishing credentials or trusted publishing only as a separately approved step.
GitHub publication is not permission to upload to PyPI.

Do not rewrite frozen scientific data, configuration or result provenance to make
a release easier. Exact historical replay may require bundles not distributed with
the repository; keep [reproducibility limits](../reproducibility.md) explicit.

Issue templates can follow recurring contributor needs. A recognized Code of
Conduct needs an actual enforcement contact before being presented as an active
policy; do not invent private contact details or response commitments.
