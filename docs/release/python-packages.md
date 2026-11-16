# Python package validation and publishing

SolarCast is distributed as **solarcast-belgium** and imported as **solarcast**.
Version **0.1.0** is published on [PyPI](https://pypi.org/project/solarcast-belgium/)
and tagged **v0.1.0** on GitHub. Install with `python -m pip install solarcast-belgium`.
It remains active-development research software. Future releases require their own
review, registry validation and explicit publication approval.

## Validate a candidate

Use the [maintainer checks](README.md), build new distributions and run:

```shell
python -m build
python scripts/check_distributions.py
python -m twine check dist/*
python scripts/check_wheel.py
```

Install Twine in the validation environment separately. Check the built METADATA,
including distribution/import distinction, version, Python bounds, SPDX licence,
extras and real project URLs. The wheel must exclude local working records, raw
PV/NWP data, weights and trained models. Source distributions intentionally include
small documented synthetic/provider fixtures; see [third-party notices](../../THIRD_PARTY.md).
The wheel includes only the small attributed geography resource, code and notices.

The single README supplies both GitHub and package-index descriptions. Its links
use absolute public GitHub URLs, and the benchmark image uses an absolute raw-file
URL. Validate the rendered Markdown and every public target before upload. Keep
the development-benchmark caveat, explicit availability assumptions and substantial
weather/model preparation requirements. Package installation does not acquire GFS
archives or model weights. Optional extras do not make those resources automatic.

## Resolve dependencies and test installations

Base dependencies provide numerical/data handling, HTTP ingestion and deterministic
solar geometry. They do not include Torch, LightGBM or the native weather stack;
pvlib does bring scientific transitive dependencies, so base is not dependency-free.

Resolve base, `lightgbm`, `weather`, `lightgbm,weather`, `experiments`, `dev`,
`chronos`, `t0` and the joint TSFM extras against the built wheel. For example:

```shell
python -m pip install --dry-run --ignore-installed --report resolve.json "dist/solarcast_belgium-0.1.0-py3-none-any.whl[lightgbm,weather]"
```

A resolver report is not an installation test. Use fresh environments for actual
base/model/weather installs, run `python -m pip check`, verify package origin outside
the repository, and run the synthetic quickstart. TSFM resolution can be tested
without downloading checkpoints; report separately whether the heavy runtime was
installed. The adapters still require explicit weight-download opt-in. Recheck on
supported platforms/Python versions; one machine's resolution is not a universal lock.

## TestPyPI owner setup

TestPyPI and production PyPI have separate accounts, projects and credentials.
An absent project in JSON and Simple APIs does not reserve its name or prove it
can be registered. Do not rename production branding to work around a test-index
collision. Report a collision and validate the local wheel instead.

Use the repository's [manual TestPyPI workflow](../../.github/workflows/publish-testpypi.yml).
The owner must sign in to TestPyPI, verify the email address, enable two-factor
authentication and configure a pending publisher under **Publishing** with:

- Project: `solarcast-belgium`.
- GitHub owner: `Junaid-Bashir`.
- Repository: `solarcast-belgium`.
- Workflow filename: `publish-testpypi.yml`.
- GitHub environment: `testpypi`.

Configure the matching GitHub environment with the owner as a required reviewer
and restrict deployments to `main`. For a single-owner repository, the owner must
be allowed to approve a run they initiated; the approval is still an intentional
human action. Do not approve the environment before reviewing the built artifacts.

The workflow is dispatched manually on `main`, checks out that exact commit,
requires successful CI for it, reruns verification and records wheel/sdist SHA-256
hashes. It uploads the verified files as `testpypi-distributions`. The isolated
publishing job consumes those same files, checks their hashes again and gets OIDC
identity only through the protected `testpypi` environment. Its upload URL is fixed
to `https://test.pypi.org/legacy/`; no production endpoint can be selected.

Record the reviewed candidate's source commit and both hashes locally before
upload. Git's canonical source bytes matter: Windows working copies can contain
CRLF while a Linux checkout uses LF. Compare against a build of the version-controlled
source, not a stale artifact from differently normalized working files. Download
and compare the workflow artifact before approving its publishing job. An unexpected
hash difference is a stop condition; do not dismiss it as a platform detail without
checking source bytes and archive metadata. Successful TestPyPI uploads are immutable.

Do not grant this TestPyPI workflow a production publisher binding. Production
publishing uses the separate release workflow and environment described below.

Alternatively the owner can upload an approved local candidate from their own
terminal using Twine's hidden password entry or an OS keyring. TestPyPI requires
its own API token; use username `__token__`, never a production token, and never put
the token in a command, repository file, chat or log. For a first project, an
account-scoped TestPyPI token may be necessary; revoke it or replace it with a
project-scoped token after validation. The explicit TestPyPI-only command is:

```shell
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/solarcast_belgium-0.1.0-py3-none-any.whl dist/solarcast_belgium-0.1.0.tar.gz
```

Do not run this command until account ownership and upload approval are clear.
TestPyPI validation does not authorize production upload. Uploaded filenames cannot
be reused casually; if the candidate changes, assess an explicitly distinct test
version before another upload rather than altering the intended production identity
or silently assuming an overwrite.

## Install the exact TestPyPI candidate

Use another new environment. Download SolarCast alone from TestPyPI, then verify
the downloaded wheel's SHA-256 against both the reviewed local candidate and the
TestPyPI JSON release metadata. Resolve dependencies from production PyPI without
letting TestPyPI participate in their resolution:

```shell
python -m pip download --no-deps --only-binary=:all: --index-url https://test.pypi.org/simple/ --dest test-download solarcast-belgium==0.1.0
python -m pip install --index-url https://pypi.org/simple/ test-download/solarcast_belgium-0.1.0-py3-none-any.whl
python -m pip check
```

This avoids a mixed-index selection for every dependency. Preserve the download
log and registry filename/hash evidence locally. Outside the checkout, verify
`solarcast.__version__ == importlib.metadata.version("solarcast-belgium") == "0.1.0"`,
package origin in the new environment, intentional public exports and the README's
synthetic benchmark. Confirm 192 scored targets, full forecast availability and
zero RMSE for its constant synthetic data. This is packaging validation, not new
forecasting evidence. Inspect the actual TestPyPI page for title, description,
README image/links, licence, Python bounds and project URLs. Do not describe local
Markdown validation as a successful remote page/installation test.

## Production Trusted Publishing design

Use PyPI's OIDC Trusted Publishing rather than a stored long-lived API token.
Only the isolated publish job gets `id-token: write`; it consumes the already
verified distributions. Build/test jobs have no index credentials. The owner must
approve creating the production `pypi` GitHub environment and pending PyPI publisher
for this exact owner/repository/workflow/environment combination. Require an owner
reviewer, prevent self-review where supported, and restrict deployment to approved
release tags. A pending publisher is not name ownership until its first successful
upload. TestPyPI configuration is independent.

The repository's [production release workflow](https://github.com/Junaid-Bashir/solarcast-belgium/blob/main/.github/workflows/publish-pypi.yml)
is triggered only by a published, non-prerelease GitHub Release for `v0.1.0` in
this repository. Bind `publish-pypi.yml` and the separate `pypi` environment in
production PyPI. Restrict that environment to the `v0.1.0` tag and require owner
review; a single owner must be allowed to approve their own initiated run.

Before creating the annotated tag, require green CI for the exact release commit.
The workflow checks out the release SHA and independently verifies that commit's
successful main-branch CI. It builds, inspects, smoke-tests and hashes distributions
once, then uploads them as `pypi-distributions`. The isolated publishing job consumes
those same files and verifies their hashes again before OIDC publication to the
fixed production endpoint `https://upload.pypi.org/legacy/`. Only this job receives
`id-token: write`, behind the protected production environment.

Review the built artifact hashes before approving deployment. The first-release
guard deliberately accepts only `v0.1.0`; update it deliberately for later versions.
No main push, pull request, arbitrary tag or other repository can publish through
this workflow. Never move a published tag to include subsequent documentation
updates. Update GitHub installation guidance in a separate commit only after the
production upload and fresh registry installation have been verified.

These official GitHub/PyPA actions are pinned to verified full commit IDs; review
upstream changes before updating them. The normal CI matrix supplies broader
platform/optional-dependency checks. Prefer reusing the same approved artifact
hashes for TestPyPI and production; document any rebuild/source changes and repeat
verification rather than claiming two different wheels are the same candidate.
Do not store PyPI credentials as long-lived GitHub secrets when OIDC is available.

## First-release note guidance

**SolarCast Belgium 0.1.0 — research software**

The first package release provides a common API for benchmarking energy forecasting
models, beginning with day-ahead Belgian aggregated solar PV:

- `benchmark`, `BenchmarkConfig`, `BenchmarkResult` and `available_models`.
- Historical Elia PV normalization and explicit provenance.
- Last-value and elapsed-time seasonal persistence baselines.
- Optional LightGBM, Chronos-2 and t0-gamma adapters.
- Historical operational GFS vintage selection and weather-informed LightGBM.
- Leakage-aware rolling-origin backtesting, common-support deterministic metrics,
  failure accounting and inspectable leaderboards.

Version 0.1.0 is pre-1.0 research software with an evolving API, not an operational
forecasting service. The published Belgian PV results are a **known-period
development benchmark**, not an untouched confirmatory evaluation or operational
validation. A fresh confirmatory period is required before generalizing performance.
NWP availability includes an explicitly disclosed approximation; optional runtimes,
local prepared weather data and separately licensed model weights have their own
prerequisites. No datasets or weights are bundled. Software is Apache-2.0; preserve
third-party notices. Keep the changelog unreleased and omit a release date until
production publication is approved and actually performed.

## Official references

- [TestPyPI accounts, upload and installation](https://packaging.python.org/en/latest/guides/using-testpypi/).
- [PyPI name retention](https://docs.pypi.org/project-management/name-retention/).
- [Pending Trusted Publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
- [Publishing with OIDC](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
- [PyPI-friendly README](https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/).
