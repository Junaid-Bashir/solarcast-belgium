# Contributing

Use Python 3.11–3.13 and the Windows or Linux/macOS virtual-environment setup in
[README.md](README.md), then install the package and development tools:

```text
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
python examples/quickstart.py
python scripts/check_docs.py
python -m build
python scripts/check_distributions.py
```

Use `python -m ruff format .` to format changes. Build release artifacts locally
with `python -m build`; building does not publish them.

Keep pull requests small and within the current roadmap phase. Describe the problem,
resulting behavior, verification, and material limitations. Add meaningful tests
with behavior changes. Scientific methodology changes need explicit justification,
availability/leakage analysis, and tests where appropriate.

**Forecasting contributions that affect evaluation methodology must explicitly
consider information availability and leakage.** Distinguish issue time, target
time and actual availability, including training labels, transforms, NWP vintages,
snapshot revisions and pretrained-model contamination. Preserve failures and shared
scoring support; never quietly replace missing forecasts or remove nighttime.

Optional tests can use `python -m pip install -e ".[dev,lightgbm]"` or
`".[dev,weather]"`. Base CI and fake-runtime TSFM tests need no provider network,
credentials or downloaded weights. Real TSFM integration tests require the explicit
environment flags documented in their model guides; do not enable them in routine PR CI.

Do not include datasets, credentials, model weights,
or machine-specific configuration. Document third-party sources and license terms
separately from the software license. Do not claim unimplemented capabilities.

Dependency ranges support installation compatibility, not exact experimental
reproduction. Future benchmark runs must record resolved dependencies, configuration,
seeds, source/model revisions, and data fingerprints. Existing development experiments
have local frozen inputs/source/environment records, not publicly distributed exact
reproduction bundles; see [reproducibility](docs/reproducibility.md). Preserve those
freezes, including line-ending-sensitive configuration hashes.

## Useful future contributions

To add a forecaster, start with the [forecasting contract](docs/forecasting-interface.md),
[architecture](docs/architecture.md) and [contract tests](tests/test_models.py).
Preserve the existing backtesting/evaluation flow and keep optional runtimes behind
their adapter boundary; no new plugin hierarchy is needed.

Discuss scope before substantial implementation. Possible future work includes
additional model adapters, European energy datasets/targets, probabilistic scoring,
additional historical forecast-weather providers and stronger reproducibility tools.
These are contribution ideas, not supported features or promises. A fresh confirmatory
period is especially valuable; do not tune on the now-known 2025 development results.

## Publication

Builds do not publish anything. Follow the [release process](docs/release/README.md)
and obtain explicit owner approval before committing, pushing or publishing. GitHub
publication and a later PyPI release are separate decisions. Do not commit credentials,
provider datasets, prediction caches, trained models, source archives or checkpoints.

## Engineering and scientific contracts

- Keep changes scoped, readable and typed; preserve Windows, Linux and macOS support.
  Inspect current behavior before editing and run relevant checks after changes.
  Avoid services, large infrastructure and dependencies without a concrete requirement.
- Keep new model families behind optional import boundaries. SolarCast is
  model-agnostic; a candidate need not win. Reuse the shared inference/backtesting
  contracts instead of adding model-specific evaluation branches.
- Store canonical timestamps in UTC and define Belgian days in Europe/Brussels.
  Distinguish issue, target and information-availability times; support 23/24/25-hour
  days. Seasonal lags are exact elapsed UTC durations, with no silent fallback.
- Historical features require an explicit cutoff and completed intervals. This is
  not proof of publication availability. Preserve capacity-vintage, revision and
  methodological-continuity uncertainties; do not manufacture availability timestamps.
- Use chronological evaluation. Reconstruct training features and eligible NWP at
  each historical issue before joining fit-time-eligible labels. Transform fitting,
  hyperparameter selection and pretrained-model evaluation must obey the same
  information restrictions. Require a successful fit before trainable prediction.
- Keep future actuals out of model context. Retain every failed-origin target row
  and report failure/availability accounting alongside common-support accuracy.
  Score raw predictions: no undeclared clipping, nighttime exclusion or ambiguous nRMSE.
  RMSE skill needs an explicit reference model on the same population.
- Never substitute future observed weather, reanalysis or later hindcasts for
  historical forecasts. Preserve forecast-run identities, radiation interval bounds,
  grid/weights identities and assumed versus observed availability. The six-hour GFS
  lag remains an assumption; retrieval time does not certify historical availability.
  Missing NWP fails the whole origin; never silently switch to history-only forecasts.
- Preserve training/prediction leakage canaries and offline availability, coverage,
  cache and sentinel tests. TSFM fake-runtime tests are mandatory; real-weight tests
  require explicit opt-in. Preserve native P50, missing elapsed slots, the full
  cutoff-to-target rollout and explicit inference-adjustment settings.
- Verify provider schemas, fields, units, semantics and terms against official
  documentation before adapter changes. Keep raw data, model weights and generated
  results outside Git, with provenance and separate licensing records.
- Preserve frozen configuration, input, source/archive, checkpoint and prediction
  identities. Do not rewrite old freezes for formatting or bypass authentication.
  Input-precision policies must remain explicit: strict radiation is the default;
  bounded GRIB-packing correction is a named, audited option, not PV clipping.
- The 2025 development results are known. Any modeling decisions informed by them
  require a new confirmation period or an explicitly labelled development study.
  Do not retune or fit an ensemble on those results and claim independent gains.

Source contracts and detailed model-specific guards live in the linked data,
feature, forecasting, backtesting, evaluation and model guides. Extend those
contracts deliberately and document uncertainty rather than encoding it silently.

## Documentation website

Install documentation tools separately from runtime dependencies:

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs serve
python -m mkdocs build --strict
python scripts/check_docs_site.py
```

See [documentation development](docs/contributing/development.md) for the Pages workflow.
