# Reproducibility: what a fresh checkout can do

| Workflow | Classification | Fresh checkout reality |
|---|---|---|
| Canonical quickstart, baseline/feature/contract examples, default tests | Fully reproducible from public source | Small offline synthetic fixtures; base dependencies plus dev tools for tests |
| Synthetic LightGBM examples/tests | Fully reproducible with optional dependency | Install the lightgbm extra; no PV/NWP download |
| New real Elia benchmark using caller schedules | Requires external download | Acquire normalized PV deliberately, preserve provenance and issue/fit configuration; not exact reproduction of an old snapshot |
| Real t0/Chronos inference | Requires optional model weights | Install the extra and obtain pinned checkpoints; large downloads and hardware/runtime requirements are separate |
| New weather-informed benchmark | Requires large external download/preparation | Prepare operational GFS vintages for every training/evaluation issue and disclose availability assumptions |
| Exact original LightGBM, TSFM and weather development experiments | Requires local frozen artifacts not publicly distributed | Original immutable PV shards, caches, sources, environment and parent manifests are required |
| Published 2025 aggregate table/chart | Development provenance only | Values are inspectable in the small published snapshot; exact raw-input recomputation needs the absent frozen bundles |
| Interactive real-data analysis notebook | Requires local frozen artifacts | Clean source is public; original snapshots/predictions/weather are not bundled. It fails clearly without them. |

## Why exact replay is not a fresh-clone promise

The research scripts authenticate parent configurations, immutable input shards,
prediction caches, source snapshots and freezes. Many required objects live under
ignored `data/` and `artifacts/`, including original source ZIPs. No public download
location for those complete bundles is configured. They may be scientifically
important without being appropriate to commit. A current provider download may
contain revisions and cannot silently replace a hash-pinned snapshot.

The original experiment source tree differs from today's public API/release tree.
The old source guards should reject that difference. Replaying an original frozen
experiment requires its archived source and environment; do not loosen guards,
rewrite manifests or claim that a current editable installation is identical.
Dependency compatibility ranges are not an exact lockfile.

## Closest legitimate replication path

1. Start with the offline quickstart and tests to validate the installation.
2. Choose a fresh chronological confirmation period and explicit issue/fit schedule,
   availability policies, context, raw-output scoring and reference model.
3. Acquire your own bounded national PV snapshot and preserve source/version/license
   and retrieval provenance. Publication/revision availability remains unresolved.
4. For NWP, prepare actual operational forecast vintages for training and evaluation;
   preserve interval bounds and distinguish observed availability from the six-hour
   approximation. Do not substitute observed weather or reanalysis.
5. Prepare the optional pinned TSFM checkpoint caches deliberately if comparing TSFMs.
6. Run the public API or advanced research components with your own local inputs.
   Record effective config, dependency versions, source revision, input fingerprints,
   trained artifacts and model/checkpoint identities. Inspect failures/common support.

That produces a new disclosed experiment, not an exact replay of the published
2025 table. The public runner does not automatically pair history/weather training
support; controlled information-value studies must arrange/check it explicitly.
The [original weather report](experiments/weather-lightgbm-v1.md) documents the
paired-training controlled design and known-period limitations.

The [public metric snapshot](benchmarks/README.md) retains original aggregate
values and their provenance, without republishing raw provider data or private archives.

## Verbatim archived metadata

Authenticated experiment configurations retain their original comments and decision
labels byte-for-byte. Historical labels are opaque provenance metadata, not required
knowledge or instructions for using SolarCast. Editing them would invalidate the
recorded configuration and parent hashes. Use the experiment names, model IDs and
explicit numerical policies documented here to interpret the studies.
