# Belgian PV data

The target is national aggregated Belgian solar PV generation in MW, at quarter-hour interval starts. It is not site-level production.

The Elia adapter retrieves explicitly bounded intervals and normalizes timestamps to UTC. Before 26 October 2026 it uses historical `ods032` measured data; the newer `ods209` source uses Belgian total measured-upscaled values. See the [Elia source contract](../data/elia.md) for the exact transition, semantics and provenance. Near-real-time values are not silently substituted.

Missing values remain missing, without zero filling or interpolation. Preserve snapshot identity and retrieval provenance. A later snapshot may contain revisions: event-time completion does not establish historical publication availability.

## Belgian days and UTC

A Belgian local day contains **92, 96 or 100** quarter-hour intervals on spring, ordinary and autumn days. Canonical timestamps are timezone-aware UTC; target days use `Europe/Brussels`. Never assume 96 slots.

Use the [real-data notebook](https://github.com/Junaid-Bashir/solarcast-belgium/blob/main/notebooks/00_real_data_quickstart.ipynb) for download-to-leaderboard code. Its June 2026 comparison is exploratory, not confirmation. Data licensing is separate from Apache-2.0 software licensing.
