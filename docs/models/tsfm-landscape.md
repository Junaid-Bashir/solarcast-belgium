# TSFM landscape and SolarCast model selection

**Source review date: 14 November 2026.** Evidence comes from publisher repositories,
model cards, configuration/packaging files and technical reports. Capabilities are
version-specific; rankings below are dated reported results, not SolarCast results.
Unknown means the inspected primary sources did not establish the fact. Native
SolarCast integration evidence is recorded separately below.

## Candidate selection rationale

| Model/checkpoint | Select? | Role and order | Main reason | Main concern |
|---|---|---|---|---|
| `amazon/chronos-2` | Yes | **Primary candidate** | Mature zero-shot API, energy/15-minute evidence, moderate size, Apache-2.0 code and declared weights | Not the current generic benchmark leader; pretraining overlap unverified |
| `theforecastingcompany/t0-gamma` | Yes | **Secondary candidate** | Strong recent reported accuracy, native covariates/quantiles, Apache-2.0 code and weights | Larger checkpoint; Windows not explicitly supported in package classifiers; gamma-specific context/corpus limits incompletely documented |
| `NX-AI/TiRex-2` | Optional | **Additional research candidate** | Recurrent xLSTM diversity, smaller univariate active model, official CPU/ONNX paths | Windows compilation/kernel caveats; public general checkpoint differs from benchmark-specific variants |
| Timestral, especially 28M/209M | Later, conditional | Competitive follow-up | Strong recent benchmark and CPU evidence; public local inference | Modified MIT **revenue restriction**, not unrestricted MIT |
| TimesFM 3.0 | No for initial general-use local track | Research-only alternative | Strong recent zero-shot evidence and covariates | Downloaded weights are **non-commercial and non-production** |
| TimesFM 2.5 200M | Reserve | Permissively licensed fallback | Mature zero-shot univariate baseline | Older than 3.0; less scientific diversity than TiRex-2; XReg is a separate fitted covariate pathway |
| Moirai 2.0 R-small | No initially | Lightweight research comparator | 11.4M parameters; efficient quantile forecasting | CC BY-NC 4.0 weights and restrictive older dependency pins |
| Toto 2.0 22M / 313M | Reserve | Alternative pretraining/size comparison | Apache-2.0; observability/synthetic pretraining; small variants | No released 2.0 exogenous support; newer Python/Torch requirements and Windows unverified |
| Granite TimeSeries PatchTST-FM-r2 | Reserve | Strong conformer comparator | Current permissive dual licensing and useful benchmark evidence | ~385M, heavier stack; exact long-output/NaN pipeline policy needs verification |

Choose **two firm candidates**, with the third justified by architecture diversity,
not an obligation to install three models. This is a practical shortlist, not an
accuracy leaderboard. Timestral could be a strong scientific comparator if the
user's applicable licensing position is established later. No restricted weights
should become an automatic SolarCast download. [C1–C5, T1–T6, X1–X6, S1–S5,
F1–F5, M1–M4, O1–O4, I1–I4]

## SolarCast constraints inspected

The package supports Python **3.11–3.13**, pandas `>=2.2,<4`, NumPy `>=1.26,<3`,
and optional LightGBM. The current environment is Python 3.12.4, pandas 3.1.1,
NumPy 2.5.4. Existing contracts pass a request, eligible PV history and an explicit
cutoff to `predict(context)`. Forecast results require finite point values on
exactly the ordered request timestamps. The backtester and evaluator are model
agnostic; errors retain failed-origin rows and scoring uses common issue/target
support. No TSFM fit method is needed.

The [first real experiment](../experiments/lightgbm-baseline-v1.md) forecasts the
next Brussels calendar day at 11:00 local, with a 30-minute safety lag and 14D
issue-anchored context. Its March–October 2025 targets contain 92/96/100 intervals.
LightGBM RMSE was 1026.881 MW versus 1027.033 MW for 48h persistence: essentially
tied. This establishes a meaningful comparator, not an expectation that TSFMs win.

### The rollout must include the gap before tomorrow

**A 100-step-capable model is not automatically sufficient.** At issue 11:00 local,
cutoff is 10:30 and latest eligible interval start is 10:15. A regularly sampled
model must predict from **10:30 today**, through tomorrow's final interval, then
select tomorrow's requested timestamps. It must not start a 96-step forecast at
tomorrow's midnight while pretending the unobserved gap was context.

For the existing experiment, the required full rollouts are **150 steps for 243
origins, 146 at spring DST and 154 at autumn DST**, calculated with the existing
target policy. Returned Forecasts still contain exactly 92/96/100 requested rows.
For arbitrary requests derive:

```text
first prediction start = last grid context start + 15 minutes = cutoff
rollout length = 1 + (last requested target - cutoff) / 15 minutes
return only the exact requested UTC timestamps from that rollout
```

Preserve a regular UTC quarter-hour grid. DST changes the requested local day,
not the model's elapsed sampling interval. No local-naive timestamps, compressed
missing rows, fixed 96-step assumption or unobserved bridge values are allowed.

## Model-by-model findings

### 1. Chronos-2: primary practical candidate

The current Amazon repository identifies Chronos-2 as its latest family. The
selected encoder-only checkpoint is ~120M parameters; smaller Chronos-2 28M and
synthetic-only 120M variants are also listed, alongside older Bolt/T5 checkpoints.
Use the **native `chronos-forecasting` package**, `Chronos2Pipeline.from_pretrained`
and its array prediction API, avoiding the full AutoGluon training/ensemble stack.
CPU and CUDA inference are explicitly supported. Adaptation complexity: **easy**
for model input/output, **moderate overall** after timestamp/missingness/bridge
canaries. [C1–C3]

The inspected checkpoint config has **8192 context steps**, output patches of 16,
and 64 output patches: **1024 steps in a native block**. That comfortably covers
SolarCast's 154-step maximum. Longer requests can roll out, but are unnecessary
here. Pass context limits explicitly; do not rely on defaults or accept silent
truncation. Arrays do not require frequency tokens. The DataFrame convenience API
requires regular timestamps and can infer/supply frequency; frequency inference is
not proof of quarter-hour alignment. [C4, C5]

Native outputs include 21 quantiles with P10/P50/P90. Select the actual **0.5
quantile** rather than assuming a convenience point field is a calibrated mean.
Past and future-known numeric/categorical covariates are supported without
Belgian-specific fitting. NaN handling is native; absent timestamps must first be
represented as missing slots, not removed. No native dedicated static-input
contract was verified. [C2, C5]

Published training includes 15-minute electricity and sub-hourly solar; its paper
also evaluates 15-minute solar-with-weather tasks and an hourly day-ahead German
electricity-price case. These support relevance, not proven Belgian PV superiority.
Most striking reported energy gains use covariates, which our first track will
withhold. Its October 2025 launch-era top-rank claims are historical, not current
rankings. Code is Apache-2.0; the selected model card declares Apache-2.0 for
weights. A separate weight LICENSE file was not present at the checked path, so
the weight declaration is sourced to the publisher's card. [C1–C3, C6]

### 2. t0-gamma: secondary accuracy-oriented candidate

**t0-gamma is publicly downloadable**, not API-only. Alpha (~102M), beta and gamma
(~256M each) remain listed. Gamma is the current third public iteration. Both
repository and gamma's actual weight LICENSE are Apache-2.0. Local PyTorch and
first-party Apple-silicon MLX runtimes are distinct from the managed Retrocast/API
service. The gamma card requires **`tfc-t0[torch]>=0.6.1`**; the repository README's
0.6.0 floor is older, so use the stricter checkpoint-specific floor later. No
alpha ONNX export should be represented as a gamma export. [T1–T4]

Use `T0Forecaster.from_pretrained(...).eval()` and `predict(context, horizon=H)`;
arrays and masks are supported, without fitting. Gamma produces 21 native
quantiles and decodes **1024 steps per pass**, then rolls out for longer horizons.
Context width is configurable. **A gamma-specific hard maximum is not documented
in the inspected config**: the current runtime validates against input width,
while the alpha/beta report uses 8192-step training/evaluation contexts. This is
evidence for the family, not a guarantee of arbitrarily long gamma inference.
Use the proposed 7D/14D inputs first; validate the pinned runtime rather than
claiming an unbounded supported context. Adaptation complexity: **moderate**.
[T2, T5, T6]

Raw arrays are frequency agnostic; supply a regular 15-minute grid. NaNs denote
missing observations; masks distinguish missing values from padding. Native past
variates and known-future dynamic covariates are available. A dedicated static
covariate input was not verified, and constant future covariates are dropped by
default. P50 is directly available. Symmetric averaging, positivity clipping,
integer snapping and leading-missing trimming are optional inference adjustments;
keep them explicitly **off** in the initial raw-output track, recording the choice
rather than inheriting a notebook's recommended postprocessing. [T1, T2, T5]

The technical report supplies electricity-demand and ERCOT electricity-price
studies for **alpha/beta**, including an independent evaluation reported within
the paper. Gamma's recent card supplies stronger generic benchmark scores, but
these energy studies do not directly establish gamma's Belgian PV performance.
Their covariates and weather-vintage assumptions also differ from SolarCast's
PV-only track; they are not evidence of an identical operational information set.
The report's older-checkpoint results and the newer gamma release must remain
distinguishable. [T2, T6]

### 3. TiRex-2: optional recurrent/accessible candidate

TiRex-2 adds bidirectional recurrent xLSTM time mixing with cross-variate attention,
not another close Chronos transformer variant. The general public checkpoint
stores **82.5M parameters**; **38.4M are active for univariate forecasting** and an
additional 44.1M serve multivariate forecasting. All stored weights still occupy
~330 MB. Do not estimate download size from the univariate active count. Code and
the checked weight LICENSE are Apache-2.0; this differs from the older TiRex
community-license release. [X1–X3]

For **`NX-AI/TiRex-2`**, the checked config is **2048 context / 320 future steps**,
with nine native deciles including P10/P50/P90. These limits cover both proposed
contexts and the complete 154-step rollout. The ONNX export exposes a fixed
2048+320 window; PyTorch allows variable request lengths and default NaN padding.
Use the default 2048-context profile for initial planning. Longer-context research
and named benchmark checkpoints are separate: for example,
`NX-AI/TiRex-2-gifteval-zs` currently declares 8192/928. Do not transfer those limits
or scores to the general checkpoint. [X3–X5]

Past/future-known covariates and missing values are supported. The array API uses
step positions, not a required sampling-frequency token; preserve the quarter-hour
grid. Dedicated static inputs were not verified. `load_model`/`forecast` wrap the
native representation; the ONNX path needs NumPy/ONNX Runtime instead of Torch.
Adaptation complexity: **moderate**, with extra mask/layout and runtime checks.
[X1, X4, X5]

The paper reports GIFT-Eval and fev results with **benchmark-specific checkpoints
and exclusion rules**, including energy/solar sources. These are evidence of the
family's relevance, not a single universal leakage-free result. Its official
Windows FAQ documents MSVC/compilation and CUDA fallback issues; CPU/ONNX is a
useful accessibility route but not a measured Windows latency guarantee. Native
P10/P50/P90 remains valuable even if its accuracy later trails persistence. [X6–X8]

### Timestral: strong evidence, conditional license

Timestral 1.0 has public **5M, 28M, 123M and 209M** weights and standalone local
CPU/GPU PyTorch inference. It is not merely an industry demo or API product.
`TimestralForecaster.from_pretrained(...).forecast(...)` accepts NaNs, supports
past/future numeric/categorical covariates and yields 21 quantiles. Context is
**32768 steps**, single-pass output **1024**, with median-fed blocks beyond that;
quarter-hour regular arrays are compatible. A separate static API was not verified.
Adapter difficulty would be **easy–moderate**. [S1, S2]

The repository code is Apache-2.0. **Weight terms are Modified MIT**, requiring
attribution and denying rights when the company/employer's global consolidated
monthly revenue exceeded **US$20 million in the preceding month**, absent separate
licensing. Public weights therefore do not imply unrestricted commercial use.
Do not label these weights ordinary MIT, redistribute them under SolarCast's
Apache license, or make them a default dependency for all corporate users. The
repository describes hosted/commercial alternatives, governed separately. This
is a reading of published terms, not legal clearance for a particular entity.
**Defer initially**, rather than reject its technical merit. [S1, S3]

Its October 2026 report/model card provide very strong GIFT/TIME/fev and industrial
IND-Eval evidence. The paper's CPU timings use a server EPYC, compiled bfloat16,
warm-up and symmetric averaging; they are not Windows laptop measurements. The
published 28M variant is an attractive later compute comparison if the license is
applicable. Post-training uses real as well as synthetic data: the released models
must not be called synthetic-only. [S2, S4, S5]

### TimesFM: distinguish 3.0 from 2.5

The repository's current generation is **TimesFM 3.0**, ~331M stored parameters,
with native multivariate/past/future covariates, nine deciles and a 15360-step
global context. It decodes 64-step patches, so longer horizons require multiple
output patches. CPU/PyTorch and an Apple MLX path are documented. The repository
code is Apache-2.0, but **3.0 weights are non-commercial/non-production**; authorized
Google Cloud services have separate commercial terms. Do not select a restricted
self-hosted checkpoint on the assumption that earlier TimesFM terms still apply.
[F1–F3]

**TimesFM 2.5 200M** remains Apache-2.0 for code and weights. Its stored checkpoint
includes the optional quantile head (~231M total), supports up to 16384 context
steps and up to ~1024 continuous-quantile horizon, and removes the old frequency
indicator. Its native zero-shot univariate path can cover SolarCast's full rollout.
XReg covariates, including static/dynamic regression inputs, are a separate fitted
path, not native zero-shot covariate attention. Keep them out of this first track.
Missing-value/preprocessing behavior and forecast flags must be checked in the
pinned version before use; no separate imputation is proposed here. Reserve 2.5
as a permissive fallback, rather than select it solely on older popularity.
Adapter difficulty: **moderate**. [F1, F4, F5]

### Moirai: investigate 2.0, do not inherit 1.x promises

The current listed research checkpoint is **Moirai-2.0-R-small**, a ~11.4M model
with nine decile outputs and missingness-aware patches. Original 1.1 and Moirai-MoE
checkpoints remain available. Version 2.0 uses a 16-step patch, up to 512 tokens
(8192 time positions as a nominal context bound), and predicts four patches per
iteration with recursive decoding for longer requests. Its 64-step native block
can be rolled out to the required 154 steps. Do not confuse a token cap with 512
quarter-hours. [M1–M3]

Uni2TS code is Apache-2.0; **2.0 weights are CC BY-NC 4.0**. CPU/GPU inference and
GluonTS-based integration are available. The inspected 2.0 wrapper has dynamic-real
and observed-mask arguments, but that alone does not establish all advertised
1.x covariate behavior for the selected pretrained 2.0 checkpoint; treat covariate
and static guarantees as requiring validation. Versioned frequency/pretraining
details are preferable to a generic “any frequency” claim. GIFT-Eval energy/domain
results are relevant but no verified Belgian zero-shot result was found. [M1–M3]

Uni2TS's current main packaging pins Torch `<2.5`, NumPy `~1.26`, SciPy `~1.11`,
older datasets/GluonTS and JAX. It would downgrade the current scientific stack and
cannot share TiRex-2's Torch `>=2.8` environment as declared. If investigated later,
use a separate environment, not relaxed core pins. Difficulty: **moderate–difficult**.
This and the non-commercial weight license exclude it from the initial general-use
shortlist despite its small size. [M4]

### Other major open/pretrained candidates

**Toto 2.0** deserves consideration, rather than only Toto 1.0: five sizes span 4M,
22M, 313M, 1B and 2.5B. Code/weights are Apache-2.0, outputs are nine deciles, and
the reported training uses internal observability and synthetic data rather than
public forecasting corpora. The 22M checkpoint is a plausible inexpensive reserve.
Its native array/mask API supports variable context/horizon and CPU/CUDA, with
single-pass versus block decoding options; a hard context maximum was not verified.
Released 2.0 exogenous support is **planned**, not present merely because 1.0 has
it. Current package metadata requires Python `>=3.12` and Torch `>=2.14`, stricter
than its README's older Torch floor. Windows/missing-value helper defaults need
validation. Strong family benchmark results do not give the 22M variant the 2.5B
model's scores. Difficulty: **moderate**. [O1–O4]

**Granite TimeSeries PatchTST-FM-r2**, ~385M, is a serious permissive reserve: code
Apache-2.0, weights offer **Apache-2.0 or OpenMDW-1.0** even though Hub metadata lists
only OpenMDW. It has 8192-step context, 99 quantiles and flexible forecast lengths
through `granite-tsfm`/`TimeSeriesForecastingPipeline`. Its saved prediction length
64 is a default, not evidence that all output must be fixed at 64. CPU is supported;
publisher estimates and weights imply more memory than our primary. Verify the
specific long-output, masking/imputation and covariate paths before promoting it;
do not assert NWP support from a generic multivariate architecture. GIFT-Eval and
mixed-domain evidence are strong but the September-era ranking is dated.
Difficulty: **moderate**. [I1–I4]

**EXAONE Forecast** appears in recent benchmark tables, but the official repository
separates the general model (technical report, no listed weight artifact) from
released Finance/Demand variants with non-commercial weights. Do not substitute
those domain checkpoints for an unreleased leaderboard model. **QiYao-M** has an
official MIT-tagged card and multimodal claims, but the inspected Hub file metadata
contained no downloadable weight artifact; code/weight completeness was not
established. **Lingjiang 2.0** appears in a vendor comparison, but a verifiable
public implementation/checkpoint was not located in this review. These are
watchlist items with unknown practical limits/licenses where unverified, not
eligible local benchmark entries. Older Sundial and other families remain outside
this small initial shortlist; coverage here is not a claim to exhaust every TSFM.
[W1–W3]

## Capability and practical comparison

Every firm/optional selected model is zero-shot: no task-specific training,
fine-tuning or per-series regression fit is required. Scaling, masking and inference
are not weight training. All operate on regular step-indexed histories; there is
no guarantee that a frequency-agnostic architecture learned Belgian solar physics.

| Candidate | Context: checked limit/profile | Native output / rollout | 15-minute / 154-step fit | Missing history | Past / future / static covariates |
|---|---|---|---|---|---|
| Chronos-2 | 8192 | 1024, rollout beyond | Good; documented 15-minute energy data | NaNs/masks; regular grid required | Yes numeric/categorical / yes / dedicated static not verified |
| t0-gamma | Configurable; gamma hard max unknown, family report 8192 | 1024, rollout beyond | Good by step API; gamma-specific corpus coverage unknown | NaNs; explicit PAD versus MISSING | Yes / yes / dedicated static not verified; constant future rows dropped by default |
| TiRex-2 general | 2048 default trained/export profile | 320 | Good; both requested contexts fit | NaNs/masks; default padding | Yes / yes / dedicated static not verified |
| Timestral | 32768 | 1024, median-fed rollout beyond | Good by array interface | NaNs and observation masks | Yes / yes, categorical too / dedicated static not verified |
| TimesFM 3.0 | 15360 global context | 64-step patches, multi-patch decode | Good by array interface | Pinned preprocessing needs verification | Yes / yes / dedicated static not verified |
| TimesFM 2.5 | 16384 | Up to ~1024 quantile horizon | Good by array interface | Version-specific preprocessing; audit before use | XReg fitted pathway, including static inputs; not native zero-shot covariates |
| Moirai 2.0 small | 512 tokens ×16 =8192 nominal time positions | Four patches =64; recursive beyond | Good through recursive wrapper; test alignment | Missing embeddings/observed masks | Wrapper fields exist; checkpoint guarantees need validation |
| Toto 2.0 | Configurable; hard max not verified | Flexible; direct or block decoding | Plausible; no frequency token required by example API | Explicit target mask; helper imputation settings need audit | Released 2.0 exogenous support not established/planned |
| Granite PatchTST-FM-r2 | 8192 | Flexible pipeline; config default 64 | Plausible; verify full bridge output | Pipeline imputation/mask choice must be explicit | Specific pretrained future/static support not verified |

All selected models natively provide P10/P50/P90. These are **marginal quantiles**,
not automatically calibrated joint trajectory samples or a full parametric
distribution. Do not fabricate samples or uncertainty metrics from deterministic outputs.
Moirai 1.x and Toto 1.x distributional outputs should not be assigned to their
quantile-output successors. Static covariate ambiguity does not impede a single
national univariate PV experiment. [C2–C5, T2–T5, X3–X5, S1–S2, F1–F5, M1–M4, O1–O3, I1]

### Qualitative selection matrix

G = good, L = limited, U = unknown/unverified. “Good” means credible evidence for
selection, not a predicted SolarCast win. No weighted score or false numeric ranking
is assigned. The previous table supplies the very-high-priority frequency/horizon
criteria; this table adds evidence and practicality.

| Candidate | Zero-shot | Accuracy evidence | Energy relevance | Open weights | Commercial clarity | CPU accessibility | GPU efficiency evidence | Windows practicality | Probabilistic | Future covariates | Adapter/docs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Chronos-2 | G | G, older than newest releases | G | G | G: permissive declaration | G | G, publisher measured | G inferred; untested here | G | G | Easy–moderate / G |
| t0-gamma | G | G, recent | G family; gamma-specific U | G | G: Apache | G, larger | G family/runtime evidence | U: Linux/macOS classifiers | G | G | Moderate / G |
| TiRex-2 | G | G, checkpoint-sensitive | G family | G | G: Apache | G, runtime caveats | G; kernel-specific | L: documented compiler/fallback cost | G | G | Moderate / G |
| Timestral | G | G, recent | G industrial; Belgian PV U | G | L: explicit revenue condition | G | G, measured setup | G inferred; untested here | G | G | Easy–moderate / G |
| TimesFM 3.0 | G | G, recent | G benchmark-family | G | L: NC/non-production | G | G, platform-specific | U for our stack | G | G | Moderate / G |
| TimesFM 2.5 | G | G, older | G | G | G: Apache | G | G | G inferred; avoid compile/JAX first | G | L: fitted XReg | Moderate / G |
| Moirai 2.0 | G | G, older | G | G | L: NC | G size, L stack | G family | L: old pins/JAX environment | G | U guarantees | Moderate–difficult / mixed versions |
| Toto 2.0 | G | G, size-specific | G benchmark, less direct PV | G | G: Apache | G small variants | G, A100 results | U | G | L: planned | Moderate / G |
| Granite r2 | G | G, dated claims | G mixed domains | G | G: Apache option | G but heavier | G publisher estimates | U | G | U | Moderate / G |

## Licensing comparison

Research and redistribution remain subject to the exact chosen artifact's notices;
“commercial-friendly” below describes published permissions, not an assessment of
every downstream use. SolarCast's license does not relicense external weights.

| Artifact family | Code | Weights | Research / redistribution / commercial implication |
|---|---|---|---|
| Chronos-2 | Apache-2.0 | Apache-2.0 declared in model card | Permissive with applicable notices; no separate weight LICENSE at checked URL |
| t0-gamma | Apache-2.0 | Apache-2.0, LICENSE inspected | Permissive with license/NOTICE conditions |
| TiRex-2 | Apache-2.0 | Apache-2.0, LICENSE inspected | Permissive; do not inherit original TiRex community terms |
| Timestral | Apache-2.0 | Modified MIT | Conditional use/redistribution; company/employer revenue exclusion applies; separate terms may be needed |
| TimesFM 3.0 | Apache-2.0 repository | TimesFM NC v1.0 | Restricted non-commercial/non-production local track; hosted service terms separate |
| TimesFM 2.5 | Apache-2.0 | Apache-2.0 card | Permissive; pin the 2.5 checkpoint explicitly |
| Moirai 2.0 R-small | Apache-2.0 | CC BY-NC 4.0 | Research/non-commercial only under that grant; attribution and NC redistribution conditions |
| Toto 2.0 | Apache-2.0 | Apache-2.0 cards | Permissive with notices |
| Granite r2 | Apache-2.0 | Apache-2.0 OR OpenMDW-1.0 | Publisher's actual LICENSE offers a permissive choice; do not infer solely from metadata |

Relevant source/license links are grouped in the source register below. No licensing
ambiguity was resolved by assuming public access means unrestricted use.

## Hardware and Python integration

Checkpoint sizes below are **file metadata**, not weight downloads. MB/GB are
decimal unless noted. Memory needed for execution is higher than parameter storage:
runtime, loading copies, attention activations, context, batch and quantiles add
overhead. No publisher-certified minimum RAM/VRAM for our exact workload was found.

| Candidate | Parameters and approximate download | Framework/install route to consider later | Hardware / Windows implication |
|---|---|---|---|
| Chronos-2 | 119.48M stored; 478 MB fp32 | `chronos-forecasting`; Torch, Transformers, Accelerate; Python >=3.10 | CPU/CUDA; plain eager CPU first on Windows; no forced flash-attention/compilation |
| t0-gamma | 255.61M; 1.022 GB fp32 | `tfc-t0[torch]>=0.6.1`; Python >=3.11, Torch >=2.4; MLX alternative | CPU/CUDA plausible; Windows unverified; first-party MLX is macOS/Apple-silicon only |
| TiRex-2 | 82.50M stored /38.4M active univariate; 330 MB; ONNX ~407 MB | `tirex-2` 0.5.0 currently; Python >=3.11, Torch >=2.8, NumPy >=2.1.3; ONNX alternative | CPU yes; CUDA Ampere+ per FAQ; Windows slow Torch recurrence fallback, possible MSVC requirement; ONNX CPU avoids Torch compiler stack |
| Timestral | 5.2/27.8/122.6/209.3M; 21/111/490/837 MB | Official Git source `timestral`; Python >=3.10, Torch >=2.4 | CPU/GPU documented; no CUDA-only core dependency; Windows execution untested here |
| TimesFM 2.5 /3.0 | 231.29M incl. 2.5 quantile head /330.71M; 925 MB /1.323 GB | `timesfm[torch]`; explicit checkpoint/version; optional MLX/JAX paths differ | CPU/CUDA; prefer no `torch.compile` first; MLX only appropriate Apple hardware |
| Moirai 2.0 | 11.39M; 45.6 MB | `uni2ts`, GluonTS/Lightning/JAX; older pinned stack | Cheap model, expensive compatibility isolation; NumPy/Torch conflicts with other candidates |
| Toto 2.0 | 4/22/313M/1/2.5B; ~16/88MB/1.2/3.9/9.1GB reported | `toto-2`; current metadata Python >=3.12, Torch >=2.14 | Small CPU models plausible; large models GPU-oriented; Windows not established |
| Granite r2 | ~385M; 1.538 GB | `granite-tsfm>=0.3.9`, Torch/Transformers pipeline | CPU or GPU; publisher estimates ~2GB VRAM, not a guaranteed full-runtime requirement |

For planning, use an ordinary **16GB-RAM CPU machine** and batch size one for the
firm candidates; this is a conservative starting budget, **not a measured minimum**.
A modest CUDA GPU can help, but no NVIDIA GPU is required by the recommended
univariate task. Start with fp32/eager execution and measure actual peak memory
before promising a particular 4/6/8GB VRAM fit. Do not prescribe H100 hardware on
the basis of benchmark throughput. Apple support is explicitly documented for
t0's MLX runtime and TimesFM's MLX backend; other backends are unverified here.

Published speed evidence is not comparable across workloads: Chronos-2 advertises
over 300 series/s on A10G, t0's backend table measures **alpha** on M1 Pro, Timestral
reports 17–113ms on 8 server CPU threads for 2048-context/96-horizon with compile,
bfloat16 and two-pass averaging, while Toto reports A100 batches. None predicts
our Windows batch-one 1342-context/150-horizon latency. TiRex's efficient streaming
research and PRO features do not imply that the public API offers persistent-state
streaming. [C2, T7, X6, S4, O2, I4]

## Accuracy, energy evidence and contamination

### What published benchmark numbers establish

The Timestral publisher's current comparison reports the following aggregate
scores. They share that table's evaluation conventions and are **not** a new
independently reproduced ranking here. fev SQL skill is not SolarCast RMSE skill.

| Checkpoint in that comparison | GIFT CRPS, lower | TIME CRPS, lower | fev SQL skill %, higher |
|---|---:|---:|---:|
| Timestral-209M | 0.446 | 0.522 | 50.39 |
| t0-gamma | 0.452 | 0.534 | 49.42 |
| TimesFM-3 | 0.456 | 0.536 | 48.99 |
| Toto-2.0-2.5B | 0.476 | 0.539 | 44.42 |
| Chronos-2 | 0.485 | 0.556 | 47.28 |

The newer gamma card instead reports **GIFT CRPS 0.4503, MASE 0.6580 and fev skill
49.40**. Its launch announcement reports 0.4520/0.6608 and 49.42. Preserve this
revision/protocol discrepancy rather than mixing the most favorable number from
each source. TiRex's paper/bench instructions use distinct zero-shot/pretrained
variants; Moirai's own report shows small-model GIFT CRPS 0.516 and MASE 0.728.
Those older snapshots must not be merged into a supposedly synchronized live
ranking. [S2, T2, T8, X7–X8, M3]

GIFT-Eval, fev-bench and TIME provide useful heterogeneous/frequency/horizon
evidence. Declared dataset overlap, failure handling, covariate availability,
context lengths and checkpoint revisions affect comparisons. Timestral's paper
explicitly uses additional covariates in GIFT/fev where available; those aggregate
gains do not isolate PV-only performance. TIME's contamination-resistant design
does not certify that Elia's 2025 test data was unseen. [S1, S4, T6]

Energy evidence is strongest as **relevance**, not a Belgian-PV result: Chronos-2
documents 15-minute electricity/solar tasks and energy-price covariates; t0 reports
demand and price case studies for older family members; TiRex discusses energy/
solar training/exclusions and public benchmarks; Timestral's IND-Eval evaluation
includes energy/industrial systems. Other families have electricity/solar tasks
in generic benchmarks or pretraining, but no verified direct national Belgian PV
zero-shot comparison was located. Social-media profiles do not establish general
superiority or clean historical information access. [C6, T6, X8, S4]

### Pretraining contamination is unresolved for SolarCast

| Family | Disclosed corpus information | What remains unknown for March–October 2025 Belgian PV |
|---|---|---|
| Chronos-2 | Chronos datasets, GIFT pretrain subsets, synthetic series; exclusions for named benchmarks | Exact Elia/derived-series membership and all-data cutoff; initial October 2025 release overlaps our evaluation calendar, and the current revision has no Elia exclusion certificate |
| t0-gamma | Family report: curated GIFT pretrain and synthetic data; alpha/beta exclusion analysis | Gamma's full corpus/cutoff not independently specified by older report; do not transfer alpha's pre-2022 claim automatically |
| TiRex-2 | LOTSA/Chronos/other sources; benchmark-specific exclusions | General checkpoint's exact relationship to every evaluation variant and Elia membership |
| Timestral | Synthetic pretraining, then real+synthetic long-context post-training | Full post-training series/cutoff and possible Elia derivatives |
| TimesFM | GIFT pretrain, web data, synthetic/augmentation; web cutoffs disclosed | Web cutoffs do not date every energy training series |
| Moirai 2.0 | GIFT pretrain/train subsets, Chronos mixup, synthetic and internal Salesforce data | Internal membership/vintages and independent Elia overlap check |
| Toto 2.0 | Internal observability plus synthetic; publisher says no public forecasting pretraining | Full internal provenance and target-specific independent verification |
| Granite r2 | GIFT pretrain subsets, other public corpora and synthetic sources | Exact Elia/derived overlap and complete temporal coverage |

“Zero-shot” should mean **no SolarCast/Belgian-specific fitting** in the first
experiment, not guaranteed absence from pretraining. Checkpoints are mostly newer
than the 2025 evaluation. Use the established historical window as a clearly
labelled **retrospective comparison**, and freeze checkpoint revisions before
collecting a genuinely future evaluation period. A synthetic-only Chronos-2
variant is a useful later contamination sensitivity study, not a fourth initial
candidate. New collection after a checkpoint freeze helps address temporal
contamination, but does not repair Elia historical revision/availability uncertainty.
[C2, C6, T6, X8, S4, F2/F4, M1/M3, O4, I1]

## Proposed first zero-shot experiment — do not run yet

1. Preserve the existing March–October 2025 targets, 245 issue times, 11:00 Brussels
   clock, 30-minute lag, normalized snapshot and next-local-day target policy.
   Preserve 48h and 7d persistence; keep last-value as an optional sanity check and
   report 24h ineligibility separately, as before.
2. Keep the frozen LightGBM model recipe/training dates. Label its input correctly:
   **PV history plus deterministic time/solar features**. TSFMs receive only PV;
   this is not identical feature engineering. All share the same eligible PV
   history budget; no observed weather or Elia forecasts enter any candidate.
3. Load one pinned pretrained checkpoint once; no fit, fine-tuning, best-checkpoint
   selection, Belgian scaling fit or validation-driven model adaptation. Use the
   provider's internal scaling and missing masks; record inference options.
4. Build a complete UTC quarter-hour grid over the allowed context, preserving
   absent/null slots as NaN and a mask where needed. Do not interpolate, forward
   fill, zero-fill or remove missing quarters in SolarCast. Record any internal
   model/wrapper missing-value transform separately. A wholly unusable history or
   non-finite output must fail the origin and remain in failure/coverage tables.
5. Predict the full cutoff-to-target-end bridge (146/150/154 here). Select target
   rows by exact UTC timestamp and return raw **P50** in `prediction_mw`, without
   clipping/nighttime masking. Use P50 for all selected TSFMs, fixed before scoring.
   P50 is a natural native deterministic output but is not the squared-error-optimal
   mean; acknowledge that limitation when ranking RMSE. Do not switch to a mean
   after inspecting scores. No quantile average should be called a true mean.
6. Reuse the existing common-support evaluator: primary all-interval RMSE, plus
   MAE, bias, 48h-reference skill, availability, n_scored and failed origins.
   Retain the daylight and horizon/local-hour/month diagnostics. Do not silently
   drop a model whose failures erase support: report the empty comparison and any
   separately declared track explicitly.
7. Freeze model/config/point-policy/context choices before scoring. The 2025
   benchmark already has published diagnostics; any decisions informed by it
   require a new holdout or a development designation. No claim of pristine
   out-of-pretraining generalization is warranted without corpus evidence.

### Small context experiment

Use **14D as primary**, matching the history-only LightGBM experiment, and **7D as a predeclared short-context
sensitivity**. These capture roughly one/two weekly cycles without assuming that
very short contexts suffice for changing solar seasonality. They fit the general
TiRex-2 checkpoint and both firm candidates. “7 days” means 672 elapsed grid slots
nominally; with the existing issue-anchored lower bound and 30-minute lag it yields
**670 eligible slots**. Likewise **14D yields 1342**, not 1344. Record nominal window,
actual grid length, observed count, NaN count and any model padding separately.

Do not automatically add 30D and 60D to the first sweep. They exceed the general
TiRex-2 2048-step profile, add compute and span evolving PV/seasonal regimes.
An optional later **Chronos-2-only** 30D/60D sensitivity fits its 8192 cap (2878/5758
eligible slots under this policy); it is not a common-context leaderboard for all
models. Gamma's longer-context validation would be a separate prerequisite.
Never silently truncate a 30D request and label the result “30D”.

Keep the current LightGBM at its 14D-trained recipe in the primary track. If 7D
context is supplied to it as an inference ablation, identify that mismatch
explicitly rather than presenting it as a newly trained 7D model. Do not choose
the best test context for each model and call that an untouched holdout comparison.

### Compute fairness

Record hardware/OS, package and checkpoint revisions, stored/active parameters,
weight bytes, dtype, backend, threads, batch size, context length, full rollout
length and returned-target count. Measure load/cache time separately from warm
forecast time. Include cold-start/compile cost where applicable; report warm p50/
p95 per origin and total run time. Synchronize CUDA around timing if used. Model
batching across origins must not expose later observations to earlier origins.

Peak process RAM and peak allocated/reserved CUDA memory are useful if practical;
label tools/sampling limitations. Use a small measurement wrapper, not an MLOps
system. Prefer batch-one CPU for accessibility, with a separately labelled GPU
track if available. Do not compare a TSFM's forward-only GPU throughput with the
entire LightGBM/data-pipeline wall time. Preserve native default transformations
explicitly; TiRex's differencing/augmentation and t0's optional two-pass settings
affect both compute and forecast values.

## Architecture and eventual dependency strategy

**No core interface/backtester/evaluator modification is required for the selected
PV-only models.** An adapter owns model loading and array conversion, receives only
`ForecastContext`, reconstructs its UTC grid and bridge, runs the pinned model,
selects P50 on `ForecastRequest.target_times`, and returns the existing `Forecast`.
It must never access the full source dataset, scoring actuals or hidden future
history. Core result validation and failed-origin behavior remain intact.

Quantile outputs can be reduced to P50 within the adapter for this deterministic
benchmark. Retaining/reliably evaluating full probability output will require a
separate future result contract. Future weather is also separate: current history
projects only PV, and target features cover requested timestamps rather than the
whole bridge. NWP past/future/vintage metadata therefore needs an explicit later
contract decision; native model covariate support does not solve data availability.
No such interface change is proposed or implemented now.

Prefer eventual **per-model extras** such as `solarcast-belgium[chronos]`, `[t0]`
and `[tirex]`, with lazy imports and actionable errors. These are proposals, not
currently available extras. Keep base/baseline/LightGBM installation free of Torch.
Avoid a default `[tsfm]` bundle: it increases download size and couples incompatible
Torch/NumPy/Transformers stacks. A convenience bundle could follow only after a
tested compatible set exists. Use separate environments when pins conflict.

Resolve against SolarCast's current NumPy/pandas ranges before adding extras; a
declared compatible range is not a tested installation. Pin exact runtime versions
and checkpoint revisions in experiment metadata. Torch CPU/CUDA wheel choice is
platform-specific and should remain documented user setup, not a hard-coded CUDA
dependency. Weights remain external, explicitly downloaded/cached when authorized
when explicitly requested, and never bundled in the SolarCast distribution or Git.
Base CI must remain offline; optional integration tests can use controlled fakes
without downloading weights during normal tests.

## Inspected revisions and source register

These are the revisions returned by publisher APIs on the research date, not
installed packages or downloaded checkpoints. Recheck metadata and pin explicitly
before implementation, rather than treating moving `main` URLs as immutable.

| Selected artifact | Checked revision | Weight bytes from Hub metadata |
|---|---|---:|
| amazon/chronos-2 | `29ec3766d36d6f73f0696f85560a422f50e8498c` | 477930472 |
| theforecastingcompany/t0-gamma | `fef67abad3eed4f227c7b536a6e8ee4e9ab1080d` | 1022492908 |
| NX-AI/TiRex-2 | `352636173653592b6dceb3fbeee9224ca97952a9` | 330038664 |

Checked source heads: Chronos `1224fafa91e1ecd6674f5204066b159e40b5e4a1`,
tfc-t0 `a83ca7669cb1acd6be9e57f0b22cee8722874fbc`, TiRex-2
`e45ec1057391344821f80725ab56e2cb6596759f`. Metadata requests enumerated file sizes;
they did not request `.safetensors` or `.onnx` contents. The model descriptions,
license files and source/config text were read only, not executed.

### Chronos official sources

- **C1:** [Repository, family/usage and code licence](https://github.com/amazon-science/chronos-forecasting), [Apache LICENSE](https://github.com/amazon-science/chronos-forecasting/blob/main/LICENSE).
- **C2:** [Official model card and weight-license declaration](https://huggingface.co/amazon/chronos-2).
- **C3:** [Package requirements](https://github.com/amazon-science/chronos-forecasting/blob/main/pyproject.toml), [official quickstart documentation](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb).
- **C4:** [Exact checkpoint configuration](https://huggingface.co/amazon/chronos-2/blob/main/config.json).
- **C5:** [Prediction, quantiles, frequency and context implementation](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py).
- **C6:** [Chronos-2 technical report, including training/energy tables](https://arxiv.org/abs/2510.15821), [synthetic-only variant card](https://huggingface.co/autogluon/chronos-2-synth).

### t0 official sources

- **T1:** [Repository/documentation/code LICENSE](https://github.com/theforecastingcompany/tfc-t0), [code licence](https://github.com/theforecastingcompany/tfc-t0/blob/main/LICENSE).
- **T2:** [Gamma model card, current requirements and benchmark results](https://huggingface.co/theforecastingcompany/t0-gamma).
- **T3:** [Actual gamma weight LICENSE](https://huggingface.co/theforecastingcompany/t0-gamma/blob/main/LICENSE), [NOTICE](https://huggingface.co/theforecastingcompany/t0-gamma/blob/main/NOTICE).
- **T4:** [Package/runtime requirements](https://github.com/theforecastingcompany/tfc-t0/blob/main/pyproject.toml), [managed API documentation](https://docs.retrocast.com/documentation).
- **T5:** [Native prediction/context/postprocessing implementation](https://github.com/theforecastingcompany/tfc-t0/blob/main/t0/forecaster.py), [gamma config](https://huggingface.co/theforecastingcompany/t0-gamma/blob/main/config.json).
- **T6:** [Technical report: alpha/beta training, context and energy studies](https://arxiv.org/abs/2609.24559).
- **T7:** [Reproducible alpha backend measurements, not gamma timings](https://github.com/theforecastingcompany/tfc-t0/blob/main/BENCHMARKS.md).
- **T8:** [Dated gamma release announcement and launch scores](https://www.theforecastingcompany.com/blog/t0-gamma/).

### TiRex official sources

- **X1:** [Repository and documentation entry point](https://github.com/NX-AI/tirex-2), [code LICENSE](https://github.com/NX-AI/tirex-2/blob/main/LICENSE).
- **X2:** [General public model card](https://huggingface.co/NX-AI/TiRex-2), [weight LICENSE](https://huggingface.co/NX-AI/TiRex-2/blob/main/LICENSE).
- **X3:** [General-checkpoint config](https://huggingface.co/NX-AI/TiRex-2/blob/main/config.json), [different GIFT-ZS config](https://huggingface.co/NX-AI/TiRex-2-gifteval-zs/blob/main/config.json).
- **X4:** [Forecasting/padding/augmentation API](https://nx-ai.github.io/tirex-2/how-to/forecasting/), [package requirements](https://github.com/NX-AI/tirex-2/blob/main/pyproject.toml).
- **X5:** [Official ONNX format and limits](https://nx-ai.github.io/tirex-2/how-to/onnx/).
- **X6:** [Windows/CUDA/compiler FAQ](https://nx-ai.github.io/tirex-2/faq/).
- **X7:** [Benchmark checkpoint/protocol distinctions](https://nx-ai.github.io/tirex-2/benchmarks/).
- **X8:** [Technical report and corpus/exclusion details](https://arxiv.org/abs/2607.01204).

### Other primary sources

- **S1:** [Timestral repository/docs and code LICENSE](https://github.com/mistralai/timestral).
- **S2:** [Timestral-209M card: limits, outputs and comparison tables](https://huggingface.co/mistralai/Timestral-209M).
- **S3:** [Actual Modified MIT weight licence](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE).
- **S4:** [Timestral technical report: post-training, benchmarks and timing methodology](https://arxiv.org/abs/2610.37585).
- **S5:** [Official model/product documentation](https://docs.mistral.ai/models/timestral-1-0-123m), [release announcement](https://mistral.ai/news/timestral/).
- **F1:** [TimesFM repository: latest/archived versions and licensing distinction](https://github.com/google-research/timesfm).
- **F2:** [3.0 model card](https://huggingface.co/google/timesfm-3.0-pytorch), [3.0 NC weight licence](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE).
- **F3:** [TimesFM code Apache LICENSE](https://github.com/google-research/timesfm/blob/master/LICENSE), [package metadata](https://github.com/google-research/timesfm/blob/master/pyproject.toml).
- **F4:** [2.5 model card, Apache declaration and usage](https://huggingface.co/google/timesfm-2.5-200m-pytorch).
- **F5:** [Original TimesFM architecture paper; not a 3.0-specific report](https://arxiv.org/abs/2310.10688).
- **M1:** [Moirai 2.0 card and NC licence declaration](https://huggingface.co/Salesforce/moirai-2.0-R-small).
- **M2:** [Uni2TS repository/docs](https://github.com/SalesforceAIResearch/uni2ts), [2.0 wrapper](https://github.com/SalesforceAIResearch/uni2ts/blob/main/src/uni2ts/model/moirai2/forecast.py), [checkpoint config](https://huggingface.co/Salesforce/moirai-2.0-R-small/blob/main/config.json).
- **M3:** [Moirai 2.0 technical report](https://arxiv.org/abs/2511.11698).
- **M4:** [Uni2TS dependency pins](https://github.com/SalesforceAIResearch/uni2ts/blob/main/pyproject.toml).
- **O1:** [Toto 2.0 current repository documentation](https://github.com/DataDog/toto/tree/main/toto2).
- **O2:** [22M model card and family sizes](https://huggingface.co/Datadog/Toto-2.0-22m).
- **O3:** [Current package metadata](https://github.com/DataDog/toto/blob/main/toto2/pyproject.toml), [model configuration/imputation defaults](https://github.com/DataDog/toto/blob/main/toto2/toto2/configuration.py).
- **O4:** [Toto 2.0 report](https://arxiv.org/abs/2605.20119), [publisher release and training disclosure](https://www.datadoghq.com/blog/ai/toto-2/).
- **I1:** [Granite r2 model card/config](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2).
- **I2:** [Actual dual-license declaration](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2/blob/main/LICENSE).
- **I3:** [Granite TSFM repository/code](https://github.com/ibm-granite/granite-tsfm).
- **I4:** [IBM release, architecture and inference guidance](https://huggingface.co/blog/ibm-research/ibm-releases-sota-granite-time-series), [publisher hardware estimates](https://www.ibm.com/granite/playground/model-selector/models).
- **W1:** [EXAONE official repository: general versus released domain models](https://github.com/LGAI-Research/EXAONE-Forecast).
- **W2:** [QiYao-M publisher model card](https://huggingface.co/DecisionIntelligence/QiYao-M).
- **W3:** [Vendor comparison mentioning Lingjiang; not a verified public release](https://www.theforecastingcompany.com/blog/t0-gamma/).

## Verification, assumptions and remaining uncertainty

The source investigation used offline contract/leakage/DST checks and left the
original LightGBM source/configuration/artifact identities unchanged. Research
claims below remain distinct from later native-integration evidence.

Assumptions: local/open-weight inference is preferred to a required paid API;
ordinary CPU access and commercial reuse clarity matter; no forecast family is
assumed superior for national PV. Unknowns include exact gamma long-context and
corpus limits, independent Elia pretraining exclusion, native Windows behavior
until smoke-tested, package resolution across the full supported Python range,
actual RAM/VRAM/speed, checkpoint-specific imputation and probabilistic calibration.
Provider model cards and licenses can change; pin and recheck before use.


## Chronos integration evidence — 2026-11-14

The dated source investigation above is preserved. The [Chronos-2 adapter](chronos.md)
uses the pinned revision and optional `chronos-forecasting` 2.3.2. Native inference passed on Windows 11,
Python 3.12.4, PyTorch 2.15.0+cpu and Transformers 5.23.0, including interior and
trailing NaN slots. A two-origin June Elia smoke run produced 192 finite/scorable
predictions through the unchanged backtester/evaluator. This resolves the narrow
Windows CPU integration question for that environment; it establishes neither
accuracy nor support across every advertised runtime/hardware combination.
GPU inference, memory profiling, pretraining exclusion and broader benchmarking
remain unverified. See the adapter document for exact commands and timing scope.

## t0 integration evidence — 2026-11-14

The selected [t0-gamma adapter](t0.md) now uses the original pinned checkpoint
`fef67abad3eed4f227c7b536a6e8ee4e9ab1080d` and optional `tfc-t0[torch]` 0.6.1.
Reverification found no material change to the selected model, licenses or API.
The package's Python floor remains >=3.11; older indexed card examples showing
>=3.10 or package >=0.6.0 do not override the current package/checkpoint floor.
Code and the actual weight LICENSE remain Apache-2.0.

Windows 11 CPU inference passed with Python 3.12.4/Torch 2.15.0+cpu, including
interior/trailing NaNs. Installation reused the existing Torch/Hub environment,
added small Python runtime dependencies, and required no compiler/WSL workaround.
The upstream package still lists Linux/macOS rather than promising Windows support.
Gamma's local loading took 3.119 seconds; the technical Elia smoke's short and
two day-ahead calls took 0.304 and 0.319/0.315 seconds. These cached single-run
measurements exclude startup/download and establish no comparative performance.
Two origins produced 192 finite/scorable predictions through the unchanged
backtester/evaluator, with raw adjustments disabled. The native 1024-step block
covers the complete 146/150/154-step rollouts; no gamma-specific hard context
maximum was newly established. Controlled accuracy, long-context studies, GPU
validation, memory profiling and pretraining exclusion remain unresolved.
