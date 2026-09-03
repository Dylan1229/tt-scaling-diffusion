# Decision-Diameter Calibration Protocol Design

## Goal

Turn the existing fixed-radius noise-neighborhood demo into a reusable calibration protocol that measures a decision diameter for every prompt, input image, and chosen parent noise.

The protocol must answer two questions before a local-neighborhood result is interpreted:

1. How close is the nearest observed semantic decision boundary to this parent?
2. At what distance have at least half of the sampled directions crossed a semantic boundary?

Semantic judgments remain manual. The software plans reproducible samples, generates them, records distances, consumes human labels, and reports bounded estimates; it does not decide whether a video succeeds.

## Operational definition

A calibration condition consists of one fixed model, scheduler, prompt, input image, generation configuration, parent seed, and human-written semantic criterion. Only initial Gaussian noise changes.

For parent noise `z_parent`, direction noise `epsilon_j`, and mixing coefficient `alpha`, generate:

`z(alpha, j) = sqrt(1 - alpha^2) * z_parent + alpha * epsilon_j`

Each direction `j` has one deterministic perturbation seed. The same `epsilon_j` is reused at every tested `alpha`, creating a reproducible radial path instead of unrelated samples at each shell.

Valid `alpha` values satisfy `0 < alpha <= 1`. At `alpha = 1`, the sample is exactly `epsilon_j`, independent of the parent. This provides a finite endpoint for the search.

The primary radial distance is normalized per-element RMS distance in noise space. Reports also include `alpha` and cosine similarity. For an ideal high-dimensional Gaussian pair, the expected RMS radius is:

`r(alpha) = sqrt(2 - 2 * sqrt(1 - alpha^2))`

The protocol uses the following operational measurements:

- **Directional crossing bracket:** for one direction, the interval between the largest tested radius retaining the parent's label and the first larger tested radius with the opposite definitive label.
- **Nearest decision radius:** the smallest observed directional crossing radius. Its estimate is reported as a lower/upper bracket, not as an exact point.
- **Typical decision radius (`r50`):** the smallest radius by which at least half of the sampled directions have shown a first label flip.
- **Nearest decision diameter:** twice the nearest decision radius.
- **Typical decision diameter (`D50`):** twice `r50`; this is the main cross-prompt comparison value.

These are sampled, operational diameters, not claims that a semantic region is spherical. Directional brackets and censored directions must always accompany the aggregate values.

## Considered approaches

### Fixed dense radius grid

Generate many independent samples at every radius. This is simple, but expensive and unable to localize a boundary along a direction because every shell uses different noise directions.

### Shared-direction adaptive scan — selected

Reuse fixed directions across radii, expand in coarse shells, and refine only brackets that contain a label flip. This directly measures directional crossings and avoids generating dense samples far from relevant boundaries.

### Automated nearest-boundary optimization

Use an optimizer or adversarial search to minimize semantic flip distance. This requires a trusted automatic semantic scorer, conflicts with the manual-review requirement, and can optimize scorer errors. It is out of scope.

## Calibration protocol

### 1. Fix and validate the condition

Record the prompt, input hash, model, scheduler, dimensions, frame count, inference steps, guidance, frame rate, parent seed, parent label, and a plain-language semantic criterion. Capture the parent's exact initial noise and verify explicit reinjection reproduces the parent video.

A changed fixed setting requires a new experiment root. Existing outputs must never be silently reused under a different condition.

### 2. Create fixed directions

Use eight directions by default, with deterministic seeds `10000` through `10007`. Direction count and seeds are recorded in the scan configuration. Every radius uses these same direction seeds.

Eight directions provide a bounded descriptive calibration consistent with the existing experiments. They do not justify population-level confidence claims.

### 3. Expand through coarse shells

Test shells in this order:

`0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00`

Generate one shell at a time and manually label every video as the parent label, the opposite label, or ambiguous. Continue outward until either:

- at least half of the directions have shown a definitive label flip, or
- `alpha = 1` has been reviewed.

The protocol may stop coarse expansion once `r50` is bracketed. It does not generate unused outer shells merely because they were listed in the schedule.

### 4. Refine observed crossings

For every direction needed to determine the nearest radius or `r50`, add the midpoint of its current parent/opposite bracket. Manually label the new sample and repeat until the bracket width is at most `0.02` in `alpha`.

Ambiguous labels do not count as either side of a boundary. They block refinement for that interval until manually resolved. If a direction flips and later returns to the parent label, retain the first observed crossing and mark the direction non-monotonic.

### 5. Report bounded results

The final report contains:

- parent condition and semantic criterion;
- all tested radii and counts by label;
- each direction's first-crossing bracket or censored status;
- nearest radius and diameter intervals;
- `r50` and `D50` intervals, or a censored result;
- actual RMS distance and cosine similarity for boundary samples;
- ambiguous and non-monotonic cases;
- number of directions and refinement tolerance.

If no direction flips by `alpha = 1`, both diameters are reported as unobserved under the sampled directions. If some directions flip but fewer than half do, the nearest diameter is finite while `D50` is right-censored.

## Data and workflow

Each calibration experiment has:

- an immutable scan configuration containing fixed generation settings, direction seeds, coarse schedule, tolerance, parent label, and semantic criterion;
- one validated parent bundle;
- an append-only sample manifest;
- generated noise tensors, videos, all-frame review sheets, and metadata;
- manually maintained labels;
- a machine-produced decision profile and next-sample plan.

Every sample specification includes a stable sample ID, direction index, `alpha`, perturbation seed, and global index. IDs encode `round(alpha * 10000)` padded to five digits, for example `d03_a02000` at `alpha = 0.2` and `d03_a10000` at `alpha = 1`. Refinement plans therefore remain unique and reproducible.

The workflow is iterative:

1. The planner emits the next shell or refinement samples.
2. A single preparation process validates and appends those samples atomically.
3. Generation may shard the newly planned samples across GPUs.
4. A human reviews every completed video and records labels.
5. The analyzer validates label completeness, updates diameter brackets, and emits either the next plan or a final profile.

Only the planning step may append to the manifest. Parallel generation starts after that atomic append, avoiding concurrent manifest writers.

## Implementation shape

Reuse the existing parent capture, variance-preserving perturbation, UniPC generation, atomic output, contact-sheet, resume, and sharding code.

Extend the existing runner minimally so that it can:

- accept explicit append-only sample plans;
- preserve one parent bundle while adding later shells or refinements;
- store and validate `direction_index`;
- reuse a perturbation seed across multiple radii;
- accept `alpha = 1`;
- generate only newly planned or explicitly selected samples;
- preserve the legacy fixed-grid behavior for existing experiments.

Add one dependency-free planning and analysis module. It produces coarse plans, validates manual labels against the manifest, computes crossing brackets and diameter summaries, flags ambiguous/non-monotonic data, and emits refinement plans. It must not import or run the video model.

No new third-party dependencies are needed.

## Validation and error handling

Reject:

- invalid, unsorted, or duplicate radii;
- `alpha` outside `(0, 1]`;
- duplicate sample IDs;
- a direction whose perturbation seed changes across radii;
- an appended sample that conflicts with an existing manifest entry;
- stale prompt, image, parent, model, or generation settings;
- labels for unknown samples;
- missing labels for a shell declared ready for analysis.

Reapplying an identical plan is a no-op. Interrupted sample preparation or generation remains resumable. A completion marker is written only after all required artifacts exist.

## Test strategy

Remote CPU-only tests cover:

- exact and reproducible neighbor construction, including `alpha = 1` producing `epsilon`;
- shared perturbation seeds across radii for each direction;
- deterministic unique sample IDs and sharding;
- append, idempotent replay, and conflict rejection for sample plans;
- stale fixed-setting validation after manifest growth;
- synthetic label sets for nearest diameter, `D50`, censored directions, ambiguous labels, and non-monotonic flips;
- deterministic midpoint refinement and the `0.02` stopping tolerance;
- all existing neighborhood-runner tests.

Model execution is not needed for these tests. After tests pass, the first end-to-end validation uses the simplified explicit flower prompt and its failed seed 0 parent.

## Out of scope

- Automatic semantic scoring
- Optimizer-based or gradient-based boundary attacks
- Claims that semantic regions are spherical
- Statistical confidence intervals from eight directions
- Changing model, scheduler, prompt, image, or generation settings during one calibration
- Rewriting or relabeling previous experiment artifacts
