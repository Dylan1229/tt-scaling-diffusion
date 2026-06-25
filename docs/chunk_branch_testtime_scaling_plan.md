# Chunk-Branch Test-Time Scaling Plan

## Goal

Verify whether last-frame I2V long-video generation benefits from test-time
scaling over chunk seeds.

The experiment fixes the first chunk and branches only the continuation chunks.
For each fixed `(prompt, root_seed)`, chunk 0 is generated once and reused for
all paths. For each later chunk, generate two candidate continuations. With 6
total chunks, chunks 1-5 branch, so each fixed chunk 0 produces `2^5 = 32`
final 30s videos.

## Fixed Scope

- Prompts: `p01`, `p02`
- Root seeds per prompt: `0`, `1`
- Fixed roots: `2 prompts x 2 root seeds = 4`
- Final branch videos: `4 roots x 32 paths = 128`
- Video format: `6 x 81` frames stitched with 1-frame overlap, `481` frames at
  16 fps, same as the previous VBench-Long 30s setting.
- Model/settings: same Wan 2.2 5B I2V config as
  `configs/long_wan22_480p_i2v.yaml`.

## Interpretation

I understand the requested experiment as:

1. Do not change chunk 0.
   - For each `(prompt, root_seed)`, generate/cache chunk 0 exactly once with
     the existing T2V path and the root seed.
   - Every branch path for that root starts from the same chunk 0 and the same
     chunk-0 last frame.
2. For every continuation chunk, choose between two seed branches.
   - Chunk 1 has 2 candidates.
   - Chunk 2 has 4 candidates because each chunk-1 candidate gets 2 children.
   - Chunk 3 has 8 candidates.
   - Chunk 4 has 16 candidates.
   - Chunk 5 has 32 candidates.
3. Save all 32 final paths per fixed first chunk, then evaluate whether the
   best path is better than a single fixed-seed I2V continuation.

## Compute Accounting

For one fixed first chunk:

- Naive independent generation: `32 paths x 5 I2V chunks = 160` continuation
  chunk calls.
- Prefix-shared tree generation: `2 + 4 + 8 + 16 + 32 = 62` continuation chunk
  calls.
- Equivalent ordinary I2V 30s continuations: `62 / 5 = 12.4` per root.

For this experiment:

- Fixed roots: `4`
- T2V chunk-0 calls: `4`
- I2V continuation chunk calls: `4 x 62 = 248`
- Equivalent ordinary I2V 30s continuations: `248 / 5 = 49.6`
- Final videos to score: `128`

This is the exact cached tree count. The user's `2 x 2 x 32 / 2 ~= 64`
full-path estimate is a reasonable conservative order-of-magnitude estimate.

## Branch Seed Rule

Keep the current deterministic seed schedule on the all-zero path so it remains
directly comparable to the existing last-frame I2V baseline.

For continuation chunk `chunk_idx in 1..5`:

```text
branch_id = integer value of the path bits through this chunk
chunk_seed = root_seed + chunk_idx * chunk_seed_stride + branch_id
chunk_seed_stride = 1_000_000
```

Examples for `root_seed=0`:

- path prefix `0` at chunk 1: `1_000_000`
- path prefix `1` at chunk 1: `1_000_001`
- path prefix `00` at chunk 2: `2_000_000`
- path prefix `01` at chunk 2: `2_000_001`
- path prefix `10` at chunk 2: `2_000_002`
- path prefix `11` at chunk 2: `2_000_003`

The all-zero path `00000` therefore uses exactly the current continuation seed
schedule:

```text
root_seed + 1_000_000
root_seed + 2_000_000
root_seed + 3_000_000
root_seed + 4_000_000
root_seed + 5_000_000
```

## Output Layout

Use a separate run root so the previous comparison outputs remain untouched:

```text
runs/baseline_long/chunk_branch_i2v_p01_p02_s0_1/
  p01/
    seed000000_path00000/
      video.mp4
      meta.json
      DONE
    ...
    seed000001_path11111/
      video.mp4
      meta.json
      DONE
  p02/
    ...
```

Metadata should include:

- `generation_mode: "last_frame_i2v_chunk_branch_tree"`
- `root_seed`
- `path_bits`
- `path_id`
- `branch_factor: 2`
- `fixed_first_chunk: true`
- `chunk_seeds`
- `parent_path_bits` for each continuation chunk

Also write:

```text
runs/baseline_long/chunk_branch_i2v_p01_p02_s0_1/branch_manifest.csv
```

with at least:

```text
prompt_id,prompt_text,root_seed,path_bits,path_id,seed_idx,video_path
```

## Minimal Implementation Plan

1. Add a separate generator module, not a large change to the existing long
   runner:

   ```text
   ttsd/runners/generate/chunk_branch_i2v.py
   ```

   This keeps the current `long_video.py` modes stable.

2. Add a small config:

   ```text
   configs/chunk_branch_i2v_p01_p02_s0_1.yaml
   ```

   It should mirror `configs/long_wan22_480p_i2v.yaml`, but add:

   ```yaml
   branch:
     prompt_ids: [p01, p02]
     root_seeds: [0, 1]
     branch_factor: 2
     branch_chunks: [1, 2, 3, 4, 5]
   ```

3. Generation algorithm per `(prompt, root_seed)`:

   - Generate chunk 0 once with T2V.
   - Store chunk 0 in memory and optionally save `chunks/chunk_000.mp4`.
   - For `chunk_idx = 1..5`, expand all current prefix states:
     - For each prefix, generate 2 I2V children from the prefix's last frame.
     - Cache each child chunk and its last frame.
   - After chunk 5, stitch chunk 0 plus the five continuation chunks for each
     of the 32 leaf paths.
   - Save one `video.mp4` and `meta.json` per leaf path.

4. Run generation on GPUs 4-7 with one fixed root per GPU:

   - GPU 4: `p01, root_seed=0`
   - GPU 5: `p01, root_seed=1`
   - GPU 6: `p02, root_seed=0`
   - GPU 7: `p02, root_seed=1`

5. Score all 128 videos with VBench-Long.

   The existing evaluator assumes an integer `meta["seed"]`, so either:

   - set `meta["seed"]` to a numeric `seed_idx = root_seed * 1000 + path_id`
     and use `branch_manifest.csv` to recover `(root_seed, path_bits)`, or
   - minimally extend the evaluator CSV to preserve optional metadata fields.

   Prefer the manifest approach first because it is less invasive.

6. Summarize test-time scaling:

   Per `(prompt, root_seed)`:

   - all-zero path score: baseline-compatible fixed continuation
   - min / mean / max / std over the 32 branch paths
   - best-of-32 score by VBench-Long overall mean
   - best overall path bits
   - best-of-32 gain over all-zero path
   - best-of-32 gain over mean/random path
   - independent-concat baseline score for the same `(prompt, root_seed)`
   - best-of-32 gain over independent concat

   Also report per-dimension best and mean for:

   - subject consistency
   - background consistency
   - motion smoothness
   - dynamic degree
   - aesthetic quality
   - imaging quality

## Comparison Targets

For each fixed `(prompt, root_seed)`, compare the 32 branch paths against two
baselines:

1. `all_zero_path`
   - Path bits `00000`.
   - Uses the same chunk seed schedule as the current last-frame I2V baseline.
   - This checks whether branching helped over a single deterministic I2V path.
2. `independent_concat`
   - Existing independent T2V chunk baseline from
     `runs/vbench_long/long_vbench_30s_20260618_190202`.
   - Compare only the same prompts and root seeds: `p01,p02` x `0,1`.
   - This answers whether the best branch path can beat independent concat.

Report two kinds of "best":

1. Overall-best path:
   - Compute each path's `overall_mean` as the mean of the six VBench-Long
     dimensions.
   - Select the path with highest `overall_mean`.
   - Compare that single path's six metric values against independent concat.
   - This is the primary fair best-of-32 result because one path is selected.
2. Per-metric oracle:
   - For each dimension separately, report the max score among the 32 paths.
   - Compare each per-metric max against independent concat.
   - This shows the upper envelope, but different metrics may choose different
     paths, so it is an oracle and not a single deployable video.

For each metric and root, write:

```text
branch_min
branch_mean
branch_max
branch_std
all_zero
overall_best_path_value
overall_best_path_bits
per_metric_best_value
per_metric_best_path_bits
independent_concat
max_minus_independent
overall_best_minus_independent
```

The aggregate table should also include:

- fraction of the 4 roots where branch max beats independent concat per metric
- fraction of the 4 roots where the overall-best path beats independent concat
  per metric
- average gain/loss versus independent concat per metric

## Expected Outputs

```text
runs/baseline_long/chunk_branch_i2v_p01_p02_s0_1/
runs/vbench_long/chunk_branch_i2v_p01_p02_s0_1/
runs/vbench_long_compare/chunk_branch_i2v_p01_p02_s0_1/
```

Important CSVs:

```text
branch_manifest.csv
vbench_long_scores_long.csv
chunk_branch_summary.csv
chunk_branch_by_root.csv
chunk_branch_by_path.csv
```

## Checks

- Static:
  - `python -m compileall` for new Python files
  - `bash -n` for any new launch script
- Generation:
  - exactly 128 final `DONE` markers
  - each final video has 481 frames
  - for each `(prompt, root_seed)`, all 32 paths share the same chunk-0 seed and
    same chunk-0 metadata
  - path `00000` uses the current deterministic continuation seeds
- Evaluation:
  - VBench-Long rows: `128 videos x 6 dimensions = 768`
  - raw results: expected `64 seed_idx groups x 6 dimensions = 384` if using
    `seed_idx = root_seed * 1000 + path_id` and staging both prompts together
  - branch summary includes best-of-32 gains for each of the 4 fixed roots

## Non-Goals For This First Check

- No training.
- No learned selector yet.
- No online pruning yet.
- No changes to chunk 0 across branch paths.
- No full 15-prompt sweep until this 4-root pilot shows whether best-of-32
  actually improves VBench-Long.
