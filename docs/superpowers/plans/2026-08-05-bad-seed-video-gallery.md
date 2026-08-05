# Bad Seed Video Gallery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and privately publish a one-page Sites gallery containing 22 directly playable Bad Seed videos, each paired with a visibly better Seed from the same Prompt.

**Architecture:** Create an isolated Sites project under `sites/bad-seed-gallery`. Keep the review data in one JSON manifest, copy the selected MP4 files into static assets, and render a responsive client-side gallery with category filters and native video controls. Validate the manifest, asset count, and file presence before building and publishing.

**Tech Stack:** Sites vinext starter, React, TypeScript, CSS, static MP4 assets, Node validation script.

## Global Constraints

- Publish privately through Sites.
- Include exactly 22 high-confidence, seed-specific failures and exactly one better same-Prompt reference for each failure.
- Exclude Prompt-wide failures where all three Seeds fail.
- Do not rank by VBench score; order by visual severity and clarity.
- Both videos in every comparison must have direct playback controls.
- The page must be understandable without experiment implementation details.

---

### Task 1: Create the gallery project and validated content manifest

**Files:**
- Create: `sites/bad-seed-gallery/` using the Sites initializer
- Create: `sites/bad-seed-gallery/data/gallery.json`
- Create: `sites/bad-seed-gallery/scripts/validate-gallery.mjs`
- Modify: `sites/bad-seed-gallery/package.json`

**Interfaces:**
- Consumes: baseline videos under `runs/baseline/prompt_dependency_v1_baseline_5b_20260728_001/<prompt_id>/seed<NNNN>/video.mp4`
- Produces: `gallery.json`, an array of `{ id, prompt, badSeed, referenceSeed, category, issue, badVideo, referenceVideo }` objects

- [ ] **Step 1: Initialize the Sites project**

Run the Sites initializer once with `sites/bad-seed-gallery` as its target, retain the setup session until installation completes, then start the development preview and open its printed local URL once.

- [ ] **Step 2: Write the failing manifest validator**

Create `scripts/validate-gallery.mjs` that reads `data/gallery.json`, asserts exactly 22 unique entries, checks that every `badSeed` differs from `referenceSeed`, requires the five allowed category values, and verifies that both referenced files exist under `public/`.

Add this package script:

```json
"validate:gallery": "node scripts/validate-gallery.mjs"
```

- [ ] **Step 3: Run the validator and verify failure**

Run: `npm run validate:gallery`

Expected: failure because `data/gallery.json` and the video assets do not exist yet.

- [ ] **Step 4: Create the exact gallery manifest**

Create these 22 mappings in severity order:

| Bad | Reference | Category | Issue |
|---|---|---|---|
| `pd041/s1` | `pd041/s0` | structural-collapse | objects disappear; road-only output |
| `pd065/s0` | `pd065/s2` | structural-collapse | harp fails to form |
| `pd066/s0` | `pd066/s1` | structural-collapse | wrestling scene fails to form |
| `pd077/s2` | `pd077/s1` | structural-collapse | swimmer is heavily distorted |
| `pd108/s2` | `pd108/s1` | prompt-drift | person replaces the Bund view |
| `pd111/s2` | `pd111/s1` | prompt-drift | unrelated person replaces the skyline |
| `pd112/s2` | `pd112/s1` | prompt-drift | foreground person obscures the skyline |
| `pd026/s0` | `pd026/s1` | object-omission | dog is missing |
| `pd029/s1` | `pd029/s0` | object-omission | cow is missing |
| `pd030/s2` | `pd030/s0` | object-omission | elephant is missing |
| `pd034/s0` | `pd034/s2` | object-omission | bird is missing |
| `pd043/s1` | `pd043/s2` | object-omission | motorcycle is missing |
| `pd046/s2` | `pd046/s1` | object-omission | stop sign is missing |
| `pd049/s2` | `pd049/s0` | object-omission | bench is missing |
| `pd050/s0` | `pd050/s1` | object-omission | bicycle is missing |
| `pd051/s1` | `pd051/s0` | object-omission | bird is missing |
| `pd053/s1` | `pd053/s0` | object-omission | dog is missing |
| `pd067/s1` | `pd067/s2` | semantic-mismatch | bicycle replaces scooter |
| `pd075/s0` | `pd075/s1` | action-mismatch | American football replaces soccer shot |
| `pd078/s0` | `pd078/s2` | semantic-mismatch | presentation scene lacks presenter and room context |
| `pd088/s0` | `pd088/s1` | semantic-mismatch | rush-hour traffic is absent |
| `pd116/s0` | `pd116/s1` | semantic-mismatch | generic dark animal replaces panda |

Use the exact Prompt text from `configs/prompt_dependency_v1_prompts.csv`. Use static asset names `/videos/<prompt_id>-bad-s<seed>.mp4` and `/videos/<prompt_id>-reference-s<seed>.mp4`.

- [ ] **Step 5: Copy the 44 videos**

Copy each selected source video into `public/videos/` using the manifest naming convention. Preserve the source videos unchanged.

- [ ] **Step 6: Run the validator and verify success**

Run: `npm run validate:gallery`

Expected: exit 0 with `22 comparisons, 44 playable assets validated`.

- [ ] **Step 7: Commit the project foundation**

Commit only the new Sites project files with message `Build bad-seed gallery content set`.

### Task 2: Build the playable comparison experience

**Files:**
- Modify: `sites/bad-seed-gallery/app/page.tsx`
- Modify: `sites/bad-seed-gallery/app/globals.css`
- Modify: `sites/bad-seed-gallery/app/layout.tsx`
- Delete: `sites/bad-seed-gallery/app/_sites-preview/`

**Interfaces:**
- Consumes: the validated `gallery.json` manifest and 44 static MP4 URLs
- Produces: a one-route responsive comparison gallery with category filters and native video controls

- [ ] **Step 1: Replace the starter page**

Render a Chinese-language page headed `Bad Seed Rescue Gallery`, with a short explanation that the set contains 22 high-confidence seed-specific failures from 360 baseline videos.

- [ ] **Step 2: Add the filter model**

Provide filter chips for `全部`, `对象缺失`, `语义不匹配`, `动作错误`, `结构崩坏`, and `Prompt 跑偏`. The default is `全部`, and the visible count updates with the filter.

- [ ] **Step 3: Render comparison cards**

Each card must show Prompt text, Prompt ID, failure label, Bad Seed on the left with a red status label, reference Seed on the right with a green status label, and native video controls with `preload="metadata"`, `playsInline`, and `muted`.

- [ ] **Step 4: Apply the visual system**

Use a neutral research-review surface, compact typographic hierarchy, strong red/green comparison cues, two-column video comparison on desktop, and stacked video comparison on mobile. Avoid dashboard chrome and decorative imagery.

- [ ] **Step 5: Remove starter-only content**

Remove the preview skeleton and temporary metadata marker. Set the site title to `Bad Seed Rescue Gallery` and the description to `Visual review of seed-specific text-to-video failures and better same-prompt references.`

- [ ] **Step 6: Run content validation**

Run: `npm run validate:gallery`

Expected: `22 comparisons, 44 playable assets validated`.

- [ ] **Step 7: Run the production build**

Run: `npm run build`

Expected: exit 0 and Cloudflare-compatible output under `dist/`.

- [ ] **Step 8: Commit the finished page**

Commit the completed page with message `Create playable bad-seed comparison gallery`.

### Task 3: Publish privately with Sites

**Files:**
- Modify: `sites/bad-seed-gallery/.openai/hosting.json`

**Interfaces:**
- Consumes: the successful production build and exact committed source
- Produces: one private deployed Sites URL

- [ ] **Step 1: Create the Sites project**

Create one private Sites project named `Bad Seed Rescue Gallery`, persist its project identifier in `.openai/hosting.json`, and keep the returned source credential private.

- [ ] **Step 2: Save and deploy the validated version**

Push the committed source using the temporary credential, package the exact build with the Sites packaging helper, save one version, and deploy it privately.

- [ ] **Step 3: Wait for deployment completion**

Poll the deployment status until it reports `succeeded` or a terminal failure.

- [ ] **Step 4: Open and hand off the published page**

Open the exact deployed URL in Codex and return that URL as the primary deliverable.
