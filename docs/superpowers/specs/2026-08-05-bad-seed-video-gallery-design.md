# Bad Seed Video Gallery

## Goal

Create a private, shareable Sites page for reviewing the 22 high-confidence Bad Seeds from the 360-video baseline set. Each failure is paired with the strongest visibly acceptable Seed from the same Prompt so the failure is immediately understandable.

## Audience and use

The page is for Peihao and the research team to review rescue candidates and use them in meeting discussions. It should prioritize rapid visual judgment over metric-heavy analysis.

## Page design

- A concise summary at the top explains that the gallery contains 22 seed-specific failures selected by visual review.
- Each comparison card shows the Bad Seed and one normal reference Seed side by side.
- Both videos have native playback controls and can be played independently.
- Each card includes the Prompt, Seed identifiers, and one short failure label.
- Filters cover object omission, spatial or semantic mismatch, structural collapse, action mismatch, and Prompt drift.
- The layout remains usable on desktop and mobile.

## Content rules

- Include only high-confidence, seed-specific failures in the main gallery.
- Exclude Prompt-wide failures where all three Seeds fail, because those are weaker rescue candidates.
- Do not use VBench score as the primary ordering signal; several obvious semantic failures score highly.
- Order the gallery by failure severity and clarity.

## Access and success criteria

- Publish privately through Sites.
- The deployed page must load all 44 videos and allow direct playback.
- Every Bad Seed must be matched to the correct Prompt and a visibly better reference Seed.
- The page should be understandable without reading experiment implementation details.

