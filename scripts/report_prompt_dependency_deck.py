"""Build a PDF deck summarizing the prompt-dependency experiment set."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PAGE_W, PAGE_H = 13.333, 7.5
TITLE = "Prompt Dependency Experiment v1"
SUBTITLE = "Step35 branching prompt set: what prompts we use and which seeds we run"
PREVIEW_DIR: Path | None = None
PREVIEW_LIMIT = 0
PAGE_INDEX = 0


def _new_page(pdf: PdfPages, title: str):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.text(0.055, 0.925, title, fontsize=23, fontweight="bold", ha="left", va="top")
    fig.text(0.055, 0.885, "tt-scaling-diffusion / Wan2.2 5B", fontsize=10, color="#666666", ha="left")
    return fig, ax


def _save(pdf: PdfPages, fig) -> None:
    global PAGE_INDEX
    PAGE_INDEX += 1
    if PREVIEW_DIR is not None and PAGE_INDEX <= PREVIEW_LIMIT:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW_DIR / f"page_{PAGE_INDEX:02d}.png", dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _wrap(text: str, width: int = 56) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def _bullet_block(fig, x: float, y: float, lines: list[str], size: int = 14, gap: float = 0.057) -> None:
    for i, line in enumerate(lines):
        fig.text(x, y - i * gap, f"- {line}", fontsize=size, ha="left", va="top")


def _draw_table(
    fig,
    ax,
    frame: pd.DataFrame,
    bbox: list[float],
    font_size: int = 10,
    header_color: str = "#e9eef6",
    scale_y: float = 1.35,
) -> None:
    table = ax.table(
        cellText=frame.values,
        colLabels=frame.columns,
        cellLoc="left",
        colLoc="left",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, scale_y)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#c4ccd8")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")


def _bar_page(pdf: PdfPages, title: str, rows: pd.DataFrame, columns: list[str]) -> None:
    fig, ax = _new_page(pdf, title)
    lefts = [0.08, 0.39, 0.70]
    for i, column in enumerate(columns):
        sub = fig.add_axes([lefts[i], 0.18, 0.23, 0.58])
        counts = rows[column].value_counts().sort_values()
        sub.barh(range(len(counts)), counts.values, color="#2f6f9f")
        sub.set_yticks(range(len(counts)), counts.index, fontsize=9)
        sub.set_xlabel("prompt count", fontsize=9)
        sub.set_title(column, fontsize=12, fontweight="bold")
        sub.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        for spine in sub.spines.values():
            spine.set_visible(False)
        for y, value in enumerate(counts.values):
            sub.text(value + 0.3, y, str(value), fontsize=9, va="center")
    _save(pdf, fig)


def _prompt_pages(pdf: PdfPages, rows: pd.DataFrame) -> None:
    classes = [
        ("static_single_or_scene", "Static Single/Scene Prompts"),
        ("multi_object", "Multi-Object Prompts"),
        ("spatial_relation", "Spatial Relationship Prompts"),
        ("human_action", "Human Action Prompts"),
        ("subject_motion", "Subject Motion Prompts"),
        ("camera_motion", "Camera / Temporal-Style Prompts"),
        ("mixed_story", "Mixed Story Prompts"),
    ]
    for prompt_class, title in classes:
        subset = rows[rows["prompt_class"] == prompt_class].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["axis", "vbench_index", "prompt_id"])
        chunks = [subset.iloc[i : i + 10].copy() for i in range(0, len(subset), 10)]
        for page_index, chunk in enumerate(chunks, start=1):
            fig, ax = _new_page(
                pdf,
                f"{title} ({page_index}/{len(chunks)})",
            )
            table_rows = []
            for _, row in chunk.iterrows():
                table_rows.append(
                    {
                        "id": row["prompt_id"],
                        "axis": row["axis"],
                        "bucket": row["motion_bucket"],
                        "seeds": row["seeds"],
                        "prompt": _wrap(row["prompt_text"], 72),
                    }
                )
            table = pd.DataFrame(table_rows)
            _draw_table(fig, ax, table, bbox=[0.055, 0.105, 0.89, 0.74], font_size=8, scale_y=1.95)
            _save(pdf, fig)


def _seed_pages(pdf: PdfPages, roots: pd.DataFrame) -> None:
    fig, ax = _new_page(pdf, "Seed Plan and Root Manifest")
    n_prompts = roots["prompt_id"].nunique()
    seeds = sorted(roots["root_seed"].unique().tolist())
    _bullet_block(
        fig,
        0.075,
        0.79,
        [
            f"Every prompt uses the same root seeds: {', '.join(map(str, seeds))}.",
            f"Total baseline roots: {n_prompts} prompts x {len(seeds)} seeds = {len(roots)} videos.",
            "For Step35 branching, each root generates 1 batched control plus 4 noisy suffix branches.",
            "Candidate seed IDs follow the existing runner convention: root_seed * 100 + candidate_index.",
            "The actual no-intervention comparison remains the batch-one baseline video, not the batched control.",
        ],
        size=15,
        gap=0.071,
    )
    seed_counts = roots.groupby("root_seed").size().reset_index(name="n_prompts")
    seed_counts["root_seed"] = seed_counts["root_seed"].astype(str)
    _draw_table(fig, ax, seed_counts, bbox=[0.075, 0.16, 0.35, 0.23], font_size=13, scale_y=1.6)
    fig.text(
        0.51,
        0.32,
        "Manifest files\n"
        "configs/prompt_dependency_v1_prompts.csv\n"
        "configs/prompt_dependency_v1_roots_3seeds.csv\n\n"
        "Configs\n"
        "configs/prompt_dependency_baseline_wan22_480p.yaml\n"
        "configs/prompt_dependency_late_branch_s35_m4_wan22_480p.yaml",
        fontsize=12,
        family="monospace",
        ha="left",
        va="top",
        color="#30343b",
    )
    _save(pdf, fig)


def build_deck(prompt_manifest: Path, root_manifest: Path, output_pdf: Path) -> None:
    prompts = pd.read_csv(prompt_manifest)
    roots = pd.read_csv(root_manifest)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        fig, ax = _new_page(pdf, TITLE)
        fig.text(0.055, 0.79, SUBTITLE, fontsize=17, ha="left", va="top")
        _bullet_block(
            fig,
            0.075,
            0.66,
            [
                f"{len(prompts)} prompts from vendored VBench prompt files.",
                f"{len(roots)} roots: each prompt runs seeds 0, 1, and 2.",
                "Prompt taxonomy is stored directly in the prompt module and CSV manifests.",
                "Primary question: which prompt types benefit from late Step35 branching?",
            ],
            size=16,
            gap=0.075,
        )
        fig.text(
            0.055,
            0.09,
            "Generated from configs/prompt_dependency_v1_prompts.csv and configs/prompt_dependency_v1_roots_3seeds.csv",
            fontsize=10,
            color="#666666",
            ha="left",
        )
        _save(pdf, fig)

        fig, ax = _new_page(pdf, "Experiment Setup")
        setup = pd.DataFrame(
            [
                ["Model", "Wan2.2 TI2V 5B"],
                ["Resolution / frames", "480 x 832, 81 frames"],
                ["Denoising steps", "50"],
                ["Branch point", "Step35, after completing denoising step 35"],
                ["Branches", "4 noisy branches + 1 batched control per root"],
                ["Perturbation", "Gaussian noise with std = scheduler_sigma * 0.10"],
                ["Evaluation", "Updated VBench quality + separate dynamic_degree and overall_consistency"],
                ["Analysis", "By prompt taxonomy and baseline quality terciles"],
            ],
            columns=["Field", "Value"],
        )
        _draw_table(fig, ax, setup, bbox=[0.08, 0.18, 0.84, 0.62], font_size=12, scale_y=1.7)
        _save(pdf, fig)

        _bar_page(pdf, "Prompt Distribution: Axes and Categories", prompts, ["axis", "prompt_class", "motion_bucket"])
        _bar_page(pdf, "Prompt Distribution: Structure Tags", prompts, ["subject_count", "relation_class", "complexity_class"])

        fig, ax = _new_page(pdf, "Taxonomy Definitions")
        taxonomy = pd.DataFrame(
            [
                ["prompt_class", "Main hand-labeled prompt family used for headline grouping."],
                ["motion_bucket", "Static prompt, human motion, subject motion, camera motion, or mixed motion."],
                ["subject_count", "Single subject/person, two subjects, multi-subject, or scene-only."],
                ["relation_class", "Whether the prompt explicitly tests co-presence or spatial relation."],
                ["camera_class", "Temporal-style camera cue such as zoom, pan, tilt, shake, steady, or slow motion."],
                ["complexity_class", "Simple, moderate, or complex grouping for coarse analysis."],
            ],
            columns=["Field", "Meaning"],
        )
        _draw_table(fig, ax, taxonomy, bbox=[0.065, 0.22, 0.87, 0.52], font_size=12, scale_y=1.95)
        _save(pdf, fig)

        _seed_pages(pdf, roots)
        _prompt_pages(pdf, prompts)


def main(argv: list[str] | None = None) -> None:
    global PREVIEW_DIR, PREVIEW_LIMIT, PAGE_INDEX
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-manifest", type=Path, default=Path("configs/prompt_dependency_v1_prompts.csv"))
    parser.add_argument("--root-manifest", type=Path, default=Path("configs/prompt_dependency_v1_roots_3seeds.csv"))
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--preview-limit", type=int, default=0)
    args = parser.parse_args(argv)
    PREVIEW_DIR = args.preview_dir
    PREVIEW_LIMIT = args.preview_limit
    PAGE_INDEX = 0
    build_deck(args.prompt_manifest, args.root_manifest, args.output_pdf)
    print(args.output_pdf)


if __name__ == "__main__":
    main()
