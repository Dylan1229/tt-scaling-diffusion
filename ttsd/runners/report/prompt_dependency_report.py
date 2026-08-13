"""Analyze prompt dependency for the Step-35, M=4 late-branching sweep."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

BG = "#F4F6F8"
PANEL = "#FFFFFF"
INK = "#17212B"
MUTED = "#5D6975"
GRID = "#D8DEE5"
BLUE = "#2764B0"
TEAL = "#188477"
ORANGE = "#D97832"
RED = "#C84B47"
PURPLE = "#7558A6"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.65,
        "grid.linewidth": 0.7,
    }
)


def _load_targets(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(columns={"seed_idx": "seed"})
    frame["seed"] = frame["seed"].astype(int)
    return frame


def _load_candidate_meta(run: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run.glob("*/seed*/meta.json")):
        meta = json.loads(path.read_text())
        if meta.get("experiment") != "late_branching_best_of_m":
            continue
        rows.append(
            {
                "prompt_id": meta["prompt_id"],
                "seed": int(meta["seed"]),
                "root_seed": int(meta["root_seed"]),
                "branch_kind": meta["branch_kind"],
                "video_path": str((path.parent / "video.mp4").resolve()),
            }
        )
    return pd.DataFrame(rows)


def _build_tables(
    manifest_path: Path,
    baseline_targets_path: Path,
    branch_targets_path: Path,
    branch_run: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    baseline = _load_targets(baseline_targets_path)
    targets = _load_targets(branch_targets_path)
    meta = _load_candidate_meta(branch_run)

    metric_columns = ["vbench_quality", "dynamic_degree", "overall_consistency"]
    candidates = meta.merge(
        targets[["prompt_id", "seed", *metric_columns]],
        on=["prompt_id", "seed"],
        how="left",
        validate="one_to_one",
    )
    if candidates[metric_columns].isna().any().any():
        raise ValueError("Missing VBench scores for generated candidates")

    baseline = baseline.rename(
        columns={
            "seed": "root_seed",
            "vbench_quality": "baseline_vbench_quality",
            "dynamic_degree": "baseline_dynamic_degree",
            "overall_consistency": "baseline_overall_consistency",
        }
    )
    baseline_columns = [
        "prompt_id",
        "root_seed",
        "baseline_vbench_quality",
        "baseline_dynamic_degree",
        "baseline_overall_consistency",
    ]
    candidates = candidates.merge(
        baseline[baseline_columns],
        on=["prompt_id", "root_seed"],
        how="left",
        validate="many_to_one",
    )
    candidates = candidates.merge(
        manifest,
        on=["prompt_id", "root_seed"],
        how="left",
        validate="many_to_one",
    )
    candidates["quality_delta"] = (
        candidates["vbench_quality"] - candidates["baseline_vbench_quality"]
    )
    candidates["dynamic_delta"] = (
        candidates["dynamic_degree"] - candidates["baseline_dynamic_degree"]
    )
    candidates["overall_delta"] = (
        candidates["overall_consistency"]
        - candidates["baseline_overall_consistency"]
    )
    candidates["quality_win"] = candidates["quality_delta"] > 1e-12
    candidates["strict_safe_win"] = (
        candidates["quality_win"]
        & (candidates["dynamic_delta"] >= 0)
        & (candidates["overall_delta"] >= 0)
    )
    candidates["pareto_win"] = (
        (candidates["quality_delta"] >= 0)
        & (candidates["dynamic_delta"] >= 0)
        & (candidates["overall_delta"] >= 0)
        & (
            (candidates["quality_delta"] > 1e-12)
            | (candidates["dynamic_delta"] > 1e-12)
            | (candidates["overall_delta"] > 1e-12)
        )
    )
    noises = candidates[candidates["branch_kind"] == "noise"].copy()

    rows = []
    for (_, _), group in noises.groupby(["prompt_id", "root_seed"], sort=True):
        best_quality = group.loc[group["quality_delta"].idxmax()]
        strict_safe = group[group["strict_safe_win"]]
        best_safe = (
            strict_safe.loc[strict_safe["quality_delta"].idxmax()]
            if len(strict_safe)
            else None
        )
        first = group.iloc[0]
        row = {
            column: first[column]
            for column in (
                "prompt_id",
                "root_seed",
                "prompt_text",
                "axis",
                "prompt_class",
                "motion_bucket",
                "subject_count",
                "relation_class",
                "camera_class",
                "complexity_class",
                "baseline_vbench_quality",
                "baseline_dynamic_degree",
                "baseline_overall_consistency",
            )
        }
        row.update(
            {
                "num_noise_branches": len(group),
                "random_quality_win_rate": group["quality_win"].mean(),
                "random_strict_safe_win_rate": group["strict_safe_win"].mean(),
                "random_pareto_win_rate": group["pareto_win"].mean(),
                "quality_oracle_win": bool(group["quality_win"].any()),
                "quality_oracle_gain": max(0.0, float(best_quality["quality_delta"])),
                "quality_best_seed": int(best_quality["seed"]),
                "quality_best_video_path": best_quality["video_path"],
                "quality_best_dynamic_delta": float(best_quality["dynamic_delta"]),
                "quality_best_overall_delta": float(best_quality["overall_delta"]),
                "strict_safe_opportunity": bool(len(strict_safe)),
                "strict_safe_gain": (
                    float(best_safe["quality_delta"]) if best_safe is not None else 0.0
                ),
                "strict_safe_seed": (
                    int(best_safe["seed"]) if best_safe is not None else -1
                ),
                "strict_safe_video_path": (
                    best_safe["video_path"] if best_safe is not None else ""
                ),
                "pareto_opportunity": bool(group["pareto_win"].any()),
            }
        )
        rows.append(row)

    roots = pd.DataFrame(rows)
    roots["motion_family"] = np.where(
        roots["motion_bucket"].eq("static_prompt"), "static", "dynamic"
    )
    rank = roots["baseline_vbench_quality"].rank(method="first")
    roots["baseline_quality_tier"] = pd.qcut(
        rank,
        3,
        labels=["low", "mid", "high"],
    ).astype(str)
    candidates = candidates.merge(
        roots[
            [
                "prompt_id",
                "root_seed",
                "motion_family",
                "baseline_quality_tier",
            ]
        ],
        on=["prompt_id", "root_seed"],
        how="left",
        validate="many_to_one",
    )
    return roots, candidates


def _aggregate_group(group: pd.DataFrame) -> dict:
    return {
        "n_prompts": group["prompt_id"].nunique(),
        "n_roots": len(group),
        "baseline_quality_mean": group["baseline_vbench_quality"].mean(),
        "quality_oracle_win_rate": group["quality_oracle_win"].mean(),
        "strict_safe_opportunity_rate": group["strict_safe_opportunity"].mean(),
        "pareto_opportunity_rate": group["pareto_opportunity"].mean(),
        "random_quality_win_rate": group["random_quality_win_rate"].mean(),
        "random_strict_safe_win_rate": group[
            "random_strict_safe_win_rate"
        ].mean(),
        "quality_oracle_mean_gain": group["quality_oracle_gain"].mean(),
        "strict_safe_mean_gain": group["strict_safe_gain"].mean(),
    }


def _build_category_summary(roots: pd.DataFrame) -> pd.DataFrame:
    frames = []
    groupings = [
        "overall",
        "baseline_quality_tier",
        "motion_family",
        "prompt_class",
        "motion_bucket",
        "complexity_class",
        "subject_count",
        "axis",
    ]
    for grouping in groupings:
        if grouping == "overall":
            rows = [{"grouping": grouping, "group": "all", **_aggregate_group(roots)}]
        else:
            rows = [
                {
                    "grouping": grouping,
                    "group": str(value),
                    **_aggregate_group(group),
                }
                for value, group in roots.groupby(grouping, sort=True)
            ]
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def _new_page(title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(0.055, 0.935, title, fontsize=25, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.055, 0.895, subtitle, fontsize=11, color=MUTED)
    return fig


def _panel(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    return ax


def _save_page(fig, pdf: PdfPages, pages_dir: Path, page_number: int) -> None:
    fig.text(0.94, 0.025, str(page_number), color=MUTED, fontsize=9)
    pdf.savefig(fig, facecolor=BG)
    fig.savefig(pages_dir / f"page_{page_number:02d}.png", dpi=120, facecolor=BG)
    plt.close(fig)


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _draw_card(fig, rect, label: str, value: str, note: str, color: str) -> None:
    ax = _panel(fig, rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.06, 0.78, label, fontsize=11, color=MUTED, transform=ax.transAxes)
    ax.text(
        0.06,
        0.48,
        value,
        fontsize=27,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
        va="center",
    )
    ax.text(
        0.06,
        0.11,
        "\n".join(textwrap.wrap(note, 33)),
        fontsize=9.5,
        color=INK,
        transform=ax.transAxes,
        va="bottom",
    )


def _bar_compare(
    ax,
    data: pd.DataFrame,
    label_column: str,
    title: str,
    max_groups: int | None = None,
    count_column: str = "n_prompts",
) -> None:
    data = data.sort_values("strict_safe_opportunity_rate", ascending=True)
    if max_groups is not None:
        data = data.tail(max_groups)
    labels = data[label_column].astype(str).str.replace("_", " ")
    y = np.arange(len(data))
    ax.barh(
        y - 0.18,
        100 * data["strict_safe_opportunity_rate"],
        height=0.34,
        label="At least one strict-safe branch",
        color=TEAL,
    )
    ax.barh(
        y + 0.18,
        100 * data["random_strict_safe_win_rate"],
        height=0.34,
        label="Random branch strict-safe win",
        color=ORANGE,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Rate (%)")
    ax.set_xlim(0, 100)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for index, (_, row) in enumerate(data.iterrows()):
        ax.text(
            101,
            index,
            f"n={int(row[count_column])}",
            va="center",
            fontsize=8.5,
            color=MUTED,
            clip_on=False,
        )


def _table(fig, rect, headers: list[str], rows: list[list[str]], widths: list[float]):
    ax = _panel(fig, rect)
    ax.axis("off")
    y_top = 0.93
    row_height = min(0.14, 0.78 / max(1, len(rows)))
    x_positions = np.cumsum([0, *widths[:-1]])
    for x, header in zip(x_positions, headers, strict=True):
        ax.text(
            x + 0.012,
            y_top,
            header,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=INK,
            va="top",
        )
    ax.plot([0.01, 0.99], [y_top - 0.055] * 2, color=GRID)
    for row_index, values in enumerate(rows):
        y = y_top - 0.09 - row_index * row_height
        for x, value, width in zip(x_positions, values, widths, strict=True):
            wrapped = "\n".join(textwrap.wrap(str(value), max(8, int(width * 75))))
            ax.text(
                x + 0.012,
                y,
                wrapped,
                transform=ax.transAxes,
                fontsize=9.2,
                color=INK,
                va="top",
            )
        ax.plot([0.01, 0.99], [y - row_height + 0.018] * 2, color=GRID, linewidth=0.6)


def _build_report(
    roots: pd.DataFrame,
    categories: pd.DataFrame,
    output: Path,
    playable_url: str | None = None,
) -> Path:
    pages_dir = output / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output / "prompt_dependency_step35_m4_report.pdf"
    overall = categories[
        (categories["grouping"] == "overall") & (categories["group"] == "all")
    ].iloc[0]

    prompt_classes = categories[categories["grouping"] == "prompt_class"].copy()
    motion_families = categories[categories["grouping"] == "motion_family"].copy()
    motion_buckets = categories[categories["grouping"] == "motion_bucket"].copy()
    quality_tiers = categories[
        categories["grouping"] == "baseline_quality_tier"
    ].copy()
    complexities = categories[
        categories["grouping"] == "complexity_class"
    ].copy()
    subjects = categories[categories["grouping"] == "subject_count"].copy()

    eligible = prompt_classes[prompt_classes["n_prompts"] >= 8]
    best = eligible.sort_values(
        ["strict_safe_opportunity_rate", "random_strict_safe_win_rate"],
        ascending=False,
    ).iloc[0]
    worst = eligible.sort_values(
        ["strict_safe_opportunity_rate", "random_strict_safe_win_rate"]
    ).iloc[0]

    with PdfPages(pdf_path) as pdf:
        page = 1
        fig = _new_page(
            "Which Prompts Benefit From Late Branching?",
            "Wan 2.2 TI2V-5B | Step 35 | M=4 noisy branches | 120 prompts x 3 seeds",
        )
        _draw_card(
            fig,
            [0.055, 0.61, 0.205, 0.20],
            "QUALITY OPPORTUNITY",
            _pct(overall["quality_oracle_win_rate"]),
            "Roots where at least one of four branches improves VBench Quality.",
            BLUE,
        )
        _draw_card(
            fig,
            [0.275, 0.61, 0.205, 0.20],
            "STRICT-SAFE OPPORTUNITY",
            _pct(overall["strict_safe_opportunity_rate"]),
            "Quality improves without lowering Dynamic Degree or Overall Consistency.",
            TEAL,
        )
        _draw_card(
            fig,
            [0.495, 0.61, 0.205, 0.20],
            "RANDOM SAFE WIN",
            _pct(overall["random_strict_safe_win_rate"]),
            "What happens without a verifier when one branch is selected at random.",
            ORANGE,
        )
        _draw_card(
            fig,
            [0.715, 0.61, 0.205, 0.20],
            "BEST PROMPT CLASS",
            str(best["group"]).replace("_", " "),
            (
                f"{_pct(best['strict_safe_opportunity_rate'])} strict-safe "
                f"opportunity across {int(best['n_prompts'])} prompts."
            ),
            PURPLE,
        )
        ax = _panel(fig, [0.055, 0.12, 0.865, 0.40])
        ax.axis("off")
        ax.text(
            0.035,
            0.84,
            "Decision-level takeaway",
            fontsize=17,
            fontweight="bold",
            color=INK,
            transform=ax.transAxes,
        )
        takeaway = (
            f"Step-35 branching has the strongest measured opportunity on "
            f"{str(best['group']).replace('_', ' ')} prompts and the weakest on "
            f"{str(worst['group']).replace('_', ' ')} prompts. "
            f"However, opportunity is an offline upper bound: the gap between "
            f"{_pct(overall['strict_safe_opportunity_rate'])} opportunity and "
            f"{_pct(overall['random_strict_safe_win_rate'])} random safe wins is "
            "the selection problem an online verifier still has to solve."
        )
        ax.text(
            0.05,
            0.61,
            "\n".join(textwrap.wrap(takeaway, 115)),
            fontsize=14,
            color=INK,
            transform=ax.transAxes,
            va="top",
            linespacing=1.45,
        )
        ax.text(
            0.05,
            0.20,
            "Scope: these conclusions apply to Step 35, M=4, Wan 5B only. "
            "They do not establish behavior at Steps 10/15/25/45.",
            fontsize=11,
            color=RED,
            transform=ax.transAxes,
        )
        if playable_url:
            ax.text(
                0.05,
                0.08,
                "Open playable baseline vs best-branch comparisons",
                fontsize=11.5,
                color=BLUE,
                fontweight="bold",
                transform=ax.transAxes,
                url=playable_url,
            )
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "How To Read The Result",
            "The report separates search-space potential from deployable selection performance.",
        )
        cards = [
            (
                "1. Quality opportunity",
                "At least one branch has higher VBench Quality than baseline.",
                "Useful for proving that a better nearby trajectory exists.",
                BLUE,
            ),
            (
                "2. Strict-safe opportunity",
                "Quality rises while motion and text-video consistency do not fall.",
                "Prevents static-video metric hacking from counting as a win.",
                TEAL,
            ),
            (
                "3. Random safe win",
                "A randomly selected branch satisfies the strict-safe rule.",
                "The honest no-verifier baseline.",
                ORANGE,
            ),
        ]
        for index, (title, body, note, color) in enumerate(cards):
            ax = _panel(fig, [0.055 + index * 0.29, 0.33, 0.265, 0.43])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.07,
                0.83,
                title,
                fontsize=15,
                fontweight="bold",
                color=color,
                transform=ax.transAxes,
            )
            ax.text(
                0.07,
                0.60,
                "\n".join(textwrap.wrap(body, 34)),
                fontsize=12,
                color=INK,
                transform=ax.transAxes,
                va="top",
                linespacing=1.4,
            )
            ax.text(
                0.07,
                0.18,
                "\n".join(textwrap.wrap(note, 34)),
                fontsize=10.5,
                color=MUTED,
                transform=ax.transAxes,
                va="top",
            )
        fig.text(
            0.055,
            0.18,
            "A high opportunity rate alone does not mean the method is ready. "
            "It means a verifier could, in principle, capture value from branching.",
            fontsize=14,
            color=INK,
        )
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "Baseline Quality: No Clear Bottom-Tier Rescue Advantage",
            (
                "The low-quality tier did not outperform the high-quality tier "
                "on strict-safe opportunity at Step 35."
            ),
        )
        ax = _panel(fig, [0.07, 0.17, 0.82, 0.63])
        order = pd.Categorical(
            quality_tiers["group"], categories=["low", "mid", "high"], ordered=True
        )
        quality_tiers = quality_tiers.assign(_order=order).sort_values("_order")
        _bar_compare(
            ax,
            quality_tiers,
            "group",
            "Opportunity vs actual random selection",
            count_column="n_roots",
        )
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "Prompt Class Dependency",
            "Each class contains multiple independent prompts and three seeds per prompt.",
        )
        ax = _panel(fig, [0.10, 0.13, 0.75, 0.70])
        _bar_compare(ax, prompt_classes, "group", "Strict-safe branching by prompt class")
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "Static vs Dynamic Prompts",
            "Prompt intent is used for this split; Dynamic Degree remains an independent safety metric.",
        )
        ax1 = _panel(fig, [0.07, 0.19, 0.39, 0.60])
        _bar_compare(ax1, motion_families, "group", "Broad motion family")
        ax2 = _panel(fig, [0.54, 0.19, 0.39, 0.60])
        _bar_compare(ax2, motion_buckets, "group", "Detailed motion category")
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "Complexity And Subject Structure",
            "Does branching behave differently as prompts become compositionally harder?",
        )
        ax1 = _panel(fig, [0.07, 0.19, 0.39, 0.60])
        _bar_compare(ax1, complexities, "group", "Prompt complexity")
        ax2 = _panel(fig, [0.54, 0.19, 0.39, 0.60])
        _bar_compare(ax2, subjects, "group", "Subject structure", max_groups=8)
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "Most And Least Promising Prompt Classes",
            "Sorted by strict-safe opportunity; random performance is shown to expose verifier dependency.",
        )
        ranked = prompt_classes.sort_values(
            ["strict_safe_opportunity_rate", "random_strict_safe_win_rate"],
            ascending=False,
        )
        rows = []
        for _, row in ranked.iterrows():
            rows.append(
                [
                    str(row["group"]).replace("_", " "),
                    str(int(row["n_prompts"])),
                    _pct(row["quality_oracle_win_rate"]),
                    _pct(row["strict_safe_opportunity_rate"]),
                    _pct(row["random_strict_safe_win_rate"]),
                    f"{100 * row['strict_safe_mean_gain']:+.2f} pp",
                ]
            )
        _table(
            fig,
            [0.055, 0.14, 0.865, 0.70],
            [
                "Prompt class",
                "Prompts",
                "Quality opportunity",
                "Strict-safe opportunity",
                "Random safe win",
                "Mean safe gain",
            ],
            rows,
            [0.25, 0.10, 0.16, 0.17, 0.15, 0.17],
        )
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "Recommendation",
            "What this experiment supports, and what it does not.",
        )
        ax = _panel(fig, [0.055, 0.15, 0.865, 0.68])
        ax.axis("off")
        recommendations = [
            (
                "Use prompt class as a prior, not a guarantee.",
                f"Prioritize {str(best['group']).replace('_', ' ')} prompts for the "
                "next verifier experiment, but keep the final decision candidate-specific.",
            ),
            (
                "Do not deploy random Step-35 branching.",
                f"Its strict-safe win rate is only "
                f"{_pct(overall['random_strict_safe_win_rate'])}; offline opportunity "
                "cannot substitute for an online selector.",
            ),
            (
                "Keep all three outcomes separate.",
                "Report VBench Quality, Dynamic Degree, and Overall Consistency independently. "
                "A quality-only win can still be a worse video.",
            ),
            (
                "Next experiment: intervention timing.",
                "Repeat M=4 at Steps 10, 15, 25, and 45 using the same prompt set, "
                "then compare each category against this Step-35 reference.",
            ),
        ]
        for index, (heading, body) in enumerate(recommendations):
            y = 0.86 - index * 0.21
            ax.text(
                0.04,
                y,
                f"{index + 1}. {heading}",
                fontsize=14,
                fontweight="bold",
                color=INK,
                transform=ax.transAxes,
            )
            ax.text(
                0.07,
                y - 0.075,
                "\n".join(textwrap.wrap(body, 105)),
                fontsize=11.5,
                color=MUTED,
                transform=ax.transAxes,
                va="top",
            )
        _save_page(fig, pdf, pages_dir, page)

    return pdf_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline-targets", required=True, type=Path)
    parser.add_argument("--branch-targets", required=True, type=Path)
    parser.add_argument("--branch-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--playable-url", default=None)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    roots, candidates = _build_tables(
        args.manifest,
        args.baseline_targets,
        args.branch_targets,
        args.branch_run,
    )
    categories = _build_category_summary(roots)
    roots.to_csv(args.output / "per_root_results.csv", index=False)
    candidates.to_csv(args.output / "per_candidate_results.csv", index=False)
    categories.to_csv(args.output / "category_summary.csv", index=False)

    overall = categories[
        (categories["grouping"] == "overall") & (categories["group"] == "all")
    ].iloc[0]
    summary = {
        "scope": "Wan 2.2 TI2V-5B, Step 35, M=4, 120 prompts x 3 seeds",
        "n_prompts": int(roots["prompt_id"].nunique()),
        "n_roots": int(len(roots)),
        "quality_oracle_win_rate": float(overall["quality_oracle_win_rate"]),
        "strict_safe_opportunity_rate": float(
            overall["strict_safe_opportunity_rate"]
        ),
        "random_quality_win_rate": float(overall["random_quality_win_rate"]),
        "random_strict_safe_win_rate": float(
            overall["random_strict_safe_win_rate"]
        ),
        "quality_oracle_mean_gain": float(overall["quality_oracle_mean_gain"]),
        "strict_safe_mean_gain": float(overall["strict_safe_mean_gain"]),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    pdf_path = _build_report(
        roots,
        categories,
        args.output,
        playable_url=args.playable_url,
    )
    print(f"[prompt_dependency] roots={len(roots)} prompts={roots['prompt_id'].nunique()}")
    print(f"[prompt_dependency] report={pdf_path}")


if __name__ == "__main__":
    main()
