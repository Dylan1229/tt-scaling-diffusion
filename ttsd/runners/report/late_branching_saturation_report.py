"""Build an intuitive PDF report for the nested late-branching experiment."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import cv2
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
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.65,
        "grid.linewidth": 0.7,
    }
)


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
    fig.savefig(
        pages_dir / f"page_{page_number:02d}.png",
        dpi=120,
        facecolor=BG,
    )
    plt.close(fig)


def _pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{100 * value:.1f}%"


def _pp(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{100 * value:+.2f} pp"


def _curve_row(curves: pd.DataFrame, run: str, scope: str, m: int) -> pd.Series:
    return curves[
        (curves["run"] == run)
        & (curves["scope"] == scope)
        & (curves["m"] == m)
    ].iloc[0]


def _draw_metric_card(fig, rect, label: str, value: str, note: str, color: str):
    ax = _panel(fig, rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.06, 0.78, label, fontsize=11, color=MUTED, transform=ax.transAxes)
    value_lines = "\n".join(textwrap.wrap(value, 20))
    value_font_size = 27 if "\n" not in value_lines else 19
    value_y = 0.43 if "\n" not in value_lines else 0.56
    ax.text(
        0.06,
        value_y,
        value_lines,
        fontsize=value_font_size,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
        va="center",
    )
    ax.text(
        0.06,
        0.12,
        "\n".join(textwrap.wrap(note, 34)),
        fontsize=9.5,
        color=INK,
        transform=ax.transAxes,
        va="bottom",
    )


def _draw_timeline(ax) -> None:
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.8, 1.2)
    ax.axis("off")
    ax.plot([0, 50], [0, 0], color=MUTED, linewidth=3)
    for step, label, color in (
        (0, "noise", MUTED),
        (35, "fork after Step 35", ORANGE),
        (50, "final video", TEAL),
    ):
        ax.scatter([step], [0], s=130, color=color, zorder=3)
        ax.text(step, 0.26, label, ha="center", color=INK, fontsize=10)
    for step in (36, 38, 40):
        ax.scatter([step], [-0.34], marker="s", s=75, color=BLUE)
        ax.text(step, -0.62, f"x0@{step}", ha="center", fontsize=9, color=BLUE)
    for offset in np.linspace(-0.42, 0.42, 8):
        ax.plot([35, 50], [0, offset], color=ORANGE, alpha=0.45, linewidth=1.3)
    ax.text(
        42.5,
        0.88,
        "M noisy suffixes share one 35-step prefix",
        ha="center",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )


def _plot_oracle_curves(ax, curves: pd.DataFrame, metric: str, title: str) -> None:
    specs = [
        ("M8_full", "all_150", "All 150 roots", BLUE, "o"),
        (
            "M16_representative",
            "representative_45",
            "Representative 45",
            TEAL,
            "s",
        ),
        ("M8_full", "bottom_15", "Bottom 15 baseline", ORANGE, "^"),
    ]
    for run, scope, label, color, marker in specs:
        subset = curves[(curves["run"] == run) & (curves["scope"] == scope)]
        values = 100 * subset[metric]
        low = 100 * subset[f"{metric}_ci_low"]
        high = 100 * subset[f"{metric}_ci_high"]
        ax.plot(
            subset["m"],
            values,
            marker=marker,
            linewidth=2.2,
            color=color,
            label=label,
        )
        ax.fill_between(subset["m"], low, high, color=color, alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_xlabel("Noisy branches available (M)")
    ax.set_ylabel("Roots with at least one gain (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)


def _verifier_plot(
    ax, summary: pd.DataFrame, run: str, random_safe_win: float
) -> None:
    subset = summary[(summary["run"] == run) & (summary["accepted"] > 0)].copy()
    fixed = subset["model"].str.startswith("fixed_")
    for mask, label, color, marker in (
        (fixed, "Fixed training-free rule", ORANGE, "s"),
        (~fixed, "Offline-trained online verifier", BLUE, "o"),
    ):
        part = subset[mask]
        ax.scatter(
            100 * part["coverage"],
            100 * part["safe_win_rate"],
            s=70,
            color=color,
            marker=marker,
            alpha=0.8,
            label=label,
        )
    ax.axhline(80, color=TEAL, linestyle="--", linewidth=1.2)
    ax.axhline(90, color=PURPLE, linestyle=":", linewidth=1.2)
    ax.axhline(
        100 * random_safe_win,
        color=MUTED,
        linestyle="-.",
        linewidth=1.2,
        label=f"Random branch ({_pct(random_safe_win)})",
    )
    ax.set_xlabel("Coverage: roots where intervention fires (%)")
    ax.set_ylabel("Strict-safe win rate among fired roots (%)")
    ax.set_title(run.replace("_", " "), loc="left", fontweight="bold")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 102)
    ax.legend(frameon=False, fontsize=9)


def _read_frames(path: str, fractions=(0.15, 0.5, 0.85)) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for fraction in fractions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((count - 1) * fraction)))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _show_sequence(fig, rect, video_path: str, title: str) -> None:
    frames = _read_frames(video_path)
    left, bottom, width, height = rect
    if not frames:
        ax = fig.add_axes(rect)
        ax.text(0.5, 0.5, "Video unavailable", ha="center")
        ax.axis("off")
        return
    gap = 0.006
    frame_width = (width - gap * (len(frames) - 1)) / len(frames)
    for index, frame in enumerate(frames):
        ax = fig.add_axes(
            [left + index * (frame_width + gap), bottom, frame_width, height]
        )
        ax.imshow(frame)
        ax.axis("off")
    fig.text(left, bottom + height + 0.008, title, fontsize=9.5, color=INK)


def _draw_examples(
    fig,
    examples: pd.DataFrame,
    baseline_run: Path,
    heading: str,
) -> None:
    fig.text(0.055, 0.855, heading, fontsize=12, color=INK, fontweight="bold")
    for row_index, (_, row) in enumerate(examples.head(3).iterrows()):
        bottom = 0.63 - row_index * 0.245
        baseline_path = (
            baseline_run
            / row["prompt_id"]
            / f"seed{int(row['root_seed']):04d}"
            / "video.mp4"
        )
        _show_sequence(
            fig,
            [0.055, bottom, 0.415, 0.17],
            str(baseline_path),
            f"{row['prompt_id']} seed{int(row['root_seed']):02d} | baseline",
        )
        _show_sequence(
            fig,
            [0.51, bottom, 0.415, 0.17],
            str(row["video_path"]),
            (
                f"selected branch {int(row['candidate_index'])} | "
                f"Q {_pp(row['quality_delta'])}, Dyn {row['dynamic_delta']:+.2f}, "
                f"Overall {_pp(row['overall_delta'])}"
            ),
        )


def _readable_table(
    fig,
    rect,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
) -> None:
    ax = _panel(fig, rect)
    ax.axis("off")
    y_top = 0.93
    row_height = 0.13
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    curves = pd.read_csv(args.analysis / "nested_oracle_curves.csv")
    verifier = pd.read_csv(args.analysis / "online_verifier_summary.csv")
    predictions = pd.read_csv(args.analysis / "online_verifier_predictions.csv")
    m8 = pd.read_csv(args.analysis / "m8_candidates.csv")
    pages_dir = args.output / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output / "late_branching_saturation_report.pdf"

    m8_8 = _curve_row(curves, "M8_full", "all_150", 8)
    m16_8 = _curve_row(curves, "M16_representative", "representative_45", 8)
    m16_16 = _curve_row(curves, "M16_representative", "representative_45", 16)
    fixed_m8 = verifier[
        (verifier["run"] == "M8_full")
        & verifier["model"].str.startswith("fixed_")
        & (verifier["accepted"] > 0)
    ].sort_values(["safe_win_rate", "accepted"], ascending=False)
    learned_m8 = verifier[
        (verifier["run"] == "M8_full")
        & ~verifier["model"].str.startswith("fixed_")
        & (verifier["accepted"] >= 5)
    ].sort_values(["safe_win_rate", "accepted"], ascending=False)

    with PdfPages(pdf_path) as pdf:
        page = 1
        fig = _new_page(
            "Late-Stage Branching: M=8 Full Sweep + M=16 Saturation Check",
            "Wan 2.2 TI2V-5B | fork after Step 35 | perturbation scale 0.10 | updated VBench",
        )
        _draw_metric_card(
            fig,
            [0.055, 0.61, 0.205, 0.20],
            "M=8 QUALITY ORACLE",
            _pct(m8_8["quality_win"]),
            "At least one branch improves updated VBench Quality, among all 150 roots.",
            BLUE,
        )
        _draw_metric_card(
            fig,
            [0.275, 0.61, 0.205, 0.20],
            "M=8 STRICT-SAFE ORACLE",
            _pct(m8_8["safe_win"]),
            "Quality rises while Dynamic Degree and Overall Consistency do not fall.",
            TEAL,
        )
        _draw_metric_card(
            fig,
            [0.495, 0.61, 0.205, 0.20],
            "M=8 -> M=16 SAFE GAIN",
            f"{100 * (m16_16['safe_win'] - m16_8['safe_win']):+.1f} pp",
            "Marginal opportunity on the same representative 45-root M16 batch.",
            ORANGE,
        )
        fixed_value = fixed_m8.iloc[0] if len(fixed_m8) else None
        _draw_metric_card(
            fig,
            [0.715, 0.61, 0.205, 0.20],
            "BEST FIXED ONLINE RULE",
            _pct(fixed_value["safe_win_rate"]) if fixed_value is not None else "n/a",
            (
                f"{fixed_value['model']} / {fixed_value['gate']}, "
                f"n={int(fixed_value['accepted'])}"
                if fixed_value is not None
                else "No accepted interventions."
            ),
            PURPLE,
        )
        ax = _panel(fig, [0.055, 0.12, 0.865, 0.40])
        ax.axis("off")
        ax.text(
            0.035,
            0.86,
            "What this experiment answers",
            fontsize=17,
            fontweight="bold",
            color=INK,
            transform=ax.transAxes,
        )
        bullets = [
            "Search-space opportunity: does increasing M keep finding strict-safe branches?",
            "Saturation: is M=8 close to M=16, or is branch diversity still the bottleneck?",
            "Selection reality: can information available by Steps 36/38/40 identify a winning branch without final VBench?",
            "Generalization: learned verifiers are evaluated leave-one-prompt-out; fixed latent rules use no labels at all.",
        ]
        for index, bullet in enumerate(bullets):
            ax.text(
                0.05,
                0.67 - index * 0.16,
                f"{index + 1}.  {bullet}",
                fontsize=12.5,
                color=INK,
                transform=ax.transAxes,
            )
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "Experimental Design",
            "The M curves are nested within each run. M8 and M16 are separate batch-size regimes.",
        )
        ax = _panel(fig, [0.055, 0.59, 0.865, 0.24])
        _draw_timeline(ax)
        cards = [
            (
                "FULL ESTIMATE",
                "150 roots x 8 noisy branches",
                "All 15 prompts x 10 seeds. Reports Best-of-1/2/4/8 and held-out verifier performance.",
                BLUE,
            ),
            (
                "SATURATION CHECK",
                "45 roots x 16 noisy branches",
                "Every prompt contributes its low, median, and high baseline-quality seed. Reports Best-of-1/2/4/8/16.",
                TEAL,
            ),
            (
                "STRICT-SAFE WIN",
                "Delta Q > 0",
                "and Delta Dynamic >= 0 and Delta Overall >= 0. Exact comparison, no tolerance.",
                ORANGE,
            ),
        ]
        for index, (label, value, note, color) in enumerate(cards):
            _draw_metric_card(
                fig,
                [0.055 + index * 0.292, 0.16, 0.27, 0.30],
                label,
                value,
                note,
                color,
            )
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "Oracle Opportunity Grows with M",
            "Final VBench is used only here to measure whether a useful branch exists.",
        )
        ax1 = _panel(fig, [0.065, 0.14, 0.405, 0.70])
        ax2 = _panel(fig, [0.535, 0.14, 0.405, 0.70])
        _plot_oracle_curves(ax1, curves, "quality_win", "Quality Oracle")
        _plot_oracle_curves(ax2, curves, "safe_win", "Strict-Safe Oracle")
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "The Important Number Is the M=8 -> M=16 Marginal Gain",
            "Both points below come from the same 45-root, batch-17 run.",
        )
        ax1 = _panel(fig, [0.065, 0.18, 0.41, 0.62])
        rep = curves[
            (curves["run"] == "M16_representative")
            & (curves["scope"] == "representative_45")
        ]
        ax1.plot(rep["m"], 100 * rep["quality_gain"], marker="o", color=BLUE, label="Quality gain")
        ax1.plot(rep["m"], 100 * rep["safe_gain"], marker="s", color=TEAL, label="Strict-safe gain")
        ax1.set_xscale("log", base=2)
        ax1.set_xticks([1, 2, 4, 8, 16])
        ax1.set_xticklabels(["1", "2", "4", "8", "16"])
        ax1.set_xlabel("M")
        ax1.set_ylabel("Mean updated-VBench Quality gain (pp)")
        ax1.set_title("Oracle mean gain", loc="left", fontweight="bold")
        ax1.legend(frameon=False)
        ax2 = _panel(fig, [0.535, 0.18, 0.405, 0.62])
        ax2.axis("off")
        marginal_rows = []
        previous = None
        for _, row in rep.iterrows():
            if previous is not None:
                marginal_rows.append(
                    [
                        f"{int(previous['m'])} -> {int(row['m'])}",
                        f"{100 * (row['quality_win'] - previous['quality_win']):+.1f} pp",
                        f"{100 * (row['safe_win'] - previous['safe_win']):+.1f} pp",
                        _pp(row["safe_gain"] - previous["safe_gain"]),
                    ]
                )
            previous = row
        _readable_table(
            fig,
            [0.535, 0.18, 0.405, 0.62],
            ["Added branches", "Quality opp.", "Safe opp.", "Safe mean gain"],
            marginal_rows,
            [0.24, 0.24, 0.24, 0.28],
        )
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "Bottom Videos Are a Separate Question",
            "A larger M can find more options; it does not imply that initially bad videos are easier to rescue.",
        )
        ax = _panel(fig, [0.065, 0.18, 0.56, 0.65])
        for scope, label, color, marker in (
            ("all_150", "All 150", BLUE, "o"),
            ("bottom_15", "Bottom 15", ORANGE, "^"),
        ):
            subset = curves[
                (curves["run"] == "M8_full") & (curves["scope"] == scope)
            ]
            ax.plot(
                subset["m"],
                100 * subset["safe_win"],
                marker=marker,
                linewidth=2.4,
                color=color,
                label=label,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_xlabel("M")
        ax.set_ylabel("Strict-safe oracle opportunity (%)")
        ax.set_title("Same intervention, different root strata", loc="left", fontweight="bold")
        ax.legend(frameon=False)
        ax2 = _panel(fig, [0.68, 0.18, 0.24, 0.65])
        ax2.axis("off")
        all8 = _curve_row(curves, "M8_full", "all_150", 8)
        bottom8 = _curve_row(curves, "M8_full", "bottom_15", 8)
        ax2.text(0.08, 0.84, "At M=8", fontsize=17, fontweight="bold", color=INK)
        ax2.text(0.08, 0.65, _pct(all8["safe_win"]), fontsize=31, color=BLUE, fontweight="bold")
        ax2.text(0.08, 0.57, "all roots", fontsize=11, color=MUTED)
        ax2.text(0.08, 0.37, _pct(bottom8["safe_win"]), fontsize=31, color=ORANGE, fontweight="bold")
        ax2.text(0.08, 0.29, "bottom-15 roots", fontsize=11, color=MUTED)
        ax2.text(
            0.08,
            0.08,
            "This comparison applies only to Step 35 and this perturbation scale.",
            fontsize=10,
            color=INK,
            wrap=True,
        )
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "Online Selection: Precision vs Coverage",
            "Every point is evaluated on final outcomes, but its decision uses only causal Step 36/38/40 inputs.",
        )
        ax1 = _panel(fig, [0.065, 0.18, 0.405, 0.65])
        ax2 = _panel(fig, [0.535, 0.18, 0.405, 0.65])
        _verifier_plot(ax1, verifier, "M8_full", m8_8["random_safe_win"])
        _verifier_plot(
            ax2,
            verifier,
            "M16_representative",
            m16_16["random_safe_win"],
        )
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "What Each Verifier Actually Means",
            "Compact table, with wrapped text; no final VBench value is available at decision time.",
        )
        table_rows = []
        candidates = pd.concat(
            [
                fixed_m8.head(2),
                learned_m8.head(3),
            ]
        )
        for _, row in candidates.iterrows():
            verifier_type = (
                "Fixed / no labels"
                if str(row["model"]).startswith("fixed_")
                else "Outcome-trained / held-out prompt"
            )
            table_rows.append(
                [
                    str(row["model"]).replace("_", " "),
                    str(row["gate"]).replace("_", " "),
                    verifier_type,
                    f"{int(row['accepted'])}/{int(row['n_roots'])} ({_pct(row['coverage'])})",
                    _pct(row["safe_win_rate"]),
                ]
            )
        table_rows.append(
            [
                "All learned models",
                "80% / 90% target",
                "Inner prompt-grouped OOF",
                "0/150 (abstain)",
                "No gate qualified",
            ]
        )
        _readable_table(
            fig,
            [0.055, 0.29, 0.865, 0.54],
            ["Signal", "Gate", "Training status", "Accepted", "Strict-safe wins"],
            table_rows,
            [0.22, 0.19, 0.25, 0.19, 0.15],
        )
        ax = _panel(fig, [0.055, 0.105, 0.865, 0.12])
        ax.axis("off")
        ax.text(
            0.025,
            0.66,
            "Interpretation guardrail",
            fontsize=11,
            fontweight="bold",
            color=RED,
            transform=ax.transAxes,
        )
        ax.text(
            0.025,
            0.23,
            "\n".join(
                textwrap.wrap(
                    (
                        "A high selective win rate with very few accepted roots is evidence of a useful "
                        "condition, not proof of a universal repair method. Prompt-held-out evaluation "
                        "reduces leakage but does not replace a larger independent prompt set."
                    ),
                    145,
                )
            ),
            fontsize=10.5,
            color=INK,
            transform=ax.transAxes,
            va="top",
        )
        _save_page(fig, pdf, pages_dir, page)

        safe_examples = (
            m8[m8["safe_win"]]
            .sort_values("quality_delta", ascending=False)
            .drop_duplicates(["prompt_id", "root_seed"])
        )
        page += 1
        fig = _new_page(
            "Visual Examples: Strong Strict-Safe Oracle Gains",
            "Each strip shows early / middle / late frames. The branch was selected offline for illustration.",
        )
        _draw_examples(
            fig,
            safe_examples,
            args.baseline_run,
            "Baseline on the left; rescued branch on the right",
        )
        _save_page(fig, pdf, pages_dir, page)

        selected_successes = predictions[
            (predictions["run"] == "M8_full")
            & (predictions["accepted"])
            & (predictions["safe_win"])
            & ~predictions["model"].str.startswith("fixed_")
        ].sort_values("quality_delta", ascending=False)
        page += 1
        fig = _new_page(
            "Visual Examples: Held-Out Online Verifier Successes",
            "These selections use a model trained without the displayed prompt.",
        )
        _draw_examples(
            fig,
            selected_successes.drop_duplicates(["prompt_id", "root_seed"]),
            args.baseline_run,
            "Decision-time input: only Step 36/38/40 latent trajectory features",
        )
        _save_page(fig, pdf, pages_dir, page)

        page += 1
        fig = _new_page(
            "Conclusions and Decision Rule",
            "Scope: Step 35 only, Wan 2.2 TI2V-5B, perturbation scale 0.10.",
        )
        ax = _panel(fig, [0.055, 0.13, 0.865, 0.72])
        ax.axis("off")
        learned_value = learned_m8.iloc[0] if len(learned_m8) else None
        conclusions = [
            (
                "1. Search opportunity",
                (
                    f"At M=8, Quality Oracle opportunity is {_pct(m8_8['quality_win'])} "
                    f"and strict-safe opportunity is {_pct(m8_8['safe_win'])}."
                ),
            ),
            (
                "2. Saturation",
                (
                    f"On the representative 45 roots, increasing M from 8 to 16 changes "
                    f"strict-safe opportunity by {100 * (m16_16['safe_win'] - m16_8['safe_win']):+.1f} pp."
                ),
            ),
            (
                "3. Fixed online rule",
                (
                    f"Best tested fixed rule reaches {_pct(fixed_value['safe_win_rate'])} "
                    f"on n={int(fixed_value['accepted'])} accepted M8 roots."
                    if fixed_value is not None
                    else "No fixed rule accepted any M8 root."
                ),
            ),
            (
                "4. Learned online verifier",
                (
                    "No 80% or 90% precision gate qualified in inner prompt-grouped OOF. "
                    f"The best forced-selection held-out result is {_pct(learned_value['safe_win_rate'])} "
                    f"on n={int(learned_value['accepted'])} M8 roots."
                    if learned_value is not None
                    else "No learned gate accepted at least five M8 roots."
                ),
            ),
            (
                "5. Recommended claim",
                "Report branching as a measurable opportunity set. Deploy only a verifier whose held-out strict-safe precision and sample count meet the team's bar; otherwise abstain.",
            ),
        ]
        for index, (heading, body) in enumerate(conclusions):
            y = 0.88 - index * 0.17
            ax.text(0.035, y, heading, fontsize=13, fontweight="bold", color=INK, transform=ax.transAxes)
            ax.text(
                0.27,
                y,
                "\n".join(textwrap.wrap(body, 94)),
                fontsize=11.5,
                color=INK,
                transform=ax.transAxes,
                va="top",
            )
        _save_page(fig, pdf, pages_dir, page)

    print(f"[report] wrote {pdf_path}")


if __name__ == "__main__":
    main()
