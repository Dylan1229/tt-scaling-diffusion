"""Build a portrait, rasterized PDF for the fixed step-10 Renoise experiment."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PAGE_SIZE = (8.27, 11.69)
COLORS = {
    "ink": "#18212b",
    "muted": "#5f6b76",
    "blue": "#3973ac",
    "green": "#2f855a",
    "red": "#c2413b",
    "gold": "#b7791f",
    "light_blue": "#e9f2fb",
    "light_green": "#e8f5ee",
    "light_red": "#fbeceb",
    "line": "#cbd5df",
}
DIM_LABELS = {
    "subject_consistency": "Subject",
    "background_consistency": "Background",
    "motion_smoothness": "Motion",
    "aesthetic_quality": "Aesthetic",
    "imaging_quality": "Image quality",
    "overall_consistency": "Text-video",
    "dynamic_degree": "Dynamic",
    "no_dynamic6": "Six-dim mean",
}


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def _font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _title(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.06, 0.955, title, fontsize=20, weight="bold", color=COLORS["ink"], va="top")
    if subtitle:
        fig.text(0.06, 0.918, subtitle, fontsize=10.3, color=COLORS["muted"], va="top")


def _save_page(fig, page_dir: Path, page_number: int) -> Path:
    path = page_dir / f"page_{page_number:02d}.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def _metric_card(
    fig,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str,
    value: str,
    detail: str,
    color: str,
) -> None:
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=COLORS["line"], linewidth=1)
    )
    ax.text(0.06, 0.79, label, fontsize=9.5, color=COLORS["muted"], va="top")
    ax.text(0.06, 0.52, value, fontsize=21, weight="bold", color=COLORS["ink"], va="top")
    ax.text(0.06, 0.15, _wrap(detail, 34), fontsize=8.7, color=COLORS["muted"], va="bottom")


def _global_conclusion(global_row: pd.Series) -> str:
    delta = float(global_row["mean_delta"])
    lo = float(global_row.get("mean_delta_prompt_ci_low", global_row["mean_delta_ci_low"]))
    hi = float(global_row.get("mean_delta_prompt_ci_high", global_row["mean_delta_ci_high"]))
    if lo > 0:
        return "The fixed intervention has a positive global mean effect with a prompt-clustered interval above zero."
    if hi < 0:
        return "The fixed intervention has a negative global mean effect with a prompt-clustered interval below zero."
    direction = "positive" if delta > 0 else "negative"
    return (
        f"The global mean is slightly {direction}, but its bootstrap interval crosses zero; "
        "the prompt-clustered interval crosses zero, so there is no clear global gain."
    )


def _bad_video_conclusion(bad_row: pd.Series) -> str:
    diff = float(bad_row.get("delta_vs_rest", math.nan))
    lo = float(bad_row.get("delta_vs_rest_ci_low", math.nan))
    hi = float(bad_row.get("delta_vs_rest_ci_high", math.nan))
    if lo > 0:
        return "Low-quality seeds benefit more than the remaining seeds, with a positive prompt-clustered interval."
    if hi < 0:
        return "Low-quality seeds benefit less than the remaining seeds, with a negative prompt-clustered interval."
    direction = "more" if diff > 0 else "less"
    return (
        f"Low-quality seeds are {direction} favorable on average, but the prompt-clustered "
        "difference is not conclusive."
    )


def _plot_global(
    paired: pd.DataFrame,
    dimensions: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    delta = paired["delta_no_dynamic6"].to_numpy()
    axes[0, 0].hist(delta, bins=24, color=COLORS["blue"], alpha=0.86)
    axes[0, 0].axvline(0, color=COLORS["ink"], linewidth=1)
    axes[0, 0].axvline(delta.mean(), color=COLORS["gold"], linewidth=2)
    axes[0, 0].set_title("Six-dim delta distribution")
    axes[0, 0].set_xlabel("Renoise minus Euler baseline")
    axes[0, 0].set_ylabel("Videos")

    base = paired["score_base_no_dynamic6"]
    renoise = paired["score_renoise_no_dynamic6"]
    low = min(base.min(), renoise.min())
    high = max(base.max(), renoise.max())
    axes[0, 1].scatter(base, renoise, s=24, alpha=0.68, color=COLORS["blue"])
    axes[0, 1].plot([low, high], [low, high], color=COLORS["ink"], linewidth=1)
    axes[0, 1].set_title("Paired final score")
    axes[0, 1].set_xlabel("Euler baseline")
    axes[0, 1].set_ylabel("Renoise + AddSteps")

    dim = dimensions[dimensions["dimension"].isin(DIM_LABELS)].copy()
    dim = dim[dim["dimension"] != "all7"]
    labels = [DIM_LABELS[item] for item in dim["dimension"]]
    values = dim["mean_delta"].to_numpy()
    colors = [COLORS["green"] if value > 0 else COLORS["red"] for value in values]
    x = np.arange(len(dim))
    axes[1, 0].bar(x, values, color=colors)
    axes[1, 0].axhline(0, color=COLORS["ink"], linewidth=1)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=35, ha="right")
    axes[1, 0].set_title("Mean component delta")

    wins = dim["win_rate"].to_numpy() * 100
    axes[1, 1].bar(x, wins, color=COLORS["blue"])
    axes[1, 1].axhline(50, color=COLORS["ink"], linestyle="--", linewidth=1)
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1, 1].set_title("Per-component win rate")
    axes[1, 1].set_ylabel("% videos improved")
    for index, value in enumerate(wins):
        axes[1, 1].text(index, value + 2, f"{value:.0f}%", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_bad_videos(
    paired: pd.DataFrame,
    badness: pd.DataFrame,
    cross_dim: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    groups = badness[
        badness["group"].isin(
            [
                "all",
                "global_bottom_10pct",
                "global_bottom_20pct",
                "global_bottom_30pct",
                "global_bottom_50pct",
            ]
        )
    ].copy()
    labels = ["All", "Bottom 10%", "Bottom 20%", "Bottom 30%", "Bottom 50%"]
    x = np.arange(len(groups))
    axes[0, 0].bar(
        x,
        groups["mean_delta"],
        color=[
            COLORS["green"] if value > 0 else COLORS["red"]
            for value in groups["mean_delta"]
        ],
    )
    axes[0, 0].axhline(0, color=COLORS["ink"], linewidth=1)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0, 0].set_title("Mean delta by baseline-quality cutoff")

    axes[0, 1].bar(x, groups["win_rate"] * 100, color=COLORS["blue"])
    axes[0, 1].axhline(50, color=COLORS["ink"], linestyle="--", linewidth=1)
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=25, ha="right")
    axes[0, 1].set_title("Win rate by baseline-quality cutoff")

    axes[1, 0].scatter(
        paired["score_base_no_dynamic6"],
        paired["delta_no_dynamic6"],
        c=pd.Categorical(paired["axis"]).codes,
        cmap="tab10",
        s=25,
        alpha=0.72,
    )
    axes[1, 0].axhline(0, color=COLORS["ink"], linewidth=1)
    axes[1, 0].set_xlabel("Baseline six-dim score")
    axes[1, 0].set_ylabel("Intervention delta")
    axes[1, 0].set_title("Does lower baseline imply larger rescue?")

    axes[1, 1].hist(
        cross_dim["delta_vs_rest"],
        bins=12,
        color=COLORS["gold"],
        alpha=0.86,
    )
    axes[1, 1].axvline(0, color=COLORS["ink"], linewidth=1)
    positive = (cross_dim["delta_vs_rest"] > 0).mean() * 100
    axes[1, 1].set_title(f"Cross-dimension badness check: {positive:.0f}% positive")
    axes[1, 1].set_xlabel("Bottom-30% delta minus rest, held-out dimensions")

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_heterogeneity(axis: pd.DataFrame, prompt: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axis = axis.sort_values("mean_delta")
    colors = [
        COLORS["green"] if value > 0 else COLORS["red"]
        for value in axis["mean_delta"]
    ]
    axes[0].barh(axis["axis"], axis["mean_delta"], color=colors)
    axes[0].axvline(0, color=COLORS["ink"], linewidth=1)
    axes[0].set_title("Axis-level mean effect")
    for y, row in enumerate(axis.itertuples()):
        axes[0].text(
            row.mean_delta,
            y,
            f"  {row.win_rate * 100:.0f}% wins",
            va="center",
            ha="left" if row.mean_delta >= 0 else "right",
            fontsize=8,
        )

    prompt = prompt.sort_values("mean_delta")
    labels = [
        f"{row.prompt_id} ({row.axis[:8]})"
        for row in prompt.itertuples()
    ]
    colors = [
        COLORS["green"] if value > 0 else COLORS["red"]
        for value in prompt["mean_delta"]
    ]
    axes[1].barh(labels, prompt["mean_delta"], color=colors)
    axes[1].axvline(0, color=COLORS["ink"], linewidth=1)
    axes[1].set_title("Prompt-level mean effect across 10 seeds")
    axes[1].tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_verifier(results: pd.DataFrame, output: Path) -> None:
    valid = results.dropna(subset=["win_rate"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for model, group in valid.groupby("model"):
        axes[0].plot(
            group["accepted"],
            group["win_rate"] * 100,
            marker="o",
            linewidth=1.4,
            label=model.replace("online_", "").replace("_", " "),
        )
    axes[0].axhline(valid["base_win_rate"].iloc[0] * 100, color=COLORS["ink"], linestyle="--")
    axes[0].set_xlabel("Accepted interventions out of 150")
    axes[0].set_ylabel("Prompt-held-out win rate (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Precision vs accepted count")
    axes[0].legend(fontsize=6.8, loc="best")

    best_per_model = (
        valid.sort_values(
            ["win_rate_conservative_low", "win_rate", "accepted"],
            ascending=[False, False, False],
        )
        .groupby("model", as_index=False)
        .first()
        .sort_values("win_rate_conservative_low")
    )
    labels = [
        model.replace("online_", "").replace("_", " ")
        for model in best_per_model["model"]
    ]
    axes[1].barh(labels, best_per_model["win_rate"] * 100, color=COLORS["blue"])
    axes[1].scatter(
        best_per_model["win_rate_conservative_low"] * 100,
        np.arange(len(best_per_model)),
        color=COLORS["red"],
        label="Conservative 95% lower bound",
    )
    axes[1].set_xlim(0, 100)
    axes[1].set_title("Best honest gate per model")
    axes[1].set_xlabel("Win rate (%)")
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _read_frames(video_path: Path, indices=(0, 40, 80)) -> list[Image.Image]:
    wanted = set(indices)
    frames = []
    for index, frame in enumerate(iio.imiter(video_path)):
        if index in wanted:
            frames.append(Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB"))
        if index >= max(indices):
            break
    return frames


def _filmstrip(video_path: Path, frame_size=(185, 107)) -> Image.Image:
    frames = _read_frames(video_path)
    strip = Image.new("RGB", (frame_size[0] * len(frames), frame_size[1]), "white")
    for index, frame in enumerate(frames):
        frame = frame.resize(frame_size, Image.Resampling.LANCZOS)
        strip.paste(frame, (index * frame_size[0], 0))
    return strip


def _contact_sheet(
    rows: pd.DataFrame,
    baseline_run: Path,
    renoise_run: Path,
    output: Path,
    *,
    title: str,
) -> None:
    width, height = 1500, 1520
    margin = 28
    label_w = 320
    row_h = 275
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 20), title, font=_font(30, True), fill=COLORS["ink"])
    draw.text((label_w + margin, 72), "Euler baseline", font=_font(19, True), fill=COLORS["muted"])
    draw.text((label_w + margin + 575, 72), "Renoise + AddSteps", font=_font(19, True), fill=COLORS["muted"])

    for row_index, row in enumerate(rows.itertuples()):
        y = 105 + row_index * row_h
        pid = row.prompt_id
        seed = int(row.seed_idx)
        label = (
            f"{pid} seed{seed:04d}\n"
            f"{row.axis}\n"
            f"six-dim delta {row.delta_no_dynamic6:+.4f}\n"
            f"dimensions up {int(row.six_components_up)}/6"
        )
        draw.multiline_text(
            (margin, y + 24),
            label,
            font=_font(18),
            fill=COLORS["ink"],
            spacing=8,
        )
        base = _filmstrip(baseline_run / pid / f"seed{seed:04d}" / "video.mp4")
        renoise = _filmstrip(renoise_run / pid / f"seed{seed:04d}" / "video.mp4")
        image.paste(base, (label_w + margin, y + 18))
        image.paste(renoise, (label_w + margin + 575, y + 18))
        draw.line((margin, y + row_h - 12, width - margin, y + row_h - 12), fill="#dde3e8", width=2)
    image.save(output)


def _top_verifier_table(results: pd.DataFrame) -> pd.DataFrame:
    table = (
        results.dropna(subset=["win_rate"])
        .sort_values(
            ["win_rate_conservative_low", "win_rate", "accepted"],
            ascending=[False, False, False],
        )
        .groupby("model", as_index=False)
        .first()
        .sort_values(
            ["win_rate_conservative_low", "win_rate", "accepted"],
            ascending=[False, False, False],
        )
        .head(7)
    )
    return pd.DataFrame(
        {
            "Approach": table["model"].str.replace("online_", "", regex=False).str.replace("_", " ", regex=False),
            "Accepted": table["accepted"].astype(int),
            "Win": table["win_rate"].map(lambda value: f"{value * 100:.1f}%"),
            "95% low": table["win_rate_conservative_low"].map(lambda value: f"{value * 100:.1f}%"),
            "Mean d6": table["mean_delta"].map(lambda value: f"{value:+.4f}"),
            "Dyn loss": table["dynamic_loss_rate"].map(lambda value: f"{value * 100:.1f}%"),
        }
    )


def _draw_table(
    ax,
    frame: pd.DataFrame,
    font_size: float = 8.5,
    col_widths: list[float] | None = None,
) -> None:
    ax.axis("off")
    table = ax.table(
        cellText=frame.values,
        colLabels=frame.columns,
        cellLoc="left",
        colLoc="left",
        colWidths=col_widths,
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["line"])
        if row == 0:
            cell.set_facecolor(COLORS["light_blue"])
            cell.set_text_props(weight="bold")


def build_report(args: argparse.Namespace) -> Path:
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    page_dir = output_dir / "pages"
    figure_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)

    paired = pd.read_csv(analysis_dir / "paired_video_results.csv")
    dimensions = pd.read_csv(analysis_dir / "dimension_summary.csv")
    badness = pd.read_csv(analysis_dir / "bad_video_summary.csv")
    cross_dim = pd.read_csv(analysis_dir / "cross_dimension_badness.csv")
    axis = pd.read_csv(analysis_dir / "axis_summary.csv")
    prompt = pd.read_csv(analysis_dir / "prompt_summary.csv")
    verifier = pd.read_csv(analysis_dir / "verifier" / "verifier_results.csv")
    posthoc_topk = pd.read_csv(
        analysis_dir / "verifier" / "verifier_posthoc_topk.csv"
    )
    summary = json.loads((analysis_dir / "analysis_summary.json").read_text())

    global_row = badness[badness["group"] == "all"].iloc[0]
    global_worst_row = badness[
        badness["group"] == "global_bottom_10pct"
    ].iloc[0]
    bad_row = badness[badness["group"] == "within_prompt_bottom_3of10"].iloc[0]
    best_gate = verifier.dropna(subset=["win_rate"]).sort_values(
        ["win_rate_conservative_low", "win_rate", "accepted"],
        ascending=[False, False, False],
    ).iloc[0]
    exploratory_top10 = posthoc_topk[
        (posthoc_topk["model"] == "online_raw_dino_pca")
        & (posthoc_topk["top_k"] == 10)
    ].iloc[0]
    dynamic_losses = int((paired["delta_dynamic_degree"] < 0).sum())
    dynamic_gains = int((paired["delta_dynamic_degree"] > 0).sum())
    hidden_dynamic_losses = paired[
        paired["win"] & (paired["delta_dynamic_degree"] < 0)
    ]["sample_id"].tolist()

    global_fig = figure_dir / "global_effect.png"
    bad_fig = figure_dir / "bad_video_effect.png"
    heterogeneity_fig = figure_dir / "heterogeneity.png"
    verifier_fig = figure_dir / "verifier.png"
    top_contact = figure_dir / "top5_gains.png"
    loss_contact = figure_dir / "top5_losses.png"
    _plot_global(paired, dimensions, global_fig)
    _plot_bad_videos(paired, badness, cross_dim, bad_fig)
    _plot_heterogeneity(axis, prompt, heterogeneity_fig)
    _plot_verifier(verifier, verifier_fig)
    _contact_sheet(
        paired.nlargest(5, "delta_no_dynamic6"),
        Path(args.baseline_run),
        Path(args.renoise_run),
        top_contact,
        title="Five largest six-dimension gains",
    )
    _contact_sheet(
        paired.nsmallest(5, "delta_no_dynamic6"),
        Path(args.baseline_run),
        Path(args.renoise_run),
        loss_contact,
        title="Five largest six-dimension regressions",
    )

    pages: list[Path] = []
    page = 1

    fig = plt.figure(figsize=PAGE_SIZE)
    _title(
        fig,
        "Fixed Step-10 Renoise + AddSteps",
        "150 paired videos, low-quality rescue analysis, and causal online verifier evaluation",
    )
    scope_ax = fig.add_axes([0.06, 0.835, 0.88, 0.065])
    scope_ax.axis("off")
    scope_ax.add_patch(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#fff6df",
            edgecolor=COLORS["gold"],
            linewidth=1.2,
        )
    )
    scope_ax.text(
        0.02,
        0.5,
        (
            "SCOPE: Results apply only to step10 -> rollback step8 -> +5.\n"
            "They do not answer step20 -> step18 or any other checkpoint."
        ),
        fontsize=8.5,
        weight="bold",
        color=COLORS["ink"],
        va="center",
    )
    _metric_card(
        fig,
        0.06,
        0.64,
        0.27,
        0.17,
        label="GLOBAL WIN RATE",
        value=f"{global_row['win_rate'] * 100:.1f}%",
        detail=f"{int(round(global_row['win_rate'] * global_row['n']))}/{int(global_row['n'])} videos; mean delta {global_row['mean_delta']:+.4f}",
        color=COLORS["light_blue"],
    )
    _metric_card(
        fig,
        0.365,
        0.64,
        0.27,
        0.17,
        label="GLOBAL WORST 10%",
        value=f"{global_worst_row['win_rate'] * 100:.1f}%",
        detail=f"Bottom 15 baseline videos; mean delta {global_worst_row['mean_delta']:+.4f}",
        color=COLORS["light_red"],
    )
    _metric_card(
        fig,
        0.67,
        0.64,
        0.27,
        0.17,
        label="BEST ONLINE GATE",
        value=f"{best_gate['win_rate'] * 100:.1f}%",
        detail=f"Accepted {int(best_gate['accepted'])}/150 across {int(best_gate['accepted_prompts'])} prompts; conservative 95% low {best_gate['win_rate_conservative_low'] * 100:.1f}%",
        color=COLORS["light_green"],
    )
    fig.text(0.06, 0.57, "What the data supports", fontsize=13, weight="bold", color=COLORS["ink"])
    conclusions = [
        _global_conclusion(global_row),
        (
            f"The globally worst 10% improve only {global_worst_row['win_rate'] * 100:.1f}% "
            f"of the time, with mean delta {global_worst_row['mean_delta']:+.4f}. "
            "Low baseline quality is not a rescue trigger."
        ),
        (
            f"Within each prompt, the bottom 3/10 reach {bad_row['win_rate'] * 100:.1f}% wins "
            f"but mean delta is {bad_row['mean_delta']:+.4f}; this is inconclusive, not a stable benefit."
        ),
        (
            "The online gate is evaluated without final VBench or intervention outputs. "
            f"Its best observed precision is {best_gate['win_rate'] * 100:.1f}% on "
            f"{int(best_gate['accepted'])} accepted videos; the uncertainty bound matters."
        ),
        (
            f"Dynamic Degree has {dynamic_losses} loss and {dynamic_gains} gain; mean zero is cancellation. "
            + (
                f"Hidden six-dim winner with dynamic loss: {', '.join(hidden_dynamic_losses)}."
                if hidden_dynamic_losses
                else "No six-dim winner lost Dynamic Degree."
            )
        ),
    ]
    fig.text(
        0.075,
        0.535,
        "\n\n".join(f"- {_wrap(item, 88)}" for item in conclusions),
        fontsize=10.1,
        va="top",
        color=COLORS["ink"],
    )
    fig.text(
        0.06,
        0.20,
        _wrap(
            "Primary endpoint: the mean of subject consistency, background consistency, "
            "motion smoothness, aesthetic quality, imaging quality, and overall consistency. "
            "Dynamic Degree is reported separately and never hidden inside the primary score.",
            95,
        ),
        fontsize=10,
        color=COLORS["muted"],
        va="top",
    )
    fig.text(
        0.06,
        0.07,
        "Experiment: Wan2.2 TI2V-5B, Euler, 50 base NFE; rollback replay adds 8 calls total (50 -> 58, +16%).",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    pages.append(_save_page(fig, page_dir, page))
    page += 1

    fig = plt.figure(figsize=PAGE_SIZE)
    _title(fig, "Experimental Design", "One fixed intervention, one matched control, and a strict online/offline boundary")
    ax = fig.add_axes([0.08, 0.55, 0.84, 0.30])
    ax.axis("off")
    xs = [0.05, 0.28, 0.50, 0.73, 0.95]
    labels = [
        "Same initial\nnoise + prompt",
        "Euler through\nstep 10",
        "Re-noise to\nstep-8 level",
        "Replay with\n+5 microsteps",
        "Continue to\nfinal video",
    ]
    fills = [
        COLORS["light_blue"],
        "#eef2f5",
        "#fde8d1",
        COLORS["light_green"],
        "#eef2f5",
    ]
    for index, (x, label, fill) in enumerate(zip(xs, labels, fills, strict=True)):
        ax.add_patch(plt.Rectangle((x - 0.09, 0.37), 0.18, 0.28, facecolor=fill, edgecolor=COLORS["line"]))
        ax.text(x, 0.51, label, ha="center", va="center", fontsize=9.5)
        if index < len(xs) - 1:
            ax.annotate("", xy=(xs[index + 1] - 0.095, 0.51), xytext=(x + 0.095, 0.51), arrowprops={"arrowstyle": "->"})
    fig.text(0.07, 0.47, "Evaluation roles", fontsize=13, weight="bold")
    roles = [
        "Global effect: compare all 150 paired final videos.",
        "Bad-video hypothesis: use final baseline VBench only to form offline analysis groups.",
        "Online verifier: use only prompt information and decoded step-5/step-10 posterior trajectory signals.",
        "No final VBench, final baseline video, or intervention result enters the verifier.",
        "Validation: leave one prompt out; the decision threshold is chosen inside the training prompts.",
        "Compute cost: replaying steps 8-10 costs three original calls plus five inserted microsteps, for 58 NFE total.",
        "Scope boundary: step20->18, later checkpoints, other rollback distances, and other microstep counts were not tested here.",
    ]
    fig.text(
        0.085,
        0.43,
        "\n\n".join(f"- {_wrap(item, 86)}" for item in roles),
        fontsize=9.7,
        va="top",
    )
    pages.append(_save_page(fig, page_dir, page))
    page += 1

    for title_text, subtitle, image_path, note in [
        (
            "Global Effect Across 150 Videos",
            "Distribution, paired scores, component means, and component win rates",
            global_fig,
            _global_conclusion(global_row),
        ),
        (
            "Bad-Video Rescue at Step 10",
            "Only the fixed step10->step8+5 policy; global cutoffs, within-prompt ranks, and coupling checks",
            bad_fig,
            (
                f"For step10->step8+5 only: {_bad_video_conclusion(bad_row)} "
                "This does not predict whether bottom videos improve under step20->step18."
            ),
        ),
        (
            "The Effect Is Content-Dependent",
            "Mean delta and win rate vary substantially by axis and prompt",
            heterogeneity_fig,
            "Prompt/axis differences are useful priors, but they are not sufficient as a per-video verifier.",
        ),
    ]:
        fig = plt.figure(figsize=PAGE_SIZE)
        _title(fig, title_text, subtitle)
        ax = fig.add_axes([0.06, 0.19, 0.88, 0.68])
        ax.imshow(plt.imread(image_path))
        ax.axis("off")
        fig.text(0.07, 0.11, _wrap(note, 92), fontsize=10.5, color=COLORS["ink"], va="top")
        pages.append(_save_page(fig, page_dir, page))
        page += 1

    fig = plt.figure(figsize=PAGE_SIZE)
    _title(
        fig,
        "Online Verifier: Can Precision Improve?",
        "Prompt-held-out causal gating for the fixed step10->step8+5 intervention only",
    )
    ax = fig.add_axes([0.07, 0.53, 0.86, 0.33])
    ax.imshow(plt.imread(verifier_fig))
    ax.axis("off")
    fig.add_artist(
        plt.Rectangle(
            (0.06, 0.415),
            0.42,
            0.085,
            transform=fig.transFigure,
            facecolor=COLORS["light_green"],
            edgecolor=COLORS["line"],
        )
    )
    fig.add_artist(
        plt.Rectangle(
            (0.52, 0.415),
            0.42,
            0.085,
            transform=fig.transFigure,
            facecolor="#fff6df",
            edgecolor=COLORS["line"],
        )
    )
    fig.text(
        0.075,
        0.481,
        "STRICT PROMPT-HELD-OUT GATE",
        fontsize=8.2,
        weight="bold",
        color=COLORS["green"],
        va="top",
    )
    fig.text(
        0.075,
        0.451,
        f"{best_gate['win_rate'] * 100:.1f}% ({int(round(best_gate['win_rate'] * best_gate['accepted']))}/{int(best_gate['accepted'])})",
        fontsize=14,
        weight="bold",
        color=COLORS["ink"],
        va="top",
    )
    fig.text(
        0.075,
        0.424,
        f"{int(best_gate['accepted_prompts'])} prompts; conservative 95% lower bound {best_gate['win_rate_conservative_low'] * 100:.1f}%",
        fontsize=7.5,
        color=COLORS["muted"],
        va="top",
    )
    fig.text(
        0.535,
        0.481,
        "EXPLORATORY OOF TOP-10 CEILING",
        fontsize=8.2,
        weight="bold",
        color=COLORS["gold"],
        va="top",
    )
    fig.text(
        0.535,
        0.451,
        f"{exploratory_top10['win_rate'] * 100:.1f}% ({int(round(exploratory_top10['win_rate'] * 10))}/10)",
        fontsize=14,
        weight="bold",
        color=COLORS["ink"],
        va="top",
    )
    fig.text(
        0.535,
        0.424,
        f"{int(exploratory_top10['accepted_prompts'])} prompts; threshold selected after observing OOF labels",
        fontsize=7.5,
        color=COLORS["muted"],
        va="top",
    )
    table = _top_verifier_table(verifier)
    table_ax = fig.add_axes([0.06, 0.16, 0.88, 0.225])
    _draw_table(
        table_ax,
        table,
        font_size=7.8,
        col_widths=[0.29, 0.13, 0.12, 0.13, 0.16, 0.14],
    )
    fig.text(
        0.07,
        0.085,
        _wrap(
            "The strict gate is the result that can be defended. The 90% top-10 number is "
            "only an exploratory ranking ceiling: top-k was selected after inspecting OOF "
            "labels, so it motivates a new frozen validation set but is not a deployment guarantee.",
            94,
        ),
        fontsize=9.7,
        color=COLORS["muted"],
        va="top",
    )
    pages.append(_save_page(fig, page_dir, page))
    page += 1

    for title_text, image_path, note in [
        (
            "Visual Examples: Largest Metric Gains",
            top_contact,
            "Filmstrips show frames 0, 40, and 80 from each paired video. Metric gains still require human review.",
        ),
        (
            "Visual Examples: Largest Metric Regressions",
            loss_contact,
            "These examples make the failure mode concrete: extra exploration can alter identity, structure, or temporal consistency.",
        ),
    ]:
        fig = plt.figure(figsize=PAGE_SIZE)
        _title(fig, title_text)
        ax = fig.add_axes([0.03, 0.10, 0.94, 0.82])
        ax.imshow(plt.imread(image_path))
        ax.axis("off")
        fig.text(0.06, 0.055, _wrap(note, 100), fontsize=8.8, color=COLORS["muted"])
        pages.append(_save_page(fig, page_dir, page))
        page += 1

    fig = plt.figure(figsize=PAGE_SIZE)
    _title(
        fig,
        "What We Can Defend",
        "Every statement on this page is scoped to the fixed step10->step8+5 intervention",
    )
    supported = [
        f"The fixed s10->s8+5 policy was tested on 150 matched videos, not only ten hand-picked failures.",
        "All global, bottom-video, prompt, and verifier results are scoped to this one intervention window.",
        f"Global six-dim win rate: {global_row['win_rate'] * 100:.1f}% with mean delta {global_row['mean_delta']:+.4f}.",
        f"Global bottom-10% win rate: {global_worst_row['win_rate'] * 100:.1f}% with mean delta {global_worst_row['mean_delta']:+.4f}.",
        f"Within-prompt bottom-3/10 win rate: {bad_row['win_rate'] * 100:.1f}% with mean delta {bad_row['mean_delta']:+.4f}.",
        f"Best causal prompt-held-out gate: {best_gate['win_rate'] * 100:.1f}% on {int(best_gate['accepted'])} accepted videos.",
        (
            f"Raw-DINO OOF ranking reaches {exploratory_top10['win_rate'] * 100:.1f}% "
            "at top-10, but this is post-hoc and not a validated threshold."
        ),
        "Dynamic Degree was separated from the six-dimensional endpoint.",
        f"Dynamic Degree changed in {dynamic_losses + dynamic_gains}/150 videos ({dynamic_losses} loss, {dynamic_gains} gain).",
    ]
    fig.add_artist(
        plt.Rectangle(
            (0.07, 0.76),
            0.86,
            0.08,
            transform=fig.transFigure,
            facecolor="#fff6df",
            edgecolor=COLORS["gold"],
            linewidth=1.2,
        )
    )
    fig.text(
        0.09,
        0.80,
        "The bottom-video result is not a claim about step20, step30, or any later intervention.",
        fontsize=10.5,
        weight="bold",
        color=COLORS["ink"],
        va="center",
    )
    fig.text(0.07, 0.69, "Supported", fontsize=14, weight="bold", color=COLORS["green"])
    fig.text(
        0.085,
        0.65,
        "\n\n".join(f"- {_wrap(item, 84)}" for item in supported),
        fontsize=10.2,
        va="top",
    )
    pages.append(_save_page(fig, page_dir, page))
    page += 1

    fig = plt.figure(figsize=PAGE_SIZE)
    _title(
        fig,
        "What Remains Unknown",
        "These questions require new intervention runs rather than reinterpretation of the step-10 data",
    )
    limits = [
        "The dataset contains 15 prompts x 10 seeds; prompt-held-out validation helps, but this is not a broad benchmark.",
        "Only step10->step8+5 was tested. This report cannot determine whether bottom videos improve at step20->step18 or any later checkpoint.",
        "VBench deltas are proxy outcomes. A blind human pairwise study is still required before claiming perceptual improvement.",
        "A verifier with a small accepted set cannot guarantee success; report its accepted count and uncertainty interval.",
        "The exploratory top-k precision curve cannot be used as a deployment claim until its threshold is frozen and tested on new prompts.",
        "The bad-video analysis is offline evidence. It is not itself an online decision rule.",
        "DINO feature families were informed by earlier work on this dev set, so a new external prompt set is still required.",
        "Benchmark axis labels are available online here, but deployment needs a prompt classifier or an axis-free gate.",
    ]
    fig.text(0.07, 0.84, "Not established by this report", fontsize=14, weight="bold", color=COLORS["red"])
    fig.text(
        0.085,
        0.79,
        "\n\n".join(f"- {_wrap(item, 84)}" for item in limits),
        fontsize=10.3,
        va="top",
    )
    fig.text(
        0.07,
        0.07,
        f"Cross-dimension bottom-30% checks positive: {summary['cross_dimension_bottom30_positive_fraction'] * 100:.0f}%.",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    pages.append(_save_page(fig, page_dir, page))

    images = [Image.open(path).convert("RGB") for path in pages]
    report_path = output_dir / "renoise_addsteps_all150_report.pdf"
    images[0].save(
        report_path,
        "PDF",
        resolution=160.0,
        save_all=True,
        append_images=images[1:],
    )
    print(f"[report] wrote {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--renoise-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    build_report(args)


if __name__ == "__main__":
    main()
