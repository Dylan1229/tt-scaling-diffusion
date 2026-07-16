from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, ImageFont


PAGE_SIZE = (8.27, 11.69)  # A4 portrait, inches
CORE6 = [
    "aesthetic_quality",
    "background_consistency",
    "imaging_quality",
    "motion_smoothness",
    "overall_consistency",
    "subject_consistency",
]
ALL7 = CORE6 + ["dynamic_degree"]
METRIC_LABELS = {
    "no_dynamic6": "no-dyn6",
    "all7": "all7",
    "imaging_quality": "IQ",
    "dynamic_degree": "Dynamic",
    "overall_consistency": "OC",
}
AXIS_LABELS = {
    "subject_consistency": "subject",
    "human_action": "human",
    "scene": "scene",
}


def _ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    fig_dir = out_dir / "figures"
    page_dir = out_dir / "pages"
    fig_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, page_dir


def _wrap(text: str, width: int = 92) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def _save_page(pdf: PdfPages, fig, page_dir: Path, page_num: int) -> int:
    pdf.savefig(fig, bbox_inches="tight")
    fig.savefig(page_dir / f"page_{page_num:02d}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return page_num + 1


def _read_mid_frame(path: Path) -> Image.Image:
    frames = []
    for i, frame in enumerate(iio.imiter(path)):
        if i in (20, 40, 60):
            frames.append(frame)
        if i >= 60:
            break
    if not frames:
        frame = iio.imread(path, index=0)
    else:
        frame = frames[len(frames) // 2]
    return Image.fromarray(np.asarray(frame).astype("uint8")).convert("RGB")


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _load_vbench_wide(base_csv: Path, renoise_csv: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(base_csv)
    renoise = pd.read_csv(renoise_csv)
    key = ["prompt_id", "prompt_text", "axis", "seed_idx", "dimension"]
    merged = base.merge(renoise, on=key, suffixes=("_euler_base", "_renoise"))
    merged["delta"] = merged["score_renoise"] - merged["score_euler_base"]

    wide = (
        merged.pivot_table(
            index=["prompt_id", "prompt_text", "axis", "seed_idx"],
            columns="dimension",
            values=["score_euler_base", "score_renoise", "delta"],
        )
        .reset_index()
    )
    wide.columns = ["_".join([str(c) for c in col if c]) for col in wide.columns.to_flat_index()]

    for prefix in ["score_euler_base", "score_renoise", "delta"]:
        wide[f"{prefix}_no_dynamic6"] = wide[[f"{prefix}_{d}" for d in CORE6]].mean(axis=1)
        wide[f"{prefix}_all7"] = wide[[f"{prefix}_{d}" for d in ALL7]].mean(axis=1)

    for metric in ["no_dynamic6", "all7", "imaging_quality", "dynamic_degree", "overall_consistency"]:
        wide[f"win_{metric}"] = wide[f"delta_{metric}"] > 0
    wide["n_components_up_core6"] = sum((wide[f"delta_{d}"] > 0).astype(int) for d in CORE6)
    wide["robust_win_no_dynamic6"] = (wide["delta_no_dynamic6"] > 0.002) & (
        wide["n_components_up_core6"] >= 4
    )

    summary = []
    for metric in ["no_dynamic6", "all7", "imaging_quality", "dynamic_degree", "overall_consistency"]:
        summary.append(
            {
                "metric": metric,
                "mean_base": wide[f"score_euler_base_{metric}"].mean(),
                "mean_renoise": wide[f"score_renoise_{metric}"].mean(),
                "mean_delta": wide[f"delta_{metric}"].mean(),
                "win_rate": wide[f"win_{metric}"].mean(),
                "n_win": int(wide[f"win_{metric}"].sum()),
                "n": len(wide),
            }
        )
    summary_df = pd.DataFrame(summary)

    merged.to_csv(out_dir / "vbench_component_long.csv", index=False)
    wide.to_csv(out_dir / "vbench_rescue_wide.csv", index=False)
    summary_df.to_csv(out_dir / "vbench_rescue_summary.csv", index=False)
    return wide, summary_df


def _make_metric_summary_figure(summary_df: pd.DataFrame, fig_dir: Path) -> Path:
    metrics = summary_df["metric"].tolist()
    deltas = summary_df["mean_delta"].to_numpy()
    wins = summary_df["win_rate"].to_numpy() * 100

    fig, ax1 = plt.subplots(figsize=(9.2, 4.8))
    colors = ["#2ca25f" if v > 0 else "#de2d26" if v < 0 else "#777777" for v in deltas]
    x = np.arange(len(metrics))
    ax1.bar(x, deltas * 1000, color=colors, width=0.62)
    ax1.axhline(0, color="#333333", linewidth=0.8)
    ax1.set_ylabel("Mean delta (x1000)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=25, ha="right")
    ax1.set_title("Worst-10 Same-Scheduler Rescue Result")

    ax2 = ax1.twinx()
    ax2.plot(x, wins, color="#1f78b4", marker="o", linewidth=2)
    ax2.set_ylabel("Win rate (%)")
    ax2.set_ylim(0, 100)
    for i, (d, w) in enumerate(zip(deltas, wins, strict=True)):
        ax1.text(i, d * 1000 + (0.3 if d >= 0 else -0.9), f"{d:+.4f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=8)
        ax2.text(i, w + 3, f"{w:.0f}%", ha="center", fontsize=8, color="#1f78b4")
    fig.tight_layout()
    out = fig_dir / "rescue_metric_summary.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _make_heatmap(wide: pd.DataFrame, fig_dir: Path) -> Path:
    dims = CORE6 + ["dynamic_degree"]
    labels = [f"{r.prompt_id}_s{int(r.seed_idx):04d}" for r in wide.itertuples()]
    matrix = wide[[f"delta_{d}" for d in dims]].to_numpy()
    vmax = max(0.015, float(np.nanmax(np.abs(matrix))))

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels(dims, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("Component Deltas: Renoise+Microsteps minus Euler Control")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="delta")
    fig.tight_layout()
    out = fig_dir / "component_delta_heatmap.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _make_timeline(fig_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.axis("off")
    xs = [0.08, 0.28, 0.48, 0.70, 0.90]
    labels = [
        "Normal denoise\nthrough step 20",
        "Detect bad local\ntrajectory",
        "Add noise back\nto step-18 level",
        "Replay step18..20\nwith +5 microsteps",
        "Resume normal\nschedule",
    ]
    colors = ["#d9eaf7", "#fee8c8", "#fdbb84", "#c7e9c0", "#dadaeb"]
    for i, (x, label, color) in enumerate(zip(xs, labels, colors, strict=True)):
        ax.add_patch(
            plt.Rectangle((x - 0.075, 0.36), 0.15, 0.26, color=color, ec="#555555", lw=1.2)
        )
        ax.text(x, 0.49, label, ha="center", va="center", fontsize=10)
        if i < len(xs) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.08, 0.49), xytext=(x + 0.08, 0.49), arrowprops={"arrowstyle": "->", "lw": 1.6})
    ax.text(
        0.5,
        0.18,
        "Key implementation choice: do not encode this as a static non-monotone scheduler. "
        "The trigger step must first land normally, then the latent is re-noised and replayed in a custom first-order loop.",
        ha="center",
        va="center",
        fontsize=10,
        wrap=True,
    )
    fig.tight_layout()
    out = fig_dir / "renoise_timeline.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _sample_rows(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.sort_values("delta_no_dynamic6", ascending=False)[
        [
            "prompt_id",
            "seed_idx",
            "axis",
            "delta_no_dynamic6",
            "delta_imaging_quality",
            "delta_dynamic_degree",
            "delta_overall_consistency",
            "n_components_up_core6",
            "robust_win_no_dynamic6",
        ]
    ]


def _make_contact_sheet(
    wide: pd.DataFrame,
    baseline_run: Path,
    renoise_run: Path,
    fig_dir: Path,
    *,
    start: int,
    count: int,
) -> Path:
    rows = _sample_rows(wide).iloc[start : start + count]
    thumb_w, thumb_h = 300, 173
    label_w = 280
    row_h = thumb_h + 34
    header_h = 58
    margin = 18
    sheet_w = margin * 3 + label_w + thumb_w * 2
    sheet_h = header_h + row_h * len(rows) + margin
    img = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(18)
    small = _load_font(14)
    bold = _load_font(18, bold=True)
    draw.text((margin, 16), "Euler control vs Renoise+Microsteps, mid-frame contact sheet", font=bold, fill="#111111")
    draw.text((margin + label_w, 34), "Euler control", font=small, fill="#555555")
    draw.text((margin + label_w + thumb_w + margin, 34), "Renoise + microsteps", font=small, fill="#555555")

    for ridx, row in enumerate(rows.itertuples(index=False)):
        y = header_h + ridx * row_h
        pid = row.prompt_id
        seed = int(row.seed_idx)
        axis = AXIS_LABELS.get(str(row.axis), str(row.axis))
        label = (
            f"{pid} seed{seed:04d}\n"
            f"{axis}\n"
            f"nd6 {row.delta_no_dynamic6:+.4f}  IQ {row.delta_imaging_quality:+.4f}\n"
            f"OC {row.delta_overall_consistency:+.4f}  up {int(row.n_components_up_core6)}/6"
        )
        draw.text((margin, y + 14), label, font=small, fill="#111111")
        base_video = baseline_run / pid / f"seed{seed:04d}" / "video.mp4"
        ren_video = renoise_run / pid / f"seed{seed:04d}" / "video.mp4"
        for col, video in enumerate([base_video, ren_video]):
            frame = _read_mid_frame(video)
            frame.thumbnail((thumb_w, thumb_h))
            x = margin + label_w + col * (thumb_w + margin)
            img.paste(frame, (x, y + 10))
            draw.rectangle((x, y + 10, x + frame.width, y + 10 + frame.height), outline="#555555")
        draw.line((margin, y + row_h - 1, sheet_w - margin, y + row_h - 1), fill="#dddddd")

    out = fig_dir / f"contact_sheet_part{start // count + 1:02d}.png"
    img.save(out)
    return out


def _load_existing_prior_stats(root: Path) -> dict[str, str]:
    stats = {}
    summary_md = root / "benefit_rules_report" / "benefit_rules_summary.md"
    if summary_md.exists():
        for line in summary_md.read_text().splitlines():
            if line.startswith("- "):
                k, _, v = line[2:].partition(":")
                if v:
                    stats[k.strip()] = v.strip()
    for name, path in {
        "Rule A selected": root / "online_causal_ruleA_step10_s004_gap4_ridge3_ge_0p6687.csv",
        "Rule B selected": root / "online_causal_ruleB_step10_patchmean_tail20_low_le_0p7369.csv",
    }.items():
        if path.exists():
            df = pd.read_csv(path)
            stats[name] = f"{int(df['six_win'].sum())}/{len(df)} six-dim wins"
    return stats


def _page_title(fig, title: str, subtitle: str | None = None) -> None:
    fig.text(0.06, 0.955, title, fontsize=20, weight="bold", va="top")
    if subtitle:
        fig.text(0.06, 0.915, subtitle, fontsize=10.5, color="#555555", va="top")


def _add_table(ax, df: pd.DataFrame, bbox=(0, 0, 1, 1), font_size=8.5) -> None:
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e8eef7")
            cell.set_text_props(weight="bold")
        cell.set_edgecolor("#b8c2d2")


def build_report(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    fig_dir, page_dir = _ensure_dirs(out_dir)

    wide, summary_df = _load_vbench_wide(Path(args.euler_vbench_csv), Path(args.renoise_vbench_csv), out_dir)
    metric_fig = _make_metric_summary_figure(summary_df, fig_dir)
    heatmap_fig = _make_heatmap(wide, fig_dir)
    timeline_fig = _make_timeline(fig_dir)
    contact1 = _make_contact_sheet(
        wide,
        Path(args.euler_run),
        Path(args.renoise_run),
        fig_dir,
        start=0,
        count=5,
    )
    contact2 = _make_contact_sheet(
        wide,
        Path(args.euler_run),
        Path(args.renoise_run),
        fig_dir,
        start=5,
        count=5,
    )

    prior_stats = _load_existing_prior_stats(Path(args.prior_analysis_root))
    sample_df = _sample_rows(wide).copy()
    for col in ["delta_no_dynamic6", "delta_imaging_quality", "delta_dynamic_degree", "delta_overall_consistency"]:
        sample_df[col] = sample_df[col].map(lambda x: f"{x:+.4f}")
    sample_df["axis"] = sample_df["axis"].map(lambda x: AXIS_LABELS.get(str(x), str(x)))
    sample_df["robust_win_no_dynamic6"] = sample_df["robust_win_no_dynamic6"].map(lambda x: "yes" if x else "no")
    sample_df = sample_df.rename(
        columns={
            "prompt_id": "prompt",
            "seed_idx": "seed",
            "axis": "axis",
            "delta_no_dynamic6": "nd6 delta",
            "delta_imaging_quality": "IQ delta",
            "delta_dynamic_degree": "dyn delta",
            "delta_overall_consistency": "OC delta",
            "n_components_up_core6": "dims up",
            "robust_win_no_dynamic6": "robust",
        }
    )

    summary_fmt = summary_df.copy()
    summary_fmt["metric"] = summary_fmt["metric"].map(lambda x: METRIC_LABELS.get(str(x), str(x)))
    summary_fmt["mean_delta"] = summary_fmt["mean_delta"].map(lambda x: f"{x:+.4f}")
    summary_fmt["mean_base"] = summary_fmt["mean_base"].map(lambda x: f"{x:.4f}")
    summary_fmt["mean_renoise"] = summary_fmt["mean_renoise"].map(lambda x: f"{x:.4f}")
    summary_fmt["win_rate"] = summary_fmt["win_rate"].map(lambda x: f"{100*x:.0f}%")
    summary_fmt["wins"] = summary_fmt["n_win"].astype(str) + "/" + summary_fmt["n"].astype(str)
    summary_fmt = summary_fmt[["metric", "mean_base", "mean_renoise", "mean_delta", "win_rate", "wins"]]
    summary_fmt = summary_fmt.rename(
        columns={
            "mean_base": "base",
            "mean_renoise": "renoise",
            "mean_delta": "delta",
            "win_rate": "win",
        }
    )

    report_path = out_dir / "renoise_microstep_weekly_report.pdf"
    page = 1
    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=PAGE_SIZE)
        _page_title(
            fig,
            "Renoise + Microsteps Weekly Report",
            "Targeted rescue, metric guardrails, and what we can defend from current evidence.",
        )
        body = (
            "Main conclusion: the implementation is now runnable and verified, but the naive step20 -> step18 "
            "+5 replay policy is not a reliable universal rescue policy on the worst-10 set. Same-scheduler "
            "Euler control shows no_dynamic6 mean delta -0.0020 with 3/10 nominal wins and 2/10 robust wins. "
            "The two human_action / harp samples improved cleanly, while several subject_consistency and scene "
            "samples regressed. This supports the mentor's direction: do targeted rescue and prompt/trajectory "
            "gating, not global add-step averaging or IQ-only selection."
        )
        fig.text(0.06, 0.84, _wrap(body, 92), fontsize=11.5, va="top")
        ax = fig.add_axes([0.06, 0.46, 0.88, 0.27])
        _add_table(ax, summary_fmt, font_size=8.5)
        checks = [
            "Code branch: codex/renoise-microstep-report, based on origin/main plus microstep grid and DLBS baselines.",
            "Generated 10/10 same-scheduler Euler controls and 10/10 Renoise+Microsteps candidates.",
            "VBench split metrics completed for subject/background/motion/dynamic/aesthetic/IQ/overall.",
            "Dynamic Degree is reported separately and is not hidden inside the primary no_dynamic6 metric.",
            "Human review is not yet a scored blind preference study; contact sheets are included for visual inspection.",
        ]
        fig.text(0.06, 0.37, "Verification status", fontsize=13, weight="bold")
        fig.text(0.08, 0.34, "\n".join(f"- {c}" for c in checks), fontsize=10.3, va="top")
        fig.text(0.06, 0.06, f"Generated from {out_dir}", fontsize=8.5, color="#666666")
        page = _save_page(pdf, fig, page_dir, page)

        fig = plt.figure(figsize=PAGE_SIZE)
        _page_title(fig, "Algorithm: What We Actually Implemented")
        img = plt.imread(timeline_fig)
        ax = fig.add_axes([0.05, 0.58, 0.90, 0.30])
        ax.imshow(img)
        ax.axis("off")
        details = [
            "Why a custom loop: a static non-monotone sigma list would change the trigger step's sigma_next. That is not the meeting algorithm.",
            "Current clean constraint: first-order Euler/Euler-SDE only. UniPC is a multi-step solver; replay would need solver-history repair.",
            "For trigger_step=20, rollback_to_step=18, extra_microsteps=5, the trace records 8 replay calls: 3 original calls in the rollback window plus 5 extra calls.",
            "Smoke run verified video output, trace JSON, and valid H264 MP4 before full worst10 generation.",
        ]
        fig.text(0.07, 0.50, "\n".join(f"- {d}" for d in details), fontsize=10.8, va="top")
        page = _save_page(pdf, fig, page_dir, page)

        fig = plt.figure(figsize=PAGE_SIZE)
        _page_title(fig, "Prior Evidence: Plain Add-Step Is Not Enough")
        prior_lines = [
            f"{k}: {v}" for k, v in prior_stats.items()
        ] or ["Prior summary files were not found."]
        text = (
            "The existing 150 x 9 add-step sweep already showed why the week should not be judged by IQ-only or global averages. "
            "IQ gain and six-dim gain rates are both around chance; clean gains are much rarer; hidden losses exist. "
            "The strongest online/non-VBench signals were early DINO trajectory gates, not prompt text alone."
        )
        fig.text(0.06, 0.86, _wrap(text, 92), fontsize=11.0, va="top")
        fig.text(0.08, 0.70, "\n".join(f"- {line}" for line in prior_lines), fontsize=10.5, va="top")
        fig.text(
            0.06,
            0.32,
            _wrap(
                "Defensible interpretation: prompt category is a useful prior, but not a verifier. "
                "The policy must condition on online trajectory signals and must be evaluated with split metrics plus visual review.",
                92,
            ),
            fontsize=11.0,
            va="top",
        )
        page = _save_page(pdf, fig, page_dir, page)

        fig = plt.figure(figsize=PAGE_SIZE)
        _page_title(fig, "Targeted Rescue Result: Summary")
        ax = fig.add_axes([0.08, 0.48, 0.84, 0.36])
        ax.imshow(plt.imread(metric_fig))
        ax.axis("off")
        notes = (
            "The naive Renoise+Microsteps setting slightly improves mean IQ (+0.0009) but hurts the primary no_dynamic6 aggregate (-0.0020) "
            "and Overall Consistency (-0.0012). Dynamic Degree stayed at 1.0 for all ten clips in this same-scheduler comparison, "
            "so this result is not caused by the static-video failure mode."
        )
        fig.text(0.07, 0.40, _wrap(notes, 92), fontsize=11.0, va="top")
        page = _save_page(pdf, fig, page_dir, page)

        fig = plt.figure(figsize=PAGE_SIZE)
        _page_title(fig, "Targeted Rescue Result: Per-Sample Components")
        ax = fig.add_axes([0.07, 0.42, 0.86, 0.47])
        ax.imshow(plt.imread(heatmap_fig))
        ax.axis("off")
        fig.text(
            0.07,
            0.34,
            _wrap(
                "The pattern is not random noise: both human_action / harp samples improved across 5/6 no-dynamic components. "
                "Most subject_consistency swimming samples regressed on the aggregate despite some IQ increases. "
                "This is exactly why the policy needs a gate rather than unconditional replay.",
                92,
            ),
            fontsize=11.0,
            va="top",
        )
        page = _save_page(pdf, fig, page_dir, page)

        fig = plt.figure(figsize=PAGE_SIZE)
        _page_title(fig, "Per-Sample Table")
        ax = fig.add_axes([0.04, 0.08, 0.92, 0.80])
        _add_table(ax, sample_df, font_size=7.2)
        page = _save_page(pdf, fig, page_dir, page)

        for title, image_path in [
            ("Visual Contact Sheet: Stronger Half", contact1),
            ("Visual Contact Sheet: Weaker Half", contact2),
        ]:
            fig = plt.figure(figsize=PAGE_SIZE)
            _page_title(fig, title)
            ax = fig.add_axes([0.03, 0.06, 0.94, 0.84])
            ax.imshow(plt.imread(image_path))
            ax.axis("off")
            page = _save_page(pdf, fig, page_dir, page)

        fig = plt.figure(figsize=PAGE_SIZE)
        _page_title(fig, "Defense Notes and Next Decisions")
        bullets = [
            "What is done: algorithmic Renoise+Microsteps runner, same-scheduler control, VBench split metrics, contact sheets, and reproducible report generation.",
            "What did not work: blindly applying step20 -> step18 +5 to the worst10 set. It is not a stable rescue policy.",
            "What looks promising: human_action / harp-style samples improved; this matches the earlier observation that some motion/texture prompts can benefit from extra local exploration.",
            "What must not be claimed: IQ improvements alone prove quality. In this run IQ win rate is 6/10 while no_dynamic6 win rate is only 3/10.",
            "Next experiment: use the DINO trajectory gate to trigger Renoise only on likely-benefit samples, then rerun worst10 plus a small prompt-stratified set.",
            "Wan13B decision: after the gated policy is defined, run a small Wan13B smoke plus blind human pairwise preference. Do not spend a full 13B sweep on an ungated policy.",
        ]
        fig.text(0.07, 0.84, "\n\n".join(f"- {b}" for b in bullets), fontsize=11.0, va="top")
        page = _save_page(pdf, fig, page_dir, page)

    print(f"[report] wrote {report_path}")
    print(f"[report] figures: {fig_dir}")
    print(f"[report] pages: {page_dir}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/analysis/renoise_microstep_worst10_001")
    parser.add_argument(
        "--euler-vbench-csv",
        default="runs/vbench_baseline/worst10_euler_control_001/vbench_scores_long.csv",
    )
    parser.add_argument(
        "--renoise-vbench-csv",
        default="runs/vbench_renoise_microsteps/renoise_worst10_s20_to_s18_x05_001/vbench_scores_long.csv",
    )
    parser.add_argument("--euler-run", default="runs/baseline/worst10_euler_control_001")
    parser.add_argument("--renoise-run", default="runs/renoise_microsteps/renoise_worst10_s20_to_s18_x05_001")
    parser.add_argument("--prior-analysis-root", default="runs/analysis/all150_comprehensive_vbench_001")
    args = parser.parse_args()
    build_report(args)


if __name__ == "__main__":
    main()
