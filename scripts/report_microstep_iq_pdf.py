"""Generate a visual PDF report for +5 microstep image-quality behavior."""

from __future__ import annotations

import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "runs/analysis/microstep_iq_patterns_001"
TAIL_ANALYSIS = ROOT / "runs/analysis/iq_tail20_other_steps_001"
OUT_PDF = ANALYSIS / "microstep_image_quality_report.pdf"
OUT_RASTER_PDF = ANALYSIS / "microstep_image_quality_report_raster.pdf"
FIG_DIR = ANALYSIS / "report_figures"
PAGE_DIR = ANALYSIS / "report_pages"
FONT_DIR = ANALYSIS / "fonts"

REGULAR_FONT = FONT_DIR / "NotoSansCJKsc-Regular.otf"
BOLD_FONT = FONT_DIR / "NotoSansCJKsc-Bold.otf"

SINGLE_VARIANTS = ["s10x05", "s15x05", "s20x05", "s25x05", "s30x05", "s35x05", "s40x05", "s45x05", "s50x05"]

COLORS = {
    "bg": "#F8F7F2",
    "ink": "#24272E",
    "muted": "#687181",
    "grid": "#DAD8D0",
    "green": "#148453",
    "red": "#BF3838",
    "amber": "#A06E23",
    "blue": "#315FA8",
    "card": "#FFFFFF",
    "soft_green": "#DDEFE6",
    "soft_red": "#F2DCDC",
    "soft_blue": "#E7EDF8",
}


def font(size: int, bold: bool = False) -> FontProperties:
    return FontProperties(fname=str(BOLD_FONT if bold else REGULAR_FONT), size=size)


def new_page() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=COLORS["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def title(ax: plt.Axes, text: str, subtitle: str | None = None) -> None:
    ax.text(0.055, 0.925, text, fontproperties=font(30, True), color=COLORS["ink"], va="top")
    if subtitle:
        ax.text(0.055, 0.865, subtitle, fontproperties=font(13), color=COLORS["muted"], va="top")


def wrapped(ax: plt.Axes, x: float, y: float, text: str, width: int, size: int = 14, color: str | None = None, bold: bool = False, leading: float = 1.3) -> float:
    color = color or COLORS["ink"]
    lines: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=width, break_long_words=True, replace_whitespace=False))
    line_h = 0.028 * (size / 14) * leading
    yy = y
    for line in lines:
        ax.text(x, yy, line, fontproperties=font(size, bold), color=color, va="top")
        yy -= line_h
    return yy


def rounded_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], fc: str = "card", ec: str = "grid", radius: float = 0.02) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            facecolor=COLORS[fc],
            edgecolor=COLORS[ec],
            linewidth=1.0,
        )
    )


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "step": pd.read_csv(ANALYSIS / "iq_single_step_summary_combined_by_step.csv"),
        "axis": pd.read_csv(ANALYSIS / "iq_single_step_summary_by_axis.csv"),
        "pattern": pd.read_csv(ANALYSIS / "iq_single_step_pattern_counts.csv"),
        "sample": pd.read_csv(ANALYSIS / "iq_single_step_sample_summary_dedup.csv"),
        "long": pd.read_csv(ANALYSIS / "iq_single_step_long_dedup.csv"),
        "prompt": pd.read_csv(ANALYSIS / "iq_single_step_prompt_summary.csv"),
        "corr": pd.read_csv(ANALYSIS / "iq_pattern_correlations_with_baseline_metrics.csv"),
    }


def delta_color(v: float) -> str:
    if v > 0.05:
        return COLORS["green"]
    if v < -0.05:
        return COLORS["red"]
    return COLORS["amber"]


def variant_label(v: str) -> str:
    return v.replace("x05", "")


def add_figure_image(ax: plt.Axes, image_path: Path, box: tuple[float, float, float, float]) -> None:
    img = Image.open(image_path).convert("RGB")
    x, y, w, h = box
    iax = ax.figure.add_axes([x, y, w, h])
    iax.imshow(img)
    iax.set_axis_off()


def save_page(pdf: PdfPages, fig: plt.Figure, page_paths: list[Path]) -> None:
    """Save both the vector page and a raster page for robust PDF export."""
    pdf.savefig(fig)
    page_path = PAGE_DIR / f"page_{len(page_paths) + 1:02d}.png"
    fig.savefig(page_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    page_paths.append(page_path)


def write_raster_pdf(page_paths: list[Path]) -> None:
    """Build a highly compatible PDF where each page is a raster image."""
    images = [Image.open(p).convert("RGB") for p in page_paths]
    if not images:
        raise RuntimeError("no raster pages to write")
    first, rest = images[0], images[1:]
    first.save(OUT_RASTER_PDF, "PDF", resolution=180.0, save_all=True, append_images=rest)
    for img in images:
        img.close()


def save_step_chart(step: pd.DataFrame) -> Path:
    path = FIG_DIR / "checkpoint_mean_delta.png"
    fig, ax1 = plt.subplots(figsize=(9.7, 3.5), facecolor=COLORS["card"])
    steps = step.sort_values("step")
    x = np.arange(len(steps))
    bars = ax1.bar(
        x,
        steps["mean_delta"],
        color=[COLORS["green"] if v >= 0 else COLORS["red"] for v in steps["mean_delta"]],
        width=0.62,
    )
    ax1.axhline(0, color=COLORS["muted"], lw=1)
    ax1.set_ylabel("平均 ΔIQ", fontproperties=font(11))
    ax1.set_xticks(x)
    ax1.set_xticklabels([variant_label(v) for v in steps["variant"]], fontproperties=font(10))
    ax1.tick_params(axis="y", labelsize=9)
    ax1.grid(axis="y", color=COLORS["grid"], alpha=0.65)
    ax1.set_title("单 checkpoint +5：平均效果很弱，s10 相对最不差", fontproperties=font(14, True), color=COLORS["ink"])
    for rect, v in zip(bars, steps["mean_delta"]):
        ax1.text(rect.get_x() + rect.get_width() / 2, v + (0.035 if v >= 0 else -0.06), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontproperties=font(9), color=delta_color(float(v)))
    ax2 = ax1.twinx()
    ax2.plot(x, steps["improve_rate"], marker="o", color=COLORS["blue"], lw=2)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("变好比例", fontproperties=font(11), color=COLORS["blue"])
    ax2.tick_params(axis="y", labelcolor=COLORS["blue"], labelsize=9)
    for spine in ax1.spines.values():
        spine.set_color(COLORS["grid"])
    for spine in ax2.spines.values():
        spine.set_color(COLORS["grid"])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_heatmap(long: pd.DataFrame, sample: pd.DataFrame) -> Path:
    path = FIG_DIR / "all34_delta_heatmap.png"
    order = sample.sort_values("mean_delta", ascending=False)["sample_id"].tolist()
    mat = long.pivot_table(index="sample_id", columns="variant", values="delta").loc[order, SINGLE_VARIANTS]
    labels = sample.set_index("sample_id").loc[order].apply(lambda r: f"{r.name.replace('_', ' ')}  {r['axis']}", axis=1).tolist()
    fig, ax = plt.subplots(figsize=(9.6, 6.3), facecolor=COLORS["card"])
    vmax = max(1.0, float(np.nanmax(np.abs(mat.to_numpy()))))
    im = ax.imshow(mat.to_numpy(), cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(SINGLE_VARIANTS)))
    ax.set_xticklabels([variant_label(v) for v in SINGLE_VARIANTS], fontproperties=font(10))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontproperties=font(6))
    ax.set_title("34 个 video × 9 个 checkpoint 的 ΔIQ 热力图（按 video 平均增益排序）", fontproperties=font(13, True))
    ax.tick_params(length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            color = "white" if abs(val) > vmax * 0.55 else COLORS["ink"]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=5.5, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.ax.set_ylabel("variant - baseline", fontproperties=font(8))
    fig.tight_layout()
    fig.savefig(path, dpi=210)
    plt.close(fig)
    return path


def save_pattern_chart(sample: pd.DataFrame) -> Path:
    path = FIG_DIR / "pattern_distribution.png"
    dist = sample["improved_count"].value_counts().sort_index()
    pattern_order = ["robust_improver", "mild_improver", "mixed_or_mild_regressor", "robust_regressor"]
    pattern_counts = sample["pattern"].value_counts().reindex(pattern_order).fillna(0)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), facecolor=COLORS["card"], gridspec_kw={"width_ratios": [1.1, 1]})
    axs[0].bar(dist.index, dist.values, color=COLORS["blue"], width=0.7)
    axs[0].set_xlabel("9 个 checkpoint 里变好的次数", fontproperties=font(10))
    axs[0].set_ylabel("video 数", fontproperties=font(10))
    axs[0].set_xticks(range(10))
    axs[0].grid(axis="y", color=COLORS["grid"], alpha=0.7)
    axs[0].set_title("很多 video 是“全好”或“全坏”", fontproperties=font(12, True))
    colors = [COLORS["green"], "#7EBD9B", COLORS["amber"], COLORS["red"]]
    axs[1].barh(np.arange(len(pattern_counts)), pattern_counts.values, color=colors)
    axs[1].set_yticks(np.arange(len(pattern_counts)))
    axs[1].set_yticklabels(["稳定变好", "轻微变好", "混合/轻微变差", "稳定变差"], fontproperties=font(10))
    axs[1].invert_yaxis()
    axs[1].grid(axis="x", color=COLORS["grid"], alpha=0.7)
    axs[1].set_title("按 video 模式分类", fontproperties=font(12, True))
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
        ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_axis_chart(axis: pd.DataFrame) -> Path:
    path = FIG_DIR / "axis_effects.png"
    df = axis.sort_values("mean_delta")
    fig, ax = plt.subplots(figsize=(9.5, 4.6), facecolor=COLORS["card"])
    y = np.arange(len(df))
    ax.barh(y, df["mean_delta"], color=[COLORS["green"] if v >= 0 else COLORS["red"] for v in df["mean_delta"]])
    ax.axvline(0, color=COLORS["muted"], lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["axis"], fontproperties=font(9))
    ax.set_xlabel("平均 ΔIQ", fontproperties=font(11))
    ax.set_title("内容类型比 checkpoint 更决定方向", fontproperties=font(14, True))
    ax.grid(axis="x", color=COLORS["grid"], alpha=0.7)
    for i, r in enumerate(df.itertuples(index=False)):
        v = float(r.mean_delta)
        ax.text(v + (0.08 if v >= 0 else -0.08), i, f"{v:+.2f}  ({int(r.n_samples)} videos)", ha="left" if v >= 0 else "right", va="center", fontproperties=font(9), color=delta_color(v))
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_scatter(sample: pd.DataFrame) -> Path:
    path = FIG_DIR / "baseline_vs_gain.png"
    fig, ax = plt.subplots(figsize=(8.8, 4.8), facecolor=COLORS["card"])
    color_map = {
        "robust_improver": COLORS["green"],
        "mild_improver": "#7EBD9B",
        "mixed_or_mild_regressor": COLORS["amber"],
        "robust_regressor": COLORS["red"],
    }
    for pat, grp in sample.groupby("pattern"):
        ax.scatter(grp["baseline_iq"], grp["mean_delta"], s=65, alpha=0.86, label=pat, color=color_map.get(pat, COLORS["blue"]), edgecolor="white", linewidth=0.8)
    ax.axhline(0, color=COLORS["muted"], lw=1)
    ax.set_xlabel("baseline image_quality", fontproperties=font(11))
    ax.set_ylabel("9 checkpoint 平均 ΔIQ", fontproperties=font(11))
    ax.set_title("不是“baseline 越差越值得加”：baseline IQ 与收益几乎无关", fontproperties=font(13, True))
    ax.grid(color=COLORS["grid"], alpha=0.65)
    ax.legend(prop=font(8), frameon=False, loc="lower right")
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def video_path(dataset: str, variant: str, prompt_id: str, seed_idx: int) -> Path:
    seed = f"seed{seed_idx:04d}"
    if dataset == "general15":
        return ROOT / f"runs/microstep_grid/microstep_all_variants_multiseed_prompt_001/{variant}/{prompt_id}/{seed}/video.mp4"
    if variant == "baseline":
        return ROOT / f"runs/baseline/20260511_224405/{prompt_id}/{seed}/video.mp4"
    if variant == "s20x05":
        return ROOT / f"runs/microstep_grid/iq_tail20_s20x05_001/s20x05/{prompt_id}/{seed}/video.mp4"
    return ROOT / f"runs/microstep_grid/iq_tail20_other_steps_001/{variant}/{prompt_id}/{seed}/video.mp4"


_FRAME_CACHE: dict[Path, Image.Image] = {}


def fit_cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = img.size
    tw, th = size
    scale = max(tw / w, th / h)
    nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def mid_frame(path: Path, size: tuple[int, int] = (320, 184)) -> Image.Image:
    path = path.resolve()
    if path in _FRAME_CACHE:
        return _FRAME_CACHE[path].copy()
    if not path.exists():
        img = Image.new("RGB", size, (230, 230, 230))
        _FRAME_CACHE[path] = img
        return img.copy()
    cap = cv2.VideoCapture(str(path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("read failed")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame).convert("RGB")
    except Exception:
        img = Image.new("RGB", size, (230, 230, 230))
    finally:
        cap.release()
    img = fit_cover(img, size)
    _FRAME_CACHE[path] = img
    return img.copy()


def save_examples(sample: pd.DataFrame, kind: str) -> Path:
    path = FIG_DIR / f"examples_{kind}.png"
    if kind == "improve":
        rows = sample.sort_values("mean_delta", ascending=False).head(3)
        header = "典型变好样本：结构简单/风格纹理/单主体，extra steps 像是在做 refinement"
    else:
        rows = sample.sort_values("mean_delta", ascending=True).head(3)
        header = "典型变差样本：多物体/人类动作/运动模糊，extra steps 容易放大结构错误"

    fig, axes = plt.subplots(len(rows), 3, figsize=(9.6, 5.2), facecolor=COLORS["card"])
    fig.suptitle(header, fontproperties=font(14, True), color=COLORS["ink"], y=0.985)
    for i, (_, row) in enumerate(rows.iterrows()):
        variants = ["baseline", row["best_variant"], row["worst_variant"]]
        titles = [
            f"baseline {row['baseline_iq']:.2f}",
            f"best {row['best_variant']} {row['best_delta']:+.2f}",
            f"worst {row['worst_variant']} {row['worst_delta']:+.2f}",
        ]
        for j, (var, t) in enumerate(zip(variants, titles)):
            ax = axes[i, j]
            img = mid_frame(video_path(str(row["dataset"]), str(var), str(row["prompt_id"]), int(row["seed_idx"])))
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(t, fontproperties=font(9, True), color=delta_color(float(row["best_delta"] if j == 1 else row["worst_delta"] if j == 2 else 0)))
        axes[i, 0].text(
            -0.02,
            -0.12,
            f"{row['sample_id']} | {row['axis']} | mean {row['mean_delta']:+.2f}",
            transform=axes[i, 0].transAxes,
            fontproperties=font(8, True),
            color=COLORS["ink"],
            va="top",
        )
        axes[i, 1].text(
            -0.02,
            -0.12,
            textwrap.fill(str(row["prompt_text"]), width=58),
            transform=axes[i, 1].transAxes,
            fontproperties=font(7),
            color=COLORS["muted"],
            va="top",
        )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_report() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    for old_page in PAGE_DIR.glob("page_*.png"):
        old_page.unlink()
    tables = load_tables()
    step = tables["step"]
    axis = tables["axis"]
    pattern = tables["pattern"]
    sample = tables["sample"]
    long = tables["long"]
    prompt = tables["prompt"]
    corr = tables["corr"]

    figs = {
        "step": save_step_chart(step),
        "heatmap": save_heatmap(long, sample),
        "pattern": save_pattern_chart(sample),
        "axis": save_axis_chart(axis),
        "scatter": save_scatter(sample),
        "examples_improve": save_examples(sample, "improve"),
        "examples_regress": save_examples(sample, "regress"),
    }

    n_samples = int(sample["sample_id"].nunique())
    n_comparisons = int(len(long))
    all_good = int((sample["improved_count"] == 9).sum())
    all_bad = int((sample["improved_count"] == 0).sum())
    robust_good = int((sample["improved_count"] >= 7).sum())
    robust_bad = int((sample["improved_count"] <= 2).sum())
    oracle_mean = float(sample["best_delta"].mean())
    best_step = step.loc[step["mean_delta"].idxmax()]
    corr_baseline = float(corr.loc[corr["metric"] == "baseline_iq", "corr_mean_delta"].iloc[0])

    page_paths: list[Path] = []
    with PdfPages(OUT_PDF) as pdf:
        # Cover.
        fig, ax = new_page()
        ax.text(0.055, 0.80, "+5 Microsteps 为什么有些 video 变好，有些变差？", fontproperties=font(32, True), color=COLORS["ink"], va="top")
        ax.text(0.055, 0.70, "Image Quality / VBench MUSIQ 分析报告", fontproperties=font(21, True), color=COLORS["blue"], va="top")
        wrapped(
            ax,
            0.055,
            0.60,
            "目标：用目前已经跑完的单 checkpoint +5 microstep 结果，解释 image_quality 的提升/退步规律，并给出直觉性的使用建议。",
            width=42,
            size=15,
            color=COLORS["muted"],
        )
        for x, label, value, color in [
            (0.07, "unique videos", str(n_samples), COLORS["blue"]),
            (0.31, "checkpoints", "9", COLORS["blue"]),
            (0.55, "comparisons", str(n_comparisons), COLORS["blue"]),
            (0.79, "metric", "ΔIQ", COLORS["blue"]),
        ]:
            rounded_box(ax, (x, 0.27), (0.16, 0.18), fc="card")
            ax.text(x + 0.08, 0.395, value, ha="center", fontproperties=font(30, True), color=color)
            ax.text(x + 0.08, 0.325, label, ha="center", fontproperties=font(11), color=COLORS["muted"])
        ax.text(0.055, 0.11, "Scope: 本报告不是全 150 videos 的完整 sweep；它覆盖本地已有 VBench 的 34 个 unique videos × 9 个单点 +5 variants。", fontproperties=font(11), color=COLORS["muted"])
        save_page(pdf, fig, page_paths)
        plt.close(fig)

        # Executive summary.
        fig, ax = new_page()
        title(ax, "一页结论", "规律不是“某个 checkpoint 一定好”，而是“video 类型决定方向”。")
        cards = [
            ("video-specific", f"{all_good} 个全变好，{all_bad} 个全变差", "很多样本对 9 个 checkpoint 的方向是一致的。", "soft_blue"),
            ("best fixed step", f"{best_step['variant']} 平均 {best_step['mean_delta']:+.2f}", "s10 相对最安全，但平均仍接近 0；不是强规律。", "soft_green"),
            ("oracle upper bound", f"best-per-video 平均 {oracle_mean:+.2f}", "即使给每个 video 挑最好 checkpoint，收益也很有限。", "soft_green"),
            ("baseline IQ", f"corr = {corr_baseline:+.2f}", "不是 baseline 分低就一定会被 microsteps 修好。", "soft_red"),
        ]
        for i, (h, v, body, fc) in enumerate(cards):
            x = 0.06 + (i % 2) * 0.45
            y = 0.58 - (i // 2) * 0.27
            rounded_box(ax, (x, y), (0.38, 0.19), fc=fc)
            ax.text(x + 0.025, y + 0.145, h, fontproperties=font(13, True), color=COLORS["ink"])
            ax.text(x + 0.025, y + 0.095, v, fontproperties=font(19, True), color=COLORS["blue"])
            wrapped(ax, x + 0.025, y + 0.055, body, width=28, size=10, color=COLORS["muted"])
        wrapped(
            ax,
            0.075,
            0.17,
            "直觉：+5 microsteps 不会凭空增加语义理解，它只是让当前 latent trajectory 在某个局部 checkpoint 附近多走几步。结构已经清楚、主要问题是纹理/细节的 video，通常能被 refine；结构本来就不稳、包含多物体或复杂人类动作的 video，额外 denoising 往往会更坚定地放大错误。",
            width=62,
            size=14,
        )
        save_page(pdf, fig, page_paths)
        plt.close(fig)

        # Mechanism page.
        fig, ax = new_page()
        title(ax, "直觉模型：+5 不是重做生成，而是局部加密采样", "把它想成在一条已经选好的 denoising 路径上，局部走得更细。")
        y = 0.58
        ax.plot([0.10, 0.90], [y, y], color=COLORS["muted"], lw=3)
        for step_num in [1, 10, 20, 30, 40, 50]:
            x = 0.10 + (step_num - 1) / 49 * 0.80
            ax.add_patch(Circle((x, y), 0.012, color=COLORS["blue"] if step_num in [10, 20, 30, 40, 50] else COLORS["muted"]))
            ax.text(x, y - 0.055, str(step_num), ha="center", fontproperties=font(10), color=COLORS["muted"])
        x20 = 0.10 + (20 - 1) / 49 * 0.80
        ax.add_patch(Rectangle((x20 - 0.055, y - 0.035), 0.11, 0.07, fill=False, edgecolor=COLORS["red"], lw=2))
        for k in range(5):
            xx = x20 - 0.035 + k * 0.017
            ax.add_patch(Circle((xx, y + 0.10), 0.006, color=COLORS["red"]))
        ax.add_patch(FancyArrowPatch((x20 - 0.02, y + 0.075), (x20 - 0.02, y + 0.025), arrowstyle="->", color=COLORS["red"], mutation_scale=16))
        ax.text(x20 + 0.055, y + 0.115, "extra 5 microsteps", fontproperties=font(12, True), color=COLORS["red"])
        rounded_box(ax, (0.07, 0.19), (0.39, 0.24), fc="soft_green")
        ax.text(0.095, 0.38, "什么时候可能变好", fontproperties=font(16, True), color=COLORS["green"])
        wrapped(ax, 0.095, 0.33, "结构已经稳定：单主体、场景/风格、纹理连续。额外步数主要用于细节收敛，像给图像做一次更细的 refinement。", width=30, size=12)
        rounded_box(ax, (0.54, 0.19), (0.39, 0.24), fc="soft_red")
        ax.text(0.565, 0.38, "什么时候可能变差", fontproperties=font(16, True), color=COLORS["red"])
        wrapped(ax, 0.565, 0.33, "结构还没稳定：多物体关系、人/手/车、强运动或局部模糊。额外步数会把错误更早锁住，或者让边界/纹理更怪。", width=30, size=12)
        save_page(pdf, fig, page_paths)
        plt.close(fig)

        # Checkpoint chart.
        fig, ax = new_page()
        title(ax, "Checkpoint 的平均效应很弱", "如果不看 video 类型，单点 +5 的平均收益基本在 0 附近波动。")
        add_figure_image(ax, figs["step"], (0.055, 0.24, 0.66, 0.50))
        rounded_box(ax, (0.75, 0.28), (0.19, 0.42), fc="card")
        wrapped(ax, 0.775, 0.65, "读图方式", width=18, size=15, bold=True, color=COLORS["ink"])
        wrapped(ax, 0.775, 0.59, "柱子：平均 ΔIQ。\n蓝线：有多少比例的 video 变好。\n\ns10 是最不差的固定选择，但平均仍是 -0.24；说明 checkpoint 不是主要解释变量。", width=18, size=11, color=COLORS["muted"])
        save_page(pdf, fig, page_paths)
        plt.close(fig)

        # Heatmap.
        fig, ax = new_page()
        title(ax, "热力图：一眼看出 video-specific 模式", "每一行是一个 video；每一列是一个 checkpoint。绿色/红色通常沿着整行出现。")
        add_figure_image(ax, figs["heatmap"], (0.055, 0.09, 0.68, 0.77))
        rounded_box(ax, (0.77, 0.20), (0.17, 0.50), fc="card")
        wrapped(ax, 0.79, 0.65, f"{robust_good} 个 video 大多数 checkpoint 变好；{robust_bad} 个 video 大多数 checkpoint 变差。", width=15, size=13, bold=True)
        wrapped(ax, 0.79, 0.51, "这说明在当前数据里，先判断 video 是否适合加 microsteps，比纠结 s20 还是 s30 更重要。", width=17, size=11, color=COLORS["muted"])
        save_page(pdf, fig, page_paths)
        plt.close(fig)

        # Pattern distribution.
        fig, ax = new_page()
        title(ax, "不是随机噪声：很多样本是稳定响应", "如果只是评估噪声，我们不会看到这么多“9/9 全好”或“0/9 全坏”。")
        add_figure_image(ax, figs["pattern"], (0.07, 0.28, 0.66, 0.48))
        rounded_box(ax, (0.77, 0.30), (0.17, 0.40), fc="soft_blue")
        wrapped(ax, 0.79, 0.65, "Interpretation", width=18, size=14, bold=True, color=COLORS["blue"])
        wrapped(ax, 0.79, 0.58, "这个分布暗示可以做 gating：先预测这个 video 属于“适合 refinement”还是“不适合”。预测对了，比固定加某一步更有价值。", width=17, size=11)
        save_page(pdf, fig, page_paths)
        plt.close(fig)

        # Content axis.
        fig, ax = new_page()
        title(ax, "内容类型的方向性最明显", "subject/style/scene 更容易受益；multi-object 和 human-action 风险最大。")
        add_figure_image(ax, figs["axis"], (0.06, 0.18, 0.63, 0.62))
        rounded_box(ax, (0.73, 0.18), (0.22, 0.62), fc="card")
        wrapped(ax, 0.755, 0.74, "经验规则", width=16, size=15, bold=True)
        wrapped(ax, 0.755, 0.68, "更可能变好：\n- 单主体结构清楚\n- 风格/纹理主导\n- 静态 scene\n\n更可能变差：\n- 多物体关系\n- 人类动作/手/车\n- 强运动或已模糊", width=18, size=11, color=COLORS["muted"])
        save_page(pdf, fig, page_paths)
        plt.close(fig)

        # Baseline IQ scatter.
        fig, ax = new_page()
        title(ax, "baseline image_quality 不是充分条件", "低分样本里既有可修的，也有越修越坏的。")
        add_figure_image(ax, figs["scatter"], (0.08, 0.22, 0.62, 0.57))
        rounded_box(ax, (0.75, 0.28), (0.19, 0.42), fc="soft_red")
        wrapped(ax, 0.775, 0.64, f"corr ≈ {corr_baseline:+.2f}", width=16, size=18, bold=True, color=COLORS["red"])
        wrapped(ax, 0.775, 0.55, "只用 baseline IQ 不能判断要不要加 microsteps。需要看内容结构：主体是否稳定、是否多物体、是否动作复杂。", width=17, size=11)
        save_page(pdf, fig, page_paths)
        plt.close(fig)

        # Examples.
        for key, page_title, subtitle in [
            ("examples_improve", "视觉例子：这些样本加步后通常更好", "注意它们多是结构清楚、纹理/风格可 refinement 的情况。"),
            ("examples_regress", "视觉例子：这些样本加步后通常更差", "多物体和人类动作里，额外 denoising 常常会放大已有结构问题。"),
        ]:
            fig, ax = new_page()
            title(ax, page_title, subtitle)
            add_figure_image(ax, figs[key], (0.055, 0.10, 0.89, 0.72))
            save_page(pdf, fig, page_paths)
            plt.close(fig)

        # Decision table.
        fig, ax = new_page()
        title(ax, "一个直觉性的使用策略", "不要无条件加 +5；先判断 video 是否适合被 refinement。")
        rows = [
            ("推荐尝试", "单主体、主体边界清楚、场景/风格/纹理主导、弱运动", "先试 s10；如果有预算，再扫少量 checkpoint"),
            ("谨慎尝试", "baseline IQ 低但结构清楚；轻微 blur；scene 里有局部 artifact", "最好做 per-video 选择，不要固定 checkpoint"),
            ("建议避免", "multiple objects、人类动作、手/车/身体局部、强运动、主体已经畸变", "extra steps 可能锁死错误或增强 artifact"),
        ]
        y = 0.68
        for label, cond, action in rows:
            fc = "soft_green" if label == "推荐尝试" else "soft_blue" if label == "谨慎尝试" else "soft_red"
            rounded_box(ax, (0.07, y - 0.14), (0.86, 0.13), fc=fc)
            ax.text(0.095, y - 0.045, label, fontproperties=font(15, True), color=COLORS["ink"])
            wrapped(ax, 0.25, y - 0.035, cond, width=34, size=11, color=COLORS["ink"])
            wrapped(ax, 0.66, y - 0.035, action, width=23, size=11, color=COLORS["muted"])
            y -= 0.18
        wrapped(
            ax,
            0.08,
            0.13,
            "下一步建议：在全 150 videos 上跑同样的 single-checkpoint sweep，并用视觉特征做一个简单 gating model，例如 optical-flow 强度、边缘/纹理复杂度、主体数量、是否 human-action。当前 34-video 结果已经足够说明“平均 checkpoint sweep”会掩盖主要规律。",
            width=78,
            size=12,
        )
        save_page(pdf, fig, page_paths)
        plt.close(fig)

        # Appendix.
        fig, ax = new_page()
        title(ax, "附录：文件与可复现性", "报告使用的 CSV 和图都保存在 runs/analysis 下。")
        files = [
            ANALYSIS / "iq_single_step_sample_summary_dedup.csv",
            ANALYSIS / "iq_single_step_summary_combined_by_step.csv",
            ANALYSIS / "iq_single_step_summary_by_axis.csv",
            ANALYSIS / "sample_summary_with_baseline_features.csv",
            TAIL_ANALYSIS / "iq_all_steps_delta_heatmap.png",
            TAIL_ANALYSIS / "iq_all_steps_contact_sheet_no_overlap_part01.png",
        ]
        y = 0.78
        for p in files:
            ax.text(0.075, y, str(p.relative_to(ROOT)), fontproperties=font(10), color=COLORS["blue"])
            y -= 0.055
        wrapped(
            ax,
            0.075,
            0.34,
            "Metric: image_quality 使用 VBench/MUSIQ；所有报告中的 ΔIQ 都已经统一到 0-100 标尺。Variants 只包含单 checkpoint +5：s10/s15/s20/s25/s30/s35/s40/s45/s50；多 checkpoint 组合没有用于本报告的主结论。",
            width=76,
            size=12,
            color=COLORS["muted"],
        )
        save_page(pdf, fig, page_paths)
        plt.close(fig)

    write_raster_pdf(page_paths)
    print(OUT_PDF)
    print(OUT_RASTER_PDF)


if __name__ == "__main__":
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    build_report()
