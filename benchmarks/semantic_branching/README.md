# Semantic branching benchmark

记录2026-09-18至2026-09-21的同一案例：草地上的橘猫 → 金毛犬。

远程权威目录：yukelab:/data/datasets/peihao/tt-scaling-diffusion/benchmarks/semantic_branching

**56个不同条件，57次完成的生成，2,834次去噪更新。** 猫对照重复生成一次，两个视频SHA256相同；注册表保留两次执行记录。

Git分支包含报告、配置、脚本和元数据；视频、图片、日志及原始数组不进入Git。下面的媒体链接需在远程完整归档或Downloads副本中打开。

## 结果入口

| 实验 | 范围 | 报告 / 可视化 |
| --- | --- | --- |
| Prompt switch | k=1–49，另有全猫/全狗seed0对照 | [完整顺序contact sheet](archive/prompt_branch_boundary_20260918_01/prompt_branching_all_steps_00-50_seed0_frame16_2026-09-18.jpg) · [严格复核](archive/prompt_branch_boundary_20260918_01/strict_review/REPORT.md) |
| Step40回噪到step4 | 新Gaussian noise seed20260921，换金毛prompt，再去噪46步 | [三栏视频](archive/renoise_step40_to04_20260921_01/comparison.mp4) · [对比图](archive/renoise_step40_to04_20260921_01/comparison.jpg) |
| 从头金毛对照 | 同一份新noise，金毛prompt完整50步 | [与回噪并排](archive/renoise_step40_to04_20260921_01/renoise_vs_full_dog.mp4) · [对比图](archive/renoise_step40_to04_20260921_01/renoise_vs_full_dog.jpg) |
| 三份不同noise | 固定同一个x40、sigma4、金毛prompt；seed20260922/23/24 | [三栏视频](archive/renoise_step40_to04_3noise_20260921_01/comparison.mp4) · [contact sheet](archive/renoise_step40_to04_3noise_20260921_01/contact_sheet.jpg) |

当前观察：四份已测试的新噪声均使后期猫latent生成可辨认的金毛外观犬类，外观、步态和背景有差异；肢体、尾巴以及运动方向仍有问题。这支持本案例的语义改变可行性，不代表画质完全通过或一般成功率。

**历史结论已分开记录：** 早期“step10成功”判断已撤回。严格复核结果和用户后续采用的step4–5工作参考不等同，见 [结论与修订记录](REPORT.md)。

## 复查与继续实验

- [协议和精确参数](PROTOCOL.md) · [机器可读配置](config.json)
- [全部56条件的视频/元数据索引](RESULTS.md) · [运行注册表](run_registry.json)
- [复现及继续记录方式](REPRODUCE.md)
- [原始数组/latent位置与存储说明](RAW_DATA.md)
- [归档清单及SHA256](artifact_manifest.json) · [归档校验结果](validation.json)

archive/保留四个原run目录的全部报告、脚本、JSON、日志、视频、PNG/JPEG和DONE文件，结构不变。旧报告原文保存，可能包含后来修正的判断；当前归纳以本目录REPORT.md为准。媒体和原始数据链接已排除Git跟踪，文档、配置、脚本和元数据可供版本管理。

Downloads副本包括全部归档媒体与报告；多GB的.npy/.pt原始数据仅保留在yukelab。实验代码仍需在远程原run布局下运行。
