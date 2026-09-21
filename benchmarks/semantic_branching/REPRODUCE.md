# 复现与继续记录

所有生成、测试、评估和图像处理都在yukelab的/data/datasets/peihao/tt-scaling-diffusion执行。本目录归档不启动新实验。

| 阶段 | 原执行位置 / 脚本 |
| --- | --- |
| 初次step5 | runs/prompt_branch_step05_20260918_01/run_pilot.py |
| k5–49扫描 | runs/prompt_branch_boundary_20260918_01/run_sweep.py |
| k1–4补充 | runs/prompt_branch_boundary_20260918_01/run_early.py |
| 单次40→4回噪 | runs/renoise_step40_to04_20260921_01/run_case.py |
| 配对从头金毛 | runs/renoise_step40_to04_20260921_01/run_full_dog_control.py |
| 三份noise | runs/renoise_step40_to04_3noise_20260921_01/run_group.py及各seed目录run_case.py |

archive/脚本是证据快照，不是重新封装的独立runner。部分脚本依赖原run布局、相对路径或已有输出目录保护；不要直接从archive启动或覆盖既有结果。

复现新结果时，在新的run目录复制对应worker，显式设置源metadata/latent路径、输出路径和noise seed；使用原.venv/bin/python、固定权重与依赖，检查空闲GPU。完整方法见PROTOCOL.md，原始参数与逐步校验见metadata.json。

## 下一次记录

新增条件至少记录：源latent路径/哈希、原/目标prompt、源完成step、目标sigma/step、噪声分布/seed、公式/系数、scheduler历史处理、实际更新数、模型快照、视频/帧、视觉结论和适用范围。使用新run目录；修订结论时保留旧版本并标明superseded。随后追加registry、媒体索引和校验清单。

本轮归档只整理已有成果，没有重新生成视频。Git分支包含文档、配置、脚本和元数据；媒体保留在远程完整归档及Downloads副本中，原始数组仅保留远程。
