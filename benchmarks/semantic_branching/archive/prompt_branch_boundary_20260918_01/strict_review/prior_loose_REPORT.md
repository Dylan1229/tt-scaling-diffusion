# Prompt branching: cat → dog, single-case step sweep

在同一个 case、seed 0 下，覆盖了第 5–49 步的每一个整数切换位置。这里的 k 表示先完成 k 步猫 prompt，再用狗 prompt 运行剩余 50−k 步。

## 目视结果

| 切换位置 | 观察 |
|---|---|
| 5–10 | 明确的目标金毛犬外形 |
| 11–14 | 仍有明显犬类外形，但偏离金毛；不可视为完全没有语义改变 |
| 15–24 | 逐渐偏猫的混合体，犬类嘴部或局部异常仍可见 |
| 25–49 | 主体保持猫；不排除局部细节变化，29–49无重新变狗的例外 |

若要求完整地变成目标金毛，已测的连续有效区间为 5–10，最晚观测成功位置为 10。11–14 仍有犬类形态，因此不能把 10 解释为一切语义影响的硬截止。一个 case/seed 不能推导模型通用阈值。

观察依据：所有切换位置的帧 0/16/32/48/64/80；第 9–14 步另看放大的首两帧。过渡区和 29–49 步另经独立目视复核。没有运行正式语义评分或全数据集统计。方向指令仍未严格遵循，结论只针对主体外形变化。

## 固定设置与核验

Wan2.2 TI2V-5B，T2V 模式，50 步，UniPC 二阶，flow_shift=5，CFG=5，832×480，81 帧，24 fps。seed=0，negative prompt 为空，文本长度 512。只切换 positive conditioning，不重加噪，不重置 scheduler。UniPC 保留切换前的历史。

每个分支在切换前所有步的 float32 latent 与完整猫基线逐字节一致，切换点模型输出历史一致；下一步 latent 开始不同。45/45 核验通过。重跑猫基线的最终 latent 与上轮一致。latent 的差异本身不作为语义变化证据。

每个分支都完整重放前缀以做状态核验；本轮实际新跑 44 个切换位置和 1 个猫对照，共 2250 个去噪步骤，没有实现或声称计算节省。

Prompt A: A realistic orange tabby cat walks slowly from left to right across a green grassy meadow. Full body side view, static camera, natural daylight, smooth continuous motion.

Prompt B: A realistic golden retriever dog walks slowly from left to right across a green grassy meadow. Full body side view, static camera, natural daylight, smooth continuous motion.

## 文件

- overview.mp4 / overview.jpg：上排 5、10、11；下排 15、20、25。
- all_steps_01.jpg 到 all_steps_06.jpg：全部 45 个位置的抽帧。
- boundary_focus_01.jpg：9–14 步放大比较。
- all_steps_validation.json / assessment.json：执行核验与目视判断。
- step_NNN/：每步视频、全帧数组、分叉 latent、完整运行元数据。
- run_sweep.py：实际运行代码。全部实验和处理都在 yukelab 执行，下载的文件仅用于查看。
