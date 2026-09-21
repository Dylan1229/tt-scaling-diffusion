# 原始数据与归档边界

archive/是独立拷贝，包含全部原报告、JSON、脚本、日志、视频、图片和DONE标记；每个拷贝均校验源/目标SHA256。查看这些成果不依赖原目录。

.npy逐帧数组与.pt latent/noise留在原远程目录，没有重复复制。远程benchmark的raw_runs/是指向原目录的相对符号链接；Downloads副本不包含链接和大数组。完整路径/大小见 [raw_data_inventory.json](raw_data_inventory.json)。归档时未重新计算大数组文件哈希；已有tensor-content哈希保存在各metadata中。

这部分原始数据仍依赖原run目录；移动/删除原目录会影响精确复现。模型权重和Python环境未打包。

远程源目录：

- yukelab:/data/datasets/peihao/tt-scaling-diffusion/runs/prompt_branch_step05_20260918_01
- yukelab:/data/datasets/peihao/tt-scaling-diffusion/runs/prompt_branch_boundary_20260918_01
- yukelab:/data/datasets/peihao/tt-scaling-diffusion/runs/renoise_step40_to04_20260921_01
- yukelab:/data/datasets/peihao/tt-scaling-diffusion/runs/renoise_step40_to04_3noise_20260921_01
