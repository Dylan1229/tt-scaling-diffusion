# 实验协议

## 共用配置

Wan2.2 TI2V-5B，WanPipeline T2V模式，无输入图像。权重快照：
/data/datasets/fanjiang/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/b8fff7315c768468a5333511427288870b2e9635

50步；UniPC order2、flow_shift5、flow_prediction/use_flow_sigmas；CFG5；negative prompt为空；max_sequence_length512。
832×480，81帧，24fps；VAE float32，其余模型bfloat16；latent及回噪算术float32。
torch 2.7.1+cu128；diffusers 0.38.0；源seed0。完整scheduler配置见 [config.json](config.json)。

Prompt A：
> A realistic orange tabby cat walks slowly from left to right across a green grassy meadow. Full body side view, static camera, natural daylight, smooth continuous motion.

Prompt B：
> A realistic golden retriever dog walks slowly from left to right across a green grassy meadow. Full body side view, static camera, natural daylight, smooth continuous motion.

## 纯prompt-switch

k指已完成的去噪更新数。A用于前k步，在callback index k−1后切换positive embedding，B用于剩余50−k步；latent与UniPC历史保持，negative embedding不变。

测试k=1–49；k=0是从头金毛对照，k=50是全猫对照。相同seed重放前缀，非实际前缀缓存复用，因此不作计算节省声明。前缀latent及历史哈希验证保存在归档validation JSON。

## 回噪：step40 → step4噪声水平

源latent：/data/datasets/peihao/tt-scaling-diffusion/runs/prompt_branch_boundary_20260918_01/step_040/branch_latent.pt
tensor-content SHA256：6963c308f0c962fcfc1c1ace911cf563bbce4ed8ee29e5ba8838500811091ac8
与全猫轨迹第40步哈希一致。

flow约定：x_sigma=(1−sigma)x0+sigma*epsilon。
sigma40=0.55678790807724；sigma4=0.982924222946167。

~~~text
a = (1 - sigma4) / (1 - sigma40)
b = sqrt(sigma4^2 - a^2 * sigma40^2)
x_restart = a * x40 + b * z_fresh
a = 0.03852732667954569
b = 0.9826901134503093
~~~

z_fresh是独立新采样单位Gaussian，seed20260921/22/23/24；没有复用旧step4 latent/noise。公式匹配名义flow边缘分布的信号系数和噪声方差，不声称实际有限步latent有已知的逐样本分解。

新建UniPC，先生成原50步调度，再同时取timesteps[4:]和sigmas[4:]，保留末尾0、清空历史。执行46次B条件更新，首次低阶暖启动，最终sigma0。与完整调度begin_index4的46次合成更新对比逐位一致。

## 配对从头金毛

仅seed20260921：保存的fresh_random_noise.pt直接作为原生pipeline初始float32 latent，金毛条件完整50步，新的完整调度。输入噪声SHA256与回噪注入z一致。

## 视觉审查

对比帧0/16/32/48/64/80（视频0到3.33秒），保留全81帧contact sheets和原分辨率帧。分别描述类别身份、姿态/方向、构图、动作、身体/肢体伪影。类别改变与画质通过分别记录；不作跨案例统计成功率声明。
