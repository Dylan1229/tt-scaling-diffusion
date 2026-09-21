# 结论与修订记录

## 范围与当前结论

一个prompt pair、一个源seed0、一个Wan2.2 TI2V-5B权重快照；定性视觉判断，无校准的自动语义/解剖评分。

已完成49个纯prompt-switch位置、全猫/全狗seed0对照、4个step40→sigma4随机回噪条件，以及1个使用同一份新噪声的从头金毛对照。共56个不同条件、57次执行；重复的是seed0全猫对照。

**后期回噪改变语义在本案例上可行。** 四个新噪声seed20260921–24均生成清楚的犬类/金毛外观。不能据此声称一般可靠性、自然解剖完全通过、最小干预强度或节省计算。Active/directional noise未测试。

## Prompt-switch判断的修订

1. Pilot在step5看到了粗粒度猫→狗类别改变，已记录方向不完全符合prompt；不是无缺陷画质认证。
2. 后续“5–10是清晰成功区间、最晚10”的宽松判断被用户指出不合格，已撤回。原判断仅作历史记录，保存在archive/prompt_branch_boundary_20260918_01/strict_review/prior_loose_*，状态为 **SUPERSEDED**。
3. 更严格的全身自然性复核将1–7记为borderline，8–49记为未成功，未确认非零位置的严格通过点。这是定性审查，不是普遍不可能的证明。
4. 用户随后把step4–5作为可接受的工作参考；本轮据此选择sigma4做可行性实验。这是操作参考，不是重新验证出的普遍safe边界。

[原始pilot报告](archive/prompt_branch_step05_20260918_01/README.md) · [严格复核报告](archive/prompt_branch_boundary_20260918_01/strict_review/REPORT.md)

## Step40随机回噪

从真实step40猫latent出发，注入独立新Gaussian noise，达到step4的名义flow噪声水平；不读取旧step4 latent或旧噪声。换金毛prompt，清空UniPC历史，沿原50步调度的后46步去噪。

初次seed20260921主体变成金毛，但腿脚有明显黑白鞋状伪影。step40只换prompt的既有对照仍是猫。[报告](archive/renoise_step40_to04_20260921_01/README.md)

这组对照同时改变回噪、solver历史与剩余计算量（10步vs46步），说明组合干预可行，不能单独归因于噪声或宣称计算节省。

## 从头金毛对照

仅seed20260921有配对从头对照：用回噪时保存的同一份z作为初始latent，金毛prompt完整50步。噪声SHA256已核对一致。

它与回噪分支还相差最初4步生成和猫信号，不是纯粹单变量的“有无猫latent”消融。[说明](archive/renoise_step40_to04_20260921_01/full_dog_fresh_noise/README.md)

## 三份新噪声

| 样本 | Noise seed | 主要可见差异 |
| --- | --- | --- |
| 1 | 20260922 | 毛色偏橘棕、抬腿明显、尾巴较低，草地有浅色斑驳。 |
| 2 | 20260923 | 初始身体较侧向且显长，尾巴低垂，转为后侧视角并逐渐走远。 |
| 3 | 20260924 | 躯干更圆、浅色尾巴高翘，后躯和腿部形状问题更明显。 |

三组保持犬类身份，没有在已检查的81帧序列中回到猫；外观、动作和场景纹理不同。共同趋势是后侧视角、朝左/远处走，未完整服从左→右侧身行走prompt。与原猫方向相似是观察，不足以单独证明因果来源。

[三噪声原报告](archive/renoise_step40_to04_3noise_20260921_01/REPORT.md) · [逐样本判断](archive/renoise_step40_to04_3noise_20260921_01/assessment.json)

## 元数据解释

Pilot metadata的通用method/switch_after_completed_steps描述整组实验，不能把全猫/全狗对照误判为step5切换；注册表依据mode、实际逐步conditioning trace和脚本分类。旧re-noise README的“未额外生成”指初次阶段，其后另加嵌套full_dog_fresh_noise对照，注册表已分别计数。process/jobs JSON中的PID是历史记录，不代表当前进程状态。
