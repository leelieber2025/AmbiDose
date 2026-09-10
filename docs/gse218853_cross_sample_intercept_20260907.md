# 不使用 allele 的跨 library native intercept 验证（2026-09-07）

脚本：`scripts/research_gse218853_cross_sample_intercept_20260907.py`。
输入是 GSE218853 的 5 个 raw libraries（rep1/2/3、nuc2/3）和对应 filtered
cell labels。脚本流式聚合每个 raw library 的 empty droplets，重新计算各自 χ；
随后调用生产 `estimate_dose_adaptive`。SNP 文件不参与拟合，只用于本报告最后的
独立排序检查。

## 模型

对每个 broad type、gene 和 library，计算 normalized type mean `y`，并以
`x = mean(dose) × χ_s,g` 作为 ambient exposure。拟合：

`y_s,t,g = intercept_t,g + slope_t,g × x_s,t,g`

其中 intercept 是 native floor。这个模型需要两个额外假设：同一 broad type 的
native 表达在 library 间相对稳定；不同 library 的 ambient χ 有足够变化。若 χ
不变，intercept 和 ambient 项仍不可识别。

## 结果

每个 library 使用的细胞/empty 数：rep1 14981/6708866，rep2 14593/6703865，
rep3 3676/6709605，nuc2 16714/6715447，nuc3 4266/6713982。

各类型 gene-level 回归 R² 中位数：

| broad type | median R² |
|---|---:|
| Collecting_duct | 0.522 |
| Endothelial | 0.559 |
| Immune | 0.759 |
| Loop_distal | 0.742 |
| Podocyte | 0.679 |
| Proximal_tubule | 0.645 |
| Stromal | 0.581 |

作为独立检查，将 intercept 与 rep1 的 allele-negative endogenous-control 原始
表达排序比较，Spearman 相关为：Collecting_duct 0.377、Endothelial 0.546、
Immune 0.768、Loop_distal 0.602、Podocyte 0.577、Proximal_tubule 0.512、
Stromal 0.288。这里 SNP 只定义检查集，未进入 intercept 拟合。

### Leave-one-library-out

为排除同一 library 内拟合造成的乐观偏差，重新对每个 held-out library 只用其余
library 拟合 intercept，再与 held-out type mean 做 gene-rank 比较。rep1/2/3
互相留出时，各 broad type 的中位 Spearman 分别为 **0.631、0.589、0.564**；
nuc2/nuc3 留出时仅为 **0.158、0.238**。rep1/2/3 的 21 个 type-library 组合中
20 个相关性 >0.3，而 nucleus 留出只有 6/13 个 >0.3。

这不是“方法已经能跨所有数据集识别”的证据，反而给出了明确边界：同一制备体系
内的 cross-library 变化足以提供 native-floor 信号；scRNA 与 nucleus 混合时，
native profile/捕获偏差改变，不能共用一个 intercept。

## 判断

这不是最终校正算法，但它第一次在不读 allele 的情况下，从真实多 library 数据中
恢复了可验证的 native expression floor；效果明显优于单个 noisy dose proxy 的模拟。
说明“跨 library χ 变化 + cross-fit dose”是可行的识别信息来源。

当前仍有三个限制：

1. native profile 跨 library 并不完全稳定，尤其不同组织制备（rep 与 nucleus）会
   破坏 intercept 假设；
2. slope/intercept 会受 dose estimator bias 和 type composition 变化影响；
3. 还没有把 intercept 转成整数 removal，也没有证明它改善 ARS/ERS。

下一步应只在 rep1/2/3 这个同制备子集内做 leave-one-library-out native floor：
在 held-out library 上只把 observed count 高于 predicted native floor 的 residual
交给 ambient subtraction；同时在 rep1 SNP 冲突集上检查 ambient recovery/native
loss。nucleus 先作为外部失配测试，不进入训练。只有该 residual 版本在冲突集上
优于当前保守结果且不损害 endogenous controls，才考虑生产化。

## Held-out residual subtraction（真实数据否决）

进一步把 rep2/rep3 的 native floor 外推到 rep1，并定义逐细胞 residual
`max(raw - predicted_native_floor, 0)`，只在 rep1 SNP conflict pairs 上评估。SNP
只作为外部真值代理，没有进入拟合。结果为 3,762 个 pair、286 个基因：
ambient recovery proxy **1.63**，native loss proxy **0.75**；median removed fraction
为 **0.513**。这不是可接受的校正：它把大量稳定的真实 marker 当成“超出 floor”而
删除。

因此 cross-library intercept 不能直接接入生产 subtraction。它提供的是识别信号，
不是 removal quota。下一步若继续，应在同制备子集内加入保守收缩（例如只移除
预测 excess 的低分位/带不确定性的部分），并以 native-loss 上限约束；不能再次使用
当前的无约束 residual 规则。

复现实验：`scripts/evaluate_cross_sample_residual_20260907.py`；结果：
`data/processed/gse218853_cross_sample_residual_20260907.json`。
