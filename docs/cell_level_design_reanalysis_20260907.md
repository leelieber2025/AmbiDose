# 回归 Fig.1 设计初衷的逐细胞剂量复核（2026-09-07）

本轮重新阅读 `draft/AmbiDose_NatureMethods_Article_revised.docx` 的 Fig.1、
Results 和 Methods，并按“污染主要是 droplet/cell-level 差异”重新检查当前算子。
生产代码和稿件均未修改。

## 逐细胞剂量捕获了部分真实异质性

`scripts/research_cell_level_dose_alignment.py` 使用低添加 PBMC 的逐细胞注入真值。
当前 `d_c` 与真实注入 UMI 的 Spearman 相关为 0.643；污染比例相关为 0.428，
高于未收缩 raw 估计的 0.188。说明细胞级估计不是噪声，收缩也改善了排序。

但真实污染比例由独立逐细胞抽样生成，类型只解释 0.146% 的方差；估计污染比例
却有 13.76% 被类型解释。六个类型内的剂量 Spearman 为 0.400–0.901，类型内
真实和估计权重的平均 total-variation distance 为 0.280。估计同时包含有效的
droplet-level 排序和不应存在的 type-level 偏差。

结果位于 `data/processed/cell_level_dose_alignment_20260907/` 和
`cell_level_posthoc_20260907/`。

## 当前汇总算子可使细胞级剂量失去作用

`scripts/research_oracle_cell_dose_weights.py` 固定每个类型的剂量总和、所有保护和
基因预算，只改变类型内权重：当前 `d_c`、类型内统一 rho，以及按真实注入 dose
重标定的 oracle 权重。

当前与类型内统一 rho 的最终矩阵逐元素完全相同；逐细胞真实 oracle 仅改变
28,356 个条目 / 38,986 UMI。当前 ARS 区间 0.207791–0.324688，oracle 完全相同；
ERS 仅由 0.868621–0.881919 改为 0.870785–0.884083。

原因与已有路径审计一致：未保护 rank-1 基因的类型级 take 大量达到该类型的
全部观测容量。预算已把某个 gene×type 清空时，后续用 `d_c` 还是统一 rho 分配
都不会改变结果。细胞级异质性只作为末端权重存在，不保证实际影响校正矩阵。

因此稿件把 type-naive quantile-floor ablation 称为细胞级主张的 “causal test”
并不充分：该 ablation 同时改变证据池、剂量估计和类型总预算，没有只操纵类型内
的 `d_c` 异质性。Fig.1 的估计方差描述仍是事实，但算子层面的因果贡献需要降格
或补充真正的固定组预算实验。

## 直接逐细胞硬上限不是答案

`scripts/research_cellwise_rank1_cap.py` 把未保护 rank-1 的类型总 take 限制为
`sum_c min(y_cg, d_c chi_g)`：

- 零污染原生保留 0.920691→0.976818；
- 低添加 ERS 0.868621–0.881919→0.948579–0.964615，重构 L1 从
  3,511,333 降到 2,263,515 UMI，但 ARS 降至 0.065605–0.206578；
- Mixture sensitivity 0.985659→0.494845，否决。

稀疏 0/1 污染计数常满足 `d_c chi_g < 1`。逐细胞截断后再相加会反复损失小数
期望；类型汇总有必要，但不能让汇总饱和抹掉全部 cell-level 信息。下一模型需要
在类型总预算内做逐细胞概率/整数分配，并让计数是否随 `d_c` 富集影响预算。

## 空滴多谱包络仍不足

`scripts/research_empty_envelope_allocation.py` 用空滴 UMI 三层谱的逐基因最大值形成
包络。零污染保留约 0.926，低添加略改善 ERS 但降低 ARS；Mixture sensitivity
为 0.9671–0.9715，仍低于 0.9857。多谱改善均值预测，但没有解决细胞级稀疏分配。

## 决策

不推广统一减量、逐细胞硬 cap、空滴最大包络或旧 correlation selector。下一项
应实现最小的“类型保护 + 类型总量 + cell-level likelihood allocation”实验。
污染组成以空滴为主；细胞表达只用于 native 竞争项，不能自由学习后把污染吸收。

所有新矩阵非负、整数且不增加原始计数；当前矩阵复现，生产源码哈希未变，
实验脚本 Ruff 通过。未运行全套产品测试，因为没有产品修改。
