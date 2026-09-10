# AmbiDose 0.3.12 设计初衷审计（2026-09-09）

## 范围与状态

按论文 Results 第一部分提出的两项设计主张，依次研究每细胞剂量的独立收益、保护误判和扣除通道归因。产品源码保持 0.3.12；本次没有修改产品参数、默认算法或论文数字。

本文对应 `data/processed/design_audit_20260909/`。研究进行中；最终结论须以每个 JSON 的 `complete: true` 和后续结论节为准。

## 可复现入口

```bash
python scripts/research_design_audit_20260909.py --dataset mixture --out data/processed/design_audit_20260909
python scripts/research_design_compartments_20260909.py --dataset nuc2 --out data/processed/design_audit_20260909
python scripts/summarize_design_audit_20260909.py
```

重新运行必须使用新的输出目录，脚本拒绝覆盖已有数据集 JSON。

剂量消融涵盖 Mixture、hgmm12k、GSE147203、zero_ambient、low_ambient、realistic_gt、kidney、fetal。SNP compartment 审计涵盖 GSE218853 的 rep1、rep2、rep3、nuc2、nuc3；组织 marker 审计使用全 kidney/fetal 人群。

## 实验控制

- 估计入口始终为 `estimate_dose_adaptive`；先估计一次，再固定原始矩阵、χ、类型和类型总剂量，研究 subtract。
- kidney、fetal 使用正式评估入口的全人群；fetal 排除 `F35_liver_CD45pos_FCAImmP7462238`。不混用 CellBender 匹配子集。
- 保存全部 15 个产品 Python 文件的运行前后 SHA-256、研究脚本 SHA-256。汇总时重新检查磁盘源码。
- `current` 的观察性 instrumentation 必须与未打补丁的普通 subtract 输出逐元素完全一致。
- 四通道实际扣除量之和必须等于最终原始矩阵减输出矩阵；输出不能有负数或超过输入。

### 剂量干预

1. `current`：产品剂量和产品分配。
2. `common_rho`：每个 sample×type 中令 `rho_bar = sum(d)/sum(n)`，`d_new = rho_bar*n`。保留类型总剂量、文库大小依赖，去掉类型内 rho 差异。
3. `common_rho_frozen_enrichment`：同上，同时固定产品原始 Pearson 重加权后的 take，隔离细胞分配与基因重加权的影响。
4. `shuffle_d_{0,1,2}_frozen_enrichment`：在 sample×type 内的文库大小分层中随机交换剂量，三个固定种子。每个分层做 5×细胞数次随机交换，只接受满足 `0 <= d_c <= n_c` 的交换。严格保持剂量多重集和类型总剂量，近似保留文库大小关联。它是受约束的负对照，不是均匀抽取所有可能排列。
5. `no_reallocation`：固定产品剂量，关闭未用 rank-1 预算再分配。诊断其净效应，不作为发布候选。

固定 enrichment 的对照逐次检查：进入 enrichment 的类型索引和原始 take 必须与基线一致。低 χ U 的 remaining-dose 限制仍按干预后的实际细胞扣除计算，这是剂量执行机制的一部分。

### 保护与通道

保护分类互斥：`is_p`、高 χ U、低 χ U、非 P/U 且高于天花板、其余非 P/U。`is_p` 是产品的保护/分配资格，不能等同于已经确认的生物学原生身份，也不一定对应高 native confidence。

四个实际分配通道：未保护 rank-1（包含再分配）、受保护 rank-1、高 χ soupOnly、低 χ soupOnly。旧 `diagnose_removal_paths.py` 的三通道编号不能直接用于当前版本。再分配不单独写矩阵；本次用 `no_reallocation` 反事实衡量其净效应，不能把其连续 take 直接当成已经确定来源的分子数量。

### 真值边界

- Barnyard：异物种计数作为污染 proxy，同物种计数作为保留 proxy。同物种也可能含污染，因此不能声称逐分子 endogenous 真值。
- zero_ambient：Poisson 生成的无新增污染控制，所有模拟细胞计数按构造均为原生。
- low_ambient / realistic_gt：知道原始基线与新增污染计数；经验 PBMC 基线可能已有污染。同一 entry 同时含二者时只报告来源归因上下界，不伪造精确分子归因。
- GSE218853：只统计 SNP mask 内 cell×gene compartment；mask 外计数不进入污染/原生判别分母。**ambient-positive entry 可同时含原生与外源等位基因，entry 的全部 UMI 不能当成真污染 UMI。** 本次 compartment 字段沿用评分接口的 `injected_umi` / `ambient_removed` 名称，但在 SNP 结果中其含义是 ambient-positive 区域总计数及该区域被删除的总计数。计数加权清除/保留与原评估的逐基因宏平均 ARS/ERS分别保留，不混用。
- 组织：on/off marker 是生物学代理，不是分子真值。共享 marker 从 off-target 集合移除，避免将自身 marker 同时计作错误类型表达；原评估 `_score` 结果另行保留供核对。
- 单个通道归因边界相对原始组成计算，通常不能把不同通道的上下界相加。最终总归因边界独立计算。互斥的纯来源 compartment 中可以核对精确扣除量。
- 保护分类覆盖量单独记录；小群体特殊分支或零剂量跳过不能凭空归入常规保护类别。

## 已确定的初步观察

Mixture、GSE147203 和 hgmm12k 的 `common_rho_frozen_enrichment` 与 current 的全局异物种清除量及同物种损失完全相同；细胞内位置可变。这些聚合成绩不能单独证明正确的细胞剂量对应关系是必要条件。它不否定类型证据池、类型总剂量或每细胞信息在其他终点上的价值。

zero_ambient 的 current 删除 920,224 / 18,017,857 UMI（5.1073%）；关闭再分配后删除 346,552（1.9234%）。同时 Mixture 异物种清除率从 98.4013% 降至 68.5653%，说明直接取消再分配无法满足现有回归约束。

SNP nuc2 的剩余 ambient-positive 区域基因总计数 2,737 中，2,724 位于 protected 区域；endogenous-control 损失 932 中，758 来自未保护 rank-1。前一个数字定位了污染证据与保护的交集，**不等于证明这 2,724 个计数全是污染，或 owner 身份判错**。项目先前的 `docs/gse218853_marker_ambient_conflict_20260907.md` 已验证同基因 native/ambient 共存；不能把该交集全部转为 U。

## 剂量消融结果

下表是保持类型总预算后的异物种 UMI 清除率，不是每细胞校正精度。

| 数据集 | current | 类型共同 rho | 共同 rho + 固定 enrichment | 关闭再分配 |
|---|---:|---:|---:|---:|
| Mixture | 98.4013% | 98.5489% | 98.4013% | 68.5653% |
| hgmm12k | 95.5561% | 96.3135% | 95.5561% | 54.1670% |
| GSE147203 | 98.0469% | 98.0774% | 98.0469% | 85.6982% |

固定 enrichment 后，三次剂量打乱也全部保持原始全局清除量和同物种损失。共同 rho + 固定 enrichment 仍然分别改变 Mixture / hgmm12k / GSE147203 的 225 / 162,271 / 248 个 entry；因此是聚合终点不敏感，不是没有执行干预。对应 L1 变化分别为 228 / 178,342 / 248 UMI。

这也解释了为什么原先的 type-naive 消融不能作为“每细胞剂量是 barnyard 高灵敏度的必要原因”的隔离证据：类型证据池和类型预算本身没有被本轮共同 rho 对照撤销。当前结果支持保留类型结构和预算机制，不能从中推导应删除每细胞剂量。

| 数据集与对照 | 新增 ambient 去除界 ARS | 基线保留界 ERS |
|---|---:|---:|
| low_ambient current | 17.69–25.92% | 89.86–90.80% |
| low_ambient 共同 rho，固定 enrichment | 18.17–26.34% | 89.32–90.25% |
| low_ambient 打乱剂量，3 seeds 范围 | 16.85–25.41% | 90.09–91.09% |
| realistic_gt current | 20.18–29.44% | 88.35–89.39% |
| realistic_gt 共同 rho，固定 enrichment | 21.04–30.09% | 87.45–88.47% |
| realistic_gt 打乱剂量，3 seeds 范围 | 19.31–28.95% | 88.48–89.59% |

打乱剂量减少了一些新增污染清除，也减少了基线损失。共同 rho 增加清除，同时增加基线损失。没有出现“当前逐细胞对应同时支配清除与保留”的结果。三个 seed 只作为稳定性检查，不足以给出泛化显著性声明。

估计 rho 与新增真实比例的 Spearman：low_ambient 0.4512，realistic_gt 0.5033。Mixture 与异物种比例相关为 0.9402，但异物种计数本身也参与估计，不能当成独立验证。低污染 calibration fixture 的估计中位 rho 0.09524925 与新增比例中位数 0.09524893 几乎相等，不意味着每细胞或每基因恢复准确。

全 kidney（16,435 cells / 24 samples）current off-target 去除 32.4838%、on-target 保留 94.7938%；共同 rho + 固定 enrichment 为 32.5787% / 94.7986%。打乱剂量的 off-target 去除为 32.4072–32.4177%，差异很小。关闭再分配为 27.8365% / 95.0035%。这些是全队列结果，不是论文 matched cohort 的分母。

### 补充：固定实际删除量的细胞分配

`research_design_fixed_margins_20260909.py` 固定 current 最终每个 sample×type×gene 的整数删除量，分别用同一个类型列分配器按 d 或 n 分配；受保护 rank-1 始终按 n 分配。每次分配以及最终每个类型×基因的计数边际都做精确相等断言。因此 column_dose / column_library 两者之间只改变权重，不改变实际删除量。它们使用类型列分配，而不是产品未饱和部分的 cell-carry；不能把它们直接当产品默认实现。

| 数据集 / 固定边际分配 | ARS 界 | ERS 界 |
|---|---:|---:|
| low_ambient，dose 权重 | 17.6107–25.9404% | 89.8513–90.7988% |
| low_ambient，library 权重 | 17.2921–25.9512% | 89.8151–90.8001% |
| realistic_gt，dose 权重 | 20.0192–29.4066% | 88.3268–89.3868% |
| realistic_gt，library 权重 | 19.5073–29.3508% | 88.2690–89.3805% |

realistic_gt 在同一列分配器内，dose 权重改善 ARS 两端约 0.5119 / 0.0558 个百分点，改善 ERS 两端约 0.0578 / 0.0063 个百分点。low_ambient 的下界改善而上界极小幅变差，不能称全面支配。这是**有限但直接的每细胞剂量分配证据**；它与 barnyard 聚合终点不敏感并不矛盾。界的移动仍不等于知道每个被删分子的真实来源，也不能据这两个开发集声称普遍收益。

## 保护与通道结果

### 零污染与低污染

零污染误删 920,224 UMI 的通道分解：未保护 rank-1 736,374（80.02%）；受保护 rank-1 130,319；高 χ U 50,896；低 χ U 2,635。关闭再分配将净误删减少 573,672 UMI，但仍有 346,552 UMI 被删除，说明再分配是主要放大器而非唯一原因。

误删最多的前 20 个基因仅占总损失 20.10%。其中包括 FTL、RPL23A、EIF1、RPL30、RPL31、HLA-A、UBC、H3F3B；EIF1、H3F3B 等在类型间的相对表达变化并不大。详细计数在 `zero_gene_losses.csv.gz`。这是观察到的广泛共享表达损失，不支持再加基因名白名单。

low_ambient 中，1,654,610 / 2,049,357 新增 UMI 位于 protected 区域；同一批区域还有 16,227,327 基线 UMI。另一端的 unowned_other 只有 1,768,936 基线 UMI，却至少损失 1,423,986。因而“保住大多数原生计数”的全局 ERS，会掩盖较小未保护部分的严重损失。这个判断使用新增真值，但基线本身仍是经验 PBMC 基线。

### 组织 marker

全 kidney 剩余 off-target marker 147,184 UMI 中，protected 占 144,691（98.31%）。on-target marker 损失 128,370 UMI 中，高 χ soupOnly 占 111,548（86.90%），未保护 rank-1 占 11,358。

因此，零污染控制的主要损失放大路径与 kidney marker 的主要损失路径不同。把所有数据集都归结为“再分配太强”或“soupOnly 太强”均不成立。组织 on/off marker 只作 proxy；分组不纯、双细胞和真正同基因污染仍需区分。

### SNP compartments

| 样本 | ambient-positive 区域剩余计数 | 其中 protected | endogenous-control 区域被删计数 | 其中未保护 rank-1 |
|---|---:|---:|---:|---:|
| rep1 | 25,710 | 25,699 | 14,026 | 12,485 |
| rep2 | 36,328 | 36,315 | 1,945 | 843 |
| rep3 | 10,338 | 10,338 | 2,287 | 521 |
| nuc2 | 2,737 | 2,724 | 932 | 758 |
| nuc3 | 3,072 | 3,069 | 1,529 | 1,300 |

五库 ambient-positive 剩余计数的 99.5–100% 在保护区域，但这不等于所有这些剩余计数都是真污染。nuc2 原评估的逐基因宏平均 ARS/ERS 为 0.42244 / 0.77038，而本次区域计数加权去除/保留为 0.21508 / 0.88996；不能混用。

进一步观察 nuc2 的实际保护来源：ownership-only 保留下来的 ambient-positive 区域计数为 2,643，native_everywhere-only 仅 2，其余显式排除之外的 type-mask 保护为 79。受保护区域剩余计数最多的基因是 Slc27a2（1,688）和 Slc34a1（656）。不能据此对这些基因撤销 ownership：需要先区分 owner 类型内的同基因混合与错误类型表达。先前跨样本 native intercept 的无约束 residual 实验也已失败，见 `docs/gse218853_cross_sample_intercept_20260907.md`。

## 目前不支持的产品改动

1. 用类型共同 rho 替换每细胞估计：本轮控制的是同样的类型总预算，未评估撤销每细胞估计后如何重新得到预算；聚合等价不等于逐细胞质量等价。
2. 取消再分配：三个 barnyard 明显退化。
3. 对 protected 的污染证据直接强制 U：同基因 native/ambient 共存，且会把真实信号交给无上限清除。
4. 一律缩小高 χ soupOnly：组织 on-target 风险需要研究，但 barnyard 依赖该通道；本次没有做出通过回归约束的新资格规则。
5. 继续用 type-naive 消融声称已经隔离证明每细胞剂量的因果收益：本轮提供了直接的反例，论文应区分类型证据的作用和每细胞分配的作用。

## 建议讨论的下一步（尚未改产品或论文）

本轮已经把“设计有没有落地”转换成可定位的机制问题。建议保留现有双通道与已验证的类型总预算机制，把后续研究目标拆开：

1. **可识别的资格错误**：在零污染控制中找出原生计数为何进入可再分配区域；在组织中检查高 χ U 的 on-target 损失是否来自混合分组、双细胞或对真实单一类型的误判。先证明新资格信息能够区分这些情况，再改变扣除量。不以 marker/SNP 测试标签直接拟合生产规则。
2. **同基因 native/ambient 混合**：将其单独列为识别边界，不能通过撤销 owner、提高删除比例或强制 U 假装解决。任何引入新来源信息的方案应有独立的验证设计。
3. **每细胞贡献的终点**：保留共同 rho / 打乱剂量对照与本轮新增的固定实际删除量分配对照，进一步做独立来源验证。后者已在 realistic_gt 给出小幅正向分配证据；尚未完成完整的生产算子等删除量前沿，不能声称 current 被另一分配全面支配，也不能用自身估计值的组内方差替代这些对照。

可讨论的 Results 表述草案（不是已写入 Word 的修改）：

> AmbiDose combines cell-specific operational dose estimates with type-conditioned evidence and native-gene protection. Estimated doses vary substantially within groups, but this variation alone does not establish their molecular accuracy. In controlled subtraction ablations that preserve sample-by-type total dose and gene reweighting, replacing individual contamination fractions with a shared within-group fraction leaves aggregate species-mixing removal and retention unchanged, while changing individual corrected entries. Thus, the strong barnyard results establish the utility of type-conditioned evidence and allocation, but do not by themselves isolate a benefit of individual dose assignment. On PBMC injection controls, individual allocation changes the trade-off between injected-count removal and baseline preservation. Additional replays at fixed realized type-by-gene removal margins show a small improvement in attribution bounds with dose-based rather than library-based column allocation in the realistic_gt fixture, with mixed bounds in the second PBMC fixture. These findings motivate evaluating cell-specific correction with count-level and source-resolved endpoints alongside aggregate leakage metrics.

此段只覆盖本轮得到的结论；原有历史数字、图与正文版本还需要单独同步，不能简单复制新版本号而保留旧指标。
