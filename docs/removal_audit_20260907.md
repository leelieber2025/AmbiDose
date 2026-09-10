# 低污染扣除审计（2026-09-07）

本轮结论：当前 PBMC 控制中，再分配明显增加计数损失；但缩减其预算
也损害真实 barnyard 去污染能力。候选未通过验收，不进入产品默认。
默认剂量、ownership、soupOnly、再分配公式和论文结果均未修改。

## 与开发历史对齐

已参考 `DEVLOG.md` 的 2026-09-04/05 结论，以及历史再分配、HVG、
交叉拟合 hurdle 和 responsibility allocation 记录，另参考
`docs/development_benchmarks.md`。不重新推广关闭再分配、ρ 上限、
放宽所有低 ceiling 基因保护、marker-only HVG 或交叉拟合 hurdle。
SoupX/DecontX 比较遵守日志锁定的无外加标签协议；本轮没有运行比较工具。

## 输入与解释边界

- 使用 `data/processed/realistic_gt_fig4_pooleddose_20260901/` 的
  `cells.h5ad`、`clean.h5ad`、`injected.npz`，核对细胞/基因顺序及
  `cells.X == clean.X + injected`。保留 3,940 个固定细胞和实测 χ，
  重新计算当前无标签 Leiden 分组和 adaptive 剂量。
- 低污染输入是向真实 PBMC 基线追加已知计数。基线可能已有 ambient，
  因此表中 ERS 是“注入前基线保留”的归属上下界，不是已知 native
  分子的精确保留率。`injected_dose` 只提供新增污染剂量，不能称为
  原始数据全部污染的 oracle。
- 零污染模拟用基线的当前 Leiden 类型汇总 profile，按每个细胞的
  library size 作 Poisson 抽样，seed=0；所有模拟计数定义为 native，
  不追加 ambient。profile 中仍含弱跨类型表达，这是识别难题的一部分。
  χ 沿用实测空液滴。它不等价于所有 native marker 都严格互斥的 toy。
- 本轮是固定细胞和 χ 后的核心估计/扣除审计，不是重跑 cell calling。
  类型数：低污染 6，零污染 7。未构成独立于开发集的验证。
- 稿件历史 ARI 对应的 `realistic_gt_fig4_res005` 缓存有 4,000 个细胞，
  T/NK/Myeloid/B 为 2149/173/1092/586；本次缓存为
  1346/575/1434/585。不能混用标签、队列及 ARI。

## 剂量与扣除路径

单位为比例，百分比表保留两位小数；完整结果见原始 JSON。

| 路径 | 零污染 native 保留 | 低污染基线保留区间 | 低污染新增 ambient 去除区间 |
|---|---:|---:|---:|
| 当前 adaptive | 93.19% | 89.64–90.92% | 16.56–27.82% |
| fixed 剂量 | 83.39% | 82.72–86.42% | 20.23–52.77% |
| mixture 剂量 | 93.19% | 89.93–91.09% | 16.43–26.64% |
| 已知新增剂量（零污染为 0） | 100.00% | 91.99–92.75% | 14.82–21.55% |
| adaptive，关闭再分配 | 97.33% | 94.44–95.98% | 8.36–21.92% |
| adaptive，关闭 soupOnly | 93.72% | 90.12–91.40% | 13.13–24.39% |

默认零污染共删 1,226,641 UMI，其中非保护 rank-1（含再分配）
938,671，保护基因 rank-1 192,000，soupOnly 95,970。
关闭再分配少删 744,943 UMI，约为原删除量的 60.7%。
默认低污染共删 2,205,436 UMI；关闭再分配少删 1,032,374。

路径追踪使用真实整数分配函数，按非保护、保护、soupOnly 三路记录，
总量与最终删除计数核对。再分配贡献用同剂量反事实对照衡量，
不将其误写成另一项可与三路重复相加的删除量。

这支持“估计偏差与分配共同作用”，不支持仅调整 selector 就能解决。
零污染 adaptive/mixture 剂量中位数均为 295.56 UMI，fixed 为 2666.80；
低污染分别为 604.80、604.80、3109.79，真实新增剂量中位数为 423。

## 候选与否决

候选只循环使用非保护、非 soupOnly 基因未花完的 χ 份额；被保护基因
省下的份额不再成为其他基因的扣除预算。不添加经验阈值或公共配置。
仅通过诊断脚本临时替换函数，产品实现未变。

| 数据集 | 当前 | 候选 | 关闭再分配 |
|---|---:|---:|---:|
| 零污染 native 保留 | 93.19% | 97.30% | 97.33% |
| Mixture sensitivity | 98.57% | 94.67% | 84.65% |
| hgmm12k sensitivity | 96.70% | 87.64% | 86.76% |

Mixture specificity 99.8589%→99.8723%；hgmm12k
99.4103%→99.4297%，改善不足以抵消 sensitivity 损失。
两个 barnyard 当前对照精确复现开发日志的 sensitivity/specificity。
所有计数上限检查通过。候选属于保留/清除权衡，并非净改进。
按 DEVLOG 的验收标准否决，不继续将此候选扩展到胎肝/肾全队列。

## 当前 HVG 与聚类诊断

在同一低污染核心输出上，raw/corrected 均无零 UMI 细胞。
中位 library size 4479→4055，中位检测基因数 1396→1104。
各自选 2,000 HVG，交集 1,229。原始 HVG 的 log-CP10K 方差比中位数
1.068，5 个超过 2；HVG 并集中 EIF1/HLA-A/H3F3B/FTL 方差比分别
12.15/9.03/7.55/6.05。这是观测到的方差变化，尚未证明全部由
细胞级剂量噪声导致，也可能含类型间差异与 library normalization。

固定 normalize/log/HVG(2000)/scale/PCA(50)/neighbors(20 PCs,15邻居)
的处理参数和随机种子。下面 ARI 对照缓存 marker 标签，非独立生物真值；
不选择最优分辨率、不修改现有论文评价函数、不代替稿件数字。

| Leiden resolution | raw | corrected，自选 HVG | corrected，固定 raw HVG |
|---|---:|---:|---:|
| 0.08 | 0.5194 | 0.5853 | 0.5846 |
| 0.20 | 0.5149 | 0.4464 | 0.4462 |
| 0.35 | 0.3977 | 0.3898 | 0.3779 |

当前缓存的变化依赖分辨率，固定 raw HVG 没有一致救回 ARI。
历史日志的 0.513→marker-only 0.956 不能直接用于解释本次版本。
结果不能支持“全部是 HVG artifact，因此 native 无损”的结论。

## 已落实的代码质量修复

`tests/test_scenarios.py` 两处无膨胀测试原先在 `denoise()` 改写 `X`
之后读取 raw，实际比较输出与自身。改为调用前复制输入，并为固定
零污染场景记录 <6% 总 native 计数损失的回归界限（当前 5.39%）。
该界限只防止该固定场景进一步退化，不是通用零污染安全承诺。

基线核心测试 133 passed；修改后场景测试 8 passed。新增脚本静态检查通过。

## 可复现文件

- `scripts/diagnose_removal_paths.py`
- `scripts/evaluate_reallocation_candidate.py`
- `scripts/diagnose_current_hvg.py`
- `data/processed/removal_paths_20260906/results.json`：含产品源码 SHA256、
  类型级指标、分路径计数；相邻 h5ad/npz 冻结估计与基线/注入矩阵。
- `data/processed/reallocation_candidate_20260907.json`
- `data/processed/current_hvg_20260907.json`：含每个分辨率的 confusion table。

复跑命令需使用新的输出路径以保留既有记录：

```bash
python scripts/diagnose_removal_paths.py --out data/processed/removal_paths_new
python scripts/evaluate_reallocation_candidate.py --audit data/processed/removal_paths_new --out data/processed/reallocation_candidate_new.json
python scripts/diagnose_current_hvg.py --audit data/processed/removal_paths_new --out data/processed/current_hvg_new.json
```

下一项有依据的研究是：固定真实新增剂量和每个类型×基因总扣除量，
区分类型间信号变形、类型内细胞分配噪声与归一化引起的方差变化。
只有在不牺牲已验证的去污染端点时才考虑替换细胞分配规则；不继续扫
总预算、ρ、fragment fold，也不把 marker-only 特征选择用于提高论文排名。
