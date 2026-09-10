# 低 purity 细胞分层验证（2026-09-07）

生产代码未修改。脚本：`scripts/validate_low_purity_subgroups_20260907.py`；结果：
`data/processed/low_purity_subgroups_20260907/results.json`。

这是 DEVLOG 中 confidence-scoped likelihood 候选的分层补充，不是新候选。使用同一
低添加 PBMC 固定件、同一 adaptive dose 和同一 KNN purity；只把默认 subtract 与已有
strength=0.1/1.0 candidate 的结果按 purity 分层。purity 是独立 PCA/KNN 近邻标签一致率，
不是生物学真值。

全体 3940 个细胞的 purity 均值 0.9970、中位数 1.0；purity<0.9 只有 46 个，
purity<0.8 只有 11 个。因此这是小样本探索，不能给出可靠的低-purity 总体效应估计。

strength=1.0 相对默认的 ERS lower 变化：

| purity 分层 | n | ΔERS lower | ΔARS lower |
|---|---:|---:|---:|
| <0.8 | 11 | +0.00216 | -0.00054 |
| 0.8–0.9 | 35 | +0.00128 | -0.00276 |
| 0.9–0.95 | 24 | +0.00076 | -0.00057 |
| ≥0.95 | 3870 | +0.00033 | -0.00059 |

strength=0.1 的变化接近噪声（<0.00015 ERS）；strength=1.0 在低-purity 组的
保留改善较大，但高-purity 细胞也同步改善，不能归因于 purity 特异机制；同时 ARS
下降，表示这是整体减少/重分配扣除的 trade-off。低-purity 子组没有显示独立的、可
区分于全体的收益。

结论：DEVLOG 中“低-purity 细胞是否隐藏局部收益”在这个 PBMC 固定件上没有得到
有力支持，也不能仅凭 11/46 个细胞关闭该问题。下一次若继续，应使用真实组织全量
细胞并按样本分层，预先固定 purity 阈值和最小细胞数；不能把本结果写成普遍安全性
结论。
