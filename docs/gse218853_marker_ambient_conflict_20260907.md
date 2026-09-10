# GSE218853 同基因 native/ambient 冲突验证（2026-09-07）

脚本：`scripts/analyze_gse218853_marker_ambient_conflict_20260907.py`。
这是独立于 AmbiDose ownership 和 corrected counts 的 SNP-level 检验；生产代码未修改。

对每个基因，先用 `ambient_fraction=0` 的 endogenous-control cell×gene 对计算各 broad
type 的平均 raw count，最高者定义为 owner（至少 2 个 control 对）。随后保留同一 owner
type 中 `ambient_fraction>0` 的 ambient-positive 对，构成“真实 marker 与 ambient marker
同时出现”的冲突集。SNP-derived ambient fraction 只是 allele-level proxy，因此不能当作
精确 UMI 真值；结果用于相对比较和识别方向，不是绝对 recovery 估计。

rep1 得到 802 个 cell×gene 冲突对、136 个基因。以
`raw × ambient_fraction` 作为 foreign ambient proxy、`raw × (1−ambient_fraction)`
作为 native proxy：

| 方法 | ambient recovery proxy | native loss proxy |
|---|---:|---:|
| AmbiDose | 4.13% | 1.44% |
| SoupX | 10.42% | 3.63% |
| DecontX | 39.90% | 13.89% |

只看 owner native mean ≥10 的较强 marker 冲突对，AmbiDose 为 4.75% / 1.42%，SoupX
为 12.04% / 3.61%，DecontX 为 41.90% / 12.57%。

结论：AmbiDose 在这种冲突上更保守，native 损失较低，但它没有识别出 foreign allele
并逐等位基因修正；只是较少删除整个 gene count。因此当前普通 count-level 输出不能
同时做到高 ambient recovery 和零 native loss。真正的识别需要 allele-aware counts、
供体 genotype 或其他独立来源证据。该结果不支持继续调 subtraction strength 来解决
来源不可识别性。
