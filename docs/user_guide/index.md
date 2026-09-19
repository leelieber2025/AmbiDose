# User Guide

The supported workflow has four stages:

1. classify raw droplets, using Cell Ranger filtered barcodes as cells when they are available;
2. estimate one ambient profile per library from empty droplets;
3. estimate a per-cell ambient dose using coarse groups (or a caller-supplied broad `type_key`);
4. subtract the dose into a new integer-count layer.

| Topic | Page |
|-------|------|
| Inputs, labels, and multi-sample contract | {doc}`workflow` |
| Command-line interface | {doc}`cli` |
| Estimator and limitations | {doc}`method` |

```{toctree}
:maxdepth: 2

workflow
cli
method
```
