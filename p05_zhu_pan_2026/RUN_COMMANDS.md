# 直接运行命令（不使用 doit）

在项目根目录 `p05_zhu_pan_2026` 下执行。PowerShell 格式。

---

## 1. data_exploration

```powershell
cd "e:\full stack\p05_zhu-pan_2026\p05_zhu_pan_2026"
python src/data_exploration.py
```

---

## 2. Table 1 replication (2021-10-01 ~ 2024-05-31)

```powershell
cd "e:\full stack\p05_zhu-pan_2026\p05_zhu_pan_2026"
$env:SAMPLE_START="2021-10-01"; $env:SAMPLE_END="2024-05-31"; $env:OUTPUT_SUFFIX="_replication"; python src/compute_portfolio_performance.py
```

---

## 3. Figure 5 replication (2021-10-01 ~ 2024-05-31)

```powershell
cd "e:\full stack\p05_zhu-pan_2026\p05_zhu_pan_2026"
$env:SAMPLE_START="2021-10-01"; $env:SAMPLE_END="2024-05-31"; $env:OUTPUT_SUFFIX="_replication"; python src/graph_trading_strategy.py
```

---

## 4. Table 1 (2024-05-31 ~ 2024-12-31)

```powershell
cd "e:\full stack\p05_zhu-pan_2026\p05_zhu_pan_2026"
$env:SAMPLE_START="2024-05-31"; $env:SAMPLE_END="2024-12-31"; $env:OUTPUT_SUFFIX="_2025"; python src/compute_portfolio_performance.py
```

---

## 5. Figure 5 (2024-05-31 ~ 2024-12-31)

```powershell
cd "e:\full stack\p05_zhu-pan_2026\p05_zhu_pan_2026"
$env:SAMPLE_START="2024-05-31"; $env:SAMPLE_END="2024-12-31"; $env:OUTPUT_SUFFIX="_2025"; python src/graph_trading_strategy.py
```

---

## 6. Table 1 full (2021-10-01 ~ 2024-12-31)

```powershell
cd "e:\full stack\p05_zhu-pan_2026\p05_zhu_pan_2026"
$env:SAMPLE_START="2021-10-01"; $env:SAMPLE_END="2024-12-31"; $env:OUTPUT_SUFFIX="_full"; python src/compute_portfolio_performance.py
```

---

## 7. Figure 5 full (2021-10-01 ~ 2024-12-31)

```powershell
cd "e:\full stack\p05_zhu-pan_2026\p05_zhu_pan_2026"
$env:SAMPLE_START="2021-10-01"; $env:SAMPLE_END="2024-12-31"; $env:OUTPUT_SUFFIX="_full"; python src/graph_trading_strategy.py
```

---

## 说明

项目统一使用 **2024-12-31** 为截止日期（CRSP 数据上限），已与 settings 等配置保持一致。

---

## 输出文件位置

- data_exploration: `_output/crsp_summary_stats.tex`, `*.png`
- Table 1: `_output/performance_table{_replication|_2025|_full}.html` 与 `.tex`
- Figure 5: `_output/cumulative_returns_paper_style{_replication|_2025|_full}.png`
