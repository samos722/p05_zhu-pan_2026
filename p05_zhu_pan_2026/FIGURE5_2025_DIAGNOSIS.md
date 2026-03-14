# Figure 5 (_2025) 中 Long-Short 与 Price > 5 缺失的原因

## 诊断结果

| 指标 | Replication (2021-10 ~ 2024-05) | _2025 (2024-05-31 ~ 2024-12-31) |
|------|--------------------------------|----------------------------------|
| Overnight 新闻数 | 大量 | 3,083 |
| 与 GPT labels 合并后 | 大量 | 2,615 |
| YES/NO 标签数量 | 676,790 | 1,258 |
| **有 YES/NO 的交易日数** | **903** | **仅 2 天** |
| overnight_headlines_cleaned | 有数据 | **0** |
| cumret_long_short / cumret_price5 | 正常 | **空** |

## 根本原因

1. **GPT 标签在 _2025 区间覆盖率极低**  
   - 2024-06 ~ 2024-12 只有 **2 个交易日** 的 overnight 新闻有 YES/NO 标签  
   - 很可能是 GPT Batch 只对 2021-10 ~ 2024-05 做了标注

2. **`overnight_headlines_cleaned` 为空**  
   - 这 2 天与 CRSP 合并后，经过 `dropna()` 得到 0 行  
   - 可能原因：  
     - 合并得到的 `next_open` / `next_close` / `prev_close` 存在较多 NaN  
     - 或 ticker / 日期在 CRSP 中匹配不足

3. **Long-Short 与 Price > 5 曲线无数据**  
   - 因 `overnight_headlines_cleaned` 为空，`cumret_long_short` 和 `cumret_price5` 都是空序列  
   - 图中自然无法绘制这两条线

## 建议

1. **补齐 _2025 区间的 GPT 标签**  
   - 对 2024-06 ~ 2024-12 的新闻运行 `label_headlines_gpt_batch.py`（包括 `--all` 和 `--fetch`）  
   - 确保该区间内的新闻都有 YES/NO 标签

2. **可选：在图例中注明数据缺失**  
   - 若短期内无法补齐标签，可在图例或注释中说明：  
     - “_2025 样本中，GPT 标签覆盖不足，Long-Short 与 Price > 5 无法计算。”
