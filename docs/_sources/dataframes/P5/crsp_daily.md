# Dataframe: `P5:crsp_daily` - 

Daily CRSP stock returns pulled from WRDS for exploratory analysis.


## DataFrame Glimpse

```
Rows: 5000
Columns: 4
$ permno          <i64> 21366
$ date   <datetime[ns]> 2022-01-03 00:00:00
$ ret             <f64> 0.020895
$ prc             <f64> 21.1557


```

## Dataframe Manifest

| Dataframe Name                 |                                                    |
|--------------------------------|--------------------------------------------------------------------------------------|
| Dataframe ID                   | [crsp_daily](../dataframes/P5/crsp_daily.md)                                       |
| Data Sources                   | CRSP                                        |
| Data Providers                 | WRDS                                      |
| Links to Providers             |                              |
| Topic Tags                     |                                           |
| Type of Data Access            |                                   |
| How is data pulled?            | Pulled via wrds python package in src/pull_CRSP_stock.py                                                    |
| Data available up to (min)     | 2022-01-03 00:00:00                                                             |
| Data available up to (max)     | 2022-01-03 00:00:00                                                             |
| Dataframe Path                 | E:\full stack\p05_zhu-pan_2026\p5\_data\crsp_daily.parquet                                                   |


**Linked Charts:**


- [P5:crsp_returns](../../charts/P5.crsp_returns.md)



## Pipeline Manifest

| Pipeline Name                   | Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models                       |
|---------------------------------|--------------------------------------------------------|
| Pipeline ID                     | [P5](../index.md)              |
| Lead Pipeline Developer         | samoszhu             |
| Contributors                    | samoszhu           |
| Git Repo URL                    |                         |
| Pipeline Web Page               | <a href="file://E:/full stack/p05_zhu-pan_2026/p5/docs/index.html">Pipeline Web Page      |
| Date of Last Code Update        | 2026-02-08 01:38:20           |
| OS Compatibility                |  |
| Linked Dataframes               |  [P5:crsp_daily](../dataframes/P5/crsp_daily.md)<br>  |


