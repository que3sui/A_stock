# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

A股量化分析数据集，用于深度学习基础大作业。纯数据仓库，无代码。

## 数据架构

### 核心数据（量价 + 元信息）

- **`basic.csv`** — 全市场股票基础信息（代码、行业、上市日期、实控人性质），是连接所有数据表的"股票族谱"
- **`trade_cal.csv`** — 交易日历（SSE/SZSE），作为统一时间轴，避免非交易日产生错误
- **`daily/`** — 日频量价数据，每个 `.csv` 文件为一个交易日截面（2016年起至今，~2500+ 个文件）。文件按日期命名 `YYYYMMDD.csv`

### 指数数据

- **`market/`** — 三大指数（000001.SH 上证、000300.SH 沪深300、399006.SZ 创业板）日频量价，字段同 `daily/`
- **`index_weight/`** — 沪深300月度成分股权重 `YYYYMM_000300.SH.csv`

### 可选特征数据

- **`metric/`** — 基本面指标（PE/PB/PS/股息率/换手率/量比/市值/股本），文件数与 `daily/` 一一对应
- **`moneyflow/`** — 资金流向（小单/中单/大单/特大单 买卖量额，按主动买卖划分），文件数与 `daily/` 一一对应
- **`news/`** — 东方财富新闻快讯（标题+内容+时间），2019年起
- **`stock_st/`** — 每日ST股票列表，2016年8月起

## 关键注意事项

- **`daily/`、`metric/`、`moneyflow/` 按交易日截面存储**。构建单只股票时间序列需跨文件读取并 merge，不能用单文件直接读取
- **除权处理**：`daily/` 的 `pre_close` 已是除权价，`pct_chg = (close - pre_close) / pre_close`，可直接使用
- **幸存者偏差**：历史回测必须用 `stock_st/` 剔除当日ST股，否则会包含已退市/ST的"幸存者"
- **PE字段**：亏损公司PE为空值（NaN），非零
- **`index_weight/` 仅覆盖沪深300**（000300.SH），月度频率

## 联合推演参考

`与各大ai的联合推演/` 目录下有三篇与不同AI模型关于如何使用此数据集进行量化分析的深度对话，可作为方法论参考：

- `01deepseek.md` — 因子构建、学院派+实战派双视角分析框架
- `02kimi.md` / `03glm.md` — 其他模型的补充视角
