# A股数据说明

## 一、基础数据

该部分包含股票基础信息、交易日历以及历史量价数据，是本项目的核心数据。

### 1. `basic.csv`

所有股票的基础信息列表。

| 名称         | 类型 | 描述                                  |
| ------------ | ---- | ------------------------------------- |
| ts_code      | str  | 股票代码                              |
| symbol       | str  | 股票代码（简）                        |
| name         | str  | 股票名称                              |
| area         | str  | 地域                                  |
| industry     | str  | 所属行业                              |
| cnspell      | str  | 拼音缩写                              |
| market       | str  | 市场类型（主板/创业板/科创板/北交所） |
| list_date    | str  | 上市日期                              |
| act_name     | str  | 实控人名称                            |
| act_ent_type | str  | 实控人企业性质                        |

### 2. `trade_cal.csv`

交易日历数据。

| 名称          | 类型 | 描述                        |
| ------------- | ---- | --------------------------- |
| exchange      | str  | 交易所 SSE上交所 SZSE深交所 |
| cal_date      | str  | 日历日期                    |
| is_open       | str  | 是否交易 0休市 1交易        |
| pretrade_date | str  | 上一个交易日                |

### 3. `daily/`

按交易日存储的A股日频量价数据目录（从2016年起至今）：

- 每个 `.csv` 文件对应一个交易日
- 文件中包含当日所有股票的量价信息

| 名称       | 类型  | 描述                                                                      |
| ---------- | ----- | ------------------------------------------------------------------------- |
| ts_code    | str   | 股票代码                                                                  |
| trade_date | str   | 交易日期                                                                  |
| open       | float | 开盘价                                                                    |
| high       | float | 最高价                                                                    |
| low        | float | 最低价                                                                    |
| close      | float | 收盘价                                                                    |
| pre_close  | float | 昨收价【除权价】                                                          |
| change     | float | 涨跌额                                                                    |
| pct_chg    | float | 涨跌幅（%） 【基于除权后的昨收计算的涨跌幅：（今收-除权昨收）/除权昨收 】 |
| vol        | float | 成交量 （手）                                                             |
| amount     | float | 成交额 （千元）                                                           |
| vwap       | float | 加权成交平均价（元）                                                      |

> **说明**：
> 数据按照日期作为截面存储，如需构建单只股票的完整历史时间序列，需要在代码中读取多个交易日文件并进行合并（merge）。

---

## 二、指数数据

该部分提供市场指数数据，可用于策略收益对比（benchmark）。

包含以下三个指数：

- 000001.SH 上证指数
- 000300.SH 沪深300
- 399006.SZ 创业板指数

### 1. `market/`

上述指数的日频量价数据。

字段描述同个股。

### 2. `index_weight/`

指数成分股及权重数据。

- 一般来说用不到
- 若选择将股票池限制在某一指数（如沪深300或创业板），可用于构建对应股票池

| 名称       | 类型  | 描述     |
| ---------- | ----- | -------- |
| index_code | str   | 指数代码 |
| con_code   | str   | 成分代码 |
| trade_date | str   | 交易日期 |
| weight     | float | 权重     |

---

## 三、进阶数据（可选）

除基础量价数据外，提供以下扩展数据，可作为特征使用：

### 1. `metric/`

个股每日重要的基本面指标数据。

| 名称            | 类型  | 描述                                   |
| --------------- | ----- | -------------------------------------- |
| ts_code         | str   | 股票代码                               |
| trade_date      | str   | 交易日期                               |
| close           | float | 当日收盘价                             |
| turnover_rate   | float | 换手率（%）                            |
| turnover_rate_f | float | 换手率（自由流通股）                   |
| volume_ratio    | float | 量比                                   |
| pe              | float | 市盈率（总市值/净利润， 亏损的PE为空） |
| pe_ttm          | float | 市盈率（TTM，亏损的PE为空）            |
| pb              | float | 市净率（总市值/净资产）                |
| ps              | float | 市销率                                 |
| ps_ttm          | float | 市销率（TTM）                          |
| dv_ratio        | float | 股息率 （%）                           |
| dv_ttm          | float | 股息率（TTM）（%）                     |
| total_share     | float | 总股本 （万股）                        |
| float_share     | float | 流通股本 （万股）                      |
| free_share      | float | 自由流通股本 （万）                    |
| total_mv        | float | 总市值 （万元）                        |
| circ_mv         | float | 流通市值（万元）                       |

### 2. `moneyflow/`

沪深A股每日资金流向数据。

| 名称            | 类型  | 描述                   |
| --------------- | ----- | ---------------------- |
| ts_code         | str   | 股票代码               |
| trade_date      | str   | 交易日期               |
| buy_sm_vol      | int   | 小单买入量（手）       |
| buy_sm_amount   | float | 小单买入金额（万元）   |
| sell_sm_vol     | int   | 小单卖出量（手）       |
| sell_sm_amount  | float | 小单卖出金额（万元）   |
| buy_md_vol      | int   | 中单买入量（手）       |
| buy_md_amount   | float | 中单买入金额（万元）   |
| sell_md_vol     | int   | 中单卖出量（手）       |
| sell_md_amount  | float | 中单卖出金额（万元）   |
| buy_lg_vol      | int   | 大单买入量（手）       |
| buy_lg_amount   | float | 大单买入金额（万元）   |
| sell_lg_vol     | int   | 大单卖出量（手）       |
| sell_lg_amount  | float | 大单卖出金额（万元）   |
| buy_elg_vol     | int   | 特大单买入量（手）     |
| buy_elg_amount  | float | 特大单买入金额（万元） |
| sell_elg_vol    | int   | 特大单卖出量（手）     |
| sell_elg_amount | float | 特大单卖出金额（万元） |
| net_mf_vol      | int   | 净流入量（手）         |
| net_mf_amount   | float | 净流入额（万元）       |

各类别统计规则如下：
**小单**：5万以下 **中单**：5万～20万 **大单**：20万～100万 **特大单**：成交额>=100万 ，数据基于主动买卖单统计

### 3. `news/`

东方财富每日新闻快讯数据。由于早期新闻数据已无法获得，故该项数据自2019年起。

| 名称     | 类型 | 描述     |
| -------- | ---- | -------- |
| datetime | str  | 新闻时间 |
| content  | str  | 内容     |
| title    | str  | 标题     |

---

## 四、其它数据

这里是补充的可能会使用到的数据。

### 1. `stock_st/`

提供每日的ST股票列表，数据自2016年8月起，过早历史已无法补齐。

| 名称       | 类型 | 描述     |
| ---------- | ---- | -------- |
| ts_code    | str  | 股票代码 |
| name       | str  | 股票名称 |
| trade_date | str  | 交易日期 |
| type       | str  | 类型     |
| type_name  | str  | 类型名称 |

---
