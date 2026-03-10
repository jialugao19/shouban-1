# shouban2 表现差原因分析与模型信息, 2026-03-10

本分析基于本次测试目录 `031001` 的最新 run:

- `out_root`: `/data-cache/shouban2/031001/firstboard_topk_v1/runs/firstboard_topk_v1_2021_2024_20260310_182751`
- `eval_kind`: `backtest`(见 `evaluation/status.yaml`)
- 区间: `20210104` 至 `20241231`

---

## 1. 结论摘要

1. 该策略当前更像 **intraday edge** 而不是 **overnight continuation edge**.
   - `baseline_t0_open_close` 为正: `ann_return≈0.43`, `sharpe≈1.07`, `mdd≈-0.30`.
   - 但 `baseline_t1_open` 为明显负: `ann_return≈-0.67`, `sharpe≈-1.50`, `mdd≈-0.99`.
2. 组合层“几乎天天买满”的行为放大了负的 overnight 漂移.
   - `coverage≈0.996`, `avg_picks≈4.64`, `max_day_picks=5`.
3. `score_threshold` 的确能改善, 但远不足以扭转.
   - `score_threshold_t1_open` 从 `sharpe≈-1.50` 改善到 `≈-1.07`, 但仍为负.

---

## 2. 模型基本信息(当前实现)

### 2.1 数据来源

只读数据:

- 交易日历: `/data/ashare/market/trade_cal.feather`
- 日频行情: `/data/ashare/market/stock1d/YYYY/YYYYMMDD.feather`
- 分钟行情: `/data/ashare/market/stock1m/YYYY/YYYYMMDD.feather`
- 可交易股票池: `/data/ashare/feature/valid_stocks.feather`

可写产物:

- `/data-cache/shouban2/<test_id>/firstboard_topk_v1/`

### 2.2 候选与路径

信号分三条路径, 最终对每个候选算三条路径分, 取 `max` 为 `score_total` 并记录 `score_path`:

- `sbdk`: 首板直板 + 低开带筛选.
- `sbgk`: 首板直板 + 高开/量能带筛选.
- `rzg`: 昨日触板未封 + 形态/强度条件.

当前路径占比(Top-K 里):

- `sbgk≈60%`, `sbdk≈25%`, `rzg≈14%`(见 `evaluation/picks.yaml` 的 `path_mix`).

### 2.3 特征与打分

特征主要来自:

- 日频: `open/high/low/close/vol/amount/vwap/rp/ma5` 等.
- 分钟: 用 `MinuteIndex==0` 的 `Vol` 作为竞价量能代理(`auction_vol`), 同时计算封板质量(`limit_quality`).

打分结构:

- 分项映射(多为 clamp 到 `[0,1]` 的线性/三角形映射).
- 各路径权重加权和得到 `score_sbdk/score_sbgk/score_rzg`.
- `score_total = max(path_scores)`.
- 排序键: `score_total` -> `auction_vol_ratio` -> `amount_cny_prev`.

### 2.4 选股与仓位管理

- 每日按排序取 `Top-K`, `K=5`.
- `filter_limit_up_at_open=True`(开盘涨停直接剔除).
- 主回测口径: `buy at d open, sell at d+1 open`.
- 成本模型: `commission_bps=3`, `stamp_duty_bps=10`, `slippage_bps=5`.

备注:

- `dynamic_k_by_score` 在本次 run 的 config 中为 `false`.
- `score_threshold=0.55` 只作为报告里的对照版本(`score_threshold_t1_open`), 不改变 baseline 的 daily_topk 落盘.

---

## 3. 症状与证据

### 3.1 按年分解(来自 nav.feather 计算)

- 2021: `return≈-12.95%`, `mdd≈-35.11%`.
- 2022: `return≈-68.18%`, `mdd≈-71.52%`.
- 2023: `return≈-68.90%`, `mdd≈-71.93%`.
- 2024: `return≈-84.74%`, `mdd≈-84.93%`.

整体 drawdown 接近 -99% 与 `baseline_t1_open` 一致, 说明风险控制与方向性风险非常突出.

### 3.2 日收益分布(来自 nav.feather)

- `median` 日收益约 `-0.46%`, 显著偏负.
- 左尾很厚, 1% 分位约 `-7.13%`, 0.1% 分位约 `-10.96%`.

这说明策略不是“偶尔踩雷”, 而是 **日常 drift 就偏负**.

### 3.3 阈值对照的含义

`score_threshold_t1_open` 改善 sharpe 的主要来源更像是:

- `coverage` 降到 `≈0.769`.
- `avg_picks` 降到 `≈1.71`.

也就是“少交易少亏钱”, 而不是“同样频率下选得更准”.

---

## 4. 可能原因(按优先级排序)

### 4.1 核心原因: 目标口径与信号形态不匹配

证据:

- `T0 open->close` 显著为正, 但 `T1 open->open` 显著为负.

解释:

- 当前信号更像“日内情绪/资金博弈”, 次日开盘存在系统性回撤或低开.
- 以 `T+1 open` 做主口径等于主动暴露隔夜风险, 把日内 edge 吃掉甚至反向.

### 4.2 路径结构问题: `sbgk` 占比过高且可能是负贡献主力

证据:

- `path_mix` 中 `sbgk` 约 60%.

解释:

- `sbgk` 倾向更接近“高开强势”, 很容易在次日出现兑现/分歧.
- 如果 `sbgk` 的 overheat 惩罚不够强, 该路径会把组合推向“追涨隔夜”的负期望区.

### 4.3 组合层问题: coverage 过高导致错误交易被放大

证据:

- `coverage≈0.996`, 几乎每天都持仓, 且接近买满.

解释:

- 在 edge 不稳定且多数日 drift 为负时, 高 coverage 会放大亏损速度.
- 阈值对照改善更多来自“少交易”, 也支持该结论.

### 4.4 数据与可交易性问题: 缺失分钟 bar/次日日频 bar 的处理会影响回测一致性

证据:

- 本次 run 过程中出现:
  - `drop_missing_auction_vol`(分钟数据缺失).
  - `drop_missing_next_day`(次日日频缺失, 典型为停牌/无 bar).

解释:

- 这些问题不一定是主要亏损原因, 但会导致样本选择偏差与复现实验不稳定.
- 后续需要在评估里统计这些 drop 的数量与集中在哪些年份/股票池.

---

## 5. 下一步建议(面向“为什么差”和“怎么改”)

1. 先把“研究目标”明确为二选一:
   - A: 做 `overnight` 策略, 继续以 `T+1 open` 为主口径, 但必须增加针对次日开盘回撤的过滤/建模.
   - B: 承认它是 `intraday` edge, 把主口径切到 `T0 close` 或 `T0` 的分钟级规则, 再谈出场优化.
2. 做最小的因果实验, 优先回答:
   - `sbgk only` vs `rzg only` vs `sbdk only`.
   - `score_threshold` 的网格(例如 `0.50/0.55/0.60`) + `dynamic_k_by_score=True`.
3. 在 evaluation 里补 3 个强约束统计, 让问题可量化:
   - `drop_missing_auction_vol` 总数与按年分布.
   - `drop_missing_next_day` 总数与按年分布.
   - `path_mix` 与 `per_path` 的收益/回撤贡献(至少先做 `T1 open` 粗分解).

