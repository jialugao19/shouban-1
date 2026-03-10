# shouban2: 新策略(重写信号)与回测展示网页计划

## 0. 背景与现有结论(来自 `shouban/plan_limit/output/plan_limit_20260301/report.md`)

- 2024 的关键问题是信号“极稀疏 + 极端集中日”(例如 `20241009`), 导致指标对少数交易日高度敏感, 资金化风险高.
- `Top-K` 上限是最有效的第一道资金化约束: `K=5` 能显著压制 `max_day_picks`, 且在 2024 all 口径上提升 `Sharpe`.
- 仅放松单一阈值(例如 rzg 的过滤条件)可能提升收益, 但会带来集中度爆发; 因此“信号更连续 + 组合 Top-K + 可复盘诊断”应绑定成一个闭环.

目标: 在 `/home/maomao/shouban2` **从零实现**一套新策略(不复用 `/home/maomao/shouban` 的代码), 重点重写信号生成逻辑, 并规划一个网页用于展示回测结果与复盘诊断.

数据约束:
- 原始数据只读来自挂载的 `/data`.
- 中间产物与回测输出写入挂载的 `/data-cache` 下新建目录(避免污染代码仓库, 也便于网页直接读取).

---

## 1. 新策略定义: `shouban2_firstboard_topk_v1`

### 1.1 核心思想

- 解决“稀疏/集中”的根因不是仅调阈值, 而是把事件信号改为可排序的 **continuous score**.
- 用 **统一候选集 + 连续打分 + 每日 Top-K** 的组合形式, 强制把收益来源从“极端日爆发”迁移为“跨日稳定排序”.

### 1.2 交易对象与基本口径(锁死, 避免漂移)

- 交易日历: `/data/ashare/market/trade_cal.feather`.
- 日频行情: `/data/ashare/market/stock1d/YYYY/YYYYMMDD.feather`.
- 分钟行情: `/data/ashare/market/stock1m/YYYY/YYYYMMDD.feather`.
- 竞价量能代理口径:
  - `auction_vol(d)` = 09:30:00 minute bar 的 `Vol`(或同口径字段).
- “触板/直板/首板”定义:
  - `touch_limit(i,d) := high(i,d) == up_limit(i,d)`.
  - `limit_close(i,d) := close(i,d) == up_limit(i,d)`.
  - `first_board(i,d) := touch_limit(i,d) and (过去 L 天内 touch_limit 为 0)`.
  - `first_straight_board(i,d) := limit_close(i,d) and (过去 L 天内 touch_limit 为 0)`.

备注: 本策略默认把“首板口径”设为 `first_board`(触板也算), 以提高 coverage, 同时用 Top-K 和连续 score 控制噪声.

### 1.3 候选集合(信号层, d 为买入日)

候选集只由“昨日事件”触发, 但打分尽量连续:

- `C_sbdk(d)`(首板低开候选):
  - `first_straight_board(i,d-1) == True`.
  - `amount(i,d-1) >= 1e8`.
  - `rp60(i,d-1) <= 0.6`(相对位置放宽, 交给 score 再筛).

- `C_sbgk(d)`(首板高开候选):
  - `first_straight_board(i,d-1) == True`.
  - `amount(i,d-1) in [5.5e8, 2.0e9]`.
  - `total_mv(i,d-1) >= 70e8 and circ_mv(i,d-1) < 520e8`(若数据可得).

- `C_rzg(d)`(弱转强候选):
  - `touch_limit_not_close(i,d-1) == True`(昨日触板但未封板, 且过去 L 天无 touch_limit).
  - `close(i,d-1)/close(i,d-4)-1 <= 0.28`(保留防极端拉升过滤).
  - 不再使用硬阈值 `close/open-1 <= -0.05` 作为生死条件, 改为打分项(见下).

最终候选集: `C(d) = union(C_sbdk(d), C_sbgk(d), C_rzg(d))`.

### 1.4 连续打分(核心, 必须完整可复盘)

打分采用“分项 score + 加权和”的形式, 每个分项都落盘以便诊断:

基础特征(全部以 d-1 为准, d 当天用 open/auction_vol):
- `open_ratio = open(i,d)/close(i,d-1)`.
- `auction_vol_ratio = auction_vol(i,d)/(vol(i,d-1))`.
- `avg_ret = vwap(i,d-1)/close(i,d-1)-1`.
- `close_open_ret = close(i,d-1)/open(i,d-1)-1`.
- `rp60 = (close(i,d-1)-lo60)/(hi60-lo60)`(60 日相对位置).
- `upper_shadow = (high(i,d-1) - max(open(i,d-1), close(i,d-1))) / (high(i,d-1) - low(i,d-1))`.
- `vol_shrinkage = vol(i,d-1) / vol(i,d-2)`.
- `limit_quality`(封板质量, 见下).

分项定义(建议全部 clamp 到 [0,1], 具体公式在实现时写死):
- `score_open_band`: open_ratio 越接近“目标区间中心”越高.
  - sbdk 目标区间: [0.955, 0.97].
  - sbgk 目标区间: [1.00, 1.06].
  - rzg 目标区间: [0.98, 1.09].
- `score_limit_quality`: 昨日封板越“早且硬”分越高(用于修正 After-Limit Effect).
  - `limit_time_score`: 昨日从开盘到“首次封住涨停”的分钟索引越早分越高.
  - `limit_hardness_score`: 封板后是否反复开板, 以及开板时的放量程度越低分越高.
  - 数据来源约束: 封板时间与开板次数可从 `stock1m` 近似计算; 若要使用“封单量/成交量”, 需要 `/data` 内存在可用的 Level2(snapshot/order)数据, 否则该子项不实现.
- `score_auction`: 竞价量能采用非线性(按子策略分别定义).
  - sbgk: 单调增, 但引入“过热阈值”(例如 `auction_vol_ratio > 0.15`)后转为惩罚, 防止过热抢筹.
  - sbdk: 寻找“缩量止跌”, auction_vol_ratio 过高视为放量杀跌风险, 得分转头向下.
- `score_liquidity`: amount(i,d-1) 单调增但饱和(避免大票无限加分).
- `score_strength`:
  - sbgk: `avg_ret_sbgk = vwap/close*1.1-1` 单调增, 强度更高得分更高.
  - rzg: `avg_ret_rzg` 单调增.
- `score_rzg_reversal`: “真伪弱转强”分解为形态 + 量能两部分.
  - `score_upper_shadow`: `upper_shadow` 越高得分越高(更偏“长上影洗盘”, 而不是大阴线砸盘).
  - `score_vol_shrinkage`: `vol_shrinkage` 接近 1(温和放量/平量)得分更高, 巨量放大(断头)与极度缩量都降分.
- `score_rp_penalty`: rp60 越高惩罚越大(避免高位追涨).

总分:
- 对每个候选, 先按其所属子策略(sbdk/sbgk/rzg)计算 `score_sbdk/score_sbgk/score_rzg`.
- `score_total = max(score_sbdk, score_sbgk, score_rzg)`(命中多类时取最强解释路径).

### 1.5 选股与组合(资金化约束优先)

- 每日从 `C(d)` 按 `score_total` 降序取 `Top-K`(默认 K=5), 等权买入.
- 交易撮合与成本模型:
  - 买入: d 日 open 成交(先用日频 open).
  - 卖出: 先做最小闭环(例如 T+1 open 卖出)以隔离信号质量; 后续再引入分钟级止盈/止损.
  - 成本: commission/stamp/slippage bps 写死在 config.
- 可交易性过滤(先做最小“防滑”):
  - `limit_up_at_open`: 若 d 日 open 已等于 up_limit, 直接过滤(不可成交的虚假收益必须剔除).

备注: 先把“信号质量”和“卖出规则复杂度”解耦, 避免策略效果来自偶然的出场规则.

---

## 2. 数据与中间产物(全部落在 `/data-cache`)

### 2.1 缓存根目录(统一命名)

- `/data-cache/shouban2/firstboard_topk_v1/`
  - `runs/<run_id>/`(单次回测)
  - `experiments/<batch_id>/`(批量实验)

`run_id/batch_id` 规则(实现时锁死):
- 建议包含: 策略名 + 年份/区间 + git hash(若可获取) + 时间戳.

### 2.2 单次回测输出契约(网页直接读取)

- `runs/<run_id>/config.yaml`(完整参数与数据路径, 复现入口).
- `runs/<run_id>/signals/`
  - `daily_candidates/YYYY/YYYYMMDD.feather`(候选 + 各分项 score + score_total).
  - `daily_topk/YYYY/YYYYMMDD.feather`(最终买入清单, 含排序键).
- `runs/<run_id>/backtest/`
  - `nav.feather`(trade_date, nav, cash, holdings, n_pos).
  - `trades.feather`(成交明细, 用于复盘与换手诊断).
- `runs/<run_id>/evaluation/report.md`(核心指标 + coverage/集中度 + 关键极端日摘要).
- `runs/<run_id>/assets/*.png`(NAV/Drawdown/Picks/Score 分布等图).

### 2.3 批量实验输出契约(复盘/对比)

- `experiments/<batch_id>/spec.yaml`(实验列表与参数).
- `experiments/<batch_id>/summary.yaml`(每个 exp_id: 指标 + coverage + 集中度 + 关键诊断统计).
- `experiments/<batch_id>/report.md`(汇总表 + 图 + shortlist 建议).
- `experiments/<batch_id>/assets/*.png`.

---

## 3. 代码组织(从零实现, 不依赖 `/shouban`)

拟目录:

- `shouban2/firstboard_topk_v1/`
  - `__init__.py`
  - `config.py`(dataclass, 写死所有参数, 不用环境变量, 不用 argparse).
  - `io_trade_dates.py`(trade_cal 读取, date range 处理).
  - `io_market.py`(stock1d/stock1m 读取, 统一字段名与 dtype).
  - `features_daily.py`(rp60/vwap/limit flags 等).
  - `signals.py`(候选生成 + 分项打分 + Top-K 选择, **必须可落盘复盘**).
  - `backtest.py`(撮合与净值, 先实现最小闭环).
  - `evaluation.py`(指标 + coverage/集中度 + 极端日摘要 + 出图).
  - `run_2024.py`(跑 2024 全年示例, 产物写入 `/data-cache/.../runs/...`).
  - `run_experiments.py`(读取 `spec.yaml` 批量跑, 写 `summary.yaml`/`report.md`).

---

## 4. 网页计划: 回测结果展示(面向复盘与改进)

### 4.1 技术选型(建议)

- Web framework: `Streamlit`.
- 数据来源: 只读 `/data-cache/shouban2/firstboard_topk_v1/` 下的 `runs/` 与 `experiments/`.

### 4.2 页面信息架构(IA)

1) **Batch Overview**
   - 选择 `batch_id`, 渲染 `summary.yaml` 表格.
   - 内置筛选: coverage, p99/max_day_picks, Sharpe, MDD.
   - 生成 shortlist(Top-N)并固定显示.

2) **Run Detail**
   - 展示 NAV/Drawdown/Picks 分布.
   - 展示 score 分布与 Top-K 的 score 时间序列(判断是否变成“单日事件驱动”).
   - 极端日 drill-down:
     - `max_day` 的候选、Top-K、分项 score, 以及排序键(auction_vol_ratio, amount_prev).

3) **Experiment Diff**
   - 选择两个 exp_id, 展示指标差分与集中度变化.
   - 自动列出“变化最大的过滤条件/分项得分”(用于定位改动的真实影响).

---

## 5. 实施里程碑与验收(重点: 复盘/实验/改进的工程闭环)

### M0: 端到端最小闭环(单年)

实施:
- 从 `/data` 读取 2024, 在 `/data-cache/shouban2/firstboard_topk_v1/runs/<run_id>/` 写完整产物.
- 信号必须落盘 `daily_candidates`(含分项), 以保证可复盘.
- 必须实现 `limit_up_at_open` 过滤, 并在报告里统计其过滤数量与对 coverage 的影响.

验收:
- `Top-K` 约束严格成立: `max_day_picks == K`.
- `evaluation/report.md` 必须包含:
  - Sharpe, MDD, ann_return/vol.
  - coverage(nonempty_days/total_days), p99/max, max_day.
  - max_day 的 Top-K 明细(含分项 score).
  - “收益贡献 Top-10 交易日”(用于确认是否仍然被少数天主导).

### M1: 批量实验框架(可复现 + 可对比)

实施:
- 引入 `spec.yaml` 定义实验网格(例如: K, 各分项权重, rzg_reversal 饱和参数, first_board_lookback).
- 把建议项作为可控实验维度(以避免一次引入过多变化):
  - `score_limit_quality` 的开关与权重(仅使用 stock1m 近似版; Level2 版另起实验).
  - `score_auction` 的非线性形状参数(过热阈值/惩罚强度; sbdk 的“缩量止跌”形状).
  - `score_rzg_reversal` 的上影线/量能项权重与饱和参数.
- 批量运行并输出 `summary.yaml` + `report.md` + 图.

验收:
- 任一实验都能通过其 `config.yaml` 复现(路径/参数/区间一致).
- `summary.yaml` 必须包含“集中度”与“极端日摘要”的结构化字段, 支持网页筛选.
 - `report.md` 必须明确: 本轮最优实验的提升来源(是 coverage 提升, 还是极端日抑制, 还是稳定日收益抬升).

### M2: 复盘模板与自动诊断(面向下一轮改进)

实施:
- 固化 `postmortem.md` 模板(每个 batch 自动生成一份), 结构包含:
  - 结论(保留/回滚/下一步).
  - 失败模式: 极端日贡献度, 行业/市值集中, 换手异常.
  - 需要新增的诊断图/字段(以 issue 列表形式写入).
- 在 evaluation 中自动输出:
  - “收益贡献 Top-10 交易日”.
  - “极端日 vs 常态日”的 picks/score 分布对比.
- 引入“市场情绪”作为实验维度(不作为默认策略, 先做对照实验):
  - `Market_Sentiment` 从日频数据计算(例如: 昨日首板数量, 昨日连板家数, 昨日晋级率等的标准化组合).
  - 动态 K: 情绪亢奋期降低 K, 冰点期 K=0/1, 回暖期提高 K(阈值写入 config 并纳入 spec.yaml 网格).

验收:
- 每个 batch 结束, 都能回答三个问题:
  1) 收益是否来自少数天.
  2) Top-K 是否有效抑制集中且不牺牲过多收益.
  3) 下一轮要改哪一个分项/阈值, 以及预期影响的方向(coverage vs noise).
 - 对动态 K 实验, 必须额外回答: 动态 K 的收益提升是否来自“避开亏损期”(冰点空仓), 还是来自“更好的排序质量”.

### M3: 网页 MVP(把复盘流程产品化)

实施:
- Streamlit 页面实现 Batch Overview / Run Detail / Experiment Diff.

验收:
- 通过网页能在 5 分钟内完成一次 batch 复盘:
  - 找到最优实验.
  - 找到其最大风险点(集中度/极端日).
  - 明确下一轮实验该改什么参数/分项.

---

## 6. 风险与硬约束

- `/data` 字段口径必须一次性对齐; 若字段名不一致, 统一在 `io_market.py` 做映射并在文档锁死.
- `total_mv/circ_mv` 若在 `/data` 不可得, 需要在 M0 前确认数据源与字段; 否则相关过滤应暂时移除并在报告中明确标注.
- 所有结构化输出优先使用 YAML, 便于 diff 与审计.
