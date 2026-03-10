# shouban2

## 1. 项目简介

`shouban2` 是一个策略研究工程, 当前聚焦单策略 `firstboard_topk_v1`:

- 目标: 重写首板/弱转强相关信号为 continuous score, 并用 `Top-K` 组合约束压制集中度.
- 输出: 每次 run 产生可复现的回测产物, 并提供网页浏览与复盘入口.

注意: 该仓库是研究代码, 不是通用量化平台.

---

## 2. 目录结构

- `firstboard_topk_v1/`: 策略实现.
- `webapp/`: Streamlit 网页, 用于浏览 `/data-cache` 下的 run/experiment 产物.
- `plan.md`: 初始策略与工程计划.
- `plan-0309.md`: 第一轮策略改造与小实验结论.
- `plan-0310.md`: 工程整理计划(README, evaluation 回归检查, 技术说明).

---

## 3. 数据来源与约束

### 3.1 数据来源

策略当前使用的数据(只读):

- `/data/ashare/market/trade_cal.feather`
- `/data/ashare/market/stock1d/YYYY/YYYYMMDD.feather`
- `/data/ashare/market/stock1m/YYYY/YYYYMMDD.feather`
- `/data/ashare/feature/valid_stocks.feather`

### 3.2 输出目录

所有中间产物与回测输出写入(可写):

- `/data-cache/shouban2/firstboard_topk_v1/`
  - `runs/<run_id>/`: 单次 run
  - `experiments/<batch_id>/`: 批量实验 batch

---

## 4. 运行方式

运行前提:

- Python 运行环境包含 `pandas`, `polars`, `pyarrow`, `yaml`, `matplotlib`, `streamlit`.
- 本机已挂载 `/data` 与 `/data-cache`.

### 4.1 单次运行(2024 全年)

```bash
PYTHONPATH=/home/maomao python -u /home/maomao/shouban2/firstboard_topk_v1/run_2024.py
```

运行结束会打印:

- `out_root=/data-cache/shouban2/firstboard_topk_v1/runs/<run_id>`

### 4.2 批量实验

`run_experiments.py` 从固定路径读取 `spec.yaml`:

- `/data-cache/shouban2/firstboard_topk_v1/experiments/<batch_id>/spec.yaml`

```bash
PYTHONPATH=/home/maomao python -u /home/maomao/shouban2/firstboard_topk_v1/run_experiments.py
```

---

## 5. 如何看结果

每次 run 会落盘:

- `config.yaml`: 可复现配置.
- `signals/`: 每日候选与 Top-K(含分项 score, 便于复盘).
- `backtest/`: `nav.feather` 与 `trades.feather`.
- `evaluation/`: `report.md` 与结构化 YAML(用于回归检查).
- `assets/`: 常用图(`nav.png`, `drawdown.png`, 等).

最常用入口:

- `/data-cache/.../evaluation/report.md`
- `/data-cache/.../assets/*.png`

---

## 6. 启动网页

```bash
PYTHONPATH=/home/maomao streamlit run /home/maomao/shouban2/webapp/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

网页会从 `/data-cache/shouban2/firstboard_topk_v1/` 读取 `runs/` 与 `experiments/`.

