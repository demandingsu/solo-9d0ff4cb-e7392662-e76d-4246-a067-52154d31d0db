# 制造计划时序数据校正

FastAPI + SQLite + Docker Compose 的分层产线计划校正工具。计划分析员上传：

1. **层级映射**：`node_id, parent_id, level, name, weight`（产线 → 车间 → 工厂，单根树）
2. **产线预测**：`node_id, period, value`（每条产线每个时段一个值）
3. **已发布车间/工厂预测**：`node_id, period, value` 或 `level, period, value`

系统按层级重算总量、找出与已发布值不一致的时段，按节点权重最小化加权调整平方和，
生成各层数量一致的新快照；网页按节点展示每个时段的调整来源，结果可下载为 CSV。

## 快速开始

```bash
docker compose up --build
# 打开 http://localhost:8000
```

首次使用：新建数据集 → 分别上传 `sample_data/mapping.csv`、`sample_data/line_forecast.csv`、
`sample_data/published.csv` → 点击「运行校正」→ 检查「调整来源」→「发布此预览为快照」。

本地不带 Docker 运行：

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
DATABASE_PATH=./app.db uvicorn app.main:create_app --factory --port 8000
pip install -r requirements-dev.txt && pytest -q
```

## 校正模型

对每个时段独立求解。设产线值为决策变量 `x_l`，内部节点值恒等于其产线后代之和，
因此层级一致性 `父 = Σ子` 天然成立。目标函数为

```
Σ_节点 w_node · (x_node − base_node)²
```

其中 `base_node` 为上传的产线预测在节点上的加总。已发布值作为线性等式约束
（节点单一值，或某层全部节点的合计）。两者都是 `x_l` 的线性函数，于是问题化为
**带线性等式约束的加权最小二乘**，用零空间约化求解（numpy SVD + 线性方程组），
不可行时明确报错而不是悄悄违反发布值。

权重语义：权重越大越"不愿被调整"，代价按 `w·Δ²` 计入；因此同等条件下低权重节点承担
更多平移。权重为 0 的节点不参与代价（求解时用极小正则项保证结果唯一、可复现）。

### 校验与拒绝发布

上传映射时即校验并在不合规时**整次拒绝**：

- 映射成环（含自环）
- 同一节点多个父节点
- 父节点不存在
- 缺层：根不是 plant / 车间下没有产线 / 层级为空 / 层级与树深度不符 / 多根
- 使用已发布预测但节点或层级不存在、同一节点同一时段两个不同发布值
- 父节点与其全部子节点的已发布值互相矛盾（预先给出可读错误）
- 约束矩阵不相容（例如工厂合计与车间合计冲突）→ `OptimizationError`

### 版本、预览与失败语义

- 每个数据集有单调递增的 `input_version`；上传内容有变化才 +1。
- 预览记录输入版本 + 三个文件的 sha256。发布时在同一个写事务内重新校验版本与哈希，
  任何一项不匹配 → `StalePreviewError`，**旧预览不得发布**。
- 校正失败不写入 preview 也不写快照；上一版已发布快照原样保留。
- 快照不可变：每次发布新建版本并记录父快照，旧版本可继续查看/下载。
- 上传 / 预览 / 发布在同一 SQLite 写事务中完成（`BEGIN IMMEDIATE`），
  并发上传与预览不会混用版本。

## 页面

- `/` 数据集列表
- `/datasets/{id}` 上传、校正、发布、历史快照、操作记录
- `/datasets/{id}/snapshots/{v}` 只读快照页
- 每个时段折叠面板显示：不一致时段明细（基线 vs 已发布 vs 差额）、已发布约束、
  按节点的调整来源（`直接发布` / `层级总量` / `层级平衡`）、产线级溯源、校正后各层数值
- 下载：`/datasets/{id}/download/snapshot.csv`、`adjustments.csv`、`conflicts.csv`
  （带 `?version=N` 或 `?preview=N`）

## JSON API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/datasets` | 新建数据集，body `{"name": "..."}` |
| GET | `/api/datasets/{id}` | 上传状态、预览、快照、事件 |
| POST | `/api/datasets/{id}/upload?kind=mapping\|line_forecast\|published` | multipart `file` |
| POST | `/api/datasets/{id}/preview` | 运行校正 |
| POST | `/api/datasets/{id}/publish` | body `{"preview_id": N}` |
| GET | `/api/datasets/{id}/snapshots` | 快照列表 |
| GET | `/api/datasets/{id}/snapshots/{v}` | 快照详情 |
| GET | `/healthz` | 健康检查 |

错误响应：`{"error": "...", "code": "stale_preview_error"}`，HTTP 400。

## 测试

```bash
pytest -q
```

覆盖：映射成环/多父/缺层、CSV 解析、约束可行与不可行、权重变化对调整分配的影响、
随机树与独立 KKT 解交叉验证、版本失效与旧预览拒绝发布、失败保留上一版、并发、
HTTP 页面与下载。

## 目录

```
app/
  main.py        FastAPI 路由（页面 + JSON API）
  service.py     上传/预览/发布编排、版本与失效判定、CSV 导出
  reconcile.py   加权最小二乘校正与一致性校验
  hierarchy.py   树构建与结构校验
  csv_io.py      CSV 解析
  db.py          SQLite schema 与事务
  templates/     Jinja2 页面
  static/        样式
tests/           78 项自动化测试
sample_data/     可直接上传的示例文件（含一个成环反例）
```

## 限制

- 单进程 SQLite；水平扩展需要换成网络数据库并加分布式锁/乐观版本。
- 已发布 `line` 层级合计会锁定每条产线（合计拆分规则未定义，不擅自猜测）。
- 不保存用户账号与权限；如需多人隔离应在反向代理层加认证。
- 不支持按周/季度等非 ISO 周期的自然排序（按字符串前缀解析年月）。
- 当前沙箱内没有 Docker 守护进程，`Dockerfile` / `docker-compose.yml` 已写好但未实际构建镜像；
  应用已在本地 venv 中以 uvicorn 启动并完成上传/校正/发布/下载全链路手工验证。
