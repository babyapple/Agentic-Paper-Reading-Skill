# Agentic Paper Reading Skill

一个能**自主阅读、检索、分析论文**的科研 Agent —— 从固定 Prompt 的 baseline 一路迭代到
Fixed vs Agentic 的对照实验。

路线（对应 `新建 文本文档.md` 的十七节路线图）：

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 1 | 固定 Prompt Baseline | ⬜ 未开始（阶段 7 做对照时才需要） |
| 2 | Paper Reading **Skill**（`SKILL.md` + references） | ✅ 完成 |
| 3 | Skill + **Tools**（PDF 读取 / 检索 / 表格分析） | ✅ 完成 |
| 4 | **Agent Loop**（READ / SEARCH / ANALYZE / FINISH + trace） | ✅ 完成 |
| 5 | **Budget / Stop Condition**（token 预算、覆盖度、无进展检测） | ✅ 完成（当前） |
| 6 | **Critic / Reflection**（判词 → 驳回 → 补缺 → 再复核） | ✅ 完成（当前） |
| 7–10 | Fixed vs Agentic / Metrics / Difficulty / Ablation | ⬜ 下一步 |

## 目录

```text
.
├── paper/                       # Step 1：10 篇舞蹈学论文（中文 PDF 语料）
├── skills/paper-reading/        # Step 2-3：Skill 定义
│   ├── SKILL.md                 #   程序性知识：怎么读、报告契约、必须守住的规则
│   ├── agents/openai.yaml       #   接入用元数据
│   ├── references/              #   按需加载：报告模板 + 结构化记录 schema
│   └── scripts/                 #   薄入口（实现复用 tools/，避免两份逻辑走偏）
├── tools/                       # Step 3：Agent 能"做什么"
│   ├── manifest.json            #   Tool 契约（name / parameters / returns / command）
│   ├── pdf_reader.py            #   Tool 1 读 PDF，保留页码
│   ├── paper_search.py          #   Tool 2 本地语料 + arXiv/Crossref 检索
│   └── table_analyzer.py        #   Tool 3 表格抽取 / 统计 / 排名 / 对比 baseline
├── agent/                       # Step 4-6：Agent Loop + 预算/停止条件 + Critic
│   ├── state.py                 #   当前信息状态 + 预算 + 指标记账
│   ├── llm.py                   #   OpenAI 兼容客户端（+ 测试用 ScriptedClient）
│   ├── planner.py               #   下一步 action 从哪来（在线 LLM / 离线启发式）
│   ├── executor.py              #   action → 三个真实 Tool → observation
│   ├── coverage.py              #   第五阶段：四维度覆盖度（问题/方法/证据/结果）
│   ├── critic.py                #   第六阶段：判词结构 + 模型版 / 规则版 Critic
│   ├── agent.py                 #   主循环 + 五类停止条件 + Critic 回路 + trace
│   └── trace_utils.py           #   trace → 动作路径 / Tool Calls / Token / Latency
├── tests/                       # 154 项回归测试（纯标准库，默认不联网）
├── .llm.env                     # 本地密钥/端点配置（已 gitignore）
├── .llm.env.example             # 上面那个文件的模板（可提交）
├── traces/                      # 运行轨迹 JSONL（已 gitignore）
└── data/cache/                  # PDF 文本抽取缓存（已 gitignore）
```

## 快速开始

```bash
# 0. 依赖：只要有 poppler 的 pdftotext 就能跑；装了 PyMuPDF 会自动优先用它
sudo apt install -y poppler-utils        # 可选：pip install pymupdf

# 1. 跑一次 Agent（没配 API key 会自动降级为离线启发式策略，循环照样跑通）
python3 -m agent.agent --paper "paper/汉代袖舞的风格类型与文化寓意_梁宇.pdf" \
    --trace traces/demo.jsonl

# 2. 配好密钥就能用真模型：编辑工作区根目录的 .llm.env（已 gitignore，不会进仓库）
#    没有这个文件就先从模板复制： cp .llm.env.example .llm.env
#    LLM_API_KEY=sk-xxx
#    LLM_BASE_URL=https://api.deepseek.com/v1
#    LLM_MODEL=deepseek-chat
python3 -m agent.agent --paper paper/某篇.pdf --report        # 收尾后写 10 节报告
python3 -m agent.agent --paper paper/某篇.pdf --config /别的/路径.env   # 或用别的配置文件
python3 -m agent.agent --paper paper/某篇.pdf --no-config     # 忽略配置（CI/测试）

# 3. 看这次运行走了什么路径、花了多少，以及 Critic 说了什么
python3 -m agent.trace_utils traces/demo.jsonl
python3 -m agent.trace_utils traces/*.jsonl --table            # 多次运行对比（阶段 8 的雏形）

# 3b. 阶段 5/6 的开关
python3 -m agent.agent --paper paper/某篇.pdf --stop-when-covered   # 覆盖度达标即停
python3 -m agent.agent --paper paper/某篇.pdf --no-critic           # 消融：去掉 Critic
python3 -m agent.agent --paper paper/某篇.pdf --disable SEARCH,ANALYZE  # 消融：去掉某个工具
python3 -m agent.agent --paper paper/某篇.pdf --max-total-tokens 40000 --max-no-progress 2

# 4. 单看工具层（阶段 3）
python3 tools/pdf_reader.py search "paper/汉代袖舞的风格类型与文化寓意_梁宇.pdf" "袖舞"
python3 tools/table_analyzer.py compare tests/fixtures/ablation.md \
    --baseline Baseline --label Method --lower-is-better Cost

# 5. 跑测试
python3 -m unittest discover -s tests -v
```

## 配置：`.llm.env`（不用设全局环境变量）

密钥与端点写在**工作区根目录的 `.llm.env`** 里，它已经在 `.gitignore` 中，不会进仓库；
模板是可提交的 `.llm.env.example`：

```bash
cp .llm.env.example .llm.env      # 然后填 LLM_API_KEY
python3 -m agent.agent --paper paper/某篇.pdf --report
```

```ini
# .llm.env —— 每行 KEY=VALUE，支持 # 注释、引号、export 前缀
LLM_API_KEY=sk-xxxxxxxx
LLM_BASE_URL=https://api.deepseek.com/v1     # 换 OpenAI 就写 https://api.openai.com/v1
LLM_MODEL=deepseek-chat                      # 如 gpt-4o-mini / qwen-max / 本地网关的模型名
#LLM_TIMEOUT=60
#LLM_MAX_RETRIES=2
#LLM_MAX_TOKENS=1200
```

| 取值来源 | 优先级 | 说明 |
| --- | --- | --- |
| 命令行参数 | 最高 | `--api-key` / `--base-url` / `--model` / `--timeout` / `--max-retries` |
| `.llm.env` | 高 | 默认读工作区根目录；可用 `--config PATH` 或环境变量 `LLM_CONFIG_FILE` 换路径 |
| 环境变量 | 低 | `LLM_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` 等，仅为兼容保留 |
| 代码默认值 | 最低 | `https://api.deepseek.com/v1` + `deepseek-chat`，超时 60s，重试 2 次 |

`--no-config` 可显式忽略配置文件（测试与 CI 用）；`LLM_API_KEY` 留空或文件缺失时会**自动降级为
离线启发式策略**而不是报错，只有显式 `--planner llm` 才会以"缺少 key"退出（exit 2）。
日志里只会写"key 来自哪个文件/变量"，永不回显 key 本身。

## Agent Loop（阶段 4）

```text
User Request → Planner → {READ | SEARCH | ANALYZE | FINISH} → Executor → Observation ─┐
                   ▲                                                                  │
                   └──────────────────────────────────────────────────────────────────┘
                                      （或 FINISH → Final Report）
```

- **决策空间只有 4 个动作**：`READ`（读论文，可指定页范围）、`SEARCH`（本地语料 / 联网）、
  `ANALYZE`（先 `extract` 抽表得到 `table:N`，再算 `rank`/`compare`/…）、`FINISH`。
- **状态驱动而不是固定顺序**：Planner 每步拿到的是"已读哪些页、还剩多少预算、上一步工具返回了什么"，
  所以同一套代码在不同论文上会走出不同路径——这正是第七节说的路径 A/B/C，也是阶段 7 要研究的东西。
- **只看行动与结果**：系统只要求模型输出一个 action JSON，不要求它暴露思维过程；
  trace 里也只有 action、工具、参数、返回摘要与耗时。
- **失败不崩、而是变成 observation**：解析不出 JSON、页码越界、表不存在、模型超时，
  都会以可读理由回到 Planner 手里，让它自己改。
- **空转保护**：被系统拒绝的提案连续超过阈值（默认 2 次）判定空转，强制收尾并写明 `finish_reason`。

停止条件（阶段 5 细化，见下节）：Planner 主动 `FINISH`；预算耗尽；无进展；覆盖度达标；空转保护；
解析连续失败。结果永远落在 `stop_cause` + `finish_reason` 里，不吞掉。

## 预算与停止条件（阶段 5）

三类约束叠在一起，每一种停止都写进结构化的 `stop_cause`，方便第七/九阶段按"怎么停的"分组比较：

| 约束 | 默认 | 触发后的 `stop_cause` |
| --- | --- | --- |
| 步数 | `--max-steps 8` | `budget_steps` |
| 工具调用 | `--max-tool-calls 12` | `budget_steps` 之前先吃掉调用额度（提案被拒） |
| 检索次数 | `--max-searches 3` | 超出的 SEARCH 提案被拒（`stop_cause=blocked_loop` 若连续空转） |
| 阅读页数 | `--max-pages 20`（只算新页） | 超预算的 READ 被拒 |
| **token** | `--max-total-tokens 60000`（0=不限） | `budget_tokens` —— 在**调用 Planner 之前**就查，不会先花掉再发现超了 |
| **无进展** | `--max-no-progress 3` | `no_progress` —— 连续若干步没拿到新页/新命中/新表格 |
| **覆盖度** | `--stop-when-covered`（默认关） | `coverage` —— Problem/Method/Evidence/Results 四维度都有原文线索即停 |
| 空转 | 被拒提案连续超过 2 次 | `blocked_loop` |

覆盖度由 `agent/coverage.py` 从**已读正文**里按关键词线索算，并且：

- 三档状态（`covered` / `weak` / `missing`）而不是布尔值——单靠一个词刷出来的不算覆盖；
- 每个维度都带**页码**，所以它既能当 Planner 的输入（"你还没读到结果类内容"），
  也能当停止条件的判据；
- 它只是**启发式线索**（`source=keyword-heuristic`），不等于"结论正确"或"内容完备"，
  所以在 prompt 里始终标注来源，不假装是人工判定。

`python3 -m agent.agent --paper paper/某篇.pdf --stop-when-covered --no-critic` 可以把
"读完四个维度就停"当成一个独立的控制变量来测。

## Critic（阶段 6）

FINISH 之后不是直接交付，而是先过一道 Critic —— 对应路线图的
`Planner → Tools → Answer → Critic → 足够？→ Finish / Continue`：

```json
{"complete": false,
 "missing": ["第6节只展开了'飘逸诡谲'一类，其余三类范型缺图像/文献证据"],
 "next_action": "READ",
 "reason": "报告契约第 5-7 部分不完整"}
```

- **判词结构**与路线图一致，解析宽容（围栏、字符串布尔、`next_action` 校验）。
- **模型版 `LlmCritic`**：把"当前草稿 + 已读页码 + 覆盖度线索 + 工具轨迹"交给模型，
  要一个 JSON 判词；检查清单就是第六阶段那四问（漏方法 / 漏实验 / 无依据结论 / 是否要再搜）。
- **规则版 `RuleCritic`**：不联网，用覆盖度 + `[p.N]` 页码引用 + 工具错误做判定。
  没有 API key 时它让第六阶段照样能跑、能测、能演示，同时也是第十阶段 `-Critic` 对照的天然实现。
- **Critic 只提意见、不执行动作**：驳回时把缺口作为 feedback 回灌给 Planner，由 Planner 决定下一步；
  决策权始终只有一个。连续驳回超过 `--max-critic-rounds`（默认 2）才收尾并标记报告未完成。
- **Critic 自己坏了就放行**：调用失败或判词解析不了时按 `complete=true` 处理并在 `reason` 里说明，
  不让评审环节把主循环拖死（`source=llm-error` / `llm-unparsed`）。

判词会写进 trace（`event=critic`，带 `after_step`），所以"Critic 说了什么、Agent 有没有照做"
在第八/九阶段是可分析的：

```bash
python3 -m agent.trace_utils traces/live_llm.jsonl        # 轨迹 + Critic 判词表
python3 -m agent.trace_utils traces/*.jsonl --table       # 多组对比：critic 轮次 / 覆盖度 / stop_cause
```

## 设计约定（后续做实验时要复用的）

- **Skill ≠ Tool ≠ Agent**：`SKILL.md` 规定"怎么读"，`tools/` 规定"能做什么"，
  `agent/` 决定"下一步做什么"，三者不互相越界。
- **页码是硬约束**：所有读取结果都带页号，报告里的事实必须写成 `[p.N]`；抽不到文本的
  扫描件会被标成 warning，而不是静默返回空。
- **算出来的数字才算数字**：均值、差值、排名、相对 baseline 的增益一律交给 `table_analyzer`，
  模型不心算；`—`/`N/A`/`未报告` 记为缺失而不是 0。
- **工具永远返回可读错误**：参数/文件/后端问题一律 stderr 单行 + exit 2，Agent 侧直接当 observation 用。
- **指标不另埋点**：`Tool Calls`（真实调用计数，被拒绝的提案不算）、`Token Cost`、`Latency`
  全部由 `agent/state.py` 记账并写进 trace 的 summary 行；`pages_read` / `searches` 同源，
  可直接对应 `references/paper_schema.md` 里的 `reading_trace`。
- **离线可跑**：`--planner heuristic` 不联网、不要 key，用来跑回归测试与做"固定工作流"参照；
  在线路径用本地 mock 服务端到端验证过（`--config` 与 `LLM_CONFIG_FILE` 两种发现方式，见 `tests/test_agent.py`）。
- **配置在文件里，不在全局**：`.llm.env`（gitignore）+ `.llm.env.example`（模板），
  优先级 命令行 > 文件 > 环境变量 > 默认值；日志只说明 key 来源，不回显 key。
  ⚠️ 仓库里存在 `.llm.env` 时，CLI 会真的去调模型——所以**测试一律加 `--no-config`**
  （`tests/test_agent.py::TestCli` 已内置该保护）。
- **停止原因是一等公民**：`stop_cause`（枚举）+ `finish_reason`（人话）双双入 trace。
  后续实验按"谁停的"分组比较：`agent_finish` / `critic_accept` / `critic_exhausted` /
  `coverage` / `budget_*` / `no_progress` / `blocked_loop` / `planner_error`。
- **Critic 只提意见**：它输出判词（是否完整 / 缺什么 / 建议动作），不执行动作、不改写草稿；
  坏了就放行（`source=llm-error` / `llm-unparsed`），不阻塞主循环。

## 下一步（阶段 7）

第五/六阶段落地后，真正的实验条件基本齐了：`stop_cause` 可分组、Critic 可开关、
工具可禁用、trace 里有路径与成本。接下来做路线图第七～十阶段：

1. **阶段 7：Fixed vs Agentic** —— `baseline/fixed_workflow.py`（固定顺序读摘要→引言→方法→实验→总结）
   对 `agent/agent.py`（状态驱动），同语料同任务跑一遍，比路径、成本与结果质量；
2. **阶段 8：指标** —— Accuracy / Completeness 需要评测集与人工或评审打分；
   Tool Calls / Token Cost / Latency 已由 `trace_utils` 直接从 trace 出表；
3. **阶段 9：难度变量** —— 给语料标 Easy/Medium/Hard，看动态决策在哪类论文上有优势；
4. **阶段 10：消融** —— `--no-critic` / `--disable SEARCH` / `--disable ANALYZE` 已经就位，
   直接构成 `Full / -Search / -Critic / -Analysis` 四组。

实跑已经暴露出一个值得进实验的问题（见本轮 `traces/live_llm_full.jsonl`）：
Critic 驳回的报告缺口是"**写**得不全"，而 action 空间里只有读/搜/算，
于是 Agent 只能再去 `READ`（甚至去抽不存在的表格），2 轮之后仍被驳回。
这提示第七/九阶段要把"重新组织答案"也纳入决策空间，或让 Critic 的
`next_action` 能表达"回炉重写"。
