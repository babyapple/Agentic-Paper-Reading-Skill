# Paper Schema / 单篇论文结构化记录

本文件定义"读完一篇论文后，机器可处理的那份记录"长什么样。
用途：跨论文对比、批处理、后续评测打分（Completeness / Accuracy 的输入）。

报告（给人看的散文）见 `review_template.md`；本文件是它的结构化对应物。
两者应保持一致：报告里写的，记录里要有；记录里标了缺的，报告里要说清。

## 顶层字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `paper_id` | string | 是 | 稳定标识；建议用文件名去掉扩展名 |
| `title` | string | 是 | 论文标题（原文） |
| `authors` | string[] | 否 | 作者 |
| `venue` | string | 否 | 期刊 / 会议 / 学位授予单位 |
| `year` | integer | 否 | 年份；不确定写 `null` |
| `language` | string | 是 | `zh` / `en` / 其他 |
| `paper_type` | enum | 是 | `empirical` \| `humanities` \| `survey` |
| `domain` | string | 否 | 领域，如 `dance studies` |

## 内容字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `problem` | string | 它要解决的问题，一句话 |
| `motivation` | string | 为什么值得解决 |
| `core_claim` | string | 作者最想让你相信的那句话 |
| `contributions` | string[] | 作者**自称**的贡献，逐条 |
| `method` | string | 实证类＝方法 / 模型 / 实验设计；人文类＝论证路径 |
| `materials` | string[] | 实证类＝数据集、实验对象；人文类＝史料、文本、作品、田野 |
| `baselines` | string[] | 对比对象（实证类＝baseline；人文类＝对比的既有观点 / 作品 / 研究） |
| `metrics` | string[] | 评价标准（实证类＝指标；人文类＝判断"更成立"的标准） |
| `results` | object[] | 见下 |
| `supporting_analysis` | string[] | 消融、稳健性检验；人文类＝对反例与边界情况的处理 |
| `strengths` | string[] | 你的判断，不是作者的自评 |
| `limitations` | string[] | 你的判断；区分"作者承认的"与"你发现的" |
| `extensions` | string[] | 可能的后续工作 |

## `results[]` 条目

```json
{
  "claim": "作者的主张（原文表述或紧贴原文的转述）",
  "evidence": "支撑它的证据：数字、表格、案例、史料",
  "page": 8,
  "evidence_strength": "strong | partial | weak | absent",
  "note": "任何需要说明的地方"
}
```

- `evidence_strength = absent` 时，`evidence` 写 `null`，`note` 说明为什么它没有依据。
- **不允许**为了让表格好看而给某个 claim 编造证据。

## 缺口与可信度

```json
{
  "gaps": ["缺消融分析", "第 3 节的方法定义依赖未引用的前置文献"],
  "confidence": {
    "paper_type": "high | medium | low",
    "core_claim": "high | medium | low",
    "limitations": "high | medium | low"
  },
  "reading_trace": {
    "pages_total": 14,
    "pages_read": [1, 2, 3, 8, 9, 10, 11, 12],
    "tool_calls": 6,
    "searches": 1
  }
}
```

`reading_trace` 与后续实验指标直接相关：`Tool Calls`、`Latency`、`Token Cost`、
以及"覆盖度不足"的诊断都从这里取数。

## 引用格式

- 文中引用一律用 `[p.N]`，N 为 PDF 页码（1-based）。
- 跨页写 `[p.8-10]`；无法确定页码时写 `[p.?]`，并在 `gaps` 里记录。
- 不引用你自己总结出来的话当作原文。

## 填写原则

1. 字段缺失写 `null` 或空数组，**不要**用猜测值占位。
2. 有争议的字段，把两种读法都写进 `note`，不要二选一硬选。
3. 先填满 `results` 与 `gaps`，再写 `strengths` / `limitations`——避免先有结论再找证据。
