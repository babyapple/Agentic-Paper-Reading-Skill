"""Agent 层：第四阶段的 Agent Loop。

模块分工（对应路线图第十八节的目录）：

- `llm.py`      —— 模型接口（OpenAI 兼容 + 测试用 ScriptedClient）
- `state.py`    —— 当前信息状态与预算记账
- `planner.py`  —— 决定下一步 action（在线 LLM / 离线启发式）
- `executor.py` —— 把 action 落到 `tools/` 的三个真实 Tool 上
- `agent.py`    —— 主循环：READ / SEARCH / ANALYZE / FINISH + 停止条件 + trace

第五阶段（预算与停止条件细化）、第六阶段（Critic）会在这一层继续长。

这里用惰性导出而不是直接 import，避免 `python3 -m agent.agent` 触发
"module found in sys.modules before execution" 的 RuntimeWarning。
"""

from typing import Any

__all__ = ["PaperAgent", "run_agent"]


def __getattr__(name: str) -> Any:  # pragma: no cover - 仅影响导入风格
    if name in __all__:
        from agent import agent as _agent

        return getattr(_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
