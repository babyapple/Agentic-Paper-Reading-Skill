#!/usr/bin/env python3
"""LLM 客户端 —— 只做一件事：把 messages 发出去，把文本与用量收回来。

为什么自己写而不是上框架（路线图第十九节）：第四阶段真正要看清的是
`LLM → Tool → Observation → Decision` 这条链，中间不该有魔法。

- `OpenAICompatibleClient`：任何 OpenAI 兼容的 `/chat/completions` 都能用
  （DeepSeek、OpenAI、vLLM、本地网关……）。配置默认来自**工作区根目录的 `.llm.env`**
  （已 gitignore），也支持命令行参数与环境变量兜底，优先级：
  **命令行 > `.llm.env` > 环境变量 > 默认值**。
- `parse_env_file` / `load_config_file`：读那个配置文件；换路径用 `--config` 或 `LLM_CONFIG_FILE`。
- `ScriptedClient`：给测试用的确定性"LLM"，按剧本返回，可断言 prompt 内容。

失败一律抛 `LlmError`（可读、带状态码），由 Agent Loop 决定是重试还是收尾。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
API_KEY_ENV_ORDER = ("LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ARK_API_KEY", "MOONSHOT_API_KEY")
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILENAME = ".llm.env"
CONFIG_PATH_ENV = "LLM_CONFIG_FILE"
# 配置文件里认得的键（也接受同义的环境变量名，见 API_KEY_ENV_ORDER）
CONFIG_KEYS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_TIMEOUT", "LLM_MAX_RETRIES", "LLM_MAX_TOKENS")


class LlmError(RuntimeError):
    """LLM 调用失败：网络、鉴权、限流、返回格式不对，都走这里。"""


@dataclass
class ChatResult:
    text: str
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int = 0


@dataclass
class ClientConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    timeout: float = 60.0
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 1200


class OpenAICompatibleClient:
    """最小的 OpenAI 兼容 chat client（stdlib urllib，无第三方依赖）。"""

    def __init__(
        self,
        config: ClientConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
        transport: Callable[[urllib.request.Request, float], bytes] | None = None,
    ) -> None:
        if not config.api_key:
            raise LlmError("缺少 API key：请设置 LLM_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY")
        self.config = config
        self.name = f"openai-compatible:{config.model}"
        self._sleep = sleep
        self._transport = transport or self._urlopen

    # ------------------------------------------------------------- 传输 --
    @staticmethod
    def _urlopen(request: urllib.request.Request, timeout: float) -> bytes:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read()

    @property
    def endpoint(self) -> str:
        return self.config.base_url.rstrip("/") + "/chat/completions"

    def complete(self, messages: Sequence[dict[str, Any]], **overrides: Any) -> ChatResult:
        payload = {
            "model": overrides.get("model", self.config.model),
            "messages": list(messages),
            "temperature": overrides.get("temperature", self.config.temperature),
            "max_tokens": overrides.get("max_tokens", self.config.max_tokens),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "Accept": "application/json",
            },
            method="POST",
        )

        started = time.time()
        last_error: str = ""
        for attempt in range(self.config.max_retries + 1):
            try:
                raw = self._transport(request, self.config.timeout)
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                except Exception:  # pragma: no cover - 读 body 失败不影响主流程
                    detail = ""
                last_error = f"HTTP {exc.code}: {detail or exc.reason}"
                if exc.code in RETRYABLE_STATUS and attempt < self.config.max_retries:
                    self._backoff(attempt)
                    continue
                raise LlmError(f"LLM 请求失败（{last_error}）") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.config.max_retries:
                    self._backoff(attempt)
                    continue
                raise LlmError(f"LLM 请求失败（{last_error}）") from exc

            latency_ms = int((time.time() - started) * 1000)
            return self._parse(raw, latency_ms)

        raise LlmError(f"LLM 请求失败（{last_error}）")  # pragma: no cover - 循环内已抛

    def _backoff(self, attempt: int) -> None:
        self._sleep(0.5 * (3**attempt))

    def _parse(self, raw: bytes, latency_ms: int) -> ChatResult:
        try:
            data = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            raise LlmError(f"LLM 返回的不是 JSON：{raw[:200]!r}") from exc
        choices = data.get("choices") or []
        if not choices:
            raise LlmError(f"LLM 返回里没有 choices：{json.dumps(data, ensure_ascii=False)[:200]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # 有些网关返回分段内容
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str) or not content.strip():
            # tool_calls 形态也认一下，方便对接到本项目的 action JSON
            if message.get("tool_calls"):
                content = json.dumps(message["tool_calls"][0].get("function", {}), ensure_ascii=False)
            else:
                raise LlmError("LLM 返回内容为空")
        usage = data.get("usage") or {}
        return ChatResult(
            text=content,
            model=data.get("model", self.config.model),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
        )


def parse_env_file(text: str) -> dict[str, str]:
    """解析 `.llm.env`：`KEY=VALUE` 每行一条。

    容忍 `#` 注释、空行、`export` 前缀、值两侧的引号、`=` 两侧空格、CRLF。
    不认得的行直接跳过——配置文件不该因为多写一行注释就整个失效。
    """

    values: dict[str, str] = {}
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key = key.strip().upper()
        value = value.strip()
        # 先去掉行尾注释：`sk-xxx  # 备注`；只在 `#` 前有空白时截断，避免误伤 URL 里的 #
        value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_config_file(path: str | Path) -> dict[str, str]:
    """读配置文件；文件不存在或读不了时抛 `LlmError`（显式指定却找不到，是配置错误）。"""

    target = Path(path).expanduser()
    if not target.exists():
        raise LlmError(f"配置文件不存在：{target}（可用 `.llm.env.example` 复制一份）")
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise LlmError(f"配置文件读不了：{target}（{exc}）") from exc
    values = parse_env_file(text)
    unknown = sorted(set(values) - set(CONFIG_KEYS) - {"OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"})
    if unknown:
        values["__unknown__"] = ",".join(unknown)  # 交给上层决定是否提示，不在这里抛错
    return values



@dataclass
class ScriptedClient:
    """按剧本返回的假 LLM：测试循环、解析、容错时用，不联网。"""

    responses: list[str]
    name: str = "scripted"
    calls: list[list[dict[str, Any]]] = field(default_factory=list)
    results: list[ChatResult] = field(default_factory=list)

    def complete(self, messages: Sequence[dict[str, Any]], **overrides: Any) -> ChatResult:
        self.calls.append(list(messages))
        if not self.responses:
            raise LlmError("剧本已用完（ScriptedClient 没有更多响应）")
        text = self.responses.pop(0)
        result = ChatResult(text=text, model="scripted", input_tokens=100, output_tokens=20, latency_ms=5)
        self.results.append(result)
        return result

    @property
    def prompts(self) -> list[str]:
        """把每次调用的 messages 拼成一个字符串，便于断言 prompt 里有什么。"""

        return ["\n".join(str(msg.get("content", "")) for msg in call) for call in self.calls]


def resolve_config(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
    config_path: str | Path | None = None,
    use_config: bool = True,
    env: dict[str, str] | None = None,
) -> tuple[ClientConfig | None, str]:
    """把配置凑齐。优先级：**命令行参数 > 配置文件 > 环境变量 > 默认值**。

    默认读工作区根目录的 `.llm.env`（已 gitignore），也可以：
    - `--config 别的路径` / `LLM_CONFIG_FILE=/path/to/file` 指定；
    - `--no-config` 明确忽略配置文件（测试/CI 要确定性时用）。

    返回 (配置, 说明)。没有 key 时配置为 None，说明里写清找过哪些地方（**不回显 key**）。
    """

    environ = os.environ if env is None else env
    file_values: dict[str, str] = {}
    origin = ""
    config_found = False

    if use_config:
        if config_path is not None:
            file_values = load_config_file(config_path)  # 显式指定：文件不存在要报错
            origin = f"配置文件 {Path(config_path).expanduser()}"
            config_found = True
        else:
            candidate = environ.get(CONFIG_PATH_ENV) or (REPO_ROOT / CONFIG_FILENAME)
            if Path(candidate).expanduser().exists():
                file_values = load_config_file(candidate)
                origin = f"配置文件 {Path(candidate).expanduser()}"
                config_found = True

    def pick(*keys: str) -> str:
        """按 配置文件 → 环境变量 取值。"""

        for key in keys:
            if file_values.get(key):
                return file_values[key]
        for key in keys:
            if environ.get(key):
                return environ[key]
        return ""

    resolved_key = api_key or pick(*API_KEY_ENV_ORDER)
    if not resolved_key:
        if config_found:
            searched = f"{origin} 里没有 LLM_API_KEY"
        elif use_config:
            searched = f"没找到 {REPO_ROOT / CONFIG_FILENAME}（可复制 .llm.env.example）"
        else:
            searched = "已用 --no-config 忽略配置文件"
        return None, f"未找到 API key（{searched}；也没设 {' / '.join(API_KEY_ENV_ORDER[:3])}）"

    if api_key:
        key_origin = "命令行 --api-key"
    elif file_values.get("LLM_API_KEY"):
        key_origin = origin or f"配置文件 {REPO_ROOT / CONFIG_FILENAME}"
    else:
        key_origin = "环境变量 " + next(name for name in API_KEY_ENV_ORDER if environ.get(name) == resolved_key)

    resolved_base = base_url or pick("LLM_BASE_URL", "OPENAI_BASE_URL") or DEFAULT_BASE_URL
    resolved_model = model or pick("LLM_MODEL", "OPENAI_MODEL") or (
        DEFAULT_MODEL if "deepseek" in resolved_base else "gpt-4o-mini"
    )
    return (
        ClientConfig(
            base_url=resolved_base.rstrip("/"),
            model=resolved_model,
            api_key=resolved_key,
            timeout=_pick_number(file_values, environ, "LLM_TIMEOUT", timeout, 60.0),
            max_retries=int(_pick_number(file_values, environ, "LLM_MAX_RETRIES", max_retries, 2)),
            max_tokens=int(_pick_number(file_values, environ, "LLM_MAX_TOKENS", max_tokens, 1200)),
        ),
        f"在线模型 {resolved_base} / {resolved_model}（key 来自{key_origin}）",
    )


def _pick_number(
    file_values: dict[str, str], environ: dict[str, str], key: str, explicit: float | None, default: float
) -> float:
    """数值型配置：命令行 > 配置文件 > 环境变量 > 默认值；解析不了就退回默认。"""

    if explicit is not None:
        return float(explicit)
    for candidate in (file_values.get(key), environ.get(key)):
        if candidate:
            try:
                return float(candidate)
            except ValueError:
                continue
    return float(default)

