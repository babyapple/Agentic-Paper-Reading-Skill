#!/usr/bin/env python3
"""Skill 内嵌入口：从 PDF 提取带页码的文本。

实现只有一份，在仓库根的 `tools/pdf_reader.py`；这里只是让 Skill 目录自包含、
便于把 `skills/paper-reading/` 单独拷走时仍能说明"脚本在哪、怎么调"。
参数与 `tools/pdf_reader.py` 完全一致：

    python3 skills/paper-reading/scripts/extract_pdf.py read paper/某篇.pdf --pages 1-3
    python3 skills/paper-reading/scripts/extract_pdf.py search paper/某篇.pdf "关键词"
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pdf_reader import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
