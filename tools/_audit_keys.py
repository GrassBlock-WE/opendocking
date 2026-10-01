# SPDX-License-Identifier: GPL-3.0-or-later
"""List literal ``tr("...")`` keys used by the GUI that the tables lack."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from odock.gui import i18n  # noqa: E402

PATTERN = re.compile(r'tr\(\s*"([A-Za-z0-9_.]+)"\s*[,)]')
also = re.compile(r'tr\(\s*f"([A-Za-z0-9_.]*)\{')

missing: dict[str, list[str]] = {}
for path in sorted((ROOT / "python" / "odock" / "gui").glob("*.py")):
    text = path.read_text(encoding="utf-8")
    for key in sorted(set(PATTERN.findall(text))):
        if key not in i18n.EN:
            missing.setdefault(key, []).append(path.name)
for key, files in sorted(missing.items()):
    print(f"{key:<34} {', '.join(files)}")
print(f"{len(missing)} missing key(s); EN has {len(i18n.EN)}")
