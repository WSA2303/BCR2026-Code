from __future__ import annotations

import re
from dataclasses import dataclass

# aceita:
# 09_100_ref
# 03_25_1000
# (e tolera sufixo extra no final, tipo _alterado)
_CASE_RE = re.compile(
    r"^(?P<C>\d{2})_(?P<X>\d+)_(?P<tag>ref|1000)(?:_.*)?$",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class CaseName:
    C: str        # "03" ou "09"
    X: int        # 25, 50, 100, 200
    tag: str      # "ref" ou "1000"
    method: str   # "nz" (ref) ou "kn" (1000)
    base: str     # "09_100"


def parse_case_name(stem: str) -> CaseName | None:
    stem = stem.split(";")[0]
    m = _CASE_RE.match(stem)
    if not m:
        return None

    C = m.group("C")
    X = int(m.group("X"))
    tag = m.group("tag").lower()
    method = "nz" if tag == "ref" else "kn"
    base = f"{C}_{X}"
    return CaseName(C=C, X=X, tag=tag, method=method, base=base)