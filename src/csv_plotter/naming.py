from __future__ import annotations

import re
from dataclasses import dataclass


# Aceita:
# 03_25_ref.csv
# 03_25_1000.csv
# 03_25_eta_ref.csv
# 03_25_eta_1000.csv

_VELOCITY_CASE_RE = re.compile(
    r"^(?P<C>\d{2})_(?P<X>\d+)_(?P<tag>ref|1000)(?:_.*)?$",
    re.IGNORECASE,
)

_ETA_CASE_RE = re.compile(
    r"^(?P<C>\d{2})_(?P<X>\d+)_eta_(?P<tag>ref|1000)(?:_.*)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CaseName:
    C: str        # "03" ou "09"
    X: int        # 25, 50, 100, 200
    tag: str      # "ref" ou "1000"
    method: str   # "nz" para ref, "kn" para 1000
    base: str     # "03_25", "09_100", etc.
    kind: str     # "velocity" ou "eta"


def parse_case_name(stem: str) -> CaseName | None:
    stem = stem.split(";")[0]

    m_eta = _ETA_CASE_RE.match(stem)
    if m_eta:
        C = m_eta.group("C")
        X = int(m_eta.group("X"))
        tag = m_eta.group("tag").lower()
        method = "nz" if tag == "ref" else "kn"
        base = f"{C}_{X}"

        return CaseName(
            C=C,
            X=X,
            tag=tag,
            method=method,
            base=base,
            kind="eta",
        )

    m_vel = _VELOCITY_CASE_RE.match(stem)
    if m_vel:
        C = m_vel.group("C")
        X = int(m_vel.group("X"))
        tag = m_vel.group("tag").lower()
        method = "nz" if tag == "ref" else "kn"
        base = f"{C}_{X}"

        return CaseName(
            C=C,
            X=X,
            tag=tag,
            method=method,
            base=base,
            kind="velocity",
        )

    return None