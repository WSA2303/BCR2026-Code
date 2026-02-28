from __future__ import annotations
import numpy as np


def l2_norm_percent_continuous(z: np.ndarray, u_num: np.ndarray, u_ref: np.ndarray) -> float:
    """
    Norma L2 (%) baseada em integral:
      100 * sqrt( ∫ (u_num-u_ref)^2 dz ) / sqrt( ∫ u_ref^2 dz )
    """
    z = np.asarray(z, dtype=float)
    u_num = np.asarray(u_num, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)

    # ordena por z
    order = np.argsort(z)
    z = z[order]
    u_num = u_num[order]
    u_ref = u_ref[order]

    num = np.trapezoid((u_num - u_ref) ** 2, z)
    den = np.trapezoid((u_ref) ** 2, z)

    if den <= 0:
        return float("nan")

    return float(100.0 * np.sqrt(num / den))