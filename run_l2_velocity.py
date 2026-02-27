from __future__ import annotations

from pathlib import Path
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str((ROOT / "src").resolve()))

from csv_plotter.io_csv import load_xy_from_csv               # noqa: E402
from csv_plotter.l2norm import l2_norm_percent_continuous     # noqa: E402
from csv_plotter.naming import parse_case_name                # noqa: E402


# Mapeamento FIXO: X -> multiplicador do Nz
# Nz -> 25, 2Nz -> 50, 4Nz -> 100, 8Nz -> 200
X_TO_MULT = {25: 1, 50: 2, 100: 4, 200: 8}


def style_like_example():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "axes.linewidth": 0.8,
    })


def label_for_method(method: str) -> str:
    return r"$\eta_0 = f(N_z)$" if method == "nz" else r"$\eta_0 = f(K_n)$"


def plot_one_C(res_df: pd.DataFrame, out_path: Path) -> None:
    style_like_example()
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=200)

    xticks = [1, 2, 4, 8]
    xticklabels = [r"$N_z$", r"$2N_z$", r"$4N_z$", r"$8N_z$"]

    styles = {
        "nz": dict(color="red", marker="o", linestyle="None", markersize=4),
        "kn": dict(color="blue", marker="*", linestyle="None", markersize=6),
    }

    for method in ["nz", "kn"]:
        sub = res_df[res_df["method"] == method].sort_values("mult")
        if sub.empty:
            continue
        ax.plot(sub["mult"], sub["L2_percent"], label=label_for_method(method), **styles[method])

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Número de volumes")
    ax.set_ylabel(r"Norma $L_2$ (\%) — Perfil de velocidade")
    ax.set_xlim(0.8, 8.2)

    ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.75")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def load_theory_csv(theory_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Lê o CSV da teoria e retorna (z_th_m, u_th_mps) em SI.
    Espera colunas tipo: z_cm,u_cm_s (como você mostrou).
    """
    if not theory_path.exists():
        raise FileNotFoundError(f"Não encontrei o CSV da teoria em: {theory_path}")

    df = pd.read_csv(theory_path, encoding="utf-8-sig", sep=None, engine="python")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # nomes esperados no seu arquivo
    if "z_cm" not in df.columns or "u_cm_s" not in df.columns:
        raise KeyError(f"Teoria precisa ter colunas z_cm e u_cm_s. Colunas: {list(df.columns)}")

    z_cm = df["z_cm"].to_numpy(dtype=float)
    u_cm_s = df["u_cm_s"].to_numpy(dtype=float)

    # converte p/ SI
    z_m = z_cm / 100.0
    u_mps = u_cm_s / 100.0

    # ordena por z e remove NaN
    mask = np.isfinite(z_m) & np.isfinite(u_mps)
    z_m = z_m[mask]
    u_mps = u_mps[mask]

    order = np.argsort(z_m)
    z_m = z_m[order]
    u_mps = u_mps[order]

    # remove duplicatas de z (às vezes aparecem)
    z_unique, idx = np.unique(z_m, return_index=True)
    u_unique = u_mps[idx]

    return z_unique, u_unique


def normalize_numeric_to_SI(df: pd.DataFrame, xcol: str, ycol: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Numérico: assume que vem em SI (m e m/s), como típico do OpenFOAM.
    """
    z = df[ycol].to_numpy(dtype=float)
    u = df[xcol].to_numpy(dtype=float)
    mask = np.isfinite(z) & np.isfinite(u)
    z = z[mask]
    u = u[mask]
    order = np.argsort(z)
    return z[order], u[order]


def prepare_domain(z_num: np.ndarray, u_num: np.ndarray, h_th: float) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Ajusta o domínio do numérico para comparar com teoria em [0, h_th].
    - se zmax ~ h_th -> usa tudo
    - se zmax ~ 2*h_th -> corta em 0..h_th
    - senão -> reescala (fallback) para 0..h_th
    """
    z0 = z_num - float(np.min(z_num))
    zmax = float(np.max(z0))
    ratio = zmax / h_th if h_th > 0 else float("inf")

    if 0.8 <= ratio <= 1.2:
        return z0, u_num, f"OK (ratio={ratio:.3f})"

    if 1.8 <= ratio <= 2.2:
        mask = z0 <= h_th
        return z0[mask], u_num[mask], f"CUT 0..h (ratio={ratio:.3f})"

    # fallback: reescala
    z_use = z0 * (h_th / zmax) if zmax > 0 else z0
    return z_use, u_num, f"RESCALE (ratio={ratio:.3f})"


def main():
    data_dir = ROOT / "data"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    # teoria vem do outputs (como você mostrou)
    theory_path = out_dir / "theory_zcm_ucms.csv"
    z_th, u_th = load_theory_csv(theory_path)
    h_th = float(np.max(z_th))

    xcol, ycol = "U_0", "z"

    csvs = [p for p in data_dir.glob("*.csv") if "z0_manual" not in p.name.lower()]

    by_C: dict[str, list[tuple[int, str, Path]]] = defaultdict(list)
    for p in csvs:
        info = parse_case_name(p.stem)  # padrão: 09_100_ref / 03_25_1000
        if info is None:
            continue
        by_C[info.C].append((info.X, info.method, p))

    if not by_C:
        raise SystemExit("Não encontrei arquivos no padrão 'C_X_ref.csv' ou 'C_X_1000.csv' em data/.")

    all_rows = []

    for C, items in by_C.items():
        for X, method, path in items:
            mult = X_TO_MULT.get(X)
            if mult is None:
                print(f"[SKIP] X={X} não está em {sorted(X_TO_MULT.keys())}: {path.name}")
                continue

            df = load_xy_from_csv(path, xcol, ycol)
            z_num, u_num = normalize_numeric_to_SI(df, xcol, ycol)

            # ajusta domínio do numérico para comparar no domínio da teoria
            z_use, u_use, msg = prepare_domain(z_num, u_num, h_th)

            # interpola teoria nos z do numérico (adaptativo)
            u_ref = np.interp(z_use, z_th, u_th)

            # L2 contínuo no grid do numérico
            err = l2_norm_percent_continuous(z_use, u_use, u_ref)

            all_rows.append({
                "C": C,
                "method": method,
                "X": X,
                "mult": mult,
                "L2_percent": err,
                "file": path.name,
                "domain_fix": msg,
            })

    all_df = pd.DataFrame(all_rows).sort_values(["C", "method", "mult"])
    all_df.to_csv(out_dir / "l2_velocity_table.csv", index=False)

    for C in sorted(by_C.keys(), key=lambda s: int(s)):
        sub = all_df[all_df["C"] == C].copy()
        out_png = out_dir / f"l2_velocity_C{C}.png"
        plot_one_C(sub, out_png)
        print("[OK] Figura:", out_png)

    print("[OK] Tabela:", out_dir / "l2_velocity_table.csv")
    print("[OK] Teoria usada:", theory_path)


if __name__ == "__main__":
    main()