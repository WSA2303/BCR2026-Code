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
    Espera colunas: z_cm,u_cm_s.
    """
    if not theory_path.exists():
        raise FileNotFoundError(f"Não encontrei o CSV da teoria em: {theory_path}")

    df = pd.read_csv(theory_path, encoding="utf-8-sig", sep=None, engine="python")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    if "z_cm" not in df.columns or "u_cm_s" not in df.columns:
        raise KeyError(f"Teoria precisa ter colunas z_cm e u_cm_s. Colunas: {list(df.columns)}")

    z_cm = df["z_cm"].to_numpy(dtype=float)
    u_cm_s = df["u_cm_s"].to_numpy(dtype=float)

    z_m = z_cm / 100.0
    u_mps = u_cm_s / 100.0

    mask = np.isfinite(z_m) & np.isfinite(u_mps)
    z_m = z_m[mask]
    u_mps = u_mps[mask]

    order = np.argsort(z_m)
    z_m = z_m[order]
    u_mps = u_mps[order]

    z_unique, idx = np.unique(z_m, return_index=True)
    u_unique = u_mps[idx]

    return z_unique, u_unique


def c_token_from_filename(stem: str) -> str:
    """
    Extrai o primeiro token do nome (antes do primeiro '_').
    Ex: '03_100_1000' -> '03'
        '09_200_ref'  -> '09'
    """
    return stem.split("_", 1)[0].strip()


def c_token_to_float(c_token: str) -> float:
    """
    Mapeia tokens tipo '03'/'09' -> 0.3/0.9.
    Aceita também '3','9','0.3','0.9'.
    """
    t = str(c_token).strip().replace(",", ".")
    if t in {"03", "3", "0.3"}:
        return 0.3
    if t in {"09", "9", "0.9"}:
        return 0.9
    raise ValueError(f"C inválido no nome do arquivo: {c_token!r}. Esperado 03 ou 09.")


def theory_path_for_C(out_dir: Path, c_token: str) -> Path:
    C = c_token_to_float(c_token)
    return out_dir / f"theory_zcm_ucms_C{C:.1f}.csv"


def normalize_numeric_to_SI(df: pd.DataFrame, xcol: str, ycol: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Numérico: assume SI (m e m/s).
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
    """
    z0 = z_num - float(np.min(z_num))
    zmax = float(np.max(z0))
    ratio = zmax / h_th if h_th > 0 else float("inf")

    if 0.8 <= ratio <= 1.2:
        return z0, u_num, f"OK (ratio={ratio:.3f})"

    if 1.8 <= ratio <= 2.2:
        mask = z0 <= h_th
        return z0[mask], u_num[mask], f"CUT 0..h (ratio={ratio:.3f})"

    z_use = z0 * (h_th / zmax) if zmax > 0 and h_th > 0 else z0
    return z_use, u_num, f"RESCALE (ratio={ratio:.3f})"


def main():
    data_dir = ROOT / "data"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    xcol, ycol = "U_0", "z"

    csvs = [p for p in data_dir.glob("*.csv") if "z0_manual" not in p.name.lower()]
    if not csvs:
        raise SystemExit("Não encontrei CSVs em data/.")

    # -------------------------
    # Agrupa por C (03/09)
    # -------------------------
    by_C: dict[str, list[tuple[int, str, Path]]] = defaultdict(list)

    for p in csvs:
        info = parse_case_name(p.stem)
        if info is not None:
            c_token = str(info.C)
            X = info.X
            method = info.method
        else:
            # fallback mínimo: pega C do nome e tenta extrair X/method do padrão esperado
            c_token = c_token_from_filename(p.stem)

            parts = p.stem.split("_")
            if len(parts) < 3:
                print(f"[SKIP] Nome fora do padrão: {p.name}")
                continue

            try:
                X = int(parts[1])
            except Exception:
                print(f"[SKIP] Não consegui ler X em: {p.name}")
                continue

            # último token: 'ref' ou '1000' (você usa 'method' = 'kn'/'nz' no seu parse_case_name;
            # se aqui cair no fallback, você pode ajustar conforme sua convenção)
            last = parts[-1].lower()
            method = "kn" if "1000" in last else "nz" if "ref" in last else "nz"

        by_C[c_token].append((X, method, p))

    # -------------------------
    # Cache da teoria por C
    # -------------------------
    theory_cache: dict[str, tuple[np.ndarray, np.ndarray, float, Path]] = {}

    def get_theory(c_token: str) -> tuple[np.ndarray, np.ndarray, float, Path]:
        if c_token in theory_cache:
            return theory_cache[c_token]
        tpath = theory_path_for_C(out_dir, c_token)
        z_th, u_th = load_theory_csv(tpath)
        h_th = float(np.max(z_th)) if len(z_th) else 0.0
        theory_cache[c_token] = (z_th, u_th, h_th, tpath)
        return theory_cache[c_token]

    # -------------------------
    # Calcula L2
    # -------------------------
    all_rows = []

    for c_token, items in by_C.items():
        z_th, u_th, h_th, tpath = get_theory(c_token)

        for X, method, path in items:
            mult = X_TO_MULT.get(X)
            if mult is None:
                print(f"[SKIP] X={X} não está em {sorted(X_TO_MULT.keys())}: {path.name}")
                continue

            df = load_xy_from_csv(path, xcol, ycol)
            z_num, u_num = normalize_numeric_to_SI(df, xcol, ycol)

            z_use, u_use, msg = prepare_domain(z_num, u_num, h_th)
            u_ref = np.interp(z_use, z_th, u_th)

            err = l2_norm_percent_continuous(z_use, u_use, u_ref)

            all_rows.append({
                "C": c_token,
                "method": method,
                "X": X,
                "mult": mult,
                "L2_percent": err,
                "file": path.name,
                "domain_fix": msg,
                "theory_file": tpath.name,
            })

    if not all_rows:
        raise SystemExit("Nenhum caso válido para calcular L2.")

    all_df = pd.DataFrame(all_rows).sort_values(["C", "method", "mult"])
    table_path = out_dir / "l2_velocity_table.csv"
    all_df.to_csv(table_path, index=False)

    # -------------------------
    # Gera UMA figura por C
    # -------------------------
    for c_token in sorted(by_C.keys(), key=lambda s: c_token_to_float(s)):
        sub = all_df[all_df["C"] == c_token].copy()
        out_png = out_dir / f"l2_velocity_C{c_token}.png"
        plot_one_C(sub, out_png)
        print("[OK] Figura:", out_png, "| Teoria:", theory_path_for_C(out_dir, c_token).name)

    print("[OK] Tabela:", table_path)


if __name__ == "__main__":
    main()