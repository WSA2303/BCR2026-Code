# run_tau_w.py
from __future__ import annotations

from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str((ROOT / "src").resolve()))

from csv_plotter.theory_params import THEORY_PARAMS, set_theory_params  # noqa: E402


# X = 25,50,100,200  ->  mult = 1,2,4,8  (Nz, 2Nz, 4Nz, 8Nz)
X_TO_MULT = {25: 1, 50: 2, 100: 4, 200: 8}


def style_like_example():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 0.8,
    })


def c_token_to_float(c_token: str) -> float:
    t = str(c_token).strip().replace(",", ".")
    if t in {"03", "3", "0.3"}:
        return 0.3
    if t in {"09", "9", "0.9"}:
        return 0.9
    raise ValueError(f"C inválido: {c_token!r}. Esperado 03 ou 09.")


def compute_tau_w_analytical(C_float: float) -> float:
    """
    tau_w = rho * g * h * sin(th)
    (rho, h, th_deg) vêm de theory_params.py, selecionados via set_theory_params(C).
    """
    set_theory_params(C_float)
    rho = float(THEORY_PARAMS["rho"])
    h = float(THEORY_PARAMS["h"])
    th_deg = float(THEORY_PARAMS["th_deg"])
    g = 9.81
    return rho * g * h * math.sin(math.radians(th_deg))


def load_tau_manual_csv(path: Path) -> pd.DataFrame:
    """
    Espera colunas:
      base, tau_w_1000, tau_w_ref
    (aceita separador automático)
    """
    if not path.exists():
        raise FileNotFoundError(f"Não encontrei: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    required = {"base", "tau_w_1000", "tau_w_ref"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"tau_manual.csv faltando colunas {sorted(missing)}. Colunas atuais: {list(df.columns)}")

    df["base"] = df["base"].astype(str).str.strip()
    df["tau_w_1000"] = pd.to_numeric(df["tau_w_1000"], errors="coerce")
    df["tau_w_ref"] = pd.to_numeric(df["tau_w_ref"], errors="coerce")

    df = df.dropna(subset=["base"]).copy()
    df = df[df["base"].str.lower() != "nan"]
    return df


def build_long_table(df_tau: pd.DataFrame) -> pd.DataFrame:
    """
    Converte de formato wide para long:
      base, method, X, mult, tau_w
    method:
      - "nz"  -> tau_w_ref
      - "kn"  -> tau_w_1000
    """
    rows = []
    for _, r in df_tau.iterrows():
        base = str(r["base"]).strip()  # ex: "03_25"
        parts = base.split("_")
        if len(parts) < 2:
            continue

        c_token = parts[0]
        try:
            X = int(parts[1])
        except Exception:
            continue

        mult = X_TO_MULT.get(X)
        if mult is None:
            continue

        # ref -> nz
        if pd.notna(r["tau_w_ref"]):
            rows.append({
                "C": c_token,
                "base": base,
                "X": X,
                "mult": mult,
                "method": "nz",
                "tau_w": float(r["tau_w_ref"]),
            })

        # 1000 -> kn
        if pd.notna(r["tau_w_1000"]):
            rows.append({
                "C": c_token,
                "base": base,
                "X": X,
                "mult": mult,
                "method": "kn",
                "tau_w": float(r["tau_w_1000"]),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("Nenhuma linha válida foi gerada a partir do tau_manual.csv.")
    return out


def plot_tau_for_C(sub: pd.DataFrame, tau_ana: float, out_path: Path) -> None:
    style_like_example()
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=200)

    xticks = [1, 2, 4, 8]
    xticklabels = [r"$N_z$", r"$2N_z$", r"$4N_z$", r"$8N_z$"]

    styles = {
        "nz": dict(color="red", marker="o", linestyle="None", markersize=4),
        "kn": dict(color="blue", marker="*", linestyle="None", markersize=7),
    }
    labels = {
        "nz": r"$\eta_0=f(N_z)$",
        "kn": r"$\eta_0=1000K_n$",
    }

    for method in ["nz", "kn"]:
        s = sub[sub["method"] == method].sort_values("mult")
        if s.empty:
            continue
        ax.plot(s["mult"], s["tau_w"], label=labels[method], **styles[method])

    # linha analítica
    ax.axhline(tau_ana, color="black", linestyle="--", linewidth=1.2, label="Analytical")

    # texto na linha analítica (posição parecida com seu exemplo)
    x_txt = 4.2
    ax.text(x_txt, tau_ana + 0.2, rf"$\tau_w={tau_ana:.2f}\ Pa$", fontsize=11)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Number of volumes")
    ax.set_ylabel(r"$\tau_w\ [Pa]$")
    ax.set_xlim(0.8, 8.2)

    ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.75")
    #ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    data_dir = ROOT / "data"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    tau_path = data_dir / "tau_manual.csv"
    df_tau = load_tau_manual_csv(tau_path)
    long_df = build_long_table(df_tau)

    # salva tabela consolidada
    table_path = out_dir / "tau_w_table.csv"
    long_df.sort_values(["C", "method", "mult"]).to_csv(table_path, index=False)

    # gera 1 figura por C encontrado (03 e 09)
    for c_token in sorted(long_df["C"].unique(), key=c_token_to_float):
        C_float = c_token_to_float(c_token)
        sub = long_df[long_df["C"] == c_token].copy()
        tau_ana = compute_tau_w_analytical(C_float)

        out_png = out_dir / f"tau_w_C{c_token}.png"
        plot_tau_for_C(sub, tau_ana, out_png)
        print("[OK] Figura:", out_png, "| tau_ana =", f"{tau_ana:.4f} Pa")

    print("[OK] Tabela:", table_path)


if __name__ == "__main__":
    main()