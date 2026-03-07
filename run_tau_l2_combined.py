from __future__ import annotations

from pathlib import Path
import sys
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str((ROOT / "src").resolve()))

from csv_plotter.theory_params import THEORY_PARAMS, set_theory_params  # noqa: E402


def style_like_example():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "axes.linewidth": 0.8,
    })


def normalize_c_token(value) -> str:
    t = str(value).strip().replace(",", ".")
    if t in {"03", "3", "3.0", "0.3"}:
        return "03"
    if t in {"09", "9", "9.0", "0.9"}:
        return "09"
    raise ValueError(f"C inválido: {value!r}")


def c_token_to_float(c_token: str) -> float:
    c_token = normalize_c_token(c_token)
    return 0.3 if c_token == "03" else 0.9


def compute_tau_w_analytical(C_float: float) -> float:
    """
    tau_w = rho * g * h * sin(th)
    """
    set_theory_params(C_float)
    rho = float(THEORY_PARAMS["rho"])
    h = float(THEORY_PARAMS["h"])
    th_deg = float(THEORY_PARAMS["th_deg"])
    g = 9.81
    return rho * g * h * math.sin(math.radians(th_deg))


def load_tau_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Não encontrei: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"C", "method", "mult", "tau_w"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} faltando colunas {sorted(missing)}")

    df["C"] = df["C"].map(normalize_c_token)
    df["method"] = df["method"].astype(str).str.strip().str.lower()
    df["mult"] = pd.to_numeric(df["mult"], errors="coerce")
    df["tau_w"] = pd.to_numeric(df["tau_w"], errors="coerce")
    df = df.dropna(subset=["C", "method", "mult", "tau_w"]).copy()
    return df


def load_l2_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Não encontrei: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"C", "method", "mult", "L2_percent"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} faltando colunas {sorted(missing)}")

    df["C"] = df["C"].map(normalize_c_token)
    df["method"] = df["method"].astype(str).str.strip().str.lower()
    df["mult"] = pd.to_numeric(df["mult"], errors="coerce")
    df["L2_percent"] = pd.to_numeric(df["L2_percent"], errors="coerce")
    df = df.dropna(subset=["C", "method", "mult", "L2_percent"]).copy()
    return df


# def _legend_handles():
#     return [
#         Line2D(
#             [0], [0],
#             color="black",
#             linestyle="--",
#             linewidth=1.4,
#             label="Analytical",
#         ),
#         Line2D(
#             [0], [0],
#             color="red",
#             marker="o",
#             markerfacecolor="none",
#             linestyle="None",
#             markersize=5,
#             label=r"$\eta_0=f(N_z)$",
#         ),
#         Line2D(
#             [0], [0],
#             color="blue",
#             marker="*",
#             linestyle="None",
#             markersize=8,
#             label=r"$\eta_0=f(K_n)$",
#         ),
#     ]


def plot_combined_for_C(
    c_token: str,
    tau_sub: pd.DataFrame,
    l2_sub: pd.DataFrame,
    out_path: Path,
    dpi: int = 300,
) -> None:
    style_like_example()

    fig, (ax_tau, ax_l2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 3.8),
        dpi=dpi,
    )

    xticks = [1, 2, 4, 8]
    xticklabels = [r"$N_z$", r"$2N_z$", r"$4N_z$", r"$8N_z$"]

    styles = {
        "nz": dict(color="red", marker="o", linestyle="None", markersize=5),
        "kn": dict(color="blue", marker="*", linestyle="None", markersize=8),
    }

    LABEL_FONTSIZE = 12
    TICK_FONTSIZE = 10
    TEXT_FONTSIZE = 10

    # -------------------------
    # Subfigura tau_w
    # -------------------------
    tau_ana = compute_tau_w_analytical(c_token_to_float(c_token))

    for method in ["nz", "kn"]:
        s = tau_sub[tau_sub["method"] == method].sort_values("mult")
        if s.empty:
            continue
        ax_tau.plot(s["mult"], s["tau_w"], **styles[method])

    ax_tau.axhline(tau_ana, color="black", linestyle="--", linewidth=1.4)

    ax_tau.text(
        4.1,
        tau_ana + 0.12,
        rf"$\tau_w={tau_ana:.2f}\ \mathrm{{Pa}}$",
        fontsize=TEXT_FONTSIZE,
    )

    ax_tau.set_xticks(xticks)
    ax_tau.set_xticklabels(xticklabels)
    ax_tau.set_xlabel("Number of volumes", fontsize=LABEL_FONTSIZE)
    ax_tau.set_ylabel(r"$\tau_w\ [Pa]$", fontsize=LABEL_FONTSIZE)
    ax_tau.set_xlim(0.8, 8.2)
    ax_tau.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.75")
    ax_tau.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    # -------------------------
    # Subfigura L2
    # -------------------------
    for method in ["nz", "kn"]:
        s = l2_sub[l2_sub["method"] == method].sort_values("mult")
        if s.empty:
            continue
        ax_l2.plot(s["mult"], s["L2_percent"], **styles[method])

    ax_l2.set_xticks(xticks)
    ax_l2.set_xticklabels(xticklabels)
    ax_l2.set_xlabel("Number of volumes", fontsize=LABEL_FONTSIZE)
    ax_l2.set_ylabel(r"$L_2$ norm (\%) — Velocity profile", fontsize=LABEL_FONTSIZE)
    ax_l2.set_xlim(0.8, 8.2)
    ax_l2.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.75")
    ax_l2.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.20,
        top=0.96,
        wspace=0.22,
    )

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    tau_table_path = out_dir / "tau_w_table.csv"
    l2_table_path = out_dir / "l2_velocity_table.csv"

    tau_df = load_tau_table(tau_table_path)
    l2_df = load_l2_table(l2_table_path)

    common_cs = sorted(
        set(tau_df["C"].unique()).intersection(set(l2_df["C"].unique())),
        key=c_token_to_float,
    )

    if not common_cs:
        raise SystemExit("Não encontrei valores de C em comum entre tau_w_table.csv e l2_velocity_table.csv.")

    for c_token in common_cs:
        tau_sub = tau_df[tau_df["C"] == c_token].copy()
        l2_sub = l2_df[l2_df["C"] == c_token].copy()

        out_path = out_dir / f"C{c_token}_combined_tau_l2.png"
        plot_combined_for_C(
            c_token=c_token,
            tau_sub=tau_sub,
            l2_sub=l2_sub,
            out_path=out_path,
            dpi=300,
        )
        print("[OK] Figura combinada:", out_path)

    print("[OK] Finalizado.")


if __name__ == "__main__":
    main()