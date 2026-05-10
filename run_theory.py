from pathlib import Path
import sys
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# garante ./src no sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str((ROOT / "src").resolve()))

from csv_plotter.theory import u_vertical  # noqa: E402
from csv_plotter.plotting import apply_plot_style  # noqa: E402
from csv_plotter.theory_params import THEORY_PARAMS, set_theory_params  # noqa: E402


def main():
    # --------- CLI ---------
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, required=True, help="Escolha C (ex: 0.3 ou 0.9)")
    args = parser.parse_args()

    # aplica o preset escolhido (atualiza THEORY_PARAMS em-place)
    set_theory_params(args.C)

    apply_plot_style()

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    # ======= PARÂMETROS (já atualizados pelo C) =======
    ty = THEORY_PARAMS["ty"]
    kn = THEORY_PARAMS["Kn"]
    n = THEORY_PARAMS["n"]
    rho = THEORY_PARAMS["rho"]
    h = THEORY_PARAMS["h"]
    th = THEORY_PARAMS["th_deg"] * math.pi / 180  # rad
    dz = THEORY_PARAMS["dz"] if THEORY_PARAMS["dz"] is not None else h / 100000
    adm = THEORY_PARAMS["adm"]
    # ================================================

    uz, u_avg, z0, gamma, eta = u_vertical(ty, kn, n, rho, th, h, dz, adm)

    # =========================================================
    # DADOS BASE
    # =========================================================
    z_m = np.linspace(0, h, len(uz))
    z_cm = z_m * 100.0
    z0_cm = z0 * 100.0

    u_cm_s = np.array(uz, dtype=float) * 100.0
    eta = np.array(eta, dtype=float)

    # =========================================================
    # 1) CSV DO PERFIL DE VELOCIDADE
    # =========================================================
    df_vel = pd.DataFrame(
        {
            "z_cm": z_cm,
            "u_cm_s": u_cm_s,
        }
    )
    csv_vel_path = out_dir / f"theory_zcm_ucms_C{args.C:.1f}.csv"
    df_vel.to_csv(csv_vel_path, index=False)
    print("[OK] CSV velocidade:", csv_vel_path)

    # =========================================================
    # 2) FIGURA DO PERFIL DE VELOCIDADE
    # =========================================================
    plt.figure(figsize=(10, 5))

    plt.plot(u_cm_s, z_cm, color="black", linewidth=3.0)
    plt.axhline(y=z0_cm, color="green", linestyle="--", linewidth=2.0)

    plt.text(
        x=float(np.max(u_cm_s)) * 0.03,
        y=z0_cm + (h * 100.0) * 0.02,
        s=f"$Z_0$ = {z0_cm:.4f} cm",
        color="green",
        fontsize=14,
    )

    yt = np.linspace(0, h * 100.0, 8)
    plt.yticks(yt, labels=np.round(yt, 2))

    plt.ylabel(r"$z(cm)$", fontsize=20)
    plt.xlabel(r"$u(cm/s)$", fontsize=20)

    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=18)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.grid(True, which="major", ls="--", linewidth=1.5, color="0.7")
    plt.tight_layout()

    fig_vel_path = out_dir / f"theory_velocity_C{args.C:.1f}.png"
    plt.savefig(fig_vel_path, dpi=150)
    plt.close()

    print("[OK] Figura velocidade:", fig_vel_path)

    # =========================================================
    # 3) CSV DO PERFIL DE VISCOSIDADE
    # =========================================================
    df_eta = pd.DataFrame(
        {
            "z_cm": z_cm,
            "eta_Pa_s": eta,
        }
    )
    csv_eta_path = out_dir / f"theory_zcm_eta_C{args.C:.1f}.csv"
    df_eta.to_csv(csv_eta_path, index=False)
    print("[OK] CSV viscosidade:", csv_eta_path)

    # =========================================================
    # 4) FIGURA DO PERFIL DE VISCOSIDADE (EIXO X LOG)
    # =========================================================
    # Para escala logarítmica, mantemos apenas valores positivos e finitos
    mask_eta = np.isfinite(eta) & (eta > 0.0)
    eta_plot = eta[mask_eta]
    z_cm_eta = z_cm[mask_eta]

    if len(eta_plot) == 0:
        print("[AVISO] Nenhum valor positivo de viscosidade encontrado para plotar em escala log.")
    else:
        plt.figure(figsize=(10, 5))

        plt.plot(eta_plot, z_cm_eta, color="black", linewidth=3.0)
        plt.axhline(y=z0_cm, color="green", linestyle="--", linewidth=2.0)

        ax = plt.gca()
        ax.set_xscale("log")

        # posição do texto adaptada para escala log
        x_min = np.min(eta_plot)
        x_max = np.max(eta_plot)
        x_text = 10 ** (np.log10(x_min) + 0.05 * (np.log10(x_max) - np.log10(x_min)))

        plt.text(
            x=x_text,
            y=z0_cm + (h * 100.0) * 0.02,
            s=f"$Z_0$ = {z0_cm:.4f} cm",
            color="green",
            fontsize=14,
        )

        yt = np.linspace(0, h * 100.0, 8)
        plt.yticks(yt, labels=np.round(yt, 2))

        plt.ylabel(r"$z(cm)$", fontsize=20)
        plt.xlabel(r"$\eta(Pa \cdot s)$", fontsize=20)

        ax.tick_params(axis="both", labelsize=18)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        plt.grid(True, which="major", ls="--", linewidth=1.5, color="0.7")
        plt.grid(True, which="minor", ls=":", linewidth=1.0, color="0.8")

        plt.tight_layout()

        fig_eta_path = out_dir / f"theory_viscosity_C{args.C:.1f}.png"
        plt.savefig(fig_eta_path, dpi=150)
        plt.close()

        print("[OK] Figura viscosidade:", fig_eta_path)


if __name__ == "__main__":
    main()