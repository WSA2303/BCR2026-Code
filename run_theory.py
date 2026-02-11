from pathlib import Path
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# garante ./src no sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str((ROOT / "src").resolve()))

from csv_plotter.theory import u_vertical          # noqa: E402
from csv_plotter.plotting import apply_plot_style  # noqa: E402


def main():
    apply_plot_style()  # <-- AQUI, antes de qualquer plt.figure/plt.plot
    
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    # ======= COPIADO/ADAPTADO DO SEU COLAB =======
    fluid_parameters = {
        "Haustein - Concreto": [39.808, 0.91, 1, 1560, 0.033482]  # [ty, Kn, n, rho, h]
    }

    h = fluid_parameters[list(fluid_parameters.keys())[0]][-1]  # m
    th = 15 * math.pi / 180  # rad
    dz = h / 100000          # m
    adm = False
    # ===========================================

    perfis = {}
    gamma = {}
    eta = {}
    z0 = {}
    u_avg = {}

    for key in fluid_parameters.keys():
        ty, kn, n, rho, _h = fluid_parameters[key]
        perfis[key], u_avg[key], z0[key], gamma[key], eta[key] = u_vertical(
            ty, kn, n, rho, th, h, dz, adm
        )

    # eixo z (em metros) com mesmo tamanho do perfil
    # (no seu colab você usa np.linspace(0, h, len(uz)))
    # aqui vamos manter igual:
    # --------------------------------------------

    # ========= 1) PERFIL DE VELOCIDADE =========
    plt.figure(figsize=(10, 5))

    for fluid, uz in perfis.items():
        z_m = np.linspace(0, h, len(uz))
        u_cm_s = np.array(uz) * 100.0
        z_cm = z_m * 100.0
        z0_cm = z0[fluid] * 100.0

        plt.plot(u_cm_s, z_cm, color="black", linewidth=3.0)
        plt.axhline(y=z0_cm, color="red", linestyle="--", linewidth=2.0)

        plt.text(
            x=max(u_cm_s) * 0.03,
            y=z0_cm + (h * 100.0) * 0.02,
            s=f"$Z_0$ = {z0_cm:.2f} cm",
            color="red",
            fontsize=14
        )

    # ticks em cm como no seu colab
    yt = np.linspace(0, h * 100.0, 8)
    plt.yticks(yt, labels=np.round(yt, 2))

    plt.ylabel(r"$z(cm)$", fontsize=16)
    plt.xlabel(r"$u(cm/s)$", fontsize=16)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.grid(True, which="major", ls="--", linewidth=1.5, color="0.7")
    plt.tight_layout()
    plt.savefig(out_dir / "theory_velocity.png", dpi=150)
    plt.close()

    # ========= 2) PERFIL DE "VISCOSIDADE" =========
    # (replica seu colab: plota (eta/rho) vs z e usa escala log)
    plt.figure(figsize=(10, 5))

    fluid0 = list(fluid_parameters.keys())[0]
    rho0 = fluid_parameters[fluid0][3]

    uz0 = perfis[fluid0]
    z_m = np.linspace(0, h, len(uz0))
    z_cm = z_m * 100.0
    z0_cm = z0[fluid0] * 100.0

    nu_like = np.array(eta[fluid0]) / rho0  # igual seu colab (eta/rho)

    plt.plot(nu_like, z_cm, color="black", linewidth=3.0)
    plt.axhline(y=z0_cm, color="red", linestyle="--", linewidth=2.0)

    plt.text(
        x=max(nu_like) * 0.05,
        y=z0_cm + (h * 100.0) * 0.02,
        s=f"$Z_0$ = {z0_cm:.2f} cm",
        color="red",
        fontsize=14
    )

    yt = np.linspace(0, h * 100.0, 8)
    plt.yticks(yt, labels=np.round(yt, 2))

    plt.ylabel(r"$z(cm)$", fontsize=16)
    plt.xlabel(r"$\eta$ [Pa.s]", fontsize=16)

    plt.xscale("log")
    plt.grid(True, which="both", ls="--", linewidth=1.5, color="0.7")

    plt.tight_layout()
    plt.savefig(out_dir / "theory_viscosity.png", dpi=150)
    plt.close()

    print("[OK] Gerados:")
    print(" -", out_dir / "theory_velocity.png")
    print(" -", out_dir / "theory_viscosity.png")


if __name__ == "__main__":
    main()
