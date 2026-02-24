# run_theory.py
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
from csv_plotter.theory_params import THEORY_PARAMS  # noqa: E402


def main():
    apply_plot_style()

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    # ======= PARÂMETROS ÚNICOS (src/csv_plotter/theory_params.py) =======
    ty = THEORY_PARAMS["ty"]
    kn = THEORY_PARAMS["Kn"]
    n = THEORY_PARAMS["n"]
    rho = THEORY_PARAMS["rho"]
    h = THEORY_PARAMS["h"]
    th = THEORY_PARAMS["th_deg"] * math.pi / 180  # rad
    dz = THEORY_PARAMS["dz"] if THEORY_PARAMS["dz"] is not None else h / 100000
    adm = THEORY_PARAMS["adm"]
    # ===================================================================

    uz, u_avg, z0, gamma, eta = u_vertical(ty, kn, n, rho, th, h, dz, adm)

    # ========= 1) PERFIL DE VELOCIDADE =========
    plt.figure(figsize=(10, 5))

    z_m = np.linspace(0, h, len(uz))
    u_cm_s = np.array(uz) * 100.0
    z_cm = z_m * 100.0
    z0_cm = z0 * 100.0

    plt.plot(u_cm_s, z_cm, color="black", linewidth=3.0)
    plt.axhline(y=z0_cm, color="red", linestyle="--", linewidth=2.0)

    plt.text(
        x=float(np.max(u_cm_s)) * 0.03,
        y=z0_cm + (h * 100.0) * 0.02,
        s=f"$Z_0$ = {z0_cm:.2f} cm",
        color="red",
        fontsize=14,
    )

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
    plt.figure(figsize=(10, 5))

    nu_like = np.array(eta) / rho  # igual ao seu colab (eta/rho)

    plt.plot(nu_like, z_cm, color="black", linewidth=3.0)
    plt.axhline(y=z0_cm, color="red", linestyle="--", linewidth=2.0)

    plt.text(
        x=float(np.max(nu_like)) * 0.05,
        y=z0_cm + (h * 100.0) * 0.02,
        s=f"$Z_0$ = {z0_cm:.2f} cm",
        color="red",
        fontsize=14,
    )

    plt.yticks(yt, labels=np.round(yt, 2))

    plt.ylabel(r"$z(cm)$", fontsize=16)
    plt.xlabel(r"$\\eta$ [Pa.s]", fontsize=16)

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