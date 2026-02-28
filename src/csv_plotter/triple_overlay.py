# src/csv_plotter/triple_overlay.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from csv_plotter.io_csv import load_xy_from_csv
from csv_plotter.plotting import apply_plot_style, style_axes
from csv_plotter.theory import u_vertical
from csv_plotter.theory_params import THEORY_PARAMS
from csv_plotter.naming import parse_case_name


def _next_available_path(out_path: Path) -> Path:
    if not out_path.exists():
        return out_path
    parent = out_path.parent
    stem = out_path.stem
    suffix = out_path.suffix
    i = 1
    while True:
        cand = parent / f"{stem} ({i}){suffix}"
        if not cand.exists():
            return cand
        i += 1


def compute_theory_df(xcol: str = "U_0", ycol: str = "z") -> tuple[pd.DataFrame, float]:
    """
    Gera o perfil analítico SEM CSV.
    Retorna (df, z0_em_metros).
    """
    ty = THEORY_PARAMS["ty"]
    Kn = THEORY_PARAMS["Kn"]
    n = THEORY_PARAMS["n"]
    rho = THEORY_PARAMS["rho"]
    th_rad = np.deg2rad(THEORY_PARAMS["th_deg"])
    h = THEORY_PARAMS["h"]
    dz = THEORY_PARAMS["dz"] if THEORY_PARAMS["dz"] is not None else h / 100000
    adm = THEORY_PARAMS["adm"]

    vel, _uavg, z0, _gamma, _eta = u_vertical(ty, Kn, n, rho, th_rad, h, dz, adm)

    z_m = np.linspace(0.0, h, len(vel))
    df = pd.DataFrame({xcol: np.array(vel, dtype=float), ycol: z_m})
    return df, float(z0)


def load_manual_z0(path: Path) -> dict[str, dict[str, float]]:
    """
    Lê data/z0_manual.csv e retorna:
      {
        "09_100": {"nz": 2.6067, "kn": 2.4552},
        ...
      }
    Valores em cm.
    """
    if not path.exists():
        return {}

    df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    if "base" not in df.columns:
        raise ValueError(f"z0_manual.csv sem coluna 'base'. Colunas: {list(df.columns)}")

    # normaliza base e converte valores
    df["base"] = df["base"].astype(str).str.strip()

    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        base = str(row.get("base", "")).strip()
        if not base or base.lower() == "nan":
            continue

        d: dict[str, float] = {}

        z_ref = pd.to_numeric(row.get("z0_ref_cm", None), errors="coerce")
        if pd.notna(z_ref):
            d["nz"] = float(z_ref)

        z_1000 = pd.to_numeric(row.get("z0_1000_cm", None), errors="coerce")
        if pd.notna(z_1000):
            d["kn"] = float(z_1000)

        if d:
            out[base] = d

    return out


def plot_triplets_with_computed_theory(
    csv_files: list[Path],
    output_dir: Path,
    xcol: str = "U_0",
    ycol: str = "z",
    dpi: int = 300,
) -> list[Path]:
    """
    Para cada base (ex: 09_100):
      - se tiver ref: plota (Analítico + ref)
      - se tiver 1000: plota (Analítico + 1000)
      - se tiver ambos: plota (Analítico + ref + 1000)

    z0_ref e z0_1000 vêm de data/z0_manual.csv (manual).
    """
    apply_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # agrupa ref/1000 por base usando parse_case_name
    groups: dict[str, dict[str, Path]] = {}
    for p in csv_files:
        info = parse_case_name(p.stem)
        if info is None:
            continue
        # info.base = "09_100", info.method = "nz" (ref) ou "kn" (1000)
        groups.setdefault(info.base, {})[info.method] = p

    if not groups:
        return []

    # z0 manuais (cm)
    data_dir = output_dir.parent / "data"
    manual_path = data_dir / "z0_manual.csv"
    manual_z0 = load_manual_z0(manual_path)

    # analítico (1 vez)
    df_ana_m, z0_ana_m = compute_theory_df(xcol=xcol, ycol=ycol)

    generated: list[Path] = []

    for base, items in groups.items():
        has_nz = "nz" in items
        has_kn = "kn" in items
        if not (has_nz or has_kn):
            continue

        df_ana = df_ana_m.copy()
        df_nz = load_xy_from_csv(items["nz"], xcol, ycol) if has_nz else None
        df_kn = load_xy_from_csv(items["kn"], xcol, ycol) if has_kn else None

        # decide unidade olhando z
        zmax_candidates = [float(df_ana[ycol].max())]
        if df_nz is not None:
            zmax_candidates.append(float(df_nz[ycol].max()))
        if df_kn is not None:
            zmax_candidates.append(float(df_kn[ycol].max()))
        zmax_all = max(zmax_candidates)

        in_meters = zmax_all <= 0.5  # se z pequeno => veio em metros

        if in_meters:
            # converte tudo pra cm e cm/s
            df_ana[xcol] *= 100.0
            df_ana[ycol] *= 100.0
            if df_nz is not None:
                df_nz[xcol] *= 100.0
                df_nz[ycol] *= 100.0
            if df_kn is not None:
                df_kn[xcol] *= 100.0
                df_kn[ycol] *= 100.0

            z0_ana = z0_ana_m * 100.0  # cm
            xlabel = r"$u\ [cm/s]$"
            ylabel = r"$z\ [cm]$"
            unit = "cm"

            z0_nz = manual_z0.get(base, {}).get("nz", None)
            z0_kn = manual_z0.get(base, {}).get("kn", None)

        else:
            # metros (raro)
            z0_ana = z0_ana_m
            xlabel = xcol
            ylabel = ycol
            unit = ""

            # manuais em cm -> m
            z0_nz = manual_z0.get(base, {}).get("nz", None)
            z0_kn = manual_z0.get(base, {}).get("kn", None)
            if z0_nz is not None:
                z0_nz = z0_nz / 100.0
            if z0_kn is not None:
                z0_kn = z0_kn / 100.0

        # WARNs
        if df_nz is not None and z0_nz is None:
            print(f"[WARN] Sem z0_ref_cm para base={base} em data/z0_manual.csv")
        if df_kn is not None and z0_kn is None:
            print(f"[WARN] Sem z0_1000_cm para base={base} em data/z0_manual.csv")

        # limites
        umax_candidates = [float(df_ana[xcol].max())]
        if df_nz is not None:
            umax_candidates.append(float(df_nz[xcol].max()))
        if df_kn is not None:
            umax_candidates.append(float(df_kn[xcol].max()))
        umax = max(umax_candidates)

        out_path = _next_available_path(output_dir / f"{base}_triple.png")

        fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

        def markevery(df: pd.DataFrame) -> int:
            n = len(df)
            return max(1, n // 40)

        # Analítico
        ax.plot(df_ana[xcol], df_ana[ycol], color="black", linewidth=3.2, label="Analítico")

        # ref (Nz)
        if df_nz is not None:
            ax.plot(
                df_nz[xcol], df_nz[ycol],
                color="red", linewidth=2.0,
                marker="o", markerfacecolor="none", markersize=7,
                markevery=markevery(df_nz),
                label=r"$\eta_0=f(N_z)$",
            )

        # 1000 (Kn)
        if df_kn is not None:
            ax.plot(
                df_kn[xcol], df_kn[ycol],
                color="blue", linewidth=2.0,
                marker="*", markersize=9,
                markevery=markevery(df_kn),
                label=r"$\eta_0=f(K_n)$",
            )

        # linhas z0
        ax.axhline(z0_ana, color="red", linestyle="--", linewidth=2.0)
        if z0_kn is not None:
            ax.axhline(z0_kn, color="green", linestyle="--", linewidth=2.0)
        if z0_nz is not None:
            ax.axhline(z0_nz, color="purple", linestyle="--", linewidth=2.0)

        # textos (posicionamento pedido)
        x_left = 0.05 * umax
        x_mid = 0.33 * umax
        dy = 0.01 * zmax   # um pouco maior pra não encostar

        # verde: abaixo da linha verde
        if z0_kn is not None:
            ax.text(
                x_left,
                z0_kn - dy,
                rf"$z_0(K_n)$ = {z0_kn:.4f}{unit}",
                color="green",
                fontsize=16,
                va="top",   # ancora pelo topo do texto (fica abaixo da linha)
            )

        # vermelho (analítico): mantém acima
        ax.text(
            x_mid,
            z0_ana + dy,
            rf"$z_0(Analítico)$ = {z0_ana:.4f}{unit}",
            color="red",
            fontsize=16,
            va="bottom",
        )

        # roxo: acima da linha roxa
        if z0_nz is not None:
            ax.text(
                x_left,
                z0_nz + dy,
                rf"$z_0(N_z)$ = {z0_nz:.4f}{unit}",
                color="purple",
                fontsize=16,
                va="bottom",  # ancora pela base do texto (fica acima da linha)
            )

        # eixos
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, umax * 1.05)
        if in_meters:          
            ax.set_ylim(0.0, 3.345)
        else:                   
            ax.set_ylim(0.0, 0.03345)

        style_axes(
            ax,
            xfmt="%.2f",
            yfmt="%.3f",
            nbins_x=6,
            nbins_y=7,
            minor_grid=False,
        )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

        generated.append(out_path)

    return generated