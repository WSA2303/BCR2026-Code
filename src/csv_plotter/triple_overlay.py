from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from csv_plotter.io_csv import load_xy_from_csv
from csv_plotter.plotting import apply_plot_style, style_axes
from csv_plotter.theory import u_vertical


# ======= PARÂMETROS FIXOS DO ANALÍTICO (sempre os mesmos) =======
THEORY_PARAMS = {
    "ty": 39.808,
    "Kn": 0.91,
    "n": 1.0,
    "rho": 1560.0,
    "th_deg": 15.0,
    "h": 0.033482,
    "dz": None,        # se None -> usa h/100000 (igual run_theory)
    "adm": False,
}
# ===============================================================


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


def _tag_from_stem(stem: str) -> str | None:
    s = stem.lower()
    if "ref" in s:
        return "nz"   # η0=f(Nz)
    if "1000" in s:
        return "kn"   # η0=f(Kn)
    return None


def _base_from_stem(stem: str) -> str:
    """
    Deixa robusto com variações tipo:
    09_100_ref; 09_100_ref_algumaCoisa 09_100_1000up etc.
    """
    s = stem.split(";")[0]  # remove tudo após ';' se existir
    low = s.lower()

    # pega o que vem antes de _ref ou _1000
    if "_ref" in low:
        base = s[: low.index("_ref")]
    elif "_1000" in low:
        base = s[: low.index("_1000")]
    else:
        base = s

    return base.rstrip("_- ").strip()


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


def estimate_z0_from_profile(df: pd.DataFrame, xcol: str, ycol: str, tol: float = 0.005) -> float:
    """
    Estima z0 como o primeiro z onde o perfil entra no "platô"
    (u >= (1-tol) u_max) e permanece assim para z acima.
    Retorna em unidades de ycol.
    """
    d = df[[xcol, ycol]].dropna().sort_values(ycol).reset_index(drop=True)
    u = d[xcol].to_numpy()
    z = d[ycol].to_numpy()

    u_max = float(np.max(u))
    thr = (1.0 - tol) * u_max

    suffix_min = np.minimum.accumulate(u[::-1])[::-1]
    idx = np.where(suffix_min >= thr)[0]
    if len(idx) == 0:
        return float(z[np.argmax(u)])
    return float(z[idx[0]])


def plot_triplets_with_computed_theory(
    csv_files: list[Path],
    output_dir: Path,
    xcol: str = "U_0",
    ycol: str = "z",
    dpi: int = 300,
) -> list[Path]:
    """
    Para cada base:
      - se tiver ref: plota (Analítico + ref)
      - se tiver 1000: plota (Analítico + 1000)
      - se tiver ambos: plota (Analítico + ref + 1000)

    Salva outputs/<base>_triple.png (sem sobrescrever).
    """
    apply_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # agrupa ref/1000 por base
    groups: dict[str, dict[str, Path]] = {}
    for p in csv_files:
        tag = _tag_from_stem(p.stem)
        if tag is None:
            continue
        base = _base_from_stem(p.stem)
        groups.setdefault(base, {})[tag] = p

    # se não tem nenhum ref/1000, não há nada a fazer
    if not groups:
        return []

    # analítico é sempre igual -> calcula 1 vez
    df_ana_m, z0_ana_m = compute_theory_df(xcol=xcol, ycol=ycol)

    generated: list[Path] = []

    for base, items in groups.items():
        has_nz = "nz" in items
        has_kn = "kn" in items

        if not (has_nz or has_kn):
            continue  # redundante, mas ok

        df_ana = df_ana_m.copy()
        df_nz = load_xy_from_csv(items["nz"], xcol, ycol) if has_nz else None
        df_kn = load_xy_from_csv(items["kn"], xcol, ycol) if has_kn else None

        # decide unidade olhando z (mais confiável)
        zmax_candidates = [float(df_ana[ycol].max())]
        if df_nz is not None:
            zmax_candidates.append(float(df_nz[ycol].max()))
        if df_kn is not None:
            zmax_candidates.append(float(df_kn[ycol].max()))
        zmax_all = max(zmax_candidates)

        in_meters = zmax_all <= 0.5  # se z pequeno, assume metros -> converte p/ cm

        if in_meters:
            df_ana[xcol] *= 100.0
            df_ana[ycol] *= 100.0
            if df_nz is not None:
                df_nz[xcol] *= 100.0
                df_nz[ycol] *= 100.0
            if df_kn is not None:
                df_kn[xcol] *= 100.0
                df_kn[ycol] *= 100.0

            z0_ana = z0_ana_m * 100.0
            xlabel = r"$u\ [cm/s]$"
            ylabel = r"$z\ [cm]$"
            unit = "cm"
        else:
            z0_ana = z0_ana_m
            xlabel = xcol
            ylabel = ycol
            unit = ""

        # z0 dos dados (se existirem)
        z0_nz = estimate_z0_from_profile(df_nz, xcol, ycol) if df_nz is not None else None
        z0_kn = estimate_z0_from_profile(df_kn, xcol, ycol) if df_kn is not None else None

        # limites
        umax_candidates = [float(df_ana[xcol].max())]
        zmax_candidates = [float(df_ana[ycol].max())]
        if df_nz is not None:
            umax_candidates.append(float(df_nz[xcol].max()))
            zmax_candidates.append(float(df_nz[ycol].max()))
        if df_kn is not None:
            umax_candidates.append(float(df_kn[xcol].max()))
            zmax_candidates.append(float(df_kn[ycol].max()))

        umax = max(umax_candidates)
        zmax = max(zmax_candidates)

        out_path = _next_available_path(output_dir / f"{base}_triple.png")

        fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

        def markevery(df: pd.DataFrame) -> int:
            n = len(df)
            return max(1, n // 40)

        # Analítico (sempre)
        ax.plot(df_ana[xcol], df_ana[ycol], color="black", linewidth=3.2, label="Analítico")

        # ref
        if df_nz is not None:
            ax.plot(
                df_nz[xcol], df_nz[ycol],
                color="red", linewidth=2.0,
                marker="o", markerfacecolor="none", markersize=7,
                markevery=markevery(df_nz),
                label=r"$\eta_0=f(N_z)$",
            )

        # 1000
        if df_kn is not None:
            ax.plot(
                df_kn[xcol], df_kn[ycol],
                color="blue", linewidth=2.0,
                marker="*", markersize=9,
                markevery=markevery(df_kn),
                label=r"$\eta_0=f(K_n)$",
            )

        # linhas z0 (só as que existem)
        ax.axhline(z0_ana, color="red", linestyle="--", linewidth=2.0)

        if z0_kn is not None:
            ax.axhline(z0_kn, color="green", linestyle="--", linewidth=2.0)
        if z0_nz is not None:
            ax.axhline(z0_nz, color="purple", linestyle="--", linewidth=2.0)

        # textos (só os que existem)
        x_left = 0.05 * umax
        x_mid  = 0.33 * umax
        dy = 0.02 * zmax

        if z0_kn is not None:
            ax.text(x_left, z0_kn + dy, rf"$z_0(K_n)$ = {z0_kn:.4f}{unit}", color="green", fontsize=16)

        ax.text(x_mid, z0_ana + dy, rf"$z_0(Analítico)$ = {z0_ana:.4f}{unit}", color="red", fontsize=16)

        if z0_nz is not None:
            ax.text(x_left, z0_nz - 2.5*dy, rf"$z_0(N_z)$ = {z0_nz:.4f}{unit}", color="purple", fontsize=16)

        # eixos / grid / legenda
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, umax * 1.05)
        ax.set_ylim(0.0, zmax * 1.05)
        
        style_axes(
            ax,
            xfmt="%.2f",
            yfmt="%.3f",
            nbins_x=6,
            nbins_y=7,
            minor_grid=False,   # deixa LIMPO igual referência
        )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

        generated.append(out_path)

    return generated

