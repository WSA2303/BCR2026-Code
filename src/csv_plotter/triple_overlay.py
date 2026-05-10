# src/csv_plotter/triple_overlay.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from csv_plotter.io_csv import load_xy_from_csv
from csv_plotter.plotting import apply_plot_style, style_axes
from csv_plotter.theory import u_vertical
from csv_plotter.theory_params import THEORY_PARAMS, set_theory_params
from csv_plotter.naming import parse_case_name
from matplotlib.ticker import FormatStrFormatter


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


def _nz_from_base(base: str) -> int:
    """
    Extrai Nz de bases como '03_25', '09_100', etc.
    """
    try:
        return int(base.split("_", 1)[1])
    except (IndexError, ValueError):
        return 10**9


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


# def _save_combined_triplets_by_c(
#     panel_cases: dict[str, list[dict]],
#     output_dir: Path,
#     dpi: int = 300,
# ) -> list[Path]:
#     """
#     Gera figuras combinadas (2x2) diretamente em subplots, uma por valor de C.
#     Sem letras (a), (b), ...
#     Sem título 'C = ...' no topo.
#     Com legenda única para as 4 subfiguras.
#     """
#     generated_panels: list[Path] = []

#     legend_handles = [
#         Line2D([0], [0], color="black", linewidth=3.2, label="Analytical"),
#         Line2D(
#             [0], [0],
#             color="red", linewidth=2.0,
#             marker="o", markerfacecolor="none", markersize=6,
#             label=r"$\eta_0=f(N_z)$",
#         ),
#         Line2D(
#             [0], [0],
#             color="blue", linewidth=2.0,
#             marker="*", markersize=8,
#             label=r"$\eta_0=f(K_n)$",
#         ),
#     ]

#     for c_token, cases in sorted(panel_cases.items()):
#         if not cases:
#             continue

#         cases = sorted(cases, key=lambda d: _nz_from_base(d["base"]))

#         n = len(cases)
#         ncols = 2
#         nrows = int(np.ceil(n / ncols))

#         fig, axes = plt.subplots(
#             nrows=nrows,
#             ncols=ncols,
#             figsize=(10, 8),
#             dpi=dpi,
#         )
#         axes = np.atleast_1d(axes).ravel()

#         group_umax = max(float(case["umax"]) for case in cases)
#         any_in_meters = any(bool(case["in_meters"]) for case in cases)
#         ytop = 3.345 if any_in_meters else 0.03345

#         xlabel = cases[0]["xlabel"]
#         ylabel = cases[0]["ylabel"]
#         unit = cases[0]["unit"]

#         LABEL_FONTSIZE = 18
#         TICK_FONTSIZE = 13
#         LEGEND_FONTSIZE = 18
#         TEXT_FONTSIZE = 12
#         TITLE_FONTSIZE = 20

#         for i, case in enumerate(cases):
#             ax = axes[i]

#             base = case["base"]
#             df_ana = case["df_ana"]
#             df_nz = case["df_nz"]
#             df_kn = case["df_kn"]
#             z0_ana = case["z0_ana"]
#             z0_nz = case["z0_nz"]
#             z0_kn = case["z0_kn"]

#             def markevery(df: pd.DataFrame) -> int:
#                 npts = len(df)
#                 return max(1, npts // 40)

#             ax.plot(df_ana["U_0"], df_ana["z"], color="black", linewidth=3.2)

#             if df_nz is not None:
#                 ax.plot(
#                     df_nz["U_0"], df_nz["z"],
#                     color="red", linewidth=2.0,
#                     marker="o", markerfacecolor="none", markersize=6,
#                     markevery=markevery(df_nz),
#                 )

#             if df_kn is not None:
#                 ax.plot(
#                     df_kn["U_0"], df_kn["z"],
#                     color="blue", linewidth=2.0,
#                     marker="*", markersize=8,
#                     markevery=markevery(df_kn),
#                 )

#             ax.axhline(z0_ana, color="green", linestyle="--", linewidth=1.8)
#             if z0_kn is not None:
#                 ax.axhline(z0_kn, color="blue", linestyle="--", linewidth=1.8)
#             if z0_nz is not None:
#                 ax.axhline(z0_nz, color="red", linestyle="--", linewidth=1.8)

#             x_left = 0.05 * group_umax
#             x_mid = 0.33 * group_umax
#             x_right = 0.58 * group_umax

#             text_box = dict(
#                 facecolor="white",
#                 edgecolor="none",
#                 alpha=0.85,
#                 pad=0.2,
#             )

#             if z0_kn is not None:
#                 ax.annotate(
#                     rf"$z_0(K_n)$ = {z0_kn:.4f}{unit}",
#                     xy=(x_left, z0_kn),
#                     xytext=(0, -8),
#                     textcoords="offset points",
#                     color="blue",
#                     fontsize=TEXT_FONTSIZE,
#                     va="top",
#                     ha="left",
#                     bbox=text_box,
#                 )

#             ax.annotate(
#                 rf"$z_0(Analytical)$ = {z0_ana:.4f}{unit}",
#                 xy=(x_mid, z0_ana),
#                 xytext=(0, 8),
#                 textcoords="offset points",
#                 color="green",
#                 fontsize=TEXT_FONTSIZE,
#                 va="bottom",
#                 ha="left",
#                 bbox=text_box,
#             )

#             if z0_nz is not None:
#                 ax.annotate(
#                     rf"$z_0(N_z)$ = {z0_nz:.4f}{unit}",
#                     xy=(x_right, z0_nz),
#                     xytext=(0, 8),
#                     textcoords="offset points",
#                     color="red",
#                     fontsize=TEXT_FONTSIZE,
#                     va="bottom",
#                     ha="left",
#                     bbox=text_box,
#                 )

#             ax.set_xlim(0.0, group_umax * 1.05)
#             ax.set_ylim(0.0, ytop)

#             style_axes(
#                 ax,
#                 xfmt="%.2f",
#                 yfmt="%.3f",
#                 nbins_x=6,
#                 nbins_y=7,
#                 minor_grid=False,
#             )
#             ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

#             nz = _nz_from_base(base)
#             ax.set_title(rf"$N_z = {nz}$", fontsize=TITLE_FONTSIZE, pad=6)

#             row = i // ncols
#             col = i % ncols

#             if row == nrows - 1:
#                 ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
#             else:
#                 ax.set_xlabel("")
#                 ax.tick_params(axis="x", labelbottom=False)

#             if col == 0:
#                 ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
#             else:
#                 ax.set_ylabel("")
#                 ax.tick_params(axis="y", labelleft=False)

#         for j in range(n, len(axes)):
#             axes[j].axis("off")

#         fig.legend(
#             handles=legend_handles,
#             loc="upper center",
#             bbox_to_anchor=(0.5, 0.975),
#             ncol=3,
#             frameon=False,
#             fontsize=LEGEND_FONTSIZE,
#         )

#         fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

#         out_path = _next_available_path(output_dir / f"C{c_token}_combined_triple.png")
#         fig.savefig(out_path, bbox_inches="tight")
#         plt.close(fig)

#         generated_panels.append(out_path)

#     return generated_panels


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

    Além das figuras individuais, gera também:
      - C03_combined_triple.png
      - C09_combined_triple.png
    """
    apply_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, dict[str, Path]] = {}
    for p in csv_files:
        info = parse_case_name(p.stem)

        if info is None:
            continue

    # Esta função é só para velocidade: U_0 x z
        if info.kind != "velocity":
            continue

        groups.setdefault(info.base, {})[info.method] = p

    if not groups:
        return []

    data_dir = output_dir.parent / "data"
    manual_path = data_dir / "z0_manual.csv"
    manual_z0 = load_manual_z0(manual_path)

    theory_cache: dict[str, tuple[pd.DataFrame, float]] = {}

    def get_theory_for_base(base: str) -> tuple[pd.DataFrame, float]:
        c_token = base.split("_", 1)[0].strip()
        if c_token in theory_cache:
            return theory_cache[c_token]

        if base.startswith("03_"):
            set_theory_params(0.3)
        elif base.startswith("09_"):
            set_theory_params(0.9)
        else:
            raise ValueError(f"Base sem C reconhecido: {base} (esperado começar com 03_ ou 09_)")

        df_ana_m, z0_ana_m = compute_theory_df(xcol=xcol, ycol=ycol)
        theory_cache[c_token] = (df_ana_m, z0_ana_m)
        return theory_cache[c_token]

    generated: list[Path] = []
    panel_cases: dict[str, list[dict]] = {}

    for base, items in groups.items():
        has_nz = "nz" in items
        has_kn = "kn" in items
        if not (has_nz or has_kn):
            continue

        df_ana_m, z0_ana_m = get_theory_for_base(base)

        df_ana = df_ana_m.copy()
        df_nz = load_xy_from_csv(items["nz"], xcol, ycol) if has_nz else None
        df_kn = load_xy_from_csv(items["kn"], xcol, ycol) if has_kn else None

        zmax_candidates = [float(df_ana[ycol].max())]
        if df_nz is not None:
            zmax_candidates.append(float(df_nz[ycol].max()))
        if df_kn is not None:
            zmax_candidates.append(float(df_kn[ycol].max()))
        zmax_all = max(zmax_candidates)

        in_meters = zmax_all <= 0.5

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

            z0_nz = manual_z0.get(base, {}).get("nz", None)
            z0_kn = manual_z0.get(base, {}).get("kn", None)

        else:
            z0_ana = z0_ana_m
            xlabel = xcol
            ylabel = ycol
            unit = ""

            z0_nz = manual_z0.get(base, {}).get("nz", None)
            z0_kn = manual_z0.get(base, {}).get("kn", None)
            if z0_nz is not None:
                z0_nz = z0_nz / 100.0
            if z0_kn is not None:
                z0_kn = z0_kn / 100.0

        if df_nz is not None and z0_nz is None:
            print(f"[WARN] Sem z0_ref_cm para base={base} em data/z0_manual.csv")
        if df_kn is not None and z0_kn is None:
            print(f"[WARN] Sem z0_1000_cm para base={base} em data/z0_manual.csv")

        umax_candidates = [float(df_ana[xcol].max())]
        if df_nz is not None:
            umax_candidates.append(float(df_nz[xcol].max()))
        if df_kn is not None:
            umax_candidates.append(float(df_kn[xcol].max()))
        umax = max(umax_candidates)

        out_path = _next_available_path(output_dir / f"{base}_triple.png")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

        LABEL_FONTSIZE = 20
        TICK_FONTSIZE = 14
        LEGEND_FONTSIZE = 20
        TEXT_FONTSIZE = 16

        def markevery(df: pd.DataFrame) -> int:
            npts = len(df)
            return max(1, npts // 40)

        ax.plot(df_ana[xcol], df_ana[ycol], color="black", linewidth=3.2, label="Analytical")

        if df_nz is not None:
            ax.plot(
                df_nz[xcol], df_nz[ycol],
                color="red", linewidth=2.0,
                marker="o", markerfacecolor="none", markersize=7,
                markevery=markevery(df_nz),
                label=r"$\eta_0=f(N_z)$",
            )

        if df_kn is not None:
            ax.plot(
                df_kn[xcol], df_kn[ycol],
                color="blue", linewidth=2.0,
                marker="*", markersize=9,
                markevery=markevery(df_kn),
                label=r"$\eta_0=f(K_n)$",
            )

        ax.axhline(z0_ana, color="green", linestyle="--", linewidth=2.0)
        if z0_kn is not None:
            ax.axhline(z0_kn, color="blue", linestyle="--", linewidth=2.0)
        if z0_nz is not None:
            ax.axhline(z0_nz, color="red", linestyle="--", linewidth=2.0)

        x_left = 0.05 * umax
        x_mid = 0.33 * umax
        x_right = 0.58 * umax

        text_box = dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
            pad=0.2,
        )

        if z0_kn is not None:
            ax.annotate(
                rf"$z_0(K_n)$ = {z0_kn:.4f}{unit}",
                xy=(x_left, z0_kn),
                xytext=(0, -10),
                textcoords="offset points",
                color="blue",
                fontsize=TEXT_FONTSIZE,
                va="top",
                ha="left",
                bbox=text_box,
            )

        ax.annotate(
            rf"$z_0(Analytical)$ = {z0_ana:.4f}{unit}",
            xy=(x_mid, z0_ana),
            xytext=(0, 10),
            textcoords="offset points",
            color="green",
            fontsize=TEXT_FONTSIZE,
            va="bottom",
            ha="left",
            bbox=text_box,
        )

        if z0_nz is not None:
            ax.annotate(
                rf"$z_0(N_z)$ = {z0_nz:.4f}{unit}",
                xy=(x_right, z0_nz),
                xytext=(0, 10),
                textcoords="offset points",
                color="red",
                fontsize=TEXT_FONTSIZE,
                va="bottom",
                ha="left",
                bbox=text_box,
            )

        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
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

        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.22),
            ncol=3,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
        )

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

        generated.append(out_path)

        c_token = base.split("_", 1)[0].strip()
        panel_cases.setdefault(c_token, []).append(
            {
                "base": base,
                "df_ana": df_ana.copy(),
                "df_nz": None if df_nz is None else df_nz.copy(),
                "df_kn": None if df_kn is None else df_kn.copy(),
                "z0_ana": z0_ana,
                "z0_nz": z0_nz,
                "z0_kn": z0_kn,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "unit": unit,
                "in_meters": in_meters,
                "umax": umax,
            }
        )

    # panel_paths = _save_combined_triplets_by_c(
    #     panel_cases=panel_cases,
    #     output_dir=output_dir,
    #     dpi=dpi,
    # )
    # generated.extend(panel_paths)

    # Geração das figuras combinadas desativada.
    # Para ativar novamente, é necessário descomentar a função
    # _save_combined_triplets_by_c definida acima.

    return generated

def compute_theory_eta_df(xcol: str = "nu1", ycol: str = "z") -> tuple[pd.DataFrame, float]:
    """
    Gera o perfil teórico de viscosidade SEM salvar CSV.

    Retorna:
        df: DataFrame com colunas [nu1, z]
        z0: altura de plug em metros
    """
    ty = THEORY_PARAMS["ty"]
    Kn = THEORY_PARAMS["Kn"]
    n = THEORY_PARAMS["n"]
    rho = THEORY_PARAMS["rho"]
    th_rad = np.deg2rad(THEORY_PARAMS["th_deg"])
    h = THEORY_PARAMS["h"]
    dz = THEORY_PARAMS["dz"] if THEORY_PARAMS["dz"] is not None else h / 100000
    adm = THEORY_PARAMS["adm"]

    _vel, _uavg, z0, _gamma, eta = u_vertical(
        ty, Kn, n, rho, th_rad, h, dz, adm
    )

    z_m = np.linspace(0.0, h, len(eta))

    df = pd.DataFrame(
        {
            xcol: np.array(eta, dtype=float),
            ycol: z_m,
        }
    )

    return df, float(z0)


def plot_eta_triplets_with_computed_theory(
    csv_files: list[Path],
    output_dir: Path,
    xcol: str = "nu1",
    ycol: str = "z",
    dpi: int = 300,
) -> list[Path]:
    """
    Gera gráficos triple de viscosidade para arquivos do tipo:

        03_25_eta_ref.csv
        03_25_eta_1000.csv
        09_100_eta_ref.csv
        09_100_eta_1000.csv

    Plota:
        nu1 x z

    Sempre usa escala logarítmica no eixo x.
    """
    apply_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, dict[str, Path]] = {}

    for p in csv_files:
        info = parse_case_name(p.stem)

        if info is None:
            continue

        # Esta função é apenas para arquivos de viscosidade
        if info.kind != "eta":
            continue

        groups.setdefault(info.base, {})[info.method] = p

    if not groups:
        return []

    data_dir = output_dir.parent / "data"
    manual_path = data_dir / "z0_manual.csv"
    manual_z0 = load_manual_z0(manual_path)

    theory_cache: dict[str, tuple[pd.DataFrame, float]] = {}

    def get_theory_for_base(base: str) -> tuple[pd.DataFrame, float]:
        c_token = base.split("_", 1)[0].strip()

        if c_token in theory_cache:
            return theory_cache[c_token]

        if base.startswith("03_"):
            set_theory_params(0.3)
        elif base.startswith("09_"):
            set_theory_params(0.9)
        else:
            raise ValueError(
                f"Base sem C reconhecido: {base} "
                f"(esperado começar com 03_ ou 09_)"
            )

        df_ana_m, z0_ana_m = compute_theory_eta_df(xcol=xcol, ycol=ycol)
        theory_cache[c_token] = (df_ana_m, z0_ana_m)

        return theory_cache[c_token]

    generated: list[Path] = []

    for base, items in groups.items():
        has_nz = "nz" in items
        has_kn = "kn" in items

        if not (has_nz or has_kn):
            continue

        df_ana_m, z0_ana_m = get_theory_for_base(base)

        df_ana = df_ana_m.copy()
        df_nz = load_xy_from_csv(items["nz"], xcol, ycol) if has_nz else None
        df_kn = load_xy_from_csv(items["kn"], xcol, ycol) if has_kn else None

        # Detecta unidade de z
        zmax_candidates = [float(df_ana[ycol].max())]

        if df_nz is not None:
            zmax_candidates.append(float(df_nz[ycol].max()))

        if df_kn is not None:
            zmax_candidates.append(float(df_kn[ycol].max()))

        zmax_all = max(zmax_candidates)
        in_meters = zmax_all <= 0.5

        if in_meters:
            df_ana[ycol] *= 100.0

            if df_nz is not None:
                df_nz[ycol] *= 100.0

            if df_kn is not None:
                df_kn[ycol] *= 100.0

            z0_ana = z0_ana_m * 100.0
            ylabel = r"$z\ [cm]$"
            unit = "cm"
            ytop = 3.345

            z0_nz = manual_z0.get(base, {}).get("nz", None)
            z0_kn = manual_z0.get(base, {}).get("kn", None)

        else:
            z0_ana = z0_ana_m
            ylabel = r"$z$"
            unit = ""
            ytop = 0.03345

            z0_nz = manual_z0.get(base, {}).get("nz", None)
            z0_kn = manual_z0.get(base, {}).get("kn", None)

            if z0_nz is not None:
                z0_nz = z0_nz / 100.0

            if z0_kn is not None:
                z0_kn = z0_kn / 100.0

        if df_nz is not None and z0_nz is None:
            print(f"[WARN] Sem z0_ref_cm para base={base} em data/z0_manual.csv")

        if df_kn is not None and z0_kn is None:
            print(f"[WARN] Sem z0_1000_cm para base={base} em data/z0_manual.csv")

        # Para escala log, remove valores nulos, negativos ou não finitos
        def clean_for_log(df: pd.DataFrame | None) -> pd.DataFrame | None:
            if df is None:
                return None

            df2 = df.copy()
            mask = (
                np.isfinite(df2[xcol])
                & np.isfinite(df2[ycol])
                & (df2[xcol] > 0.0)
            )

            return df2.loc[mask].sort_values(ycol)

        df_ana = clean_for_log(df_ana)
        df_nz = clean_for_log(df_nz)
        df_kn = clean_for_log(df_kn)

        x_candidates_min = []
        x_candidates_max = []

        for df in [df_ana, df_nz, df_kn]:
            if df is not None and len(df) > 0:
                x_candidates_min.append(float(df[xcol].min()))
                x_candidates_max.append(float(df[xcol].max()))

        if not x_candidates_min:
            print(f"[SKIP] {base}: nenhum valor positivo de {xcol} para escala log.")
            continue

        xmin = min(x_candidates_min)
        xmax = max(x_candidates_max)

        out_path = _next_available_path(output_dir / f"{base}_eta_triple.png")

        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

        LABEL_FONTSIZE = 20
        TICK_FONTSIZE = 14
        LEGEND_FONTSIZE = 20
        TEXT_FONTSIZE = 16

        def markevery(df: pd.DataFrame) -> int:
            npts = len(df)
            return max(1, npts // 40)

        if df_ana is not None and len(df_ana) > 0:
            ax.plot(
                df_ana[xcol],
                df_ana[ycol],
                color="black",
                linewidth=3.2,
                label="Analytical",
            )

        if df_nz is not None and len(df_nz) > 0:
            ax.plot(
                df_nz[xcol],
                df_nz[ycol],
                color="red",
                linewidth=2.0,
                marker="o",
                markerfacecolor="none",
                markersize=7,
                markevery=markevery(df_nz),
                label=r"$\eta_0=f(N_z)$",
            )

        if df_kn is not None and len(df_kn) > 0:
            ax.plot(
                df_kn[xcol],
                df_kn[ycol],
                color="blue",
                linewidth=2.0,
                marker="*",
                markersize=9,
                markevery=markevery(df_kn),
                label=r"$\eta_0=f(K_n)$",
            )

        # Linhas de z0
        ax.axhline(z0_ana, color="green", linestyle="--", linewidth=2.0)

        if z0_kn is not None:
            ax.axhline(z0_kn, color="blue", linestyle="--", linewidth=2.0)

        if z0_nz is not None:
            ax.axhline(z0_nz, color="red", linestyle="--", linewidth=2.0)

        # Eixo x logarítmico
        ax.set_xscale("log")

        # Posições de texto adequadas para eixo log
        log_xmin = np.log10(xmin)
        log_xmax = np.log10(xmax)

        def xlog_pos(frac: float) -> float:
            return 10 ** (log_xmin + frac * (log_xmax - log_xmin))

        x_left = xlog_pos(0.05)
        x_mid = xlog_pos(0.33)
        x_right = xlog_pos(0.58)

        text_box = dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
            pad=0.2,
        )

        if z0_kn is not None:
            ax.annotate(
                rf"$z_0(K_n)$ = {z0_kn:.4f}{unit}",
                xy=(x_left, z0_kn),
                xytext=(0, -10),
                textcoords="offset points",
                color="blue",
                fontsize=TEXT_FONTSIZE,
                va="top",
                ha="left",
                bbox=text_box,
            )

        ax.annotate(
            rf"$z_0(Analytical)$ = {z0_ana:.4f}{unit}",
            xy=(x_mid, z0_ana),
            xytext=(0, 10),
            textcoords="offset points",
            color="green",
            fontsize=TEXT_FONTSIZE,
            va="bottom",
            ha="left",
            bbox=text_box,
        )

        if z0_nz is not None:
            ax.annotate(
                rf"$z_0(N_z)$ = {z0_nz:.4f}{unit}",
                xy=(x_right, z0_nz),
                xytext=(0, 10),
                textcoords="offset points",
                color="red",
                fontsize=TEXT_FONTSIZE,
                va="bottom",
                ha="left",
                bbox=text_box,
            )

        ax.set_xlabel(r"$\nu_1$", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)

        ax.set_xlim(xmin * 0.8, xmax * 1.25)
        ax.set_ylim(0.0, ytop)

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.tick_params(which="major", length=7, width=1.2, direction="out")
        ax.tick_params(which="minor", length=4, width=1.0, direction="out")

        ax.set_axisbelow(True)
        ax.grid(True, which="major", linestyle="--", linewidth=1.2, color="0.70", alpha=0.9)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.7, color="0.85", alpha=0.8)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.22),
            ncol=3,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
        )

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

        generated.append(out_path)

    return generated