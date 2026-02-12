from __future__ import annotations

import argparse
import math
from pathlib import Path

from csv_plotter.io_csv import list_csv_files, load_xy_from_csv
from csv_plotter.plotting import plot_xy
from csv_plotter.triple_overlay import plot_triplets_with_computed_theory


def project_root() -> Path:
    # .../SEU_PROJETO/src/csv_plotter/cli.py -> parents[2] = SEU_PROJETO
    return Path(__file__).resolve().parents[2]


def resolve_from_root(p: str | None, default_rel: str) -> Path:
    root = project_root()
    if p is None:
        return (root / default_rel).resolve()
    path = Path(p).expanduser()
    return (path if path.is_absolute() else (root / path)).resolve()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gera gráficos a partir de CSVs.")
    p.add_argument("--input_dir", default=None, help="Pasta dos CSVs (default: ./data)")
    p.add_argument("--output_dir", default=None, help="Pasta de saída (default: ./outputs)")
    p.add_argument("--xcol", default="U_0", help="Coluna do eixo x (default: U_0)")
    p.add_argument("--ycol", default="z", help="Coluna do eixo y (default: z)")
    p.add_argument("--swap", action="store_true", help="Inverte os eixos")
    p.add_argument("--dpi", type=int, default=300, help="DPI do PNG (default: 300)")
    p.add_argument("--zmax", type=float, default=0.036, help="Limite superior do eixo z (default: 0.036)")

    return p

def next_available_path(out_path: Path) -> Path:
    """Se out_path já existir, retorna out_path com ' (1)', ' (2)', ..."""
    if not out_path.exists():
        return out_path

    parent = out_path.parent
    stem = out_path.stem
    suffix = out_path.suffix

    i = 1
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def label_from_filename(stem: str) -> str | None:
    name = stem.lower()
    if "ref" in name:
        return r"$\eta_0 = f(N_z)$"
    if "1000" in name:
        return r"$\eta_0 = f(K_n)$"
    return None


def maybe_convert_units_to_cm(df, xcol: str, ycol: str, zmax_m: float):
    """
    Se detectar que z está em metros (valores pequenos), converte z -> cm.
    Se detectar que U_0 está em m/s (valores típicos), converte U_0 -> cm/s.
    Retorna: (df_convertido, ylim, xlabel, ylabel, xfmt, yfmt)
    """
    df2 = df.copy()

    xlabel = xcol
    ylabel = ycol
    xfmt = "%.2f"
    yfmt = "%.2f"
    ylim = None

    # --- z em cm ---
    if ycol == "z":
        zmax_val = float(df2[ycol].max())
        # heurística: se z_max <= 0.5, quase certeza que está em metros
        if zmax_val <= 0.5:
            df2[ycol] = df2[ycol] * 100.0
            ylabel = r"$z\ [cm]$"
            yfmt = "%.2f"
            ylim = (0.0, zmax_m * 100.0)  # zmax veio em metros
        else:
            # já parece estar em cm
            ylabel = r"$z\ [cm]$"
            yfmt = "%.2f"
            ylim = (0.0, zmax_m)  # assume que usuário passou em cm (raro)

    # --- U_0 em cm/s ---
    if xcol == "U_0":
        umax = float(df2[xcol].abs().max())
        # heurística: velocidades típicas em m/s costumam ser < 5
        if umax <= 5.0:
            df2[xcol] = df2[xcol] * 100.0
            xlabel = r"$u\ [cm/s]$"
            xfmt = "%.4f"
        else:
            xlabel = r"$u\ [cm/s]$"
            xfmt = "%.4f"

    return df2, ylim, xlabel, ylabel, xfmt, yfmt

def main() -> None:
    args = build_parser().parse_args()

    input_dir = resolve_from_root(args.input_dir, "data")
    output_dir = resolve_from_root(args.output_dir, "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Limite do eixo vertical APENAS quando "z" estiver no eixo Y do gráfico
    ylim = None
    if (not args.swap and args.ycol == "z") or (args.swap and args.xcol == "z"):
        ylim = (0.0, args.zmax)

    csv_files = list_csv_files(input_dir)

    # 1) PNGs dos CSVs
    for csv_path in csv_files:
        df = load_xy_from_csv(csv_path, args.xcol, args.ycol)
        
        # legenda baseada no nome do arquivo
        label = label_from_filename(csv_path.stem)

        # converte unidades (se precisar) e define ylim/labels/formatos
        df_plot, ylim_plot, xlabel, ylabel, xfmt, yfmt = maybe_convert_units_to_cm(
            df, args.xcol, args.ycol, args.zmax
        )

        if args.swap:
            out_name = f"{csv_path.stem}_{args.ycol}_vs_{args.xcol}.png"
        else:
            out_name = f"{csv_path.stem}_{args.xcol}_vs_{args.ycol}.png"

        out_path = next_available_path(output_dir / out_name)

        plot_xy(
            df=df_plot,
            xcol=args.xcol,
            ycol=args.ycol,
            out_path=out_path,
            title=None,            # <- igual ao estilo “limpo”
            label=label,           # <- aqui entra sua legenda automática
            swap=args.swap,
            dpi=args.dpi,
            ylim=ylim_plot,
            xlabel=xlabel,
            ylabel=ylabel,
            xfmt=xfmt,
            yfmt=yfmt,
        )
        print(f"[OK] {out_path}")
    
    # 2) Triplet: ref + 1000 + theory_velocity (global)
    generated = plot_triplets_with_computed_theory(
    csv_files=csv_files,
    output_dir=output_dir,
    xcol=args.xcol,
    ycol=args.ycol,
    dpi=args.dpi,
    )

    for p in generated:
        print(f"[OK][TRIPLE] {p}")

    print(f"\nPronto! Imagens em: {output_dir}")
