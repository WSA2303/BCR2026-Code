from __future__ import annotations

import argparse
import math
from pathlib import Path

from csv_plotter.io_csv import list_csv_files, load_xy_from_csv
from csv_plotter.plotting import plot_xy


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

        if args.swap:
            out_name = f"{csv_path.stem}_{args.ycol}_vs_{args.xcol}.png"
        else:
            out_name = f"{csv_path.stem}_{args.xcol}_vs_{args.ycol}.png"

        out_path = next_available_path(output_dir / out_name)

        plot_xy(
            df=df,
            xcol=args.xcol,
            ycol=args.ycol,
            out_path=out_path,
            title=csv_path.stem,
            swap=args.swap,
            dpi=args.dpi,
            ylim=ylim,  
            xlabel=r"$u\ [cm/s]$",       # se você estiver em cm/s
            ylabel=r"$z\ [cm]$",         # se você estiver em cm
            yfmt="%.3f",                 # opcional: combina com z em cm
        )

        print(f"[OK] {out_path}")

    print(f"\nPronto! Imagens em: {output_dir}")
