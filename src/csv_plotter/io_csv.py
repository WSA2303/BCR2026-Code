from __future__ import annotations

from pathlib import Path
import pandas as pd


def list_csv_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {input_dir}")

    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum .csv encontrado em: {input_dir}")

    return files


def read_csv_flex(path: Path) -> pd.DataFrame:
    """Tenta detectar separador automaticamente (',', ';', '\\t') e limpa nomes de colunas."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = None
        for sep in (";", ",", "\t"):
            try:
                df = pd.read_csv(path, sep=sep)
                break
            except Exception:
                pass
        if df is None:
            raise

    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_xy_from_csv(path: Path, xcol: str, ycol: str) -> pd.DataFrame:
    df = read_csv_flex(path)

    xcol = xcol.strip()
    ycol = ycol.strip()

    if xcol not in df.columns or ycol not in df.columns:
        raise KeyError(
            f"No arquivo '{path.name}', não achei '{xcol}' e/ou '{ycol}'. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    out = df[[xcol, ycol]].copy()
    out[xcol] = pd.to_numeric(out[xcol], errors="coerce")
    out[ycol] = pd.to_numeric(out[ycol], errors="coerce")
    out = out.dropna(subset=[xcol, ycol]).sort_values(by=ycol)

    return out
