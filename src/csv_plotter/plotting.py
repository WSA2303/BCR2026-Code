from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


_STYLE_APPLIED = False


def apply_plot_style() -> None:
    """Aplica um estilo global parecido com o da sua referência."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    plt.rcParams.update({
        # fonte / matemática (bem parecido com LaTeX)
        "font.family": "serif",
        "mathtext.fontset": "cm",

        # tamanhos
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 18,

        # linhas / bordas
        "lines.linewidth": 2.8,
        "axes.linewidth": 1.3,
    })

    _STYLE_APPLIED = True


def plot_xy(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    out_path: Path,
    title: str | None = None,
    swap: bool = False,
    dpi: int = 300,
    ylim: tuple[float, float] | None = None,
    label: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xfmt: str = "%.4f",
    yfmt: str = "%.4f",
    legend_top: bool = True,
) -> None:
    apply_plot_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

    if swap:
        x, y = df[ycol], df[xcol]
        _xlabel = xlabel or ycol
        _ylabel = ylabel or xcol
    else:
        x, y = df[xcol], df[ycol]
        _xlabel = xlabel or xcol
        _ylabel = ylabel or ycol

    ax.plot(x, y, label=label)

    if title:
        ax.set_title(title)

    ax.set_xlabel(_xlabel)
    ax.set_ylabel(_ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    # formatadores de tick
    ax.xaxis.set_major_formatter(FormatStrFormatter(xfmt))
    ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt))

    # grid tracejado como na imagem
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=1.5, color="0.7")

    # legenda no topo (igual referência) — só aparece se label != None
    if label:
        if legend_top:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)
        else:
            ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
