from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator, MultipleLocator

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

def _nice_step(span: float, nbins: int) -> float:
    """Escolhe um passo 'bonito' para ter ~nbins divisões."""
    import math

    span = abs(span)
    if span == 0:
        return 1.0

    raw = span / max(1, nbins)
    mag = 10 ** math.floor(math.log10(raw))
    for s in (1, 2, 2.5, 5, 10):
        step = s * mag
        if step >= raw:
            return step
    return 10 * mag


def style_axes(
    ax,
    xfmt: str = "%.2f",
    yfmt: str = "%.3f",
    nbins_x: int = 6,
    nbins_y: int = 7,
    minor_sub: int = 2,
    minor_grid: bool = False,
) -> None:
    # define major step baseado no range atual do eixo (isso evita tick demais)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xstep = _nice_step(xmax - xmin, nbins_x)
    ystep = _nice_step(ymax - ymin, nbins_y)

    ax.xaxis.set_major_locator(MultipleLocator(xstep))
    ax.yaxis.set_major_locator(MultipleLocator(ystep))

    # minor ticks (sem “papel quadriculado” por padrão)
    ax.xaxis.set_minor_locator(AutoMinorLocator(minor_sub))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_sub))

    # formatação dos números
    ax.xaxis.set_major_formatter(FormatStrFormatter(xfmt))
    ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt))

    # ticks
    ax.tick_params(which="major", length=7, width=1.2, direction="out")
    ax.tick_params(which="minor", length=4, width=1.0, direction="out")

    # grid (major suave e limpo)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=1.2, color="0.70", alpha=0.9)

    # minor grid opcional (bem fraquinho)
    if minor_grid:
        ax.grid(True, which="minor", linestyle=":", linewidth=0.7, color="0.85", alpha=0.8)



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

    style_axes(ax, xfmt=xfmt, yfmt=yfmt, minor_grid=False)

    # legenda no topo (igual referência) — só aparece se label != None
    if label:
        if legend_top:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)
        else:
            ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
