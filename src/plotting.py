"""
Publication-quality plotting for the expertise evaluation pipeline.

All figures follow a consistent aesthetic:
- ``seaborn`` *white* theme with enlarged fonts for print.
- Harmonious colour palette via ``Set2``.
- Every function accepts an optional ``save_path`` for deterministic
  figure export (PNG, 300 dpi).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
_PALETTE = "Set2"

def _set_style() -> None:
    """Apply consistent publication-quality styling."""
    sns.set_theme(style="white", font_scale=1.25)
    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })


def _save(fig, save_path: Path | str | None) -> None:
    """Save figure if a path is provided, then close."""
    if save_path is not None:
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight", transparent=True)
        logger.info("Figure saved to %s", save_path)
    plt.close(fig)


# =========================================================================
# 1.  Heatmaps (2 × 3 grid)
# =========================================================================
def plot_heatmaps(
    score_dfs: list[pd.DataFrame],
    titles: list[str],
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (24, 14),
) -> None:
    """Plot a 2 × 3 grid of score heatmaps.

    Parameters
    ----------
    score_dfs : list[pd.DataFrame]
        Score matrices (reviewers × proposals), one per method.
    titles : list[str]
        Human-readable method names.
    save_path : Path or str, optional
        If given, save figure to this path.
    figsize : tuple[int, int]
        Figure dimensions in inches.
    """
    _set_style()
    n = len(score_dfs)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.asarray(axes).flatten()

    for ax, df, title in zip(axes, score_dfs, titles):
        data = df.values if not hasattr(df, "toarray") else df.toarray()
        sns.heatmap(data, ax=ax, cmap="viridis", xticklabels=False,
                    yticklabels=False, cbar_kws={"shrink": 0.6})
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Proposals")
        ax.set_ylabel("Reviewers")

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    _save(fig, save_path)


# =========================================================================
# 2.  KDE distributions
# =========================================================================
def plot_score_distributions(
    score_dfs: list[pd.DataFrame],
    labels: list[str],
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """Overlay KDE plots of score distributions for each method.

    Parameters
    ----------
    score_dfs : list[pd.DataFrame]
        Score matrices.
    labels : list[str]
        Method names for the legend.
    save_path : Path or str, optional
        If given, save figure to this path.
    figsize : tuple[int, int]
        Figure dimensions.
    """
    _set_style()
    fig, ax = plt.subplots(figsize=figsize)

    for df, label in zip(score_dfs, labels):
        vals = df.values.flatten()
        vals = vals[~np.isnan(vals)]
        sns.kdeplot(vals, label=label, fill=True, alpha=0.35, ax=ax)

    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)


# =========================================================================
# 3.  Rank boxplot
# =========================================================================
def plot_rank_boxplot(
    method_metrics: dict[str, dict[str, dict]],
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """Boxplot of rank distributions across methods.

    Methods are sorted by median rank (best on the left).

    Parameters
    ----------
    method_metrics : dict
        ``{method_name: {proposal_id: {"rank": int, ...}, ...}}``.
    save_path : Path or str, optional
        If given, save figure to this path.
    figsize : tuple[int, int]
        Figure dimensions.
    """
    _set_style()

    # Collect ranks per method
    data = {}
    for name, m in method_metrics.items():
        data[name] = np.array([m[pid]["rank"] for pid in m])

    # Sort by median rank
    sorted_names = sorted(data, key=lambda n: np.median(data[n]))

    fig, ax = plt.subplots(figsize=figsize)
    bplot = ax.boxplot(
        [data[n] for n in sorted_names],
        patch_artist=True,
        labels=sorted_names,
        showfliers=True,
    )

    colours = sns.color_palette(_PALETTE, len(sorted_names))
    for patch, colour in zip(bplot["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.8)

    ax.set_ylabel("Rank of True Reviewer")
    ax.set_title("Distribution of True-Reviewer Ranks by Method")
    fig.tight_layout()
    _save(fig, save_path)


# =========================================================================
# 4.  Raincloud / violin plot
# =========================================================================
def plot_rainclouds(
    long_df: pd.DataFrame,
    score_columns: list[str],
    expertise_col: str = "expertise_label",
    save_path: Path | str | None = None,
    figsize_per_method: tuple[int, int] = (10, 3),
) -> None:
    """Violin + strip plots of score vs. expertise level per method.

    If ``ptitprince`` is installed, true raincloud plots are used;
    otherwise, standard violin + strip plots are produced.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-form DataFrame with expertise labels and score columns.
    score_columns : list[str]
        Column names containing each method's scores.
    expertise_col : str
        Column with expertise labels (``"Expert"``, ``"Intermediate"``,
        ``"Non-Expert"``).
    save_path : Path or str, optional
        If given, save figure to this path.
    figsize_per_method : tuple[int, int]
        Figure size per subplot row.
    """
    _set_style()
    expertise_order = ["Expert", "Intermediate", "Non-Expert"]
    n = len(score_columns)
    fig_h = figsize_per_method[1] * n
    fig, axes = plt.subplots(n, 1, figsize=(figsize_per_method[0], fig_h))
    if n == 1:
        axes = [axes]

    try:
        import ptitprince as pt
        use_rain = True
    except ImportError:
        use_rain = False
        logger.info("ptitprince not installed — falling back to violin plots")

    for ax, col in zip(axes, score_columns):
        sub = long_df[[expertise_col, col]].dropna()
        if use_rain:
            pt.RainCloud(
                x=expertise_col, y=col, data=sub,
                order=expertise_order, palette=_PALETTE,
                hue=expertise_col, alpha=0.9,
                width_viol=0.6, width_box=0.25,
                point_size=2.0, ax=ax, legend=False,
            )
        else:
            sns.violinplot(
                x=expertise_col, y=col, data=sub,
                order=expertise_order, palette=_PALETTE, ax=ax,
            )
            sns.stripplot(
                x=expertise_col, y=col, data=sub,
                order=expertise_order, color="0.3", size=2, ax=ax,
            )
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Score")

    fig.tight_layout()
    _save(fig, save_path)
