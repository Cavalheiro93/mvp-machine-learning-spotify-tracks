from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None  # graceful degradation if seaborn isn't available

try:  # lazy/optional import to avoid hard failure if not installed
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore


def _ensure_dir(path: Optional[str]) -> None:
    if path is None:
        return
    os.makedirs(path, exist_ok=True)


def _display_or_save(fig: Any, title: str, save_dir: Optional[str]) -> None:
    if save_dir:
        safe = (
            title.lower()
            .replace(" ", "_")
            .replace("/", "-")
            .replace("\\", "-")
            .replace("%", "pct")
        )
        fig.savefig(os.path.join(save_dir, f"{safe}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        fig.show()


def _grid_size(n: int) -> Tuple[int, int]:
    if n <= 0:
        return 0, 0
    cols = 3 if n >= 3 else n
    rows = int(np.ceil(n / cols))
    return rows, cols


def _is_categorical(s: pd.Series, max_unique: int = 30) -> bool:
    if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
        return True
    if pd.api.types.is_bool_dtype(s):
        return True
    # treat low-cardinality numerics as categorical for visualization
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        try:
            nun = s.nunique(dropna=True)
            return nun <= max_unique
        except Exception:
            return False
    return False


def quick_eda(
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    datetime_cols: Optional[List[str]] = None,
    max_cat: int = 30,
    max_num_plots: int = 18,
    pairplot: bool = False,
    pairplot_sample: int = 1500,
    corr_max_cols: int = 30,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Quick, opinionated EDA for a pandas DataFrame.

    - Prints shape, dtypes, memory, missingness, duplicates, unique counts
    - Plots: numeric histograms + boxplots, correlation heatmap,
             categorical bar charts (low-cardinality), missingness bar
    - Optional target-aware summaries and pairplot (sampled)

    Returns a dict of figures keyed by a short name.
    If save_dir is provided, figures are saved there and closed; otherwise shown inline.
    """

    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    _ensure_dir(save_dir)
    figs: Dict[str, Any] = {}

    # Coerce datetime columns if provided
    if datetime_cols:
        for c in datetime_cols:
            if c in df.columns:
                with pd.option_context("mode.chained_assignment", None):
                    df[c] = pd.to_datetime(df[c], errors="coerce")

    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)

    if verbose:
        print("===== Overview =====")
        print(f"Shape: {n_rows} rows x {n_cols} cols")
        print(f"Memory: {mem_mb:.2f} MB\n")

        print("===== Dtypes =====")
        print(df.dtypes.sort_index())
        print()

        print("===== Head =====")
        print(df.head(5))
        print()

    # Missingness
    mis = df.isna().sum().sort_values(ascending=False)
    mis_pct = (mis / len(df) * 100).round(2)
    miss_tbl = pd.DataFrame({"missing": mis, "missing_%": mis_pct})
    if verbose:
        print("===== Missing Values =====")
        print(miss_tbl[miss_tbl["missing"] > 0].head(30))
        print()

    # Duplicates
    dup_count = int(df.duplicated().sum())
    if verbose:
        print("===== Duplicates =====")
        print(f"Duplicate row count: {dup_count}")
        print()

    # Unique counts
    unique_counts = df.nunique(dropna=True).sort_values(ascending=True)
    if verbose:
        print("===== Unique Counts (ascending) =====")
        print(unique_counts.head(30))
        print()

    # Numeric / categorical columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if _is_categorical(df[c], max_unique=max_cat)]

    # Numeric describe
    if num_cols and verbose:
        print("===== Numeric Describe =====")
        print(df[num_cols].describe().T)
        print()

    # Categorical describe (only small ones)
    small_cat_cols = [c for c in cat_cols if df[c].nunique(dropna=True) <= max_cat]
    if small_cat_cols and verbose:
        print("===== Categorical (low-cardinality) Top Values =====")
        for c in small_cat_cols[:10]:
            vc = df[c].value_counts(dropna=False).head(10)
            print(f"- {c}:")
            print(vc)
        print()

    # Missingness bar plot
    if mis.sum() > 0 and plt is not None:
        fig, ax = plt.subplots(figsize=(min(12, max(6, n_cols * 0.5)), 4))
        miss_pct_sorted = mis_pct[mis_pct > 0].sort_values(ascending=False)
        ax.bar(miss_pct_sorted.index.astype(str), miss_pct_sorted.values, color="#d62728")
        ax.set_title("Missing Values (%) by Column")
        ax.set_ylabel("%")
        ax.set_xticklabels(miss_pct_sorted.index.astype(str), rotation=60, ha="right")
        fig.tight_layout()
        figs["missing"] = fig
        _display_or_save(fig, "missing_values_pct", save_dir)

    # Numeric histograms
    if num_cols and plt is not None:
        take = min(len(num_cols), max_num_plots)
        cols_take = num_cols[:take]
        rows, cols = _grid_size(take)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = np.atleast_1d(axes).flatten()
        for ax, c in zip(axes, cols_take):
            s = df[c].dropna()
            if sns is not None:
                sns.histplot(s, kde=True, ax=ax, color="#1f77b4")
            else:
                ax.hist(s, bins=30, color="#1f77b4", alpha=0.85)
            ax.set_title(f"Hist: {c}")
        for ax in axes[len(cols_take):]:
            ax.axis("off")
        fig.suptitle("Numeric Histograms", y=1.02)
        fig.tight_layout()
        figs["num_hist"] = fig
        _display_or_save(fig, "numeric_histograms", save_dir)

    # Numeric boxplots (quick outlier scan)
    if num_cols and plt is not None:
        take = min(len(num_cols), max(6, max_num_plots // 2))
        cols_take = num_cols[:take]
        fig, ax = plt.subplots(figsize=(min(16, 1 + 0.6 * take), 5))
        if sns is not None:
            sns.boxplot(data=df[cols_take], orient="h", ax=ax, color="#2ca02c")
        else:
            # Fallback: individual boxplots
            ax.boxplot([df[c].dropna().values for c in cols_take], vert=False, labels=cols_take)
        ax.set_title("Numeric Boxplots (subset)")
        fig.tight_layout()
        figs["num_box"] = fig
        _display_or_save(fig, "numeric_boxplots", save_dir)

    # Correlation heatmap for top-variance numeric columns
    if len(num_cols) >= 2 and plt is not None:
        # choose up to corr_max_cols columns by variance (drop NaNs)
        variances = df[num_cols].var(numeric_only=True).sort_values(ascending=False)
        chosen = list(variances.head(corr_max_cols).index)
        corr = df[chosen].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(min(1 + 0.4 * len(chosen), 18), min(1 + 0.4 * len(chosen), 18)))
        if sns is not None:
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True)
        else:
            cax = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(chosen)))
            ax.set_yticks(range(len(chosen)))
            ax.set_xticklabels(chosen, rotation=90)
            ax.set_yticklabels(chosen)
            fig.colorbar(cax)
        ax.set_title("Correlation Heatmap (top variance)")
        fig.tight_layout()
        figs["corr_heatmap"] = fig
        _display_or_save(fig, "correlation_heatmap", save_dir)

    # Categorical bar plots (only low-cardinality)
    if small_cat_cols and plt is not None:
        take = min(len(small_cat_cols), max_num_plots)
        cols_take = small_cat_cols[:take]
        rows, cols = _grid_size(take)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
        axes = np.atleast_1d(axes).flatten()
        for ax, c in zip(axes, cols_take):
            series = df[c].astype(str).fillna("<NA>")
            vc = series.value_counts().head(15)
            if sns is not None:
                sns.barplot(x=vc.values, y=vc.index, ax=ax, color="#9467bd")
            else:
                ax.barh(vc.index, vc.values, color="#9467bd")
            ax.set_title(f"Top categories: {c}")
        for ax in axes[len(cols_take):]:
            ax.axis("off")
        fig.tight_layout()
        figs["cat_bars"] = fig
        _display_or_save(fig, "categorical_bars", save_dir)

    # Target-aware quick look
    if target and target in df.columns:
        tgt = df[target]
        if pd.api.types.is_numeric_dtype(tgt) and plt is not None:
            # correlation with numeric features
            if num_cols:
                corr_t = df[num_cols].corrwith(tgt, numeric_only=True).sort_values(key=lambda s: s.abs(), ascending=False)
                if verbose:
                    print("===== Target Correlation (numeric) =====")
                    print(corr_t.head(20))
                    print()

                top = corr_t.dropna().head(min(12, len(corr_t)))
                if len(top) > 0:
                    fig, ax = plt.subplots(figsize=(6, 0.4 * len(top) + 1))
                    vals = top.values
                    idx = top.index
                    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
                    ax.barh(idx, vals, color=colors)
                    ax.axvline(0, color="black", lw=1)
                    ax.set_title(f"Top correlations with {target}")
                    fig.tight_layout()
                    figs["target_corr"] = fig
                    _display_or_save(fig, f"target_{target}_correlations", save_dir)
        else:
            # categorical target: show mean of top numeric features by class
            if num_cols and plt is not None:
                # choose top-variance numeric columns to summarize
                variances = df[num_cols].var(numeric_only=True).sort_values(ascending=False)
                chosen = list(variances.head(min(6, len(variances))).index)
                agg = df.groupby(target, dropna=False)[chosen].mean(numeric_only=True)
                fig, axes = plt.subplots(len(chosen), 1, figsize=(7, 2.6 * len(chosen)))
                axes = np.atleast_1d(axes)
                for ax, c in zip(axes, chosen):
                    vals = agg[c]
                    if sns is not None:
                        sns.barplot(x=vals.index.astype(str), y=vals.values, ax=ax, color="#1f77b4")
                    else:
                        ax.bar(vals.index.astype(str), vals.values, color="#1f77b4")
                    ax.set_title(f"Mean {c} by {target}")
                    ax.set_xlabel(target)
                fig.tight_layout()
                figs["target_categorical_summary"] = fig
                _display_or_save(fig, f"target_{target}_by_class_means", save_dir)

    # Optional pairplot (sampled and only if seaborn available)
    if pairplot and sns is not None and plt is not None and len(num_cols) >= 2:
        subset_cols = num_cols[: min(6, len(num_cols))]
        data = df[subset_cols].dropna()
        if len(data) > pairplot_sample:
            data = data.sample(pairplot_sample, random_state=42)
        g = sns.pairplot(data)
        figs["pairplot"] = g.fig
        _display_or_save(g.fig, "pairplot_numeric_subset", save_dir)

    if verbose:
        print("EDA complete. Figures:")
        print(", ".join(figs.keys()) if figs else "(no figures generated)")

    return figs


def profile_df(
    df: pd.DataFrame,
    **kwargs,
) -> Dict[str, plt.Figure]:
    """Alias to quick_eda for convenience."""
    return quick_eda(df, **kwargs)
