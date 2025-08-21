# xclim_tools.core.memory
# =================================================

# This module provides a utility to persist the output of an xclim indicator run.
# Each call creates a UUID-named folder and stores:
# - a CSV file with the indicator values
# - a JSON file with computed statistics
# - a PNG plot of the time series
# - optionally, a YAML file with input arguments and a markdown summary.

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from xclim_ai.utils.paths import OUTPUT_RESULTS
#from xclim_tools.core.memory import save_to_memory  # circular, only needed if reused elsewhere


def save_to_memory(
    ind_name: str,
    result: xr.DataArray | xr.Dataset,
    stats_dict: Dict[str, Any],
    root_dir: Path,
    *,
    save_args: Optional[Dict[str, Any]] = None,
    summary: Optional[str] = None,
) -> Path:
    """Persist the result of a single indicator run to disk.

    Parameters
    ----------
    ind_name : str
        Name of the indicator (used as file prefix).
    result : xr.DataArray or xr.Dataset
        xarray object returned by the indicator.
    stats_dict : dict
        JSON-serializable dictionary with aggregated statistics.
    root_dir : Path
        Root folder where the UUID-named directory will be created.
    save_args : dict, optional
        Arguments passed to the indicator, saved in YAML if provided.
    summary : str, optional
        Optional markdown summary to save.

    Returns
    -------
    Path
        Path to the directory where files were saved.
    """
    run_dir = Path(root_dir)

    # Save values as CSV
    if isinstance(result, xr.DataArray):
        df = result.mean("model") if "model" in result.dims else result
        df = df.to_dataframe()
    else:
        df = result.to_dataframe()

    # Prepare plotting DataFrame
    if isinstance(result, xr.DataArray):
        result_res = result.resample(time="1Y").mean()
        p10 = result_res.quantile(0.10, dim="model")
        mean = result_res.mean(dim="model")
        p90 = result_res.quantile(0.90, dim="model")

        df_plot = pd.concat(
            [
                p10.to_dataframe(name="p10"),
                mean.to_dataframe(name="mean"),
                p90.to_dataframe(name="p90"),
            ],
            axis=1,
        )
    else:
        df_plot = result.to_dataframe()

    df.drop(columns=[c for c in ["lat", "lon"] if c in df.columns], inplace=True)
    csv_path = run_dir / f"{ind_name}.csv"
    df.to_csv(csv_path, index=True)

    # Save statistics as JSON
    json_path = run_dir / f"{ind_name}_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)

    # Save plot
    png_path = run_dir / f"{ind_name}.png"
    _save_plot(df_plot, png_path, ind_name)

    # Save markdown summary (if provided)
    if summary:
        summary_path = run_dir / f"{ind_name}_summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

    # Save YAML input arguments (if provided)
    if save_args:
        try:
            import yaml

            yaml_path = run_dir / f"{ind_name}_args.yaml"
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(save_args, f, allow_unicode=True)
        except ModuleNotFoundError:
            pass  # If PyYAML is not installed, silently skip

    return run_dir


def _save_plot(df: pd.DataFrame, path: Path, ind_name: str):
    """Generate a basic time series plot and save it to disk."""
    try:
        if df.empty:
            return

        fig, ax = plt.subplots(figsize=(9, 5))

        # If standard columns exist, use the nicer plot with bands
        has_quantiles = all(col in df.columns for col in ["mean", "p10", "p90"])

        if has_quantiles:
            # Rolling mean for smoother visualization (10-year window when index is datetime)
            try:
                df_rolling = df.rolling(window=10, center=True).mean()
            except Exception:
                df_rolling = df

            # Plot rolling mean and percentiles
            ax.plot(df_rolling.index, df_rolling["mean"], color="tab:red", label="10-year rolling mean", linewidth=2.5)
            ax.plot(df_rolling.index, df_rolling["p10"], color="tab:red", ls="--", alpha=0.7, lw=1.5)
            ax.plot(df_rolling.index, df_rolling["p90"], color="tab:red", ls=":", alpha=0.7, lw=1.5)

            # Plot original mean and percentiles
            ax.plot(df.index, df["mean"], label="Multi-model mean", color="black", lw=1)
            ax.fill_between(df.index, df["p10"], df["p90"], color="black", alpha=0.08, label="10–90% range")
        else:
            # Fallback: plot up to first 5 numeric columns
            plotted = 0
            for col in df.columns:
                if plotted >= 5:
                    break
                try:
                    ax.plot(df.index, df[col], label=str(col), lw=1.5)
                    plotted += 1
                except Exception:
                    continue

        # Styling
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Time")
        ax.set_ylabel(ind_name)
        ax.set_title(f"{ind_name} time series")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        # Do not raise plotting errors; silently skip plot generation
        try:
            plt.close('all')
        except Exception:
            pass