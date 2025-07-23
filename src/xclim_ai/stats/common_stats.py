# xclim_tools.utils.statistics
# =================================================
# Statistical summaries for climate indicator results.
# Includes annual and monthly aggregation, trend detection (Mann-Kendall).

from __future__ import annotations

import pandas as pd
import pymannkendall as mk
import xarray as xr
from typing import Any, Dict

__all__ = ["common_stats", "aggregate_stats"]


def common_stats(df: pd.DataFrame) -> Dict[str, Any]:

    x = df.index.year.to_numpy()
    y = df.values.flatten()
    mk_test = mk.original_test(y, alpha=0.05)

    return {
        "mean": float(df.mean().values[0].round(2)),
        "std":  float(df.std().values[0].round(2)),
        "min":  float(df.min().values[0].round(2)),
        "max":  float(df.max().values[0].round(2)),
        "trend": {
            "slope":     float(mk_test.slope.round(2)),
            "intercept": float(mk_test.intercept.round(2)),
            "p_value":   float(mk_test.p.round(5)),
        },
    }


def aggregate_stats(result: xr.DataArray) -> Dict[str, Any]:

    dat = (
        result.mean("model")    # media ensemble
              .to_dataframe()
              .drop(columns=["lat", "lon"], errors="ignore")
    )

    historical   = dat.loc["1981":"2020"]
    short_term   = dat.loc["2021":"2035"]
    medium_term  = dat.loc["2036":"2050"]

    yearly = {
        period: common_stats(df.resample("Y").mean())
        for period, df in {
            "historical (1981-2020)":  historical,
            "short_term (2021-2035)":  short_term,
            "medium_term (2036-2050)": medium_term,
        }.items()
    }

    freq = pd.infer_freq(dat.index)
    if freq in {"MS", "D"}:
        months = ["jan","feb","mar","apr","may","jun",
                  "jul","aug","sep","oct","nov","dec"]
        monthly = {}
        for i, m in enumerate(months, start=1):
            for period, df in {
                "historical (1981-2020)": historical,
                "short_term (2021-2035)": short_term,
                "medium_term (2036-2050)": medium_term,
            }.items():
                month_df = df[df.index.month == i]
                if not month_df.empty:
                    monthly[f"{m}_{period}"] = common_stats(
                        month_df.resample("Y").mean()
                    )
        return {"yearly": yearly, "monthly": monthly}

    return {"yearly": yearly}