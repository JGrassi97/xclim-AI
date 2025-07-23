# xclim_tools.datasets.cmip_openmeteo
# =================================================

# This module provides helper functions to download, cache, and convert Open-Meteo climate API
# responses into pandas DataFrames or xarray Datasets. Results are cached locally as Parquet files
# using a hash of the query parameters.

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import logging

import pandas as pd
import xarray as xr
import openmeteo_requests


def _as_csv(value: Any) -> str:
    """Convert iterable values to comma-separated strings, otherwise return as-is."""
    if isinstance(value, (list, tuple, set)):
        return ",".join(map(str, value))
    return value


def _load_openmeteo_df(
    *,
    url: str,
    params: Dict[str, Any],
    cache_dir: Path | str,
    cache_on_disk: bool = True,
) -> pd.DataFrame:
    """Download data from Open-Meteo API and return a multi-model DataFrame indexed by (date, model)."""
    cache_dir = Path(cache_dir)
    if cache_on_disk:
        cache_dir.mkdir(parents=True, exist_ok=True)

    req_string = json.dumps({"url": url, "params": params}, sort_keys=True)
    query_hash = hashlib.md5(req_string.encode()).hexdigest()
    parquet_path = cache_dir / f"openmeteo_{query_hash}.parquet"

   #logger = logging.getLogger("XclimAI")

    if cache_on_disk and parquet_path.exists():
        #logger.info(f"📁 Loading from cache: {parquet_path.name}")
        return pd.read_parquet(parquet_path)

    #logger.info("⬇️  Downloading data from Open-Meteo…")
    client = openmeteo_requests.Client()
    responses = client.weather_api(url, params=params)

    if not responses:
        raise RuntimeError("Open-Meteo API returned an empty response.")

    requested_models: list[str] = params.get("models", "").split(",")
    all_dfs: list[pd.DataFrame] = []

    for i, response in enumerate(responses):
        model_name = requested_models[i] if i < len(requested_models) else f"model_{i}"

        daily = response.Daily()
        variables = [
            daily.Variables(j).ValuesAsNumpy() for j in range(daily.VariablesLength())
        ]
        dates = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )

        data: dict[str, Iterable[Any]] = {"date": dates}
        for name, values in zip(params["daily"].split(","), variables):
            data[name] = values

        df = pd.DataFrame(data)
        df["model"] = model_name
        df.set_index(["date", "model"], inplace=True)
        all_dfs.append(df)

    final_df = pd.concat(all_dfs).sort_index()

    if cache_on_disk:
        final_df.to_parquet(parquet_path, compression="snappy")
        #logger.info(f"✅ Saved to cache: {parquet_path.name}")

    return final_df


def load_openmeteo(
    source: str | None,
    request: Dict[str, Any],
    *,
    cache_dir: Path | str,
    cache_on_disk: bool = True,
    as_xarray: bool = False,
) -> Union[pd.DataFrame, xr.Dataset]:
    """
    High-level wrapper for the Open-Meteo climate API.

    Parameters
    ----------
    source : str | None
        Use "climate" (default) or provide a full API URL.
    request : dict
        Query parameters including latitude, longitude, variables, models, etc.
    cache_dir : Path | str
        Directory for local cache storage.
    cache_on_disk : bool
        Whether to save and reuse query cache.
    as_xarray : bool
        If True, return an xarray.Dataset. Otherwise, return a pandas.DataFrame.

    Returns
    -------
    pd.DataFrame or xr.Dataset
    """
    url_aliases = {
        None: "https://climate-api.open-meteo.com/v1/climate",
        "": "https://climate-api.open-meteo.com/v1/climate",
        "climate": "https://climate-api.open-meteo.com/v1/climate",
    }
    url = url_aliases.get(source, source)

    for key in ("daily", "models"):
        if key in request:
            request[key] = _as_csv(request[key])

    df = _load_openmeteo_df(
        url=url,
        params=request,
        cache_dir=cache_dir,
        cache_on_disk=cache_on_disk,
    )

    return xr.Dataset.from_dataframe(df) if as_xarray else df
