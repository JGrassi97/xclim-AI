# xclim_tools.datasets.standard_ensemble
# =================================================

# This module provides a high-level wrapper to download, clean, and return
# an xarray.Dataset of daily CMIP6-HR variables from the Open-Meteo API.
# It standardizes variable names to CF-compliant conventions and adds
# appropriate metadata and derived indicators like the water budget.

import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Iterable, List, Optional, Union

from xclim_ai.datasets.cmip_openmeteo import load_openmeteo
from xclim.indicators.atmos import water_budget_from_tas


_DEFAULT_DAILY: List[str] = [
    "temperature_2m_max",
    "temperature_2m_mean",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "dew_point_2m_mean",
]

_DEFAULT_MODELS: List[str] = [
    "CMCC_CM2_VHR4",
    "FGOALS_f3_H",
    "HiRAM_SIT_HR",
    "MRI_AGCM3_2_S",
    "EC_Earth3P_HR",
    "MPI_ESM1_2_XR",
    "NICAM16_8S",
]

_RENAME_MAP = {
    "temperature_2m_mean": "tas",
    "temperature_2m_max": "tasmax",
    "temperature_2m_min": "tasmin",
    "precipitation_sum": "pr",
    "wind_speed_10m_mean": "sfcWind",
    "wind_speed_10m_max": "sfcWindmax",
    "relative_humidity_2m_mean": "hurs",
    "dew_point_2m_mean": "tdps",
    "date": "time",
}


def load_standard_ensemble(
    lat: float,
    lon: float,
    *,
    start_date: str = "2026-01-01",
    end_date: str = "2050-12-31",
    name: str = "openmeteo_standard_ensemble",
    daily: Optional[Iterable[str]] = None,
    models: Optional[Iterable[str]] = None,
    cache_dir: Union[str, Path] = Path("./.cache"),
    source: str = "climate",
) -> xr.Dataset:
    """
    Download a default ensemble of daily variables for the specified point
    and return an xarray.Dataset with CF-compliant names and attributes.

    Parameters
    ----------
    lat, lon : float
        Geographic coordinates (°N, °E).
    start_date, end_date : str
        ISO-8601 date range.
    daily : iterable of str, optional
        List of variables to request; uses default if None.
    models : iterable of str, optional
        List of model names; uses default if None.
    cache_dir : path-like
        Directory for storing Parquet cache files.
    source : str
        API alias or full URL (passed to `load_openmeteo`).

    Returns
    -------
    xr.Dataset
        The loaded and formatted dataset with metadata and added indicators.
    """
    daily_vars = list(daily) if daily is not None else _DEFAULT_DAILY
    model_list = list(models) if models is not None else _DEFAULT_MODELS

    request = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": daily_vars,
        "models": model_list,
    }

    ds = load_openmeteo(
        source,
        request,
        cache_dir=Path(cache_dir),
        as_xarray=True,
    ).load()

    ds = ds.rename(_RENAME_MAP)
    ds = ds.assign_coords(time=pd.to_datetime(ds.time))
    ds = ds.assign_coords({"lat": lat, "lon": lon}).set_coords(["lat", "lon"])

    # Air temperature attributes
    for var, method in [("tas", "mean"), ("tasmax", "maximum"), ("tasmin", "minimum")]:
        ds[var].attrs.update({
            "standard_name": "air_temperature",
            "units": "degC",
            "cell_methods": f"time: {method}",
        })

    # Wind speed attributes
    ds["sfcWind"].attrs.update({
        "standard_name": "wind_speed",
        "units": "m s-1",
        "cell_methods": "time: mean",
    })
    ds["sfcWindmax"].attrs.update({
        "standard_name": "wind_speed",
        "units": "m s-1",
        "cell_methods": "time: maximum",
    })

    # Humidity and dew point
    ds["hurs"].attrs.update({
        "standard_name": "relative_humidity",
        "units": "1",
        "cell_methods": "time: mean",
    })
    ds["tdps"].attrs.update({
        "standard_name": "dew_point_temperature",
        "units": "degC",
        "cell_methods": "time: mean",
    })

    # Precipitation
    ds["pr"].attrs.update({
        "standard_name": "precipitation_flux",
        "units": "mm day-1",
        "cell_methods": "time: mean",
    })

    # Coordinate attributes
    ds["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    ds["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})

    # Derived variable: water budget
    wb = water_budget_from_tas(ds=ds)
    ds["wb"] = wb
    ds["wb"].attrs.update({
        "standard_name": "water_budget",
        "units": "mm day-1",
        "cell_methods": "time: mean",
        "description": (
            "Water budget computed from daily mean temperature (tas) "
            "using the xclim indicator water_budget_from_tas."
        ),
    })

    return ds