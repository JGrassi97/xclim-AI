# xclim_tools.datasets.__init__.py
# =================================================\

from typing import Any, Callable, Dict, Tuple

import logging

import xarray as xr

from .openmeteo_standard_ensemble import load_standard_ensemble
from ..utils.config import load_config
from ..utils.paths import OPENMETEO_CACHE_DIR

# Registry of available dataset loader functions
DATASET_LOADERS: Dict[str, Callable[..., xr.Dataset]] = {
    "openmeteo_standard_ensemble": load_standard_ensemble,
}


def load_dataset_from_config(*, return_params: bool = False, **overrides: Any) -> xr.Dataset | Tuple[xr.Dataset, Dict[str, Any], str]:
    """Load a dataset based on configuration settings.

    Parameters
    ----------
    **overrides: Any
        Optional keyword arguments overriding the parameters in the
        configuration file.

    Returns
    -------
    xr.Dataset
        The loaded xarray dataset.
    """
    cfg = load_config()
    ds_cfg = cfg.get("dataset", {})
    name = ds_cfg.get("name", "openmeteo_standard_ensemble")
    params = ds_cfg.get("params", {})
    params.update(overrides)
    #logger = logging.getLogger("XclimAI")
    #logger.info(f"Loading dataset '{name}' with parameters: {params}")

    if "cache_dir" not in params:
        params["cache_dir"] = OPENMETEO_CACHE_DIR

    loader = DATASET_LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset loader '{name}'")

    ds = loader(**params)
    return (ds, params, name) if return_params else ds
