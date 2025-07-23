# xclim_tools.utils.config
# =================================================

# Load configuration elements for xclim tools, including:
# - list of indicators to expose
# - fields to exclude from input schema generation
# - general configuration settings (e.g., from ~/.xclim_config.yaml)

from importlib import resources
from pathlib import Path
from typing import Union

import yaml

from xclim_ai.utils.paths import CONFIG_PATH


def load_indicator_list() -> list[str]:
    """
    Load the list of available indicators from the package YAML file.

    Returns
    -------
    list of str
        List of indicator names.
    """
    try:
        txt = resources.files("xclim_ai.data").joinpath("indicators.yaml").read_text()
        return yaml.safe_load(txt)["indicators"]
    except Exception as e:
        raise RuntimeError(f"Failed to load indicators list: {e}")


def load_exclusions_list() -> list[str]:
    """
    Load the list of fields to exclude from dynamic input schema.

    Returns
    -------
    list of str
        List of field names to exclude.
    """
    try:
        txt = resources.files("xclim_ai.data").joinpath("exclude.yaml").read_text()
        return yaml.safe_load(txt)["exclude_fields"]
    except Exception as e:
        raise RuntimeError(f"Failed to load exclude_fields list: {e}")


def load_config(path: Union[Path, None] = CONFIG_PATH) -> dict:
    """
    Load the global configuration file.

    Parameters
    ----------
    path : Path or None
        Path to the YAML config file. Defaults to CONFIG_PATH.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if path is None:
        path = CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)