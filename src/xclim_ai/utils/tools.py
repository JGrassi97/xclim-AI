# xclim_tools.utils.tools
# =================================================

# Utility to retrieve only the tools marked as valid for use,
# based on an external YAML configuration file.

from xclim_ai import TOOLS
from xclim_ai.utils.paths import VALID_TOOLS_PATH
import yaml


def get_valid_tools():
    """
    Load and return a list of validated indicator tools based on an external list.

    Returns
    -------
    list
        A list of tool instances whose names are included in the VALID_TOOLS_PATH file.
    """
    with open(VALID_TOOLS_PATH, "r", encoding="utf-8") as f:
        valid_tools_names = yaml.safe_load(f)

    valid_tools = [tool for name, tool in TOOLS.items() if name in valid_tools_names]
    return valid_tools