# xclim_tools.core.registry
# =================================================

# This module builds and exposes the registry of available xclim tools.
# It dynamically loads the list of indicators and constructs their classes via factory.

from ..utils.config import load_indicator_list
from .factory import build_tools

TOOLS = build_tools(load_indicator_list())

__all__ = ["TOOLS"]