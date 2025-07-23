# xclim_tools.core.base
# =================================================

# This module defines a base class for xclim indicator tools to be used with LangChain agents.
# It provides a consistent interface, input schema, and dataset handling mechanism.


from typing import Type, Any
from pydantic import BaseModel, PrivateAttr
from langchain.tools import BaseTool


class XclimBaseInput(BaseModel):
    """Base input schema for xclim tools."""
    model_config = dict(arbitrary_types_allowed=True)


class XclimIndicatorTool(BaseTool):
    """
    Abstract base class for xclim tools used within LangChain.

    Subclasses must implement the `_run` and `_arun` methods.
    """

    name: str = "base_indicator_tool"
    description: str = "Generic xclim tool"
    args_schema: Type[XclimBaseInput] = XclimBaseInput

    _ds: Any = PrivateAttr(default=None)

    def __init__(self, ds: Any, **kwargs):
        super().__init__(**kwargs)
        self._ds = ds

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement the _run method.")

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement the _arun method.")