# xclim_tools.core.factory
# =================================================

# This module dynamically builds LangChain-compatible tools from xclim indicators.
# It creates custom input schemas using Pydantic, wraps xclim functions into executable tools,
# and enables optional statistical summarization via an LLM.

import inspect
from typing import Type, Dict
from pathlib import Path

import xarray as xr
import xclim.indicators.atmos as xind
from pydantic import create_model, Field
from typing import Literal

from .base import XclimBaseInput, XclimIndicatorTool
from xclim_ai.utils.annotations import clean_annotation
from xclim_ai.core.memory import save_to_memory
from xclim_ai.utils.config import load_exclusions_list
from xclim_ai.utils.llm import initialize_llm
from xclim_ai.stats.common_stats import aggregate_stats
from xclim_ai.utils.prompts import load_prompt

EXCLUDE_FIELDS = load_exclusions_list()
llm = initialize_llm()


def _make_run(func, ind_name: str | None = None):
    """Creates the _run method for a dynamic tool class."""

    def _run(self, safe: bool = True, **kwargs):
        try:
            result = func(ds=self._ds, **kwargs)
            stats_dict = aggregate_stats(result)
            to_context = stats_dict
            summ = None

            if self.llm_summary:
                summary_prompt = load_prompt("stat_summarizer.md")
                summary_prompt = (
                    summary_prompt.replace("{result.attrs}`", str(result.attrs))
                    .replace("{stats_dict}", str(stats_dict))
                )
                to_context = llm.invoke(summary_prompt).content
                summ = to_context

            if getattr(self, "out_dir", None) is not None:
                save_to_memory(
                    ind_name,
                    result,
                    stats_dict,
                    root_dir=self.out_dir,
                    save_args=kwargs,
                    summary=summ,
                )

            return to_context

        except Exception as e:
            raise
            return f"Error during execution: {e}" if safe else (_ for _ in ()).throw(e)

    return _run


def build_input(ind_name: str, indicator) -> Type[XclimBaseInput]:
    """Builds a custom input schema for a given xclim indicator."""
    sig = inspect.signature(indicator)
    fields = {
        pname: (
            clean_annotation(param.annotation),
            Field(default=(param.default if param.default != inspect._empty else ...)),
        )
        for pname, param in sig.parameters.items()
        if pname not in EXCLUDE_FIELDS
    }

    cls_name = f"{ind_name.title().replace('_', '')}Input"
    input_model = create_model(cls_name, __base__=XclimBaseInput, **fields)
    input_model.model_rebuild(
        _types_namespace={
            "DayOfYearStr": str,
            "Quantified": str,
            "rv_continuous": str,
            "DateStr": str,
            "Literal": Literal,
        }
    )

    return input_model


def build_tool(
    ind_name: str, input_model: Type[XclimBaseInput]
) -> Type[XclimIndicatorTool]:
    """Dynamically builds a LangChain-compatible tool class for a specific indicator."""
    func = getattr(xind, ind_name)

    def __init__(
        self,
        ds: xr.Dataset,
        *,
        llm_summary: bool = False,
        out_dir: Path | None = None,
    ):
        super(XclimIndicatorTool, self).__init__(ds=ds)
        object.__setattr__(self, "_ds", ds)
        object.__setattr__(self, "llm_summary", llm_summary)
        object.__setattr__(self, "out_dir", Path(out_dir) if out_dir else None)

    cls_name = f"{ind_name.title().replace('_', '')}Tool"
    return type(
        cls_name,
        (XclimIndicatorTool,),
        {
            "__module__": __name__,
            "__init__": __init__,
            "_run": _make_run(func, ind_name),
            "name": ind_name,
            "description": inspect.getdoc(func) or f"xclim tool {ind_name}",
            "args_schema": input_model,
            "__annotations__": {
                "name": str,
                "description": str,
                "args_schema": Type[input_model],
            },
        },
    )


def build_tools(names) -> Dict[str, Type[XclimIndicatorTool]]:
    """Creates a dictionary of dynamically generated tool classes."""
    tools = {}
    for nm in names:
        ind = getattr(xind, nm)
        inp = build_input(nm, ind)
        tools[nm] = build_tool(nm, inp)
    return tools