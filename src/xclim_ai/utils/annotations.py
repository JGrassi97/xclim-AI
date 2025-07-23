# xclim_tools.utils.annotations
# =================================================

# Helper for cleaning type annotations when building dynamic models.

import inspect
from typing import Any, Type


def clean_annotation(ann: Any) -> Type:
    """
    Clean and normalize a function argument annotation.

    Parameters
    ----------
    ann : Any
        The raw annotation object (may be inspect._empty or a string).

    Returns
    -------
    type
        A valid Python type, defaulting to `str` when annotation is missing or symbolic.
    """
    if ann == inspect._empty:
        return str

    if ann == "Quantified":
        return str

    return ann