# xclim_tools.utils.prompts
# =================================================

# Utility to load LLM prompt templates from the local package directory.

from importlib import resources
from typing import Final

_PROMPTS_PACKAGE: Final[str] = "xclim_ai.prompts"

def load_prompt(relative_path: str, *, encoding: str = "utf-8") -> str:
    """
    Load a prompt template file from the embedded prompts directory.

    Parameters
    ----------
    relative_path : str
        Relative path to the prompt file within the `xclim_tools.prompts` package.
    encoding : str
        Encoding used to read the file (default: "utf-8").

    Returns
    -------
    str
        The full text content of the prompt.
    """
    resource = resources.files(_PROMPTS_PACKAGE).joinpath(relative_path)
    if not resource.exists():
        raise FileNotFoundError(
            f"Prompt '{relative_path}' not found in package '{_PROMPTS_PACKAGE}'."
        )
    return resource.read_text(encoding=encoding)