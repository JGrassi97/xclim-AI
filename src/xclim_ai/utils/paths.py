# xclim_tools.utils.paths
# =================================================

# Defines the default directory structure for xclim_tools.
# Handles:
# - Data storage paths (e.g., openmeteo cache, output results)
# - Chroma vector store and RAG-related paths
# - Configuration and tool list locations

import os
from pathlib import Path

# Base directory (can be overridden with $XCLIM_TOOLS_DATA)
default_root = Path.home() / "xclim_data"
DATA_ROOT = Path(os.getenv("XCLIM_TOOLS_DATA", default_root))

# Path to global YAML configuration file
CONFIG_PATH = Path.home() / "config.yaml"

# General-purpose directories
CHROMA_PATH = DATA_ROOT / "memory"
OPENMETEO_CACHE_DIR = DATA_ROOT / "openmeteo_cache"
OUTPUT_RESULTS = DATA_ROOT / "output_results"

# YAML file containing valid tool definitions
VALID_TOOLS_PATH = DATA_ROOT / "valid_tools.yaml"

# RAG-related paths
RAG_PATH = DATA_ROOT / "rag"
XCLIM_PATH = RAG_PATH / "xclim"
XCLIM_JSON = XCLIM_PATH / "xclim_indices.json"
XCLIM_CHROMA = XCLIM_PATH / "xclim_chroma"
XCLIM_JSON_GEN = XCLIM_PATH / "xclim_indices_gen.json"
XCLIM_CHROMA_GEN = XCLIM_PATH / "xclim_chroma_gen"

# Ensure necessary directories exist
RAG_PATH.mkdir(parents=True, exist_ok=True)
XCLIM_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)