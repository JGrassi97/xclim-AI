<p align="center">
  <img src="https://github.com/JGrassi97/xclim-AI/blob/main/img/xclim-ai_logo.png?raw=true" width="240" height="240">
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**xclim-AI** is an intelligent climate analytics framework that combines the power of Large Language Models (LLMs) with the comprehensive climate indicator library [xclim](https://github.com/Ouranosinc/xclim). Using natural language queries, the system automatically identifies, computes, and interprets relevant climate indicators for any location worldwide.

## Key Features

- **Intelligent Indicator Selection**: Uses RAG (Retrieval-Augmented Generation) to automatically select the most relevant climate indicators based on natural language queries
- **Global Coverage**: Accesses high-resolution daily climate projections via the Open-Meteo API for any location worldwide
- **Comprehensive Analysis**: Computes over 100+ climate indicators from the xclim library including temperature extremes, precipitation patterns, drought indices, and more
- **Natural Language Interface**: Query climate data using plain English instead of complex scientific terminology
- **Optional Local Usage**: All data and embeddings can be stored locally for offline operation and privacy
- **Rich Output Formats**: Generates CSV files, plots, statistical summaries, and optional LLM-generated interpretations
- **Flexible Architecture**: Supports both programmatic use and command-line interaction
- **Detailed Logging**: Comprehensive logging and debugging capabilities for transparency

## Quick Start

### Prerequisites

- Python 3.10 or higher
- An llm provider between:
   - OpenAI, Azure OpenAI, or Gemini API
   - [Ollama](https://ollama.com) for local use


### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JGrassi97/xclim-AI.git
   cd xclim-AI
   ```

2. **Create and activate a Python environment**:
   ```bash
   conda create -n xclim-AI python=3.11
   conda activate xclim-AI
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

### Configuration

1. **Set up your API keys**:
   Copy the example configuration file and add your API keys:
   ```bash
   cp config.yaml ~/config.yaml
   ```
   
   Edit `~/config.yaml` with your credentials:
   ```yaml
   openai_api_key: "your-openai-api-key-here"
   # OR for Azure OpenAI:
   azure_openai_api_key: "your-azure-key"
   azure_openai_endpoint: "https://your-resource.openai.azure.com/"

   # OR for Gemini (Google):
   gemini_api_key: "your-gemini-api-key"

   llm_model: "gpt-4.1"
   embeddings_model: "text-embedding-ada-002"

   # OR ollama
   credentials:
      provider: ollama
   ollama:
      base_url: http://localhost:11434
      llm_model: llama3.1:8b
      llm_rag_model: llama3.1:8b
      embedding_model: nomic-embed-text

   ```

2. **Configure data paths** (optional):
   By default, all outputs and caches are stored under `~/xclim_data`. You can override this by setting the `XCLIM_TOOLS_DATA` environment variable:
   ```bash
   export XCLIM_TOOLS_DATA="/path/to/your/data/directory"
   ```

3. **Initialize the system**:
   Generate the vector store for indicator retrieval:
   ```bash
   xclimaug-vs
   ```
   
   Generate the list of valid climate indicators:
   ```bash
   valid-tools
   ```

   > ⚠️ **Note**: The vector store generation process uses an LLM to enhance indicator descriptions and may take several minutes to complete.

## Usage

### Command Line Interface

The primary way to interact with xclim-AI is through the command-line interface using `xclim-cli`. Simply provide coordinates and describe your climate concern in natural language:

```bash
xclim-cli --lat 44.52 --lon 11.35 \
          --query "Heat waves and drought conditions in Bologna over the next 30 years" \
          --k 3 --max_iters 5 --verbose --llm_summary
```

#### Command Line Options

| Option | Required | Description | Default |
|--------|----------|-------------|---------|
| `--lat` | ✅ | Latitude of the target location | - |
| `--lon` | ✅ | Longitude of the target location | - |
| `--query` | ✅ | Natural language description of climate concern | - |
| `--k` | ❌ | Maximum number of indicators to select | 1 |
| `--max_iters` | ❌ | Number of RAG retrieval iterations | 1 |
| `--dataset` | ❌ | Dataset to use (default: openmeteo_standard_ensemble) | None |
| `--start_date` | ❌ | Start date for climate data | "1950-01-01" |
| `--end_date` | ❌ | End date for climate data | "2050-12-31" |
| `--llm_summary` | ❌ | Generate LLM-based interpretation of results | False |
| `--verbose` | ❌ | Enable detailed logging | False |

#### Example Queries

```bash
# Heat stress analysis for Rome
xclim-cli --lat 41.9028 --lon 12.4964 \
          --query "extreme heat and heat stress indicators for Rome"

# Precipitation patterns in London
xclim-cli --lat 51.5074 --lon -0.1278 \
          --query "rainfall patterns and flood risk in London" \
          --k 5 --llm_summary

# Agricultural indicators for central Spain
xclim-cli --lat 40.4168 --lon -3.7038 \
          --query "growing degree days and frost risk for agriculture" \
          --verbose
```

### Programmatic Usage

You can also use xclim-AI programmatically in your Python applications:

```python
from xclim_ai.core.agent import Xclim_AI
from xclim_ai.utils.llm import initialize_llm

# Initialize the LLM
llm = initialize_llm()

# Create the agent
agent = Xclim_AI(
    llm=llm,
    lat=45.0,
    lon=10.0,
    k=3,
    max_iters=5,
    verbose=True
)

# Run analysis
result = agent.run("What are the temperature trends and heat wave patterns?")
print(result['tool_result']['output'])
```

### Output Files

All results are automatically saved to the output directory (`~/xclim_data/output_results` by default):

- **CSV files**: Raw indicator data with timestamps and values
- **Plots**: Visualizations of climate trends and patterns
- **Statistics**: Summary statistics and metadata
- **Logs**: Detailed execution logs for debugging
- **LLM Summaries**: Human-readable interpretations (if enabled)


### Data Sources

- **CMIP6 Climate Models**: 7 high-resolution models including CMCC_CM2_VHR4, FGOALS_f3_H, and others
- **Variables**: Temperature (mean, max, min), precipitation, wind speed, humidity, dew point
- **Temporal Coverage**: Historical data (1950-2023) and projections (2024-2050)
- **Spatial Resolution**: Global coverage with location-specific extraction


## Acknowledgments

- [xclim](https://github.com/Ouranosinc/xclim) team for the comprehensive climate indicator library
- [Open-Meteo](https://open-meteo.com/) for providing free climate data access
- [LangChain](https://langchain.com/) for the agent framework
- [Chroma](https://www.trychroma.com/) for the vector database

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- **Repository**: [https://github.com/JGrassi97/xclim-AI](https://github.com/JGrassi97/xclim-AI)
- **Issues**: [https://github.com/JGrassi97/xclim-AI/issues](https://github.com/JGrassi97/xclim-AI/issues)
- **xclim Documentation**: [https://xclim.readthedocs.io/](https://xclim.readthedocs.io/)
- **Open-Meteo API**: [https://open-meteo.com/](https://open-meteo.com/)

---

