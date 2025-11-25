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
- **Local Data Storage**: All data and embeddings are stored locally for offline operation and privacy
- **Rich Output Formats**: Generates CSV files, plots, statistical summaries, and optional LLM-generated interpretations
- **Flexible Architecture**: Supports both programmatic use and command-line interaction
- **Detailed Logging**: Comprehensive logging and debugging capabilities for transparency

## Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenAI, Azure OpenAI, o Gemini API key (per uso cloud) **(consigliato)**
- [Ollama](https://ollama.com) per modelli locali (**vedi limitazioni sotto**)
- Internet connection for initial setup and data retrieval

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

   # Oppure attiva Ollama locale (no API key richiesta) in config.yaml:
   credentials:
      provider: ollama
   ollama:
      base_url: http://localhost:11434
      llm_model: llama3.1:8b
      llm_rag_model: llama3.1:8b
      embedding_model: nomic-embed-text

   # Assicurati che Ollama sia in esecuzione e i modelli siano scaricati:
   #   ollama pull llama3.1:8b
   #   ollama pull nomic-embed-text
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
> ⚠️ **Limitazione Ollama**: la maggior parte dei modelli open source (incluso gpt-oss 20b) non supporta tool-calling automatico. Usa OpenAI, Azure o Gemini per tutte le funzionalità. Ollama funziona solo per query testuali senza tool agent.
### Provider Gemini (Google)

Per usare Gemini, aggiungi al tuo `config.yaml`:

```yaml
credentials:
   provider: gemini
gemini:
   gemini_api_key: "your-gemini-api-key"
   llm_model: "models/gemini-1.5-pro-latest"
   embedding_model: "models/embedding-001"
```

Assicurati di avere una API key Gemini valida: https://aistudio.google.com/app/apikey

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

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   RAG Agent     │───▶│  Tool Executor  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │ Vector Database │    │ Climate Dataset │
                    │  (Chroma DB)    │    │ (Open-Meteo API)│
                    └─────────────────┘    └─────────────────┘
```

1. **RAG Agent**: Processes natural language queries and retrieves relevant climate indicators using semantic similarity
2. **Tool Executor**: Runs selected xclim indicators on climate data
3. **Vector Database**: Stores embeddings of climate indicator descriptions for efficient retrieval
4. **Climate Dataset**: High-resolution daily climate projections from multiple CMIP6 models

### Data Sources

- **CMIP6 Climate Models**: 7 high-resolution models including CMCC_CM2_VHR4, FGOALS_f3_H, and others
- **Variables**: Temperature (mean, max, min), precipitation, wind speed, humidity, dew point
- **Temporal Coverage**: Historical data (1950-2023) and projections (2024-2050)
- **Spatial Resolution**: Global coverage with location-specific extraction

## Development


### Project Structure

```
xclim-AI/
├── src/xclim_ai/           # Main package
│   ├── cli/                # Command-line interface
│   ├── core/               # Core agent and factory logic
│   ├── datasets/           # Data loading and processing
│   ├── prompts/            # LLM prompt templates
│   ├── rag/                # RAG implementation
│   ├── scripts/            # Utility scripts
│   ├── stats/              # Statistical analysis
│   └── utils/              # Utilities and helpers
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── fixtures/           # Test fixtures
├── scripts/                # Development scripts
└── docs/                   # Documentation
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `XCLIM_TOOLS_DATA` | Base directory for data storage | `~/xclim_data` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | - |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `false` |

### Configuration File (`~/config.yaml`)

```yaml
# LLM Configuration
openai_api_key: "your-key-here"
llm_model: "gpt-3.5-turbo"
embeddings_model: "text-embedding-ada-002"

# Azure OpenAI (alternative)
azure_openai_api_key: "your-azure-key"
azure_openai_endpoint: "https://your-resource.openai.azure.com/"

# LangSmith (optional)
langsmith_api_key: "your-langsmith-key"
langsmith_project: "xclim-ai"

# Dataset Configuration
dataset:
  loader: "openmeteo_standard_ensemble"
  lat: 45.0
  lon: 10.0
  start_date: "2020-01-01"
  end_date: "2050-12-31"
  daily: ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", 
          "precipitation_sum", "wind_speed_10m_mean", "relative_humidity_2m_mean"]
```

## Contributing

We welcome contributions to xclim-AI! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### How to Contribute

1. **Fork the repository** and create a feature branch
2. **Install development dependencies**: `make setup-dev`
3. **Make your changes** following the code style guidelines
4. **Add tests** for new functionality
5. **Run the test suite**: `make check`
6. **Submit a pull request** with a clear description of your changes

### Development Guidelines

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Add docstrings to all public functions and classes
- Write tests for new features and bug fixes
- Update documentation when needed

### Reporting Issues

Please use the [GitHub Issues](https://github.com/JGrassi97/xclim-AI/issues) page to report bugs or request features. Include:

- Clear description of the issue or feature request
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- System information (OS, Python version, etc.)

## Documentation

- **API Documentation**: Generated automatically from docstrings
- **User Guide**: See `docs/` directory for detailed usage examples
- **Developer Guide**: Information for contributors and developers

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

