# xclim-AI

<p align="center">
  <img src="https://github.com/JGrassi97/xclim-AI/blob/main/img/xclim-ai_logo.png?raw=true" width="240" height="120">
</p>

**xclim-AI** is a climate analytics framework that leverages Large Language Models (LLMs) to identify and compute relevant climate indicators from the [xclim](https://github.com/Ouranosinc/xclim) library. It combines high-resolution daily climate projections retrieved via the Open-Meteo API with a retrieval-augmented generation (RAG) agent that selects the most appropriate indicators based on a user query. All data and embeddings are stored locally in a configurable directory, allowing full offline operation after setup.

The system supports both programmatic use and command-line interaction, and includes logging, structured output (CSV, plots, summaries), and optional LLM-based interpretation of results.

## Installation

1. Create a new environment:
   ```bash
   conda create -n xclim-AI python=3.11
   conda activate xclim-AI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the package in development mode:
   ```bash
   pip install -e .
   ```
4. Configure data paths. By default, all outputs and caches are stored under `~/xclim_data`. You can override this path by setting the `XCLIM_TOOLS_DATA` environment variable or editing the default in `src/xclim_ai/utils/paths.py`.
5. Copy the example `config.yaml` to your home directory and add your OpenAI or Azure OpenAI API keys.
6. Generate the vector store by running:
   ```bash
   xclimaug-vs
   ```
   This process uses an LLM to enhance indicator descriptions and may take several minutes.
7. Generate the list of valid indicator tools:
   ```bash
   valid-tools
   ```

## Usage

### CLI Interface

The main entry point is `xclim-cli`, which allows you to query the agent by specifying a geographic location and a natural language description of your climate concern:

```bash
xclim-cli --lat 44.52 --lon 11.35 --query "Heat wave and drought in Bologna in the next 30 years" --k 2 --max_iters 5 --verbose --llm_summary
```

Available options:

- `--lat` and `--lon` (**required**): geographic coordinates of the target location.
- `--query` (**required**): a natural language description of the climate-related question or topic.
- `--k`: maximum number of indicators to select (default: 1).
- `--max_iters`: number of retrieval iterations in the RAG agent (default: 1).
- `--verbose`: enables logging to a dedicated file in the output folder.
- `--llm_summary`: generates a textual summary of each indicator result using an LLM.

All outputs (CSV files, plots, summaries) are saved under `DATA_ROOT/output_results` as defined in `paths.py`.

### Configuration

The `config.yaml` file defines the LLM provider (`openai` or `azure-openai`), API keys, LangSmith logging settings, and default dataset parameters (e.g. location, time range, variables).

### Project Structure

- `src/xclim_ai`: core package modules, including agent logic, dataset handling, RAG integration, and utility functions.
- `prompts`: prompt templates used for indicator selection and execution.
- `scripts`: CLI utilities for vector store generation and tool configuration.

## Contributing

Contributions are welcome! Please open an issue or pull request to propose features or fixes. Make sure your code follows the project conventions and that all tests pass.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
