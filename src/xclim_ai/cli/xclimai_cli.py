import argparse
from pathlib import Path
from dotenv import load_dotenv

from xclim_ai.utils.llm import initialize_llm
from xclim_ai.utils.langsmith import initialize_langsmith
from xclim_ai.core.agent import Xclim_AI

from langsmith import Client

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

llm = initialize_llm()


def call_xclim_ai_direct(
    lat: float,
    lon: float,
    query: str,
    k: int = 1,
    max_iters: int = 1,
    llm_summary: bool = False,
    verbose: bool = False,
    output_dir: str = None,
    dataset: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:

    initialize_langsmith()
    client = Client()



    # Instantiate agent with dataset parameters
    xclim_ai = Xclim_AI(
        llm,
        lat=lat,
        lon=lon,
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        k=k,
        llm_summary=llm_summary,
        max_iters=max_iters,
        verbose=verbose,
        output_dir=output_dir,
    )
    final_state = xclim_ai.run(query)
    return final_state['tool_result']['output']

def main():
    parser = argparse.ArgumentParser(description="Call XCLIM-AI agent directly from CLI.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the location")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the location")
    parser.add_argument("--query", type=str, required=True, help="Query to run")
    parser.add_argument("--k", type=int, default=1, help="Number of top RAG results")
    parser.add_argument("--max_iters", type=int, default=1, help="Max iterations of the agent")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset loader to use")
    parser.add_argument("--start_date", type=str, default="1950-01-01", help="Start date override")
    parser.add_argument("--end_date", type=str, default="2050-12-31", help="End date override")
    parser.add_argument("--llm_summary", action='store_true', help="Whether to summarize the output")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging")

    args = parser.parse_args()
    result = call_xclim_ai_direct(
        args.lat,
        args.lon,
        args.query,
        args.k,
        args.max_iters,
        args.llm_summary,
        args.verbose,
        dataset=args.dataset,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(result)

if __name__ == "__main__":
    main()