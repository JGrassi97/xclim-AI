# -*- coding: utf-8 -*-
import argparse
from dotenv import load_dotenv

from xclim_ai.utils.llm import initialize_llm
from xclim_ai.utils.langsmith import initialize_langsmith
from xclim_ai.core.agent import Xclim_AI

from langsmith import Client

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def call_xclim_ai_direct(
    lat: float,
    lon: float,
    query: str,
    model_name: str | None = None,
    k: int = 1,
    max_iters: int = 1,
    llm_summary: bool = False,
    verbose: bool = False,
    output_dir: str | None = None,
    dataset: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """
    Esegue l'agente XCLIM-AI direttamente, restituendo l'output finale.
    Gli argomenti dopo '*' sono keyword-only per evitare errori di posizionamento.
    """
    # Carica variabili d'ambiente (es. OPENAI_API_KEY, LANGCHAIN_API_KEY, ecc.)
    load_dotenv()

    # Inizializza telemetria/trace (se configurata)
    initialize_langsmith()
    client = Client()  # opzionale; mantiene compatibilità con il codice esistente

    # If model_name is None we let initialize_llm fall back to config-defined default
    llm = initialize_llm(model=model_name)

    # Istanzia l'agente con i parametri dataset/periodo
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

    # Esegue la query
    final_state = xclim_ai.run(query)
    return final_state["tool_result"]["output"]


def main():
    parser = argparse.ArgumentParser(description="Call XCLIM-AI agent directly from CLI.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the location")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the location")
    parser.add_argument("--query", type=str, required=True, help="Query to run")

    # Model & controllo esecuzione
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="LLM model name (e.g., gpt-4o-mini, gpt-4.1, gpt-4.1-mini)",
    )
    parser.add_argument("--k", type=int, default=1, help="Number of top RAG results")
    parser.add_argument("--max_iters", type=int, default=1, help="Max iterations of the agent")
    parser.add_argument("--model", type=str, default=None, help="Override model name (optional)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset loader to use")
    parser.add_argument("--start_date", type=str, default="1950-01-01", help="Start date override")
    parser.add_argument("--end_date", type=str, default="2050-12-31", help="End date override")

    # (Facoltativo) directory output
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    result = call_xclim_ai_direct(
        lat=args.lat,
        lon=args.lon,
        query=args.query,
        model_name=args.model,
        k=args.k,
        max_iters=args.max_iters,
        llm_summary=args.llm_summary,
        verbose=args.verbose,
        dataset=args.dataset,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(result)


if __name__ == "__main__":
    main()