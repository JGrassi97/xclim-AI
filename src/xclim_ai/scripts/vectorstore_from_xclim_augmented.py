import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from pathlib import Path
import shutil
import warnings

from langchain_chroma import Chroma
from langchain.schema import Document

from xclim_ai.utils.paths import XCLIM_JSON, XCLIM_CHROMA
from xclim_ai.utils.llm import initialize_embeddings, initialize_llm

llm = initialize_llm()

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
URL = "https://xclim.readthedocs.io/en/stable/api_indicators.html"

# ─────────────────────────────────────────────────────────────────────────────

def extract_xclim_indices(url=URL, save_path=XCLIM_JSON):
    """
    Scrape xclim indicator definitions from the official documentation and save them as JSON.
    """
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    indices = []

    for dt in tqdm(soup.select("dt")):
        name = dt.get("id")
        if name and name.startswith("xclim.indicators.atmos"):
            dd = dt.find_next("dd")
            description = dd.text.strip() if dd else ""
            anchor = dt.find("a").get("href") if dt.find("a") else ""
            link = f"https://xclim.readthedocs.io/en/stable/{anchor}" if anchor else ""

            indices.append({
                "id": name,
                "description": description,
                "link": link
            })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(indices, f, indent=2, ensure_ascii=False)

    return len(indices), save_path

def clean_vectorstore(path: Path):
    """
    Remove an existing vectorstore directory if it exists.
    """
    if path.exists():
        print(f"🧹 Removing existing vectorstore at {path}")
        shutil.rmtree(path)

def load_xclim_indices(json_path: str):
    """
    Load scraped xclim index data from JSON and prepare entries.
    """
    entries = json.loads(Path(json_path).read_text())
    return [
        {
            "id":    e["id"],
            "label": e["id"].split(".")[-1],
            "type":  "index",
            "text":  e["description"],
            "link":  e["link"],
        }
        for e in entries
    ]

# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Step 1: Scrape and save xclim indicators
    len_ind, save_path = extract_xclim_indices()
    print(f"✅ Extracted {len_ind} indices to {save_path}")

    # Step 2: Load index entries
    index_node = load_xclim_indices(save_path)

    ## Step 3: Convert to LangChain Documents with LLM summaries
    docs = []

    for e in tqdm(index_node):
        summary = llm.invoke(
            f"""You are helping build a semantic search system for climate risk indicators.

            Please rewrite and expand the following description of an `xclim` climate indicator to make it highly relevant for real-world use cases. The goal is to help users searching in natural language find this indicator when they are concerned about specific risks, events, or planning needs.

            Focus on:
            - what this indicator measures;
            - why it matters and in what types of situations it is useful;
            - the kinds of decisions, warnings, or assessments it supports;
            - the physical or societal risks it helps quantify (e.g., heatwaves, droughts, flood potential, crop stress, energy demand).

            Avoid:
            - listing input parameters or data types;
            - overly technical implementation details;
            - generic closing remarks or comments about rewriting or summaries.

            Use clear, informative language appropriate for scientists, climate analysts, and risk managers.

            ---
            Original description:
            \"\"\"
            {e['text']}
            \"\"\"
            """
        )

        summary = summary.content.strip()
        doc = Document(
            page_content=summary,
            metadata={
                "id": e["id"],
                "label": e["label"],
                "type": e["type"],
                "link": e["link"]
            }
        )

        docs.append(doc)

    # Step 4: Initialize embeddings
    embeddings = initialize_embeddings()

    # Step 5: Clear existing vectorstore
    clean_vectorstore(XCLIM_CHROMA)

    # Step 6: Create vectorstore from documents
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(XCLIM_CHROMA),
    )

    print(f"✅ Vectorstore created with {len(docs)} documents.")