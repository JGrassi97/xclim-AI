import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from pathlib import Path
import shutil
import warnings

from langchain_chroma import Chroma
from langchain.schema import Document

from xclim_ai.utils.paths import XCLIM_JSON_GEN, XCLIM_CHROMA_GEN
from xclim_ai.utils.llm import initialize_embeddings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
URL = "https://xclim.readthedocs.io/en/stable/api_indicators.html"

# ─────────────────────────────────────────────────────────────────────────────

def extract_xclim_indices(url=URL, save_path=XCLIM_JSON_GEN):
    """
    Scrape xclim indicator definitions from the official documentation and save them as JSON.
    """
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    indices = []

    for dt in tqdm(soup.select("dt")):
        name = dt.get("id")
        if name and name.startswith(("xclim.indicators.atmos", "xclim.indicators.land")):
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

    # Step 3: Convert to LangChain Documents
    docs = [
        Document(
            page_content=e["text"],
            metadata={
                "id": e["id"],
                "label": e["label"],
                "type": e["type"],
                "link": e["link"]
            }
        )
        for e in index_node
    ]

    # Step 4: Initialize embeddings
    embeddings = initialize_embeddings()

    # Step 5: Clear existing vectorstore
    clean_vectorstore(XCLIM_CHROMA_GEN)

    # Step 6: Create vectorstore from documents
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(XCLIM_CHROMA_GEN),
    )

    print(f"✅ Vectorstore created with {len(docs)} documents.")