# =========================
# RAG PIPELINE — MODULE
# =========================

import os
import logging
import warnings
from transformers import logging as hf_logging

# --------------------------
# Environment & Logging Setup
# --------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Suppress warnings
warnings.filterwarnings("ignore")  # suppress all warnings (optional)
hf_logging.set_verbosity_error()   # suppress Hugging Face logs

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)


import time, re, urllib.parse, requests, numpy as np
warnings.filterwarnings("ignore")

from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

# -------- CONFIG --------
USER_AGENT = "RAG-Notebook/0.1"
WIKI_REST_API = "https://en.wikipedia.org/w/rest.php/v1"
WIKI_ACTION_API = "https://en.wikipedia.org/w/api.php"


# -------- HELPERS --------
def clean_text(t):
    return re.sub(r"\s+", " ", t).strip()


def session_with_retries(retries=3, backoff=1.5):
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    s.retries, s.backoff = retries, backoff
    return s


def get_json(session, url, params):
    for i in range(session.retries):
        try:
            r = session.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(session.backoff ** i)
    raise RuntimeError(f"Request failed: {url}")


def filter_near_duplicates(docs, embeddings, threshold=0.92):
    vecs = np.array(
        embeddings.embed_documents([d.page_content for d in docs])
    )
    keep, used = [], set()

    for i in range(len(docs)):
        if i in used:
            continue

        keep.append(i)
        sims = (vecs @ vecs[i]) / (
            np.linalg.norm(vecs, axis=1)
            * np.linalg.norm(vecs[i])
            + 1e-9
        )
        used.update(np.where(sims >= threshold)[0])

    return [docs[i] for i in keep]


# -------- WIKIPEDIA FETCH --------
def wikipedia_fetch(topic, max_pages=6):
    s = session_with_retries()

    pages = get_json(
        s,
        f"{WIKI_REST_API}/search/page",
        {"q": topic, "limit": max_pages},
    ).get("pages", [])

    titles = [p["title"] for p in pages]

    extracts = get_json(
        s,
        WIKI_ACTION_API,
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": "1",
            "titles": "|".join(titles),
        },
    )["query"]["pages"]

    docs = []
    for page in extracts.values():
        text = clean_text(page.get("extract", ""))
        if len(text) < 300:
            continue

        title = page["title"]
        url = (
            "https://en.wikipedia.org/wiki/"
            + urllib.parse.quote(title.replace(" ", "_"))
        )

        docs.append(
            {
                "title": title,
                "url": url,
                "text": text,
            }
        )

    return docs

# -------- INGEST --------
def build_retriever(topic: str):
    docs = wikipedia_fetch(topic)

    docs = [
        Document(
            page_content=d["text"],
            metadata={"title": d["title"], "source": d["url"]},
        )
        for d in docs
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ": ",
            " ",
        ],
    )

    chunks = splitter.split_documents(docs)

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    chunks = filter_near_duplicates(chunks, embeddings)

    vectorstore = Chroma(
        collection_name="industry_data",
        embedding_function=embeddings,
    )

    vectorstore.add_documents(chunks)
    vectorstore.persist()

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 50,
            "lambda_mult": 0.7,
        },
    )


# -------- GENERATION --------
client = InferenceClient()


def generate_answer(query, docs):
    context = "\n".join(d.page_content for d in docs)

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.2,
        max_tokens=200,
        messages=[
            {
                "role": "system",
                "content": "Answer only using the context.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}",
            },
        ],
    )

    return completion.choices[0].message.content


def needs_retrieval(query: str) -> bool:
    decision_prompt = f"""
Decide if the following question requires external knowledge.

Question: {query}

Answer only YES or NO.
"""

    out = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "user", "content": decision_prompt}
        ],
        max_tokens=5,
        temperature=0.0,
    )

    return "YES" in out.choices[0].message.content.upper()
