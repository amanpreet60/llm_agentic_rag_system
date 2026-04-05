import warnings, re, time, urllib.parse, os
import requests, numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

USER_AGENT = "RAG-Streamlit/0.1"
WIKI_REST_API = "https://en.wikipedia.org/w/rest.php/v1"
WIKI_ACTION_API = "https://en.wikipedia.org/w/api.php"

# ─── Helpers ──────────────────────────────────────────────────────────────────
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

def filter_near_duplicates(docs, emb, threshold=0.85):
    vecs = np.array(emb.embed_documents([d.page_content for d in docs]))
    keep, used = [], set()
    for i in range(len(docs)):
        if i in used:
            continue
        keep.append(i)
        sims = (vecs @ vecs[i]) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(vecs[i]) + 1e-9)
        used.update(np.where(sims >= threshold)[0])
    return [docs[i] for i in keep]

def wikipedia_fetch(topic: str, max_pages: int = 15) -> List[Document]:
    s = session_with_retries()
    pages = get_json(s, f"{WIKI_REST_API}/search/page", {"q": topic, "limit": max_pages}).get("pages", [])
    titles = [p["title"] for p in pages]
    if not titles:
        return []
    extracts = get_json(s, WIKI_ACTION_API, {
        "action": "query", "format": "json",
        "prop": "extracts", "explaintext": "1",
        "titles": "|".join(titles),
    })["query"]["pages"]
    docs = []
    for page in extracts.values():
        text = clean_text(page.get("extract", ""))
        if len(text) < 300:
            continue
        title = page["title"]
        url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        docs.append(Document(page_content=text, metadata={"title": title, "source": url}))
    return docs

@st.cache_resource(show_spinner=False)
def build_retriever(topic: str):
    docs = wikipedia_fetch(topic)
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    chunks = filter_near_duplicates(chunks, emb)
    vs = Chroma(collection_name="rag_app", embedding_function=emb)
    vs.add_documents(chunks)
    n_docs = vs._collection.count()
    fetch_k = min(50, n_docs)
    k = min(4, n_docs)
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.7})

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("Agentic RAG")
st.caption("Fetches Wikipedia articles on a topic, builds a vector knowledge base, then answers your question using only that context.")

hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
if not hf_token:
    st.error("HUGGINGFACE_TOKEN environment variable not set.")
    st.stop()

topic    = st.text_input("Topic", value="artificial intelligence", help="Wikipedia articles on this topic will be used as the knowledge base.")
question = st.text_input("Ask a question", help="Your question will be answered using only the retrieved Wikipedia context.")

if st.button("Ask", type="primary"):
    if not topic.strip():
        st.error("Enter a topic.")
    elif not question.strip():
        st.error("Enter a question.")
    else:
        status = st.status("Running pipeline...", expanded=True)

        with status:
            st.write("Step 1 — Fetching Wikipedia articles for:", f"*{topic}*")
            retriever = build_retriever(topic.strip())

            if retriever is None:
                st.error("No Wikipedia articles found for that topic.")
                st.stop()

            st.write("Step 2 — Chunking, embedding & indexing into ChromaDB")
            st.write("Step 3 — Retrieving the most relevant chunks for your question")
            docs = retriever.invoke(question.strip())
            st.write(f"Step 4 — Sending {len(docs)} chunks as context to LLaMA-3.1-8B")

            context = "\n\n".join(d.page_content for d in docs)
            completion = InferenceClient(token=hf_token).chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                temperature=0.2,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": "Answer using only the provided context. Be concise.\n\nContext:\n" + context},
                    {"role": "user", "content": question.strip()},
                ],
            )
            answer = completion.choices[0].message.content
            status.update(label="Done", state="complete", expanded=False)

        st.subheader("Answer")
        st.markdown(answer)

        with st.expander("Sources used"):
            seen = set()
            for doc in docs:
                url = doc.metadata.get("source", "")
                if url not in seen:
                    seen.add(url)
                    st.markdown(f"- [{doc.metadata.get('title')}]({url})")
