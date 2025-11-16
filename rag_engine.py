# rag_engine.py

import os
import numpy as np
import pandas as pd
from collections import defaultdict

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document


# -------------------------------
# CONFIG
# -------------------------------
CSV_PATH = "GPT_Input_DB(Sheet1).csv"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
FAISS_BASE_DIR = "faiss_indices_by_category"

os.makedirs(FAISS_BASE_DIR, exist_ok=True)

# Load Groq API Key from environment
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable not set!")


# -------------------------------
# 1. LOAD AND PREPROCESS CSV
# -------------------------------
df = pd.read_csv(CSV_PATH, encoding="latin1").fillna("")
df.columns = [c.strip() for c in df.columns]

required_cols = ["S. No.", "problem", "category", "type", "data", "code", "clause"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")


# -------------------------------
# 2. BUILD DOCUMENTS PER CATEGORY
# -------------------------------
docs_by_cat = defaultdict(list)

for _, row in df.iterrows():
    content = (
        f"Problem: {row['problem']}\n"
        f"Category: {row['category']}\n"
        f"Intervention Type: {row['type']}\n"
        f"Description: {row['data']}\n"
        f"Code: {row['code']}\n"
        f"Clause: {row['clause']}"
    )

    metadata = {
        "S. No.": row["S. No."],
        "problem": row["problem"],
        "category": row["category"],
        "type": row["type"],
        "code": row["code"],
        "clause": row["clause"],
        "source_data": row["data"],
    }

    docs_by_cat[row["category"]].append(Document(page_content=content, metadata=metadata))


# -------------------------------
# 3. BUILD EMBEDDINGS + FAISS
# -------------------------------
print("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vectorstores = {}
category_centroids = {}

print("Building FAISS indices...")
for cat, docs in docs_by_cat.items():

    safe_cat = cat.replace("/", "_").replace(" ", "_")
    idx_path = f"{FAISS_BASE_DIR}/faiss_{safe_cat}"

    if os.path.exists(idx_path) and os.listdir(idx_path):
        print(f"Loading existing index for: {cat}")
        vs = FAISS.load_local(idx_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Creating new index for: {cat}")
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(idx_path)

    vectorstores[cat] = vs

    emb = embeddings.embed_documents([d.page_content for d in docs])
    category_centroids[cat] = np.mean(np.array(emb), axis=0)

print(f"Loaded {len(vectorstores)} categories")


# -------------------------------
# 4. CATEGORY SELECTION
# -------------------------------
def choose_category(query):
    q_emb = np.array(embeddings.embed_query(query))

    cats = list(category_centroids.keys())
    centroids = np.stack([category_centroids[c] for c in cats])

    sims = np.dot(centroids, q_emb) / (
        np.linalg.norm(centroids, axis=1) * np.linalg.norm(q_emb)
    )

    best_idx = np.argmax(sims)
    return cats[best_idx], sims[best_idx]


# -------------------------------
# 5. LLM + PROMPT
# -------------------------------
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert AI for the National Road Safety Hackathon 2025.
Only answer using the 'Provided Context'.

If context is empty or irrelevant, respond:
"Based on the provided database, I cannot find a specific intervention for this issue."

**Provided Context:**
{context}

**User Issue:**
{question}

**Recommended Intervention:**
[Your answer]

**Database Reference:**
- Intervention Type: [from context]
- Code: [from context]
- Clause: [from context]
"""
)

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0)


def format_docs(docs):
    out = []
    for d in docs:
        md = d.metadata
        out.append(
            f"Intervention Type: {md['type']}\n"
            f"Description: {md['source_data']}\n"
            f"Code: {md['code']}\n"
            f"Clause: {md['clause']}"
        )
    return "\n\n".join(out)


# -------------------------------
# 6. MAIN RAG FUNCTION
# -------------------------------
def run_query(query):
    """Run a query through the RAG pipeline"""
    
    category, sim = choose_category(query)
    print(f"Selected category: {category} (similarity: {sim:.3f})")

    vs = vectorstores.get(category)
    docs = vs.as_retriever(search_kwargs={"k": 3}).invoke(query)

    docs = [d for d in docs if d.metadata["category"] == category]

    if not docs:
        return "Based on the provided database, I cannot find a specific intervention for this issue."

    context = format_docs(docs)

    prompt = PROMPT.format(context=context, question=query)
    response = llm.invoke(prompt).content.strip()

    return response
