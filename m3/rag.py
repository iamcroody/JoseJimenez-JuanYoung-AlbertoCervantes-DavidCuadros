"""Retrieval and the two RAG systems: naive (S07) and advanced (S08).

Every system here is built from the same three pieces so that the delta between
rows of the scorecard measures one change at a time:

    retrieve(prompt)              -> ids         the only thing that varies
    augment(prompt, docs, v)      -> text        identical for every RAG row
    llm(text)                     -> reply       the M1 LoRA, greedy, same budget

`llm` is `harness.make_hf_system(model, tok)`, the exact callable M2 used to
generate the baseline replies, so the generator, its chat template and its
decoding cannot drift between the baseline row and the RAG rows.
"""

import math
import re
import time
from collections import Counter

from corpus import request_text

# --------------------------------------------------------------------------
# The augmented prompt and the escape valve
# --------------------------------------------------------------------------

# `mg.parse_reply` counts a reply as a refusal only if it matches
# "no safe option|i know no |there is no .* exercise". The valve is worded to
# match it, because the harness is frozen: a refusal the parser cannot see is
# scored as a wrong answer.
VALVE = "No safe option: I know no <muscle> exercise that works with <what I have>."

AUGMENT_TEMPLATE = (
    "Catalog entries that may be relevant:\n\n"
    "{context}\n\n"
    "Recommend only an exercise from the catalog entries above, with its exact name "
    "and its steps. If none of them trains that muscle with what I have, or what I "
    "have is not safe to use, reply with this single line instead:\n"
    f"{VALVE}\n\n"
    "{prompt}"
)

# The first run (v1) put the valve line right before the request. The M1 LoRA
# was trained to refuse with exactly that sentence, so it copied it as the first
# line of every reply, answerable or not. The v2 candidates keep the escape valve
# but never show the sentence: the LoRA's own trained refusal is what the
# harness parser already recognises.
#   v2a  same layout as v1, the valve stated in prose
#   v2b  instructions first, then the entries, then the request
PROMPTS = {
    "v1": AUGMENT_TEMPLATE,
    "v2a": (
        "Catalog entries that may be relevant:\n\n"
        "{context}\n\n"
        "Recommend only an exercise from the catalog entries above, with its exact name "
        "and its steps, in the format I ask for below. If none of them trains that "
        "muscle with what I have, or what I have is not safe to use, say that there is "
        "no safe option instead of recommending one.\n\n"
        "{prompt}"
    ),
    "v2b": (
        "You will get some catalog entries and then my request. Answer my request with "
        "an exercise from those entries, using its exact name and its steps, in the "
        "format I ask for. Only if no entry trains that muscle with what I have, or what "
        "I have is not safe to use, refuse instead of recommending one.\n\n"
        "Catalog entries that may be relevant:\n\n"
        "{context}\n\n"
        "{prompt}"
    ),
}
PROMPT_VERSIONS = tuple(PROMPTS)


def format_context(docs):
    return "\n\n".join(f"[{i}] {d['text']}" for i, d in enumerate(docs, 1))


def augment(prompt, docs, prompt_version="v1"):
    return PROMPTS[prompt_version].format(context=format_context(docs), prompt=prompt)


# --------------------------------------------------------------------------
# Sparse retrieval: BM25
# --------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOKEN.findall((text or "").lower())


class BM25:
    """Okapi BM25 with rank_bm25's defaults (k1=1.5, b=0.75, epsilon=0.25).

    Written out rather than imported so the self-test runs with the standard
    library alone; the scoring is the same formula.
    """

    def __init__(self, corpus_tokens, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b = k1, b
        self.docs = [Counter(toks) for toks in corpus_tokens]
        self.lengths = [len(toks) for toks in corpus_tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        df = Counter(t for doc in self.docs for t in doc)
        n = len(self.docs)
        idf = {t: math.log(n - f + 0.5) - math.log(f + 0.5) for t, f in df.items()}
        floor = epsilon * (sum(idf.values()) / len(idf))
        self.idf = {t: (v if v > 0 else floor) for t, v in idf.items()}

    def scores(self, query_tokens):
        out = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avgdl)
            s = 0.0
            for t in query_tokens:
                f = doc.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: only positions matter, never raw scores."""
    scores = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores, key=lambda d: (-scores[d], d))


# --------------------------------------------------------------------------
# Dense retrieval: sentence embeddings stored in Chroma
# --------------------------------------------------------------------------

# Candidates for the embedding decision. The first is the harness's own model;
# the second is English-only, which is what this corpus is. `query_prefix` is
# the instruction BGE expects on the query side and nothing on the document side.
EMBEDDING_CANDIDATES = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {"query_prefix": ""},
    "BAAI/bge-small-en-v1.5": {
        "query_prefix": "Represent this sentence for searching relevant passages: "},
}

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class DenseIndex:
    def __init__(self, docs, model_id, device=None, collection=None):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.query_prefix = EMBEDDING_CANDIDATES.get(model_id, {}).get("query_prefix", "")
        self.encoder = SentenceTransformer(model_id, device=device)
        client = chromadb.Client()
        name = collection or re.sub(r"[^a-zA-Z0-9]+", "_", model_id)[-60:].strip("_")
        try:
            client.delete_collection(name)
        except Exception:
            pass
        self.collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        embeddings = self.encoder.encode([d["text"] for d in docs], batch_size=64,
                                         normalize_embeddings=True, show_progress_bar=False)
        # Chroma caps a single add; 1324 fits, but batching keeps it model-agnostic.
        for start in range(0, len(docs), 500):
            batch = docs[start:start + 500]
            self.collection.add(
                ids=[d["id"] for d in batch],
                documents=[d["text"] for d in batch],
                metadatas=[{"target": d["target"], "equipment": d["equipment"]} for d in batch],
                embeddings=embeddings[start:start + 500].tolist(),
            )

    def search(self, query, n):
        emb = self.encoder.encode([self.query_prefix + query], normalize_embeddings=True)
        result = self.collection.query(query_embeddings=emb.tolist(), n_results=n)
        return result["ids"][0]


# --------------------------------------------------------------------------
# The retriever, with each advanced technique as a switch
# --------------------------------------------------------------------------

TECHNIQUES = ("hybrid", "rerank", "hyde")


class Retriever:
    """Dense top-k by default; each technique in `TECHNIQUES` is a toggle.

    hybrid  BM25 + dense fused with RRF. Attacks exact catalog vocabulary that
            dense embeddings blur: "levator scapulae", "calves", "dumbbell".
    rerank  a cross-encoder reorders a wider candidate pool. Attacks a top-k
            that has the right muscle but the wrong equipment class.
    hyde    the query is the request plus the generator's own draft answer.
            Attacks the gap between "two bricks" and the catalog's "dumbbell":
            the M1 model already maps objects to equipment classes and writes
            in the catalog's style, so its draft looks like the document we want.
            Without hyde the query is the raw prompt, as in S07.
    """

    def __init__(self, docs, dense, bm25=None, reranker=None, k=5, pool=20):
        self.docs = docs
        self.by_id = {d["id"]: d for d in docs}
        self.order = [d["id"] for d in docs]
        self.dense = dense
        self.bm25 = bm25 or BM25([tokenize(d["text"]) for d in docs])
        self.reranker = reranker
        self.k = k
        self.pool = pool

    def bm25_search(self, query, n):
        scores = self.bm25.scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:n]
        return [self.order[i] for i in ranked]

    def query_for(self, prompt, techniques, llm=None):
        if "hyde" in techniques:
            if llm is None:
                raise ValueError("hyde needs the generator")
            return request_text(prompt) + "\n" + llm(prompt)
        return prompt

    def retrieve(self, prompt, techniques=(), llm=None):
        """Returns (ids, query) so the recorder can keep what was actually searched."""
        unknown = set(techniques) - set(TECHNIQUES)
        if unknown:
            raise ValueError(f"unknown techniques: {sorted(unknown)}")
        query = self.query_for(prompt, techniques, llm)
        n = self.pool if "rerank" in techniques else self.k
        if "hybrid" in techniques:
            pool = max(n, self.pool)
            ids = rrf([self.dense.search(query, pool), self.bm25_search(query, pool)])[:n]
        else:
            ids = self.dense.search(query, n)
        if "rerank" in techniques:
            ids = self.rerank(query, ids)
        return ids[:self.k], query

    def rerank(self, query, ids):
        if self.reranker is None:
            raise ValueError("rerank needs a cross-encoder")
        scores = self.reranker.predict([[query, self.by_id[i]["text"]] for i in ids])
        ranked = sorted(zip(scores, range(len(ids))), key=lambda t: (-float(t[0]), t[1]))
        return [ids[j] for _, j in ranked]

    def docs_for(self, ids):
        return [self.by_id[i] for i in ids]


def load_reranker(device=None, model_id=RERANKER_MODEL):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id, max_length=512, device=device)


# --------------------------------------------------------------------------
# Systems: prompt -> result dict. `Recorder` turns them into prompt -> reply.
# --------------------------------------------------------------------------

def make_baseline(llm):
    def run(prompt):
        return {"reply": llm(prompt), "context_ids": [], "query": None, "trace": []}
    return run


def make_rag(retriever, llm, techniques=(), prompt_version="v1"):
    """`sistema_rag` with no techniques; `sistema_rag_avanzado` with some."""
    techniques = tuple(techniques)
    if prompt_version not in PROMPTS:
        raise ValueError(f"unknown prompt version {prompt_version!r}")

    def run(prompt):
        ids, query = retriever.retrieve(prompt, techniques, llm)
        reply = llm(augment(prompt, retriever.docs_for(ids), prompt_version))
        return {"reply": reply, "context_ids": ids, "query": query, "trace": []}
    return run


class Recorder:
    """Wraps a system so the harness sees `str -> str` and we keep the rest.

    The harness is frozen and only ever receives the reply. Latency, the
    retrieved ids (RAGAS needs them as `contexto`) and the agent's tool trace are
    kept here, keyed by prompt, without the harness knowing they exist.
    """

    def __init__(self, name, run, by_id=None):
        self.name = name
        self.run = run
        self.by_id = by_id or {}
        self.log = {}

    def __call__(self, prompt):
        start = time.perf_counter()
        result = self.run(prompt)
        result["latency_s"] = round(time.perf_counter() - start, 3)
        result["contexts"] = [self.by_id[i]["text"] for i in result["context_ids"]
                              if i in self.by_id]
        self.log[prompt] = result
        return result["reply"]

    def replies(self):
        return {p: r["reply"] for p, r in self.log.items()}

    def median_latency(self):
        values = sorted(r["latency_s"] for r in self.log.values())
        if not values:
            return None
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else round((values[mid - 1] + values[mid]) / 2, 3)


def classify_failure(example, row, context_ids, valid):
    """Retrieval, generation, or out of corpus — the S07 three failure modes.

    `valid` is the set of document ids that would satisfy dimension 3. If it is
    empty the corpus holds no right answer, so the only correct behaviour is the
    valve. Otherwise a miss with a valid document in the prompt is the
    generator's fault, and a miss without one is the retriever's.
    """
    if row["hit"]:
        return "ok"
    if not valid:
        return "out of corpus (the right answer is a refusal or a decline)"
    if set(context_ids) & valid:
        return "generation (valid exercise was in the context)"
    if not context_ids:
        return "no retrieval"
    return "retrieval (no valid exercise in the context)"
