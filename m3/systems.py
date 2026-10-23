"""The three RAG systems under the names the assignment uses: `sistema(pregunta) -> respuesta`.

    import systems
    systems.bind(retriever, llm, docs)
    systems.sistema_rag(prompt)            # naive: dense top-k + escape valve
    systems.sistema_rag_avanzado(prompt)   # hybrid + rerank + HyDE
    systems.sistema_final(prompt)          # advanced + the find_exercises tool

Each one returns only the reply string, so it can be passed straight to
`harness(eval_set, sistema)`. They are the same closures `make_rag` and
`make_agent` build in the notebook; nothing about their behaviour changes.
`prompt_version` picks the augmented prompt (see `rag.PROMPTS`); all three
systems always share it. The default is the version the final run chose on the
validation probe; "v1" reproduces the first run.
"""

from agent import make_agent
from rag import TECHNIQUES, make_rag

PROMPT_VERSION = "v2a"

_BOUND = {}


def bind(retriever, llm, docs, techniques=TECHNIQUES, prompt_version=PROMPT_VERSION):
    _BOUND["naive"] = make_rag(retriever, llm, prompt_version=prompt_version)
    _BOUND["advanced"] = make_rag(retriever, llm, techniques, prompt_version)
    _BOUND["final"] = make_agent(retriever, llm, docs, techniques,
                                 prompt_version=prompt_version)


def _run(name, pregunta):
    if name not in _BOUND:
        raise RuntimeError("call systems.bind(retriever, llm, docs) first")
    return _BOUND[name](pregunta)["reply"]


def sistema_rag(pregunta):
    return _run("naive", pregunta)


def sistema_rag_avanzado(pregunta):
    return _run("advanced", pregunta)


def sistema_final(pregunta):
    return _run("final", pregunta)
