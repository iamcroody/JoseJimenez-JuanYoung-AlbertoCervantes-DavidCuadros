"""The M3 corpus: the exercise catalog, one document per exercise.

ingest -> chunk lives here; embed -> store lives in `rag.py`.

The chunk is the whole exercise record, never a slice of it. Dimension 3 of the
harness grades an answer by resolving the exercise *name* against the catalog
and ROUGE-L compares the recited *steps* with that exercise's real steps, so a
chunk that carries the steps without the name (or the name without the
equipment) is a chunk the generator cannot use correctly. Every record already
fits comfortably inside one embedding window, which `corpus_stats` measures
rather than assumes.
"""

import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))

import macgyver as mg  # noqa: E402
import harness as H  # noqa: E402

SOURCE = "https://github.com/jayounghoyos/exercises-dataset (submodule data/exercises-dataset)"
LICENCE = "MIT for the exercise metadata and text; images and GIFs (c) Gym visual, not used"

# The four equipment classes a household object can stand in for. The tool in
# `agent.py` takes one of these, and the diagnosis uses them to decide whether
# a valid answer exists in the corpus at all.
EQUIPMENT_CLASSES = tuple(mg.OBJECT_MAP)


def render(record):
    """One exercise as the text the retriever indexes and the generator reads.

    Laid out like the answer format the M1 model was trained to produce, so the
    name and steps it should copy look the same in the context as in its output.
    """
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(record["instruction_steps"]["en"], 1))
    secondary = ", ".join(record.get("secondary_muscles") or []) or "none"
    return (
        f"Exercise: {record['name']}\n"
        f"Target muscle: {record['target']}\n"
        f"Equipment: {record['equipment']}\n"
        f"Body part: {record['body_part']}\n"
        f"Secondary muscles: {secondary}\n"
        f"Steps:\n{steps}"
    )


def build_documents(catalog=None):
    """Every catalog record as a document with the metadata retrieval needs.

    `reachable` uses the same apparatus filter the harness grades with, so the
    diagnosis can tell "no valid exercise exists" apart from "a valid one exists
    but was not retrieved".
    """
    catalog = mg.load_catalog() if catalog is None else catalog
    needs_apparatus = H._data_script().needs_apparatus
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "target": r["target"],
            "equipment": r["equipment"],
            "body_part": r["body_part"],
            "reachable": not needs_apparatus(r),
            "text": render(r),
        }
        for r in catalog
    ]


def corpus_stats(docs):
    chars = [len(d["text"]) for d in docs]
    words = [len(d["text"].split()) for d in docs]
    return {
        "source": SOURCE,
        "licence": LICENCE,
        "documents": len(docs),
        "chunks": len(docs),
        "reachable_documents": sum(d["reachable"] for d in docs),
        "chars_min": min(chars),
        "chars_median": statistics.median(chars),
        "chars_max": max(chars),
        "words_median": statistics.median(words),
        "words_p95": sorted(words)[int(0.95 * (len(words) - 1))],
        "words_max": max(words),
        "targets": len({d["target"] for d in docs}),
        "equipment_classes": len({d["equipment"] for d in docs}),
    }


def valid_ids(docs, target, equipment):
    """Documents that would satisfy dimension 3 for (target, equipment)."""
    if not target or not equipment:
        return set()
    return {d["id"] for d in docs
            if d["reachable"] and d["target"] == target and d["equipment"] == equipment}


def valid_ids_for(example, docs):
    meta = example.get("meta", {})
    return valid_ids(docs, meta.get("target"), meta.get("equipment"))


_FORMAT_BLOCK = re.compile(r"\n\s*Answer in exactly this format:.*\Z", re.S)


def request_text(prompt):
    """The user's request without the answer-format boilerplate.

    The eval prompts end with a format template ("Exercise: <name> ... Safety:
    <one line>") and a refusal instruction that are identical in every case. As
    a retrieval query they are pure noise: they say nothing about the muscle or
    the object and they look like every document in the corpus.
    """
    return _FORMAT_BLOCK.sub("", prompt).strip()
