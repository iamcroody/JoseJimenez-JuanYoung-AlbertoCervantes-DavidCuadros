"""RAGAS's four metrics, computed the way the S10 lab does it, with a local judge.

What changed from the lab, and why:

  - The judge is its own model, `Qwen/Qwen2.5-3B-Instruct`, loaded only after
    the generator and the harness judge are gone. If it does not fit, it falls
    back to the M2 judge size and says so. The harness's dimension-2 judge is
    untouched; this one only grades context.
  - An unparseable verdict is None and is counted, never a default. The lab's
    `juez` returns 3 when it cannot find a digit; M2 already showed why that
    turns a broken judge into an average-looking one.
  - Prompts are in English, because the corpus and the replies are.
  - context precision is rank-weighted (average precision over the retrieved
    order), as RAGAS defines it, instead of the lab's plain fraction; context
    recall is per reference sentence, as RAGAS defines it, instead of one
    yes/no for the whole reference.

Each case needs the question, the reply, the retrieved contexts and the
reference — the `esperado` of the eval set.
"""

import gc
import re

from corpus import request_text
import macgyver as mg

JUDGE_CANDIDATES = ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct")
SYSTEM = "You are a strict, objective evaluator. Follow the output format exactly."
MIN_CLAIM_CHARS = 12

_YES = re.compile(r"^\W*(yes|y)\b", re.I)
_NO = re.compile(r"^\W*(no|n)\b", re.I)


def parse_yes_no(text):
    text = (text or "").strip()
    if _YES.search(text):
        return 1
    if _NO.search(text):
        return 0
    return None


class RagasJudge:
    def __init__(self, model_id, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(mg.SEED)
        picked_device, dtype = mg.pick_device_dtype()
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.model = self.model.to(device or picked_device).eval()
        self.torch = torch
        self.calls = 0
        self.unparsed = 0

    def generate(self, user, max_new_tokens):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        self.calls += 1
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()

    def yes_no(self, user):
        verdict = parse_yes_no(self.generate(user + "\n\nAnswer with yes or no only.", 3))
        if verdict is None:
            self.unparsed += 1
        return verdict

    def free(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def load_judge(candidates=JUDGE_CANDIDATES, device=None):
    """The largest candidate that fits. Returns (judge, notes on what failed)."""
    import torch

    notes = []
    for model_id in candidates:
        try:
            judge = RagasJudge(model_id, device=device)
            judge.generate("Say yes.", 2)
            return judge, notes
        except torch.cuda.OutOfMemoryError as exc:
            notes.append(f"{model_id}: out of memory ({str(exc).splitlines()[0]})")
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            notes.append(f"{model_id}: out of memory ({str(exc).splitlines()[0]})")
        gc.collect()
        torch.cuda.empty_cache()
    raise RuntimeError("no RAGAS judge fits on this device: " + "; ".join(notes))


# --------------------------------------------------------------------------
# The four metrics
# --------------------------------------------------------------------------

def _claims(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    claims = []
    for p in parts:
        p = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", p).strip()
        if len(p) >= MIN_CLAIM_CHARS and not p.endswith(":"):
            claims.append(p)
    return claims


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def faithfulness(judge, case):
    """Share of the reply's statements that the retrieved context supports."""
    if not case["contexts"]:
        return None
    context = "\n\n".join(case["contexts"])
    verdicts = [judge.yes_no(f"Context:\n{context}\n\nIs this statement supported by the "
                             f"context above?\nStatement: \"{claim}\"")
                for claim in _claims(case["answer"])]
    return _mean(verdicts)


def context_precision(judge, case):
    """Average precision of the retrieved order: relevant entries should come first."""
    if not case["contexts"]:
        return None
    verdicts = [judge.yes_no(f"Request: {case['question']}\n\nCatalog entry:\n{ctx}\n\n"
                             "Is this catalog entry useful for answering the request?")
                for ctx in case["contexts"]]
    hits, precisions = 0, []
    for rank, verdict in enumerate(verdicts, 1):
        if verdict == 1:
            hits += 1
            precisions.append(hits / rank)
    if all(v is None for v in verdicts):
        return None
    return round(sum(precisions) / hits, 4) if hits else 0.0


def context_recall(judge, case):
    """Share of the reference's sentences that can be attributed to the context."""
    if not case["contexts"]:
        return None
    context = "\n\n".join(case["contexts"])
    verdicts = [judge.yes_no(f"Context:\n{context}\n\nCan this sentence be attributed to "
                             f"the context above?\nSentence: \"{claim}\"")
                for claim in _claims(case["reference"])]
    return _mean(verdicts)


def answer_relevancy(judge, similarity, case):
    """Cosine between the request and a question regenerated from the reply."""
    question = judge.generate(
        "Write the one question that this answer is responding to. Output only the "
        f"question.\n\nAnswer:\n{case['answer']}", 40)
    return round(similarity(case["question"], question), 4)


METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def build_cases(eval_set, log):
    """One RAGAS case per eval example, from a system's recorder log."""
    cases = []
    for example in eval_set:
        entry = log[example["input"]]
        cases.append({
            "id": example["id"],
            "question": request_text(example["input"]),
            "answer": entry["reply"],
            "contexts": entry["contexts"],
            "reference": example["esperado"],
        })
    return cases


def evaluate(judge, similarity, cases, verbose=True):
    unparsed_before = judge.unparsed
    rows = []
    for case in cases:
        row = {
            "id": case["id"],
            "faithfulness": faithfulness(judge, case),
            "answer_relevancy": answer_relevancy(judge, similarity, case),
            "context_precision": context_precision(judge, case),
            "context_recall": context_recall(judge, case),
        }
        rows.append(row)
        if verbose:
            print(f"  {case['id']:26} " + " ".join(
                f"{m[:6]}={'n/a' if row[m] is None else f'{row[m]:.2f}'}" for m in METRICS))
    summary = {m: _mean([r[m] for r in rows]) for m in METRICS}
    summary["unparsed_verdicts"] = judge.unparsed - unparsed_before
    summary["judge_model"] = judge.model_id
    return summary, rows
