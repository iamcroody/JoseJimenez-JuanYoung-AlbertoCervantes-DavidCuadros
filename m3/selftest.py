"""Checks that need no GPU, no model weights and nothing outside the standard library.

    python m3/selftest.py

On Windows run it as `PYTHONUTF8=1 python m3/selftest.py`: `mg.load_catalog`
reads the catalog with the platform encoding, which is UTF-8 on Colab and Linux
but cp1252 on Windows. Run first in the notebook too: finding a broken parser here costs a second
rather than a Colab session.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus as C  # noqa: E402
import rag as R  # noqa: E402
import agent as A  # noqa: E402
import ragas_m3 as G  # noqa: E402
import systems as S  # noqa: E402
import probe as P  # noqa: E402
import macgyver as mg  # noqa: E402
import harness as H  # noqa: E402

FAILURES = []

# The first run's prompts, so a later edit cannot silently change what v1 means.
V1_AUGMENTED_SHA256 = "946fbf1b866d4141f83937a5622ffb56d54cdd4a4d87f9833e9df9185a6da4f7"
V1_DECISION_SHA256 = "c658a09abe59da6001debd7778355c0f9e1c59df007431291abb61a270bc8589"


def check(name, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


class FakeDense:
    """Dense search stand-in: BM25 over names only, so ranking differs from full BM25."""

    def __init__(self, docs):
        self.order = [d["id"] for d in docs]
        self.bm25 = R.BM25([R.tokenize(d["name"]) for d in docs])

    def search(self, query, n):
        scores = self.bm25.scores(R.tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:n]
        return [self.order[i] for i in ranked]


def scripted_llm(outputs):
    """A generator that replays canned outputs and records every prompt it got."""
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return outputs[min(len(calls) - 1, len(outputs) - 1)]
    llm.calls = calls
    return llm


def main():
    eval_set = H.load_eval_set()
    docs = C.build_documents()
    stats = C.corpus_stats(docs)
    by_id = {d["id"]: d for d in docs}

    print("corpus")
    check("one document per catalog exercise", stats["documents"] == 1324, stats["documents"])
    check("document ids are unique", len(by_id) == len(docs))
    check("reachable count matches the harness filter",
          stats["reachable_documents"] == len(H.reachable_catalog()),
          stats["reachable_documents"])
    check("every document carries its name and its steps",
          all(d["text"].startswith(f"Exercise: {d['name']}\n") and "\n1. " in d["text"]
              for d in docs))
    print(f"       {stats}")

    print("eval set vs corpus")
    out_of_corpus = [e["id"] for e in eval_set if not C.valid_ids_for(e, docs)]
    check("at least two cases have no valid answer in the corpus", len(out_of_corpus) >= 2,
          ", ".join(out_of_corpus))
    answerable = [e for e in eval_set if e["kind"] == "answerable"]
    check("every answerable case has a valid document",
          all(C.valid_ids_for(e, docs) for e in answerable))
    check("every gold exercise is among the valid documents",
          all(e["meta"]["gold_exercise_id"] in C.valid_ids_for(e, docs) for e in answerable))
    stripped = C.request_text(eval_set[4]["input"])
    check("request_text drops the format block",
          "Answer in exactly this format" not in stripped and "two bricks" in stripped, stripped)
    adv03 = next(e for e in eval_set if e["id"] == "adv03-out-of-domain")
    check("request_text leaves a prompt with no format block alone",
          C.request_text(adv03["input"]) == adv03["input"].strip())

    print("retrieval")
    bm25 = R.BM25([R.tokenize(d["text"]) for d in docs])
    retriever = R.Retriever(docs, FakeDense(docs), bm25=bm25, k=5)
    top = retriever.bm25_search("levator scapulae neck stretch body weight", 5)
    check("BM25 finds an exact rare term",
          any(by_id[i]["target"] == "levator scapulae" for i in top),
          [by_id[i]["name"] for i in top])
    check("RRF rewards agreement and keeps one-list items",
          R.rrf([["a", "b", "c"], ["b", "d"]]) == ["b", "a", "d", "c"],
          R.rrf([["a", "b", "c"], ["b", "d"]]))
    ids, query = retriever.retrieve(eval_set[0]["input"], ("hybrid",))
    check("hybrid returns k unique ids", len(ids) == 5 and len(set(ids)) == 5)
    draft = "Exercise: dumbbell biceps curl\nGym equivalent: dumbbell"
    llm = scripted_llm([draft])
    ids, query = retriever.retrieve(eval_set[4]["input"], ("hyde",), llm)
    check("hyde searches with the request plus the draft",
          query.endswith(draft) and "Answer in exactly this format" not in query)
    try:
        retriever.retrieve(eval_set[0]["input"], ("rerank",))
        check("rerank without a cross-encoder fails loudly", False)
    except ValueError:
        check("rerank without a cross-encoder fails loudly", True)

    print("escape valve")
    valve = "No safe option: I know no lats exercise that works with two full cans of food."
    check("the valve is detected as a refusal by the frozen harness parser",
          mg.parse_reply(valve)["refused"])
    check("the valve template itself is detected as a refusal",
          mg.parse_reply(R.VALVE)["refused"])
    adv01 = next(e for e in eval_set if e["id"] == "adv01-impossible-pair")
    detail = H.domain_check(adv01, valve, mg.build_name_index(H.reachable_catalog()))
    check("the valve scores a hit on the impossible pair", detail["hit"])
    prompt = R.augment(eval_set[0]["input"], docs[:2])
    check("the augmented prompt ends with the untouched eval prompt",
          prompt.endswith(eval_set[0]["input"]))

    print("prompt versions")
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    check("v1 augmented prompt is byte-identical to the first run",
          digest(R.augment(eval_set[0]["input"], docs[:2])) == V1_AUGMENTED_SHA256)
    check("v1 decision prompt is byte-identical to the first run",
          digest(A.DECISION_TEMPLATE) == V1_DECISION_SHA256)
    for version in R.PROMPT_VERSIONS:
        text = R.augment(eval_set[0]["input"], docs[:2], version)
        check(f"{version}: ends with the untouched eval prompt and holds the entries",
              text.endswith(eval_set[0]["input"]) and docs[0]["text"] in text
              and docs[1]["text"] in text)
    for version in ("v2a", "v2b"):
        check(f"{version}: never shows the valve sentence the LoRA copies",
              "No safe option:" not in R.PROMPTS[version]
              and "No safe option:" not in A.DECISION_TEMPLATES[version])
    try:
        R.make_rag(retriever, scripted_llm(["x"]), prompt_version="v3")
        check("an unknown prompt version fails loudly", False)
    except ValueError:
        check("an unknown prompt version fails loudly", True)
    seen = scripted_llm(["reply"])
    R.make_rag(retriever, seen, prompt_version="v2b")(eval_set[0]["input"])
    check("make_rag sends the chosen version to the generator",
          seen.calls[-1].startswith("You will get some catalog entries"))

    print("prompt probe")
    fake_rows = (
        [{"kind": "answerable", "target": "biceps", "object": f"object {i}", "exercise_id": str(i),
          "prompt": [{"content": f"answerable {i}"}]} for i in range(12)]
        + [{"kind": "refusal", "target": "lats", "object": f"object {i}", "exercise_id": None,
            "prompt": [{"content": f"refusal {i}"}]} for i in range(6)]
        + [{"kind": "answerable", "target": adv01["meta"]["target"],
            "object": adv01["meta"]["object"], "exercise_id": "x",
            "prompt": [{"content": "taken"}]}])
    rows = P.dev_rows(eval_set, fake_rows)
    check("the probe sample has the fixed size and no eval-set pair",
          sum(r["kind"] == "answerable" for r in rows) == P.N_ANSWERABLE
          and sum(r["kind"] == "refusal" for r in rows) == P.N_REFUSAL
          and all(r["prompt"] != "taken" for r in rows))
    check("the probe sample is deterministic", P.dev_rows(eval_set, fake_rows) == rows)
    answer = eval_set[4]["esperado"]
    collapsed = P.score([valve] * len(rows), rows)
    never = P.score([answer] * len(rows), rows)
    good = P.score([answer if r["kind"] == "answerable" else valve for r in rows], rows)
    check("a prompt that refuses everything does not fix the collapse",
          P.verdict(collapsed) == "still refuses almost everything", collapsed)
    check("a prompt that never refuses does not fix the collapse",
          P.verdict(never) == "never refuses", never)
    check("a prompt that answers and refuses where it should fixes it",
          P.verdict(good) == "fixes the collapse", good)
    check("no qualifying candidate means no choice",
          P.choose({"v2a": collapsed, "v2b": never}) is None)
    check("the better qualifying candidate wins, ties go to v2a",
          P.choose({"v2a": collapsed, "v2b": good}) == "v2b"
          and P.choose({"v2a": good, "v2b": good}) == "v2a")

    print("tool parser")
    schema = A.tool_schema({d["target"] for d in docs})
    good = {"tool": "find_exercises", "args": {"target": "biceps", "equipment": "dumbbell"}}
    cases = {
        "plain json": ('{"tool": "find_exercises", "args": {"target": "biceps", '
                       '"equipment": "dumbbell"}}', good),
        "fenced json": ('```json\n{"tool": "find_exercises", "args": {"target": "biceps", '
                        '"equipment": "dumbbell"}}\n```', good),
        "prose around it": ('Sure! I will call {"tool": "find_exercises", "args": '
                            '{"target": "biceps", "equipment": "dumbbell"}} now.', good),
        "single quotes": ("{'tool': 'find_exercises', 'args': {'target': 'biceps', "
                          "'equipment': 'dumbbell'}}", good),
        "trailing commas": ('{"tool": "find_exercises", "args": {"target": "biceps", '
                            '"equipment": "dumbbell",},}', good),
        "function style": ('find_exercises(target="biceps", equipment="dumbbell")', good),
        "no tool key": ('{"target": "Biceps", "equipment": "dumbbells"}', good),
        "arguments as a string": ('{"name": "find_exercises", "arguments": '
                                  '"{\\"target\\": \\"biceps\\", \\"equipment\\": \\"dumbbell\\"}"}',
                                  good),
        "thinking block first": ('<think>hmm {"x": 1}</think>{"tool": "find_exercises", '
                                 '"args": {"target": "biceps", "equipment": "dumbbell"}}', good),
    }
    for name, (text, expected) in cases.items():
        call, error = A.parse_tool_call(text, schema)
        check(f"parses: {name}", call == expected, error or call)
    rejects = {
        "unknown tool": '{"tool": "web_search", "args": {"q": "biceps"}}',
        "bad enum": '{"tool": "find_exercises", "args": {"target": "lower biceps", '
                    '"equipment": "dumbbell"}}',
        "missing argument": '{"tool": "find_exercises", "args": {"target": "biceps"}}',
        "broken json": '{"tool": "find_exercises", "args": {"target": "biceps"',
    }
    for name, text in rejects.items():
        call, error = A.parse_tool_call(text, schema)
        check(f"rejects with a reason: {name}", call is None and error, error)
    call, error = A.parse_tool_call(eval_set[0]["esperado"], schema)
    check("a normal answer is not a tool call", call is None and error is None, error)

    print("tool")
    check("no lats exercise with dumbbells (the adv01 premise)",
          A.find_exercises(docs, "lats", "dumbbell") == [])
    found = A.find_exercises(docs, "biceps", "dumbbell")
    check("biceps with dumbbells returns real reachable matches",
          found and all(d["reachable"] and d["target"] == "biceps"
                        and d["equipment"] == "dumbbell" for d in found)
          and len(found) <= A.MAX_RESULTS)

    print("agent")
    call_text = '{"tool": "find_exercises", "args": {"target": "lats", "equipment": "dumbbell"}}'
    looping = scripted_llm([call_text, call_text, call_text, valve])
    result = A.make_agent(retriever, looping, docs, ())(adv01["input"])
    check("a repeated call ends the loop", any(t["action"] == "repeat_call"
                                               for t in result["trace"]), result["trace"])
    check("an empty tool result reaches the final prompt",
          "found no exercise" in looping.calls[-1])
    llms = [scripted_llm([f'{{"tool": "find_exercises", "args": {{"target": "{t}", '
                          f'"equipment": "dumbbell"}}}}' for t in
                          ("biceps", "triceps", "forearms", "delts", "biceps")] + ["done"])]
    result = A.make_agent(retriever, llms[0], docs, ())(eval_set[4]["input"])
    check("the step cap stops a model that never stops calling",
          len(A.tool_calls(result["trace"])) == A.MAX_STEPS
          and result["trace"][-1]["action"] == "step_cap", result["trace"])
    direct = scripted_llm([eval_set[4]["esperado"]])
    result = A.make_agent(retriever, direct, docs, ())(eval_set[4]["input"])
    check("a direct answer makes no tool call and one generation",
          A.tool_calls(result["trace"]) == [] and len(direct.calls) == 1)
    garbage = scripted_llm(['{"tool": "find_exercises", "args": ', eval_set[4]["esperado"]])
    result = A.make_agent(retriever, garbage, docs, ())(eval_set[4]["input"])
    check("an unparseable call is fed back and the agent recovers",
          result["trace"][0]["action"] == "parse_error"
          and result["reply"] == eval_set[4]["esperado"], result["trace"])

    print("ragas helpers")
    check("yes/no verdicts parse, anything else is None",
          [G.parse_yes_no(t) for t in ("Yes.", "no", "**Yes**", "maybe", "")]
          == [1, 0, 1, None, None])
    claims = G._claims(eval_set[4]["esperado"])
    check("claims skip headers and step numbers",
          "Steps:" not in claims and not any(c[:2] == "1." for c in claims)
          and len(claims) >= 6, len(claims))

    print("recorder")
    rec = R.Recorder("naive", R.make_rag(retriever, scripted_llm(["reply"])), by_id)
    reply = rec(eval_set[0]["input"])
    entry = rec.log[eval_set[0]["input"]]
    check("the harness sees only the reply", reply == "reply")
    check("contexts and latency are kept on the side",
          len(entry["contexts"]) == 5 and entry["latency_s"] >= 0)

    print("assignment names")
    try:
        S.sistema_rag(eval_set[0]["input"])
        check("an unbound system fails loudly", False)
    except RuntimeError:
        check("an unbound system fails loudly", True)
    plain = R.Retriever(docs, FakeDense(docs), bm25=bm25, k=5)
    S.bind(plain, scripted_llm(["reply"]), docs, techniques=("hybrid",))
    check("sistema_rag and sistema_rag_avanzado return the reply string",
          S.sistema_rag(eval_set[0]["input"]) == "reply"
          and S.sistema_rag_avanzado(eval_set[0]["input"]) == "reply")
    S.bind(plain, scripted_llm([call_text, call_text, valve]), docs, techniques=("hybrid",))
    check("sistema_final runs the agent and returns its last reply",
          S.sistema_final(adv01["input"]) == valve)
    check("the default prompt is the one the final run chose", S.PROMPT_VERSION == "v2a",
          S.PROMPT_VERSION)
    for name, wrapper, outputs, build in (
            ("sistema_rag", S.sistema_rag, ["reply"],
             lambda llm: R.make_rag(plain, llm, prompt_version="v2a")),
            ("sistema_rag_avanzado", S.sistema_rag_avanzado, ["reply"],
             lambda llm: R.make_rag(plain, llm, ("hybrid",), "v2a")),
            ("sistema_final", S.sistema_final, [call_text, call_text, valve],
             lambda llm: A.make_agent(plain, llm, docs, ("hybrid",), prompt_version="v2a"))):
        bound_llm, direct_llm = scripted_llm(outputs), scripted_llm(outputs)
        S.bind(plain, bound_llm, docs, techniques=("hybrid",))
        bound_reply = wrapper(adv01["input"])
        direct_reply = build(direct_llm)(adv01["input"])["reply"]
        check(f"{name} is the notebook's system: same prompts, same reply",
              bound_reply == direct_reply and bound_llm.calls == direct_llm.calls)

    print("\nself-test", "PASSED" if not FAILURES else f"FAILED: {FAILURES}")
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
