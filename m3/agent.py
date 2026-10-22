"""The agentic layer: one domain tool on top of the advanced RAG.

The tool is a structured lookup over the catalog, `find_exercises(target,
equipment)`. Retrieval answers "what reads like this request"; the tool answers
"what exists for exactly this muscle and this equipment class", including the
answer retrieval can never give: *nothing does*. That empty result is the
evidence the escape valve needs on a case like lats with two cans of food,
where the top-k always returns k plausible-looking documents anyway.

When it should be called: the retrieved entries hold no exercise for the
requested muscle with the equipment class the object stands in for. When it
should not: they already do, or the request is not an exercise lookup at all
(a dosing question, an unsafe object).
"""

import ast
import json
import re

from corpus import EQUIPMENT_CLASSES
from rag import VALVE, augment, format_context

MAX_STEPS = 3
MAX_RESULTS = 5

_EQUIPMENT_ALIASES = {
    "dumbbells": "dumbbell", "kettlebells": "kettlebell", "bodyweight": "body weight",
    "body-weight": "body weight", "none": "body weight", "weight": "weighted",
    "weights": "weighted",
}


def tool_schema(targets):
    return {
        "name": "find_exercises",
        "description": ("Look up real catalog exercises for one target muscle and one "
                        "equipment class. Returns up to five exercises with their steps, "
                        "or says that the catalog has none."),
        "parameters": {
            "target": {"type": "string", "enum": sorted(targets),
                       "description": "the muscle the user wants to train"},
            "equipment": {"type": "string", "enum": list(EQUIPMENT_CLASSES),
                          "description": ("the class the user's object stands in for: "
                                          "dumbbell = a matched pair, one per hand; "
                                          "kettlebell = one load carried by a handle; "
                                          "weighted = dead weight added to a bodyweight "
                                          "movement; body weight = nothing at all")},
        },
        "required": ["target", "equipment"],
    }


def find_exercises(docs, target, equipment, limit=MAX_RESULTS):
    """Reachable exercises for (target, equipment), lowest id first."""
    matches = [d for d in docs if d["reachable"]
               and d["target"] == target and d["equipment"] == equipment]
    return sorted(matches, key=lambda d: d["id"])[:limit]


# --------------------------------------------------------------------------
# Parsing what the model proposes, tolerantly
# --------------------------------------------------------------------------

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_CALL = re.compile(r"find_exercises\s*[\(\[](.*?)[\)\]]", re.S)
_KWARG = re.compile(r"(\w+)\s*[=:]\s*[\"']?([^,\"'\)\]]+)[\"']?")


def _balanced_object(text):
    start = text.find("{")
    while start != -1:
        depth, in_str, quote = 0, False, ""
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if ch == quote and text[i - 1] != "\\":
                    in_str = False
            elif ch in "\"'":
                in_str, quote = True, ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _loads(raw):
    """json.loads, then the repairs small models need, then Python literals."""
    attempts = [raw, re.sub(r",\s*([}\]])", r"\1", raw)]
    if '"' not in raw:
        attempts.append(attempts[-1].replace("'", '"'))
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            continue
    try:
        value = ast.literal_eval(attempts[1])
        return value if isinstance(value, dict) else None
    except (ValueError, SyntaxError):
        return None


def _extract(text):
    for fenced in _FENCE.findall(text):
        obj = _balanced_object(fenced)
        if obj and _loads(obj) is not None:
            return _loads(obj)
    obj = _balanced_object(text)
    if obj is not None:
        loaded = _loads(obj)
        if loaded is not None:
            return loaded
    call = _CALL.search(text)
    if call:
        return {"tool": "find_exercises",
                "args": {k.lower(): v.strip() for k, v in _KWARG.findall(call.group(1))}}
    return None


def looks_like_call(text):
    text = _THINK.sub("", text or "")
    return "find_exercises" in text or text.lstrip().startswith(("{", "```"))


def parse_tool_call(text, schema):
    """Returns (call, error). call is {"tool", "args"} or None; error explains why.

    (None, None) means the model answered in prose: no call was attempted.
    """
    text = _THINK.sub("", text or "")
    payload = _extract(text)
    if payload is None:
        return (None, "could not parse a tool call") if looks_like_call(text) else (None, None)
    if not isinstance(payload, dict):
        return None, "tool call is not a JSON object"

    name = payload.get("tool") or payload.get("name") or payload.get("function")
    args = payload.get("args", payload.get("arguments", payload.get("parameters")))
    if isinstance(args, str):
        args = _loads(args)
    if name is None and args is None and {"target", "equipment"} & set(payload):
        name, args = schema["name"], payload
    if name != schema["name"]:
        return None, f"unknown tool {name!r}"
    if not isinstance(args, dict):
        return None, "arguments are not an object"

    clean = {}
    for key in schema["required"]:
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            return None, f"missing argument {key!r}"
        value = value.strip().lower()
        if key == "equipment":
            value = _EQUIPMENT_ALIASES.get(value, value)
        allowed = schema["parameters"][key]["enum"]
        if value not in allowed:
            return None, f"{key}={value!r} is not one of {allowed}"
        clean[key] = value
    return {"tool": schema["name"], "args": clean}, None


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------

DECISION_TEMPLATE = (
    "Catalog entries that may be relevant:\n\n"
    "{context}\n\n"
    "{observations}"
    "You can look up the catalog with this tool:\n"
    "{schema}\n\n"
    "If the entries above include an exercise that trains that muscle with what I "
    "have, answer now. If they do not, reply with only this JSON and nothing else:\n"
    '{{"tool": "find_exercises", "args": {{"target": "<muscle>", "equipment": "<class>"}}}}\n'
    "If the tool already found no exercise, reply with this single line instead:\n"
    f"{VALVE}\n\n"
    "{prompt}"
)

# v2 drops the valve sentence here too, for the reason given next to
# `rag.PROMPTS`; the decision layout is the same for both v2 candidates.
_DECISION_V2 = DECISION_TEMPLATE.replace(
    "If the tool already found no exercise, reply with this single line instead:\n"
    f"{VALVE}\n\n",
    "If the tool already found no exercise, say that there is no safe option.\n\n")
DECISION_TEMPLATES = {"v1": DECISION_TEMPLATE, "v2a": _DECISION_V2, "v2b": _DECISION_V2}


def _observation(call, found):
    args = call["args"]
    head = f"Tool result for find_exercises(target={args['target']}, equipment={args['equipment']}):"
    if not found:
        return f"{head} the catalog has no such exercise.\n\n"
    return f"{head}\n{format_context(found)}\n\n"


def make_agent(retriever, llm, docs, techniques, max_steps=MAX_STEPS,
               prompt_version="v1"):
    """`sistema_final`: advanced retrieval, then up to `max_steps` tool calls.

    The step cap is the guard against a loop; a repeated identical call ends the
    loop early for the same reason. Whatever happens, the last word goes through
    the same augmented prompt every other RAG row uses, with the tool's results
    added to the context.
    """
    schema = tool_schema({d["target"] for d in docs})
    schema_text = json.dumps(schema, indent=1)
    techniques = tuple(techniques)
    decision = DECISION_TEMPLATES[prompt_version]

    def run(prompt):
        ids, query = retriever.retrieve(prompt, techniques, llm)
        retrieved = retriever.docs_for(ids)
        tool_docs, observations, trace, seen = [], "", [], set()

        for step in range(1, max_steps + 1):
            text = decision.format(context=format_context(retrieved),
                                   observations=observations,
                                   schema=schema_text, prompt=prompt)
            out = llm(text)
            call, error = parse_tool_call(out, schema)
            if call is None and error is None:
                trace.append({"step": step, "action": "answer"})
                return _result(out, ids, tool_docs, query, trace)
            if call is None:
                trace.append({"step": step, "action": "parse_error", "error": error,
                              "raw": out[:300]})
                observations += f"Your last tool call was invalid ({error}).\n\n"
                continue
            key = (call["args"]["target"], call["args"]["equipment"])
            if key in seen:
                trace.append({"step": step, "action": "repeat_call", "args": call["args"]})
                break
            seen.add(key)
            found = find_exercises(docs, *key)
            tool_docs += [d for d in found if d["id"] not in {t["id"] for t in tool_docs}]
            observations += _observation(call, found)
            trace.append({"step": step, "action": "tool_call", "args": call["args"],
                          "results": [d["id"] for d in found]})
        else:
            trace.append({"step": max_steps, "action": "step_cap"})

        context = retrieved + [d for d in tool_docs if d["id"] not in set(ids)]
        final_prompt = augment(prompt, context, prompt_version)
        if any(t["action"] == "tool_call" and not t["results"] for t in trace):
            final_prompt = _observation_note(trace) + final_prompt
        return _result(llm(final_prompt), ids, tool_docs, query, trace)

    return run


def _observation_note(trace):
    empty = [t["args"] for t in trace if t["action"] == "tool_call" and not t["results"]]
    lines = [f"A catalog lookup for {a['target']} with {a['equipment']} found no exercise."
             for a in empty]
    return "\n".join(lines) + "\n\n"


def _result(reply, ids, tool_docs, query, trace):
    context_ids = list(ids) + [d["id"] for d in tool_docs if d["id"] not in set(ids)]
    return {"reply": reply, "context_ids": context_ids, "query": query, "trace": trace}


def tool_calls(trace):
    return [t for t in trace if t["action"] == "tool_call"]
