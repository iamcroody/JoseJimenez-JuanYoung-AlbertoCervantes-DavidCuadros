"""Choose the augmented prompt on M1's validation split, before the full run.

The first run (prompt v1) refused all 14 eval cases. Picking its replacement on
the eval set would tune the prompt on the test, so the choice is made here, on
rows of M1's validation split whose (muscle, object) pair is not in the eval set:

  answerable rows   a valid catalog exercise exists; a good prompt answers them
  refusal rows      the pair is impossible; a good prompt refuses them

Every candidate runs through the naive RAG with the same retriever, generator
and k, so the prompt is the only difference. The rule is fixed before the run:

  a candidate fixes the collapse if it answers at least MIN_ANSWERED of the
  answerable rows and refuses at least MIN_REFUSED of the refusal rows;
  among those, the highest answered + refused wins, ties go to the earlier
  candidate. If none qualifies, the notebook stops.
"""

import random
import subprocess
import sys

import macgyver as mg

CANDIDATES = ("v2a", "v2b")
CONTROL = "v1"
N_ANSWERABLE = 8
N_REFUSAL = 4
MIN_ANSWERED = 0.5
MIN_REFUSED = 0.25


def ensure_splits():
    """M1's splits are deterministic and gitignored: rebuild them if missing."""
    if not (mg.PROCESSED / "val.jsonl").exists():
        subprocess.run([sys.executable, "scripts/01_macgyver_data.py"], check=True)


def dev_rows(eval_set, rows=None, n_answerable=N_ANSWERABLE, n_refusal=N_REFUSAL,
             seed=mg.SEED):
    """A fixed sample of validation rows, none sharing a (muscle, object) pair with the eval set."""
    rows = mg.load_split("val") if rows is None else rows
    taken = {(e["meta"].get("target"), e["meta"].get("object")) for e in eval_set}
    pool = [r for r in rows if (r["target"], r["object"]) not in taken]
    rng = random.Random(seed)
    sample = []
    for kind, n in (("answerable", n_answerable), ("refusal", n_refusal)):
        members = sorted((r for r in pool if r["kind"] == kind),
                         key=lambda r: (r["target"], r["object"], r["exercise_id"] or ""))
        if len(members) < n:
            raise ValueError(f"only {len(members)} {kind} rows left in the validation split")
        sample += rng.sample(members, n)
    return [{"prompt": r["prompt"][0]["content"], "kind": r["kind"], "target": r["target"],
             "object": r["object"]} for r in sample]


def score(replies, rows):
    """How a candidate did: share of answerable rows answered, of refusal rows refused."""
    answerable = [rep for rep, r in zip(replies, rows) if r["kind"] == "answerable"]
    refusal = [rep for rep, r in zip(replies, rows) if r["kind"] == "refusal"]
    refused = lambda reply: mg.parse_reply(reply)["refused"]
    return {
        "answered": round(sum(not refused(x) for x in answerable) / len(answerable), 3),
        "refused": round(sum(refused(x) for x in refusal) / len(refusal), 3),
        "n_answerable": len(answerable),
        "n_refusal": len(refusal),
    }


def fixes_collapse(s):
    return s["answered"] >= MIN_ANSWERED and s["refused"] >= MIN_REFUSED


def choose(scores, candidates=CANDIDATES):
    """The winning candidate under the rule above, or None if none qualifies."""
    passing = [c for c in candidates if fixes_collapse(scores[c])]
    if not passing:
        return None
    return max(passing, key=lambda c: (scores[c]["answered"] + scores[c]["refused"],
                                       -candidates.index(c)))


def verdict(s):
    if fixes_collapse(s):
        return "fixes the collapse"
    if s["answered"] < MIN_ANSWERED:
        return "still refuses almost everything"
    return "never refuses"
