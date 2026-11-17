"""Runs the cells of notebooks/M5_efficiency.ipynb locally (RTX 4060 8 GB).

Two deviations from the Colab T4 notebook, both to keep the comparison fair:
  * the generator is forced to fp16 (what pick_device_dtype returns on a T4);
    on an Ada GPU it would pick bf16 and the "before" row would not be M3's.
  * the LLM judge runs on CPU (fp32) so generator + judge fit in 8 GB without
    spilling into shared memory. Latency is timed before the harness runs, so
    the judge's placement does not touch the latency columns.
"""
import gc, json, os, sys, time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
os.chdir(ROOT)
for p in ("scripts", "eval", "m3", "final/efficiency"):
    sys.path.insert(0, str(ROOT / p))

import torch
from transformers import set_seed
import macgyver as mg

_orig_pick = mg.pick_device_dtype
def _t4_like():
    return "cuda", torch.float16
mg.pick_device_dtype = _t4_like

import harness as H
import corpus as C
import rag as R
import systems as S
import efficiency as E
import measure as MEA
import invariance as INV

set_seed(H.SEED)
assert torch.cuda.is_available()
print("gpu:", torch.cuda.get_device_name(0), "| seed:", H.SEED, flush=True)
REPORTS = ROOT / "reports"

def free_gpu():
    gc.collect(); torch.cuda.empty_cache()

eval_set = H.load_eval_set()
docs = C.build_documents()
print(f"{len(eval_set)} casos, {len(docs)} docs", flush=True)
retriever = E.build_retriever(docs, device="cuda")
ADV = R.TECHNIQUES
PV = S.PROMPT_VERSION
print("retriever:", E.EMBED_MODEL, "k=", E.TOP_K, ADV, PV, flush=True)

similarity = H.Similarity()
mg.pick_device_dtype = lambda: ("cpu", torch.float32)
judge = H.Judge(device="cpu")
mg.pick_device_dtype = _t4_like
print("judge on", judge.model.device, flush=True)

tok = mg.load_tokenizer()
t0 = time.time()
model_fp16 = E.load_fp16_model()
gen = E.make_generator(model_fp16, tok)
S.bind(retriever, gen, docs, techniques=ADV, prompt_version=PV)
before = MEA.measure_system("antes-fp16", S.sistema_rag_avanzado, model_fp16,
                            eval_set, judge=judge, similarity=similarity, verbose=True)
(REPORTS / "_m5_before_checkpoint.json").write_text(json.dumps(before, indent=2, ensure_ascii=False))
print("BEFORE", {k: before[k] for k in ("latency_p50_s", "latency_p95_s", "peak_vram_mb", "footprint_mb")},
      before["quality"], f"({time.time()-t0:.0f}s)", flush=True)

# Non-accuracy probes on fp16 too, so int4's can be read against a reference.
inv_b = INV.invariance_probe(eval_set, S.sistema_rag_avanzado, verbose=False)
valve_b = INV.valve_check(eval_set, S.sistema_rag_avanzado, verbose=False)
adv_b = INV.adversarial_check(eval_set, S.sistema_rag_avanzado, verbose=False)
before["non_accuracy"] = {"invariance": inv_b, "valve": valve_b, "adversarial": adv_b}
del model_fp16, gen
free_gpu()

t0 = time.time()
model_int4 = E.load_int4_model()
gen = E.make_generator(model_int4, tok)
S.bind(retriever, gen, docs, techniques=ADV, prompt_version=PV)
after = MEA.measure_system("despues-int4", S.sistema_rag_avanzado, model_int4,
                           eval_set, judge=judge, similarity=similarity, verbose=True)
print("AFTER", {k: after[k] for k in ("latency_p50_s", "latency_p95_s", "peak_vram_mb", "footprint_mb")},
      after["quality"], f"({time.time()-t0:.0f}s)", flush=True)

inv = INV.invariance_probe(eval_set, S.sistema_rag_avanzado, verbose=True)
valve = INV.valve_check(eval_set, S.sistema_rag_avanzado, verbose=True)
adv = INV.adversarial_check(eval_set, S.sistema_rag_avanzado, verbose=True)
after["non_accuracy"] = {"invariance": inv, "valve": valve, "adversarial": adv}

machine = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "generator_dtype": "fp16 (forced, T4-equivalent)", "judge_device": "cpu fp32"}
for r in (before, after):
    r["machine"] = machine
csv_path, json_path = MEA.write_outputs([before, after], REPORTS / "efficiency_m5.csv",
                                        REPORTS / "efficiency_m5_detail.json")
print(Path(csv_path).read_text())

qb, qa = before["quality"]["domain_hit_rate"], after["quality"]["domain_hit_rate"]
drop = round((qb - qa) * 100, 2)
vram_lower = after["peak_vram_mb"] < before["peak_vram_mb"]
p95_ok = after["latency_p95_s"] <= before["latency_p95_s"]
print(f"dim3 antes={qb} despues={qa} caida={drop} pts | VRAM menor={vram_lower} | p95 no peor={p95_ok}")
print("PRESUPUESTO CUMPLIDO:", drop <= 5 and vram_lower and p95_ok)
print("fp16 non-acc: inv", inv_b["invariance_rate"], "valve", valve_b["all_fired"], "adv", adv_b["adversarial_hit_rate"])
print("int4 non-acc: inv", inv["invariance_rate"], "valve", valve["all_fired"], "adv", adv["adversarial_hit_rate"])
print("DONE", flush=True)
