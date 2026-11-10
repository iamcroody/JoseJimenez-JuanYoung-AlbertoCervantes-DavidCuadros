"""Self-test sin GPU de la plomería de eficiencia del Módulo 5.

    python final/efficiency/selftest.py

Verifica, SIN cargar ningún modelo y con solo numpy disponible (ni torch, ni
bitsandbytes, ni transformers):

  - la aritmética de percentiles de `measure.latency_percentiles` sobre una
    lista fija de latencias (p50/p95 esperados);
  - la función de perturbación `invariance.perturb` sobre cadenas de muestra
    (preserva significado, cambia la forma, y es determinista);
  - el esquema del dict que `measure.build_result` produce y que el notebook
    serializa, incluido que `write_outputs` escribe un CSV/JSON coherente.

No importa `efficiency.py` (que sí necesita torch/transformers/peft): este
archivo solo toca las partes puras. Es el análogo de `m3/selftest.py` para la
feature de eficiencia: encontrar un error de plomería aquí cuesta un segundo en
lugar de una sesión de Colab T4.
"""

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import measure as M  # noqa: E402
import invariance as I  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main():
    print("percentile math")
    # 0..100 en pasos de 10: p50 = 50, p95 = 95 (interpolación lineal de numpy).
    timings = [float(x) for x in range(0, 101, 10)]
    stats = M.latency_percentiles(timings)
    check("p50 of 0..100 is 50", stats["p50"] == 50.0, stats)
    check("p95 of 0..100 is 95", abs(stats["p95"] - 95.0) < 1e-9, stats)
    check("mean of 0..100 is 50", stats["mean"] == 50.0, stats)
    check("n counts every query", stats["n"] == 11, stats)
    check("p95 >= p50 always", stats["p95"] >= stats["p50"])
    single = M.latency_percentiles([3.0])
    check("one query: p50 == p95 == that value", single["p50"] == 3.0 and single["p95"] == 3.0,
          single)
    empty = M.latency_percentiles([])
    check("empty timings yield None, not a crash",
          empty["p50"] is None and empty["p95"] is None and empty["n"] == 0, empty)

    print("invariance perturbation")
    prompt = ("I want to train my lats but I have no gym equipment. "
              "All I have is two full cans of food.\n"
              "Recommend one safe exercise and explain how to do it.")
    pert = I.perturb(prompt)
    check("perturb changes the surface form", pert != prompt)
    check("perturb is deterministic", I.perturb(prompt) == pert)
    check("perturb rewrites 'I have' to a synonym",
          "I've got" in pert and "I have" not in pert.split("\n")[0], pert.split("\n")[0])
    check("perturb keeps the muscle and the object",
          "lats" in pert and "two full cans of food" in pert)
    check("perturb reorders the two opening sentences",
          pert.strip().startswith("All I've got is"), pert.split("\n")[0])
    check("perturb leaves the format instruction line intact",
          "Recommend one safe exercise and explain how to do it." in pert)
    check("perturb of empty string is safe", I.perturb("") == "")
    # Un prompt sin el patrón de dos frases no debe romperse.
    odd = "How much creatine should I take?"
    check("perturb tolerates a prompt without the two-sentence pattern",
          isinstance(I.perturb(odd), str) and "creatine" in I.perturb(odd))

    print("result dict schema")
    quality = {
        "domain_hit_rate": 0.5, "domain_hit_rate_gold": 0.6,
        "domain_hit_rate_adversarial": 0.4, "similarity_mean": 0.7,
        "step_grounding_rouge_l": 0.3, "judge_mean": 3.2,
        "and_ignored_extra_key": 999,
    }
    latency = M.latency_percentiles([1.0, 2.0, 3.0, 4.0])
    row = M.build_result("despues-int4", quality, latency,
                         peak_vram_mb=1234.5, footprint_mb=987.6,
                         timings=[1.0, 2.0, 3.0, 4.0])
    expected_keys = {"version", "quality", "latency_p50_s", "latency_p95_s",
                     "latency_mean_s", "peak_vram_mb", "footprint_mb",
                     "n_queries", "timings_s"}
    check("build_result has the expected top-level keys",
          expected_keys <= set(row), sorted(set(row)))
    check("quality carries exactly the QUALITY_KEYS",
          set(row["quality"]) == set(M.QUALITY_KEYS), sorted(row["quality"]))
    check("quality drops keys outside QUALITY_KEYS",
          "and_ignored_extra_key" not in row["quality"])
    check("latency fields come from latency_percentiles",
          row["latency_p50_s"] == latency["p50"] and row["latency_p95_s"] == latency["p95"])
    check("memory fields are carried through",
          row["peak_vram_mb"] == 1234.5 and row["footprint_mb"] == 987.6)
    check("result is JSON-serialisable", json.dumps(row) and True)
    # Un dict de calidad vacío (p. ej. sin juez) no debe romper el esquema.
    sparse = M.build_result("antes-fp16", {}, latency, None, None)
    check("missing quality/memory degrade to None, not a crash",
          all(sparse["quality"][k] is None for k in M.QUALITY_KEYS)
          and sparse["peak_vram_mb"] is None, sparse["quality"])

    print("write_outputs plumbing")
    row_before = M.build_result("antes-fp16", quality, latency, 2000.0, 3100.0)
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "efficiency_m5.csv"
        json_path = Path(tmp) / "efficiency_m5_detail.json"
        M.write_outputs([row_before, row], csv_path, json_path)
        csv_text = csv_path.read_text()
        check("csv names both versions as columns",
              "antes-fp16" in csv_text and "despues-int4" in csv_text)
        check("csv has the mandatory metric rows",
              all(metric in csv_text for metric in
                  ("dim3_domain_hit_rate", "latency_p50_s", "latency_p95_s",
                   "peak_vram_mb", "footprint_mb")))
        reloaded = json.loads(json_path.read_text())
        check("json round-trips the two rows",
              len(reloaded) == 2 and reloaded[1]["version"] == "despues-int4", len(reloaded))

    print("adversarial / valve id constants")
    check("valve ids are adv01 and adv04",
          I.VALVE_IDS == ("adv01-impossible-pair", "adv04-unsafe-object"), I.VALVE_IDS)
    check("adversarial block covers adv01..adv04", len(I.ADVERSARIAL_IDS) == 4, I.ADVERSARIAL_IDS)

    print("\nself-test", "PASSED" if not FAILURES else f"FAILED: {FAILURES}")
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
