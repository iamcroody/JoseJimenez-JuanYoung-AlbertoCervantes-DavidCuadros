"""Mide una versión del sistema sobre el MISMO eval set y la MISMA máquina.

Produce las columnas que la tabla antes/después de la asignación exige:

  - Calidad: las tres dimensiones del harness CONGELADO
    (`harness.harness(eval_set, system)`). No se reimplementa ninguna dimensión;
    se pasa el `system(prompt)->reply` ligado (p. ej. `sistema_rag_avanzado`
    sobre el generador fp16 o int4) y se leen `domain_hit_rate*`, `judge_mean*`,
    `similarity_mean`, etc.
  - Latencia p50 y p95: cada una de las 14 consultas se cronometra en la misma
    máquina y se agregan con `numpy.percentile`. La asignación pide p50 Y p95.
  - Memoria/tamaño: pico de VRAM con `torch.cuda.max_memory_allocated` y la
    huella de pesos con `efficiency.model_footprint_mb` (fp16 vs int4).

`measure_system(...)` hace una pasada de latencia + una pasada de harness y
devuelve un dict que el notebook escribe a `reports/efficiency_m5.csv` y
`reports/efficiency_m5_detail.json`.

La aritmética de percentiles (`latency_percentiles`) y el armado del dict de
resultado (`build_result`) son PUROS: solo necesitan numpy y se ejercen en
`selftest.py` sin cargar ningún modelo. Todo lo que toca torch/harness vive en
funciones que solo se llaman en Colab.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _sub in ("scripts", "eval", "m3"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Claves de calidad que se extraen del scorecard del harness para la tabla.
# La dimensión 3 (domain_hit_rate) es la columna "Quality (harness)" principal,
# por la postura de evaluación honesta del proyecto (se confía en la tasa de
# acierto de dominio por catálogo por encima del juez LLM).
QUALITY_KEYS = (
    "domain_hit_rate",
    "domain_hit_rate_gold",
    "domain_hit_rate_adversarial",
    "similarity_mean",
    "step_grounding_rouge_l",
    "judge_mean",
)


def latency_percentiles(timings):
    """p50 y p95 (segundos) de una lista de latencias por consulta.

    Puro: solo numpy. Se ejerce en selftest.py con una lista fija. Usa
    interpolación lineal (la de `numpy.percentile` por defecto), la misma para
    las filas 'antes' y 'después', que es lo que mantiene la comparación justa.
    """
    import numpy as np

    arr = np.asarray(list(timings), dtype=float)
    if arr.size == 0:
        return {"p50": None, "p95": None, "mean": None, "n": 0}
    return {
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "mean": round(float(arr.mean()), 3),
        "n": int(arr.size),
    }


def build_result(version, quality, latency, peak_vram_mb, footprint_mb,
                 timings=None, extra=None):
    """Arma el dict de una fila de la tabla. Puro: no toca torch ni el harness.

    `quality` es el scorecard del harness (o un sub-dict con las QUALITY_KEYS),
    `latency` es la salida de `latency_percentiles`. El esquema resultante es el
    que `selftest.py` valida y el notebook serializa.
    """
    quality = quality or {}
    result = {
        "version": version,
        "quality": {k: quality.get(k) for k in QUALITY_KEYS},
        "latency_p50_s": latency.get("p50") if latency else None,
        "latency_p95_s": latency.get("p95") if latency else None,
        "latency_mean_s": latency.get("mean") if latency else None,
        "peak_vram_mb": peak_vram_mb,
        "footprint_mb": footprint_mb,
        "n_queries": latency.get("n") if latency else None,
    }
    if timings is not None:
        result["timings_s"] = [round(float(t), 3) for t in timings]
    if extra:
        result.update(extra)
    return result


def time_queries(eval_set, system, warmup=True):
    """Cronometra cada consulta del eval set en la misma máquina.

    `system(prompt)->reply` es el sistema ya ligado. Una consulta de calentamiento
    (que no se cuenta) absorbe la compilación/caché de la primera generación,
    igual que `warm_up` en M3_rag.ipynb. Devuelve (timings, replies).
    """
    if warmup and eval_set:
        system(eval_set[0]["input"])

    timings, replies = [], {}
    for e in eval_set:
        start = time.perf_counter()
        reply = system(e["input"])
        timings.append(time.perf_counter() - start)
        replies[e["input"]] = reply
    return timings, replies


def peak_vram_mb():
    """Pico de VRAM asignada desde el último reset, en MB. None sin CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)


def reset_vram_peak():
    """Reinicia el contador de pico de VRAM antes de medir una versión."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def measure_system(version, system, model, eval_set, judge=None, similarity=None,
                   catalog=None, verbose=True):
    """Mide calidad + latencia + memoria de UNA versión del sistema.

    Pasos, en el orden que exige medir en la misma máquina:
      1. resetear el pico de VRAM;
      2. cronometrar las 14 consultas (latencia p50/p95) con calentamiento;
      3. correr el harness CONGELADO para las tres dimensiones;
      4. leer pico de VRAM y huella de pesos del `model`.

    `system` es el callable ligado (`sistema_rag_avanzado`/`sistema_final`),
    `model` es el modelo HF subyacente (para la huella). Reusa el harness; no
    reimplementa dimensiones. En Colab conviene pasar el mismo `judge` y
    `similarity` a 'antes' y 'después' para no recargarlos.
    """
    import harness as H
    from efficiency import model_footprint_mb

    reset_vram_peak()

    timings, _replies = time_queries(eval_set, system, warmup=True)
    latency = latency_percentiles(timings)

    scorecard = H.harness(eval_set, system, judge=judge, similarity=similarity,
                          catalog=catalog, label=version, verbose=verbose)

    return build_result(
        version=version,
        quality=scorecard,
        latency=latency,
        peak_vram_mb=peak_vram_mb(),
        footprint_mb=model_footprint_mb(model),
        timings=timings,
        extra={"scorecard": scorecard},
    )


def write_outputs(results, csv_path, json_path):
    """Escribe reports/efficiency_m5.csv y reports/efficiency_m5_detail.json.

    `results` es la lista [fila_antes, fila_despues] de `build_result`/
    `measure_system`. El CSV lleva una columna por versión, igual que el formato
    de `harness.write_scorecard` (agregar una versión es agregar una columna).
    Puro salvo la E/S: no toca torch.
    """
    import csv
    import json

    csv_path, json_path = Path(csv_path), Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    names = [r["version"] for r in results]
    rows = [
        ("dim3_domain_hit_rate", lambda r: r["quality"].get("domain_hit_rate")),
        ("dim3_domain_hit_rate_gold", lambda r: r["quality"].get("domain_hit_rate_gold")),
        ("dim3_domain_hit_rate_adversarial",
         lambda r: r["quality"].get("domain_hit_rate_adversarial")),
        ("dim1_similarity_mean", lambda r: r["quality"].get("similarity_mean")),
        ("dim1_step_grounding_rouge_l",
         lambda r: r["quality"].get("step_grounding_rouge_l")),
        ("dim2_judge_mean", lambda r: r["quality"].get("judge_mean")),
        ("latency_p50_s", lambda r: r.get("latency_p50_s")),
        ("latency_p95_s", lambda r: r.get("latency_p95_s")),
        ("latency_mean_s", lambda r: r.get("latency_mean_s")),
        ("peak_vram_mb", lambda r: r.get("peak_vram_mb")),
        ("footprint_mb", lambda r: r.get("footprint_mb")),
        ("n_queries", lambda r: r.get("n_queries")),
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric"] + names)
        for key, getter in rows:
            writer.writerow([key] + ["" if getter(r) is None else getter(r)
                                     for r in results])

    # El detalle lleva timings crudos y scorecards completos para auditoría.
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    return csv_path, json_path
