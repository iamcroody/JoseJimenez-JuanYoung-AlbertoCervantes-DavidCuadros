"""Self-test sin GPU ni red del despliegue del Módulo 5.

    python final/deploy/selftest.py

Verifica, con SOLO stdlib + numpy disponibles (sin gradio, torch, wandb,
chromadb ni sentence_transformers), las piezas puras del Space:

  (a) el esquema del log JSON por petición (logging_utils) y la ausencia de PII;
  (b) la derivación de `components_used` y `valve_fired` a partir de respuestas
      falsas (app.py), incluyendo técnicas activas y presencia de imagen;
  (c) `health.health_check` con los loaders MONKEYPATCHED/mockeados, de modo que
      devuelve un dict de estado bien formado sin cargar pesos ni Chroma;
  (d) el FILTRO de producción (satisfacción de restricciones, sección 7) sobre un
      catálogo FALSO diminuto, reutilizando la dimensión 3 del harness, y la
      resolución de imagen/GIF del ejercicio nombrado sobre un dataset falso.

Es el análogo de `m3/selftest.py` y de `final/efficiency/selftest.py` para la
feature de despliegue: encontrar un error de plomería aquí cuesta un segundo en
lugar de una sesión de Colab T4 o un deploy fallido.
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import logging_utils as L  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# Catálogo e índices falsos: evitan tocar el catálogo real y mantienen el
# self-test corriendo con solo stdlib+numpy.
# --------------------------------------------------------------------------

def fake_catalog():
    """Dos ejercicios reales-de-mentira y uno inalcanzable para probar el filtro."""
    return [
        {"id": "0001", "name": "test biceps curl", "target": "biceps",
         "equipment": "dumbbell", "body_part": "upper arms",
         "secondary_muscles": [],
         "instruction_steps": {"en": ["Stand up.", "Curl the weight."]},
         "image": "images/0001-x.jpg", "gif_url": "videos/0001-x.gif"},
        {"id": "0002", "name": "test air squat", "target": "quads",
         "equipment": "body weight", "body_part": "upper legs",
         "secondary_muscles": [],
         "instruction_steps": {"en": ["Stand.", "Squat down."]},
         "image": "images/0002-y.jpg", "gif_url": "videos/0002-y.gif"},
    ]


def main():
    import app  # importa sin gradio/torch gracias a imports perezosos
    import health as health_mod
    import macgyver as mg

    print("logging schema / no PII")
    rec = L.build_record(
        latency_ms=123.456, muscle="Biceps  ",
        object_category_value="dumbbell",
        components_used=["rag", "hyde", "vision", "not-a-real-component"],
        valve_fired=False, passed_filter=True, recommended=True)
    problems = L.validate_record(rec)
    check("a well-formed record validates clean", problems == [], problems)
    check("timestamp is ISO 8601 with timezone", rec["timestamp"].endswith("+00:00")
          or rec["timestamp"].endswith("Z"), rec["timestamp"])
    check("latency is numeric", isinstance(rec["latency_ms"], float))
    check("muscle is lowercased and trimmed", rec["muscle"] == "biceps", rec["muscle"])
    check("unknown components are dropped from the record",
          "not-a-real-component" not in rec["components_used"], rec["components_used"])
    check("components keep the schema order",
          rec["components_used"] == ["rag", "hyde", "vision"], rec["components_used"])

    # Anti-PII: un texto libre con datos personales nunca debe caber entero.
    leaky = L.redact("My name is Juan Perez and I live at Calle 10 #5, phone 3001234567")
    check("redact truncates free text hard", len(leaky) <= L.MAX_FIELD_CHARS, len(leaky))
    check("object_category never stores the raw object phrase",
          L.object_category("two filled water bottles") == "unknown",
          L.object_category("two filled water bottles"))
    check("object_category accepts a real equipment class",
          L.object_category("dumbbell") == "dumbbell")

    # Un registro con un componente inventado inyectado a mano debe fallar la
    # validación (defensa en profundidad del esquema).
    bad = dict(rec)
    bad["components_used"] = ["telepathy"]
    check("validate_record rejects an out-of-schema component",
          L.validate_record(bad) != [])
    bad2 = dict(rec)
    bad2["muscle"] = "x" * (L.MAX_FIELD_CHARS + 5)
    check("validate_record flags an over-long field as a possible PII leak",
          any("PII" in p for p in L.validate_record(bad2)))

    print("components_used / valve_fired derivation")
    valve_reply = ("No safe option: I know no lats exercise that works with "
                   "two full cans of food.")
    answer_reply = ("Exercise: test biceps curl\n"
                    "Gym equivalent: dumbbell\n"
                    "Steps:\n1. Stand up.\n2. Curl the weight.\n"
                    "Safety: keep it controlled.")
    check("the valve reply is detected as a refusal", app.valve_fired(valve_reply))
    check("a normal answer is not a refusal", not app.valve_fired(answer_reply))

    comps = app.components_used(("hybrid", "rerank", "hyde"), answer_reply,
                                has_media=True, used_tool=True)
    check("components_used always includes rag", "rag" in comps)
    check("components_used lists active rerank and hyde",
          "rerank" in comps and "hyde" in comps, comps)
    check("components_used adds tool and vision when present",
          "tool" in comps and "vision" in comps, comps)
    comps_norerank = app.components_used(("hybrid",), answer_reply, has_media=False)
    check("components_used omits vision when there is no media",
          "vision" not in comps_norerank, comps_norerank)
    check("components_used omits inactive techniques",
          "rerank" not in comps_norerank and "hyde" not in comps_norerank,
          comps_norerank)

    print("production constraint-satisfaction filter (dim 3 reuse)")
    catalog = fake_catalog()
    index = mg.build_name_index(catalog)   # alcanzable == el catálogo falso entero
    obj = "two filled water bottles"        # stands for a dumbbell
    passed, detail = app.passes_production_filter(
        answer_reply, "biceps", obj, catalog_index=index, full_index=index)
    check("a verified real exercise passes the filter", passed, detail.get("why"))

    # Un ejercicio inventado (no está en el catálogo) debe FALLAR el filtro.
    invented = ("Exercise: mega turbo biceps blaster\n"
                "Gym equivalent: dumbbell\n"
                "Steps:\n1. Do the thing.\nSafety: ok.")
    passed_inv, detail_inv = app.passes_production_filter(
        invented, "biceps", obj, catalog_index=index, full_index=index)
    check("an invented exercise fails the filter", not passed_inv, detail_inv.get("why"))

    # Un ejercicio real pero para el músculo equivocado debe fallar.
    wrong_muscle = ("Exercise: test air squat\n"
                    "Gym equivalent: body weight\n"
                    "Steps:\n1. Stand.\n2. Squat down.\nSafety: ok.")
    passed_wm, _ = app.passes_production_filter(
        wrong_muscle, "biceps", obj, catalog_index=index, full_index=index)
    check("a real exercise for the wrong muscle fails the filter", not passed_wm)

    print("media resolution for the visual component")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "videos").mkdir()
        (tmp / "images").mkdir()
        gif = tmp / "videos" / "0001-x.gif"
        gif.write_bytes(b"GIF89a")
        media, record = app.resolve_media(answer_reply, catalog_index=index,
                                          dataset_dir=tmp)
        check("resolve_media returns the catalog GIF for a named exercise",
              media == str(gif), media)
        check("resolve_media returns the matching record",
              record is not None and record["id"] == "0001")
        # Sin archivo presente y sin imagen de respaldo -> None.
        media_none, _ = app.resolve_media(invented, catalog_index=index, dataset_dir=tmp)
        check("resolve_media returns None for an unresolvable exercise",
              media_none is None)

    print("health_check with mocked loaders")

    class FakeCollection:
        def query(self, query_embeddings=None, n_results=1):
            return {"ids": [["0001"]]}

        def count(self):
            return 2

    class FakeEncoder:
        def get_sentence_embedding_dimension(self):
            return 8

    class FakeDense:
        def __init__(self):
            self.collection = FakeCollection()
            self.encoder = FakeEncoder()

    class FakeRetriever:
        def __init__(self):
            self.dense = FakeDense()

    class FakeState:
        ready = True
        quantized = False
        retriever = FakeRetriever()
        systems = object()

    status = health_mod.health_check(FakeState())
    for key in ("status", "models_loaded", "index_ready", "git_commit", "timestamp"):
        check(f"health_check reports {key}", key in status, sorted(status))
    check("health_check is ok when models and index respond",
          status["status"] == "ok" and status["models_loaded"]
          and status["index_ready"], status["status"])

    # Estado no cargado -> 'down', sin tocar ningún modelo.
    class NotReady:
        ready = False
    down = health_mod.health_check(NotReady())
    check("health_check is 'down' when models are not loaded",
          down["status"] == "down" and not down["models_loaded"], down["status"])

    # Índice que falla -> 'degraded' (modelos cargados, índice no responde).
    class BrokenCollection:
        def query(self, **kw):
            raise RuntimeError("chroma offline")

        def count(self):
            raise RuntimeError("chroma offline")

    class BrokenDense:
        def __init__(self):
            self.collection = BrokenCollection()
            self.encoder = FakeEncoder()

    class BrokenRetriever:
        def __init__(self):
            self.dense = BrokenDense()

    class DegradedState:
        ready = True
        quantized = True
        retriever = BrokenRetriever()
        systems = object()

    degraded = health_mod.health_check(DegradedState())
    check("health_check is 'degraded' when the index does not respond",
          degraded["status"] == "degraded" and degraded["models_loaded"]
          and not degraded["index_ready"], degraded["status"])
    check("health_check carries the quantized flag", degraded["quantized"] is True)

    print("wandb observer no-ops without a key")
    with mock.patch.dict("os.environ", {}, clear=True):
        import wandb_utils as W

        observer = W.init_observer(project="test-proj")
        check("observer is disabled without WANDB_API_KEY", not observer.enabled)
        # log_request debe actualizar contadores en memoria sin tocar la red.
        observer.log_request(120.0, ["rag", "vision"], valve_fired=False,
                             passed_filter=True, recommended=True)
        observer.log_request(90.0, ["rag"], valve_fired=True,
                             passed_filter=False, recommended=False)
        check("observer counts recommendations and refusals",
              observer.counters["recommendations"] == 1
              and observer.counters["refusals"] == 1, observer.counters)
        check("observer counts a filter block",
              observer.counters["filter_blocks"] == 1, observer.counters)
        check("observer tallies component usage",
              observer.component_counts["rag"] == 2
              and observer.component_counts["vision"] == 1, observer.component_counts)
        observer.finish()  # no-op seguro cuando está deshabilitado

    print("ZeroGPU wiring and the M3 retriever settings")
    check("without the spaces package the GPU decorator is a no-op",
          app.spaces is None and app.run_advanced.__name__ == "run_advanced")
    check("final/ is on sys.path so the efficiency module imports",
          str(app.ROOT / "final") in sys.path)
    from efficiency import efficiency as E
    check("the app serves the retriever M3 measured (bge-small, top 8, pool 20)",
          (E.EMBED_MODEL, E.TOP_K, E.POOL) == ("BAAI/bge-small-en-v1.5", 8, 20),
          (E.EMBED_MODEL, E.TOP_K, E.POOL))

    print("handle_request end to end with a fake system")

    class FakeSystems:
        @staticmethod
        def sistema_rag_avanzado(prompt):
            raise AssertionError("the injected run should be used, not this one")

    class ServingState:
        ready = True
        techniques = ("hybrid", "rerank", "hyde")
        systems = FakeSystems()
        catalog_index = index
        full_index = index

    with mock.patch.dict("os.environ", {}, clear=True):
        import wandb_utils as W

        serving_observer = W.init_observer(project="test-proj")
    seen = []

    def fake_run(prompt):
        seen.append(prompt)
        return answer_reply

    text, _media = app.handle_request(ServingState(), serving_observer, "biceps", obj,
                                      run=fake_run)
    check("handle_request sends the M1 prompt to the injected run",
          seen == [mg.build_prompt("biceps", obj)])
    check("a verified answer is shown as the model wrote it", text == answer_reply)

    text_inv, media_inv = app.handle_request(ServingState(), serving_observer, "biceps",
                                             obj, run=lambda prompt: invented)
    check("an unverified answer is replaced by the valve message, with no image",
          text_inv == app.VALVE_MESSAGE and media_inv is None)

    print("health as JSON for the /health API endpoint")
    import json

    health = json.loads(app.health_json(NotReady()))
    check("health_json is valid JSON and reports 'down' before loading",
          health["status"] == "down", health["status"])

    print("\nself-test", "PASSED" if not FAILURES else f"FAILED: {FAILURES}")
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
