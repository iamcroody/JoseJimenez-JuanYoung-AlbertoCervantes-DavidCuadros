"""App Gradio del Módulo 5: el sistema MacGyver Gym Rat de punta a punta.

Entrada: músculo objetivo + objeto casero. Salida: la recomendación del sistema
M3 (por defecto `sistema_rag_avanzado`, el mejor en la dimensión 3 del harness) y,
cuando la respuesta nombra un ejercicio real del catálogo, la IMAGEN/GIF de ese
ejercicio (el componente visual que exige M4). En un rechazo se muestra el mensaje
de la válvula y ninguna imagen.

Este archivo NO reimplementa el pipeline ni el scoring: reutiliza
  - `m3/rag.py` + `m3/systems.py` para el sistema RAG,
  - `m3/corpus.py` para construir los documentos y resolver el ejercicio a su id,
  - `eval/harness.py` (dimensión 3) para el FILTRO de satisfacción de
    restricciones de producción (sección 7 de PROJECT_DEFINITION.md),
  - `final/efficiency/efficiency.py` para el generador int4 cuando `QUANTIZED=1`.

Rutas estilo Colab: el repo se clona en
`/content/JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros`; agregamos
`scripts/`, `eval/` y `m3/` a `sys.path` igual que en los notebooks, para que el
traslado al Space o a Colab sea directo.

Importaciones PEREZOSAS: cargar este módulo NO importa gradio, torch, chromadb ni
sentence_transformers. Así `ast.parse`/`py_compile` y `selftest.py` quedan limpios
sin esas dependencias. El filtro de producción y la resolución de imagen son puros
(stdlib + el catálogo) y el self-test los ejercita con un catálogo falso.

Determinismo: seed 42 y decodificación greedy, los mismos de M1/M2/M3 (los fija
`harness.make_hf_system`, que usan tanto M3 como el generador int4).
"""

import os
import sys
import time
from pathlib import Path

# ROOT = raíz del repo (…/JoseJimenez-...). final/deploy/app.py -> parents[2].
ROOT = Path(__file__).resolve().parents[2]
for _sub in ("scripts", "eval", "m3"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)
_DEPLOY = str(Path(__file__).resolve().parent)
if _DEPLOY not in sys.path:
    sys.path.insert(0, _DEPLOY)

# Estos módulos del repo son puros (sus deps pesadas son perezosas), así que
# importarlos no arrastra torch/chromadb y el self-test puede cargar este archivo.
import macgyver as mg  # noqa: E402
import harness as H  # noqa: E402
import corpus as C  # noqa: E402

import logging_utils as L  # noqa: E402

# Carpeta del dataset (submódulo). Las imágenes/GIFs viven en
# data/exercises-dataset/{images,videos}/<id>-<media_id>.{jpg,gif} y cada record
# del catálogo ya trae las rutas relativas en `image` y `gif_url`.
DATASET_DIR = ROOT / "data" / "exercises-dataset"

# El objeto casero por defecto en la UI. La app pide el objeto como frase libre
# del catálogo de equivalencias (macgyver.OBJECT_MAP) para poder mapearlo a su
# clase de equipo sin registrar la frase cruda en el log.
DEFAULT_OBJECTS = [o for eq in mg.OBJECT_MAP
                   for o in mg.objects_for(eq, include_holdout=True)]

VALVE_MESSAGE = (
    "No safe option: I know no exercise for that muscle that works safely with "
    "what you have. This tool only covers four equipment classes (dumbbell, "
    "kettlebell, weighted, body weight) and suggests; it does not replace a "
    "trainer or a health professional."
)

# Aviso de alcance y de que el sistema SUGIERE, una persona decide (requisito del
# PDF para dominios sensibles; aquí el dominio es salud/ejercicio). Las imágenes
# provienen del CATÁLOGO, no son generadas: se marca explícitamente.
DISCLAIMER = (
    "⚠️ Esta herramienta **sugiere**; la decisión es tuya y no sustituye a un "
    "entrenador ni a un profesional de la salud. Cubre solo 4 clases de equipo "
    "(dumbbell, kettlebell, weighted, body weight). La imagen/GIF proviene del "
    "catálogo de ejercicios (NO es generada por IA)."
)


# --------------------------------------------------------------------------
# Filtro de satisfacción de restricciones (producción), sección 7
# --------------------------------------------------------------------------

def passes_production_filter(reply, target, obj, catalog_index=None,
                             full_index=None):
    """¿La respuesta nombra un ejercicio REAL que entrena el músculo con el objeto?

    Reutiliza la MISMA lógica de la dimensión 3 del harness (`H.domain_check`
    sobre un caso 'answerable'): resuelve el ejercicio nombrado contra el catálogo
    alcanzable y verifica que existe, entrena el músculo pedido y usa la clase de
    equipo que el objeto representa. No duplica el scoring: arma un `example` y
    delega en `harness.domain_check`.

    Devuelve (passed, detail): `detail` es el dict de `domain_check`, útil para el
    log y para decidir si disparó la válvula.
    """
    if catalog_index is None:
        catalog_index = mg.build_name_index(H.reachable_catalog())
    if full_index is None:
        full_index = mg.build_name_index(mg.load_catalog())

    example = {
        "id": "live-request",
        "kind": "answerable",
        "input": "",
        "esperado": "",
        "criterio": "",
        "meta": {"target": (target or "").strip().lower(), "object": obj},
    }
    detail = H.domain_check(example, reply, catalog_index, full_index)
    return bool(detail["hit"]), detail


# --------------------------------------------------------------------------
# Resolver el ejercicio nombrado a su imagen/GIF del catálogo (componente visual)
# --------------------------------------------------------------------------

def resolve_media(reply, catalog_index=None, dataset_dir=DATASET_DIR):
    """Mapea el ejercicio nombrado en la respuesta a su GIF (o imagen) del catálogo.

    Reutiliza el parser de respuestas (`mg.parse_reply`) y el índice de nombres
    (`mg.build_name_index`) de M3. Prefiere el GIF (`gif_url`), con la imagen fija
    (`image`) como respaldo. Devuelve la ruta absoluta existente o None si el
    ejercicio no se resuelve o el archivo no está presente.
    """
    if catalog_index is None:
        catalog_index = mg.build_name_index(mg.load_catalog())
    parsed = mg.parse_reply(reply)
    if not parsed["exercise"]:
        return None, None
    record = catalog_index.get(mg.normalise_name(parsed["exercise"]))
    if record is None:
        return None, None

    for key in ("gif_url", "image"):
        rel = record.get(key)
        if not rel:
            continue
        path = Path(dataset_dir) / rel
        if path.exists():
            return str(path), record
    return None, record


# --------------------------------------------------------------------------
# Derivación de components_used a partir de las técnicas y la traza
# --------------------------------------------------------------------------

def components_used(techniques, reply, has_media, used_tool=False):
    """Lista de componentes para el log, a partir de técnicas activas y resultado.

    'rag' siempre (el sistema desplegado es RAG). 'hybrid' del pipeline cuenta
    como 'rag'; 'rerank'/'hyde' se listan si están activos. 'tool' si el agente
    llamó a la herramienta. 'vision' si se resolvió y mostró una imagen/GIF.
    """
    comps = ["rag"]
    for technique in ("rerank", "hyde"):
        if technique in set(techniques or ()):
            comps.append(technique)
    if used_tool:
        comps.append("tool")
    if has_media:
        comps.append("vision")
    return comps


def valve_fired(reply):
    """¿La respuesta es un rechazo según el parser congelado del harness?"""
    return bool(mg.parse_reply(reply)["refused"])


# --------------------------------------------------------------------------
# Estado del sistema (modelos + índice), cargado perezosamente
# --------------------------------------------------------------------------

class SystemState:
    """Mantiene el sistema M3 bound, el retriever, los docs y los índices.

    Se construye una sola vez al arrancar el Space (o bajo demanda). Centraliza
    la elección fp16 vs int4 según la variable de entorno `QUANTIZED`.
    """

    def __init__(self):
        self.ready = False
        self.quantized = os.environ.get("QUANTIZED", "0") == "1"
        self.techniques = None
        self.docs = None
        self.catalog_index = None
        self.full_index = None
        self.retriever = None
        self.systems = None

    def load(self):
        """Carga modelos e índice. Importa torch/chromadb/etc. SOLO aquí."""
        import systems as S
        import rag as R
        from transformers import set_seed

        set_seed(H.SEED)

        self.docs = C.build_documents()
        self.catalog_index = mg.build_name_index(H.reachable_catalog())
        self.full_index = mg.build_name_index(mg.load_catalog())
        self.techniques = R.TECHNIQUES  # hybrid + rerank + hyde (avanzado)

        # Generador: int4 si QUANTIZED=1 (puede caber en hardware gratuito del
        # Space), fp16 en caso contrario. Ambas ramas reutilizan los atajos de
        # final/efficiency/efficiency.py, así el generador desplegado y la fila
        # medida en la tabla antes/después salen del MISMO código (una sola
        # fuente de verdad, sin deriva entre el Space y la medición).
        from efficiency import efficiency as E

        if self.quantized:
            llm = E.load_int4_generator()
        else:
            llm = E.load_fp16_generator()

        # Índice denso + reranker (los deps pesados se importan dentro de rag.py).
        embedder_id = next(iter(R.EMBEDDING_CANDIDATES))
        dense = R.DenseIndex(self.docs, embedder_id)
        reranker = R.load_reranker()
        self.retriever = R.Retriever(self.docs, dense, reranker=reranker)
        S.bind(self.retriever, llm, self.docs, techniques=self.techniques)
        self.systems = S
        self.ready = True
        return self


# --------------------------------------------------------------------------
# El manejador de una petición (texto + imagen + log + observabilidad)
# --------------------------------------------------------------------------

def handle_request(state, observer, target, obj):
    """Procesa una petición de punta a punta y devuelve (texto, ruta_media).

    Pasos: construir el prompt (misma definición que M1/M2/M3), correr el sistema
    avanzado, aplicar el FILTRO de producción, resolver la imagen si procede,
    registrar el log JSON sin PII y la observabilidad W&B.
    """
    start = time.perf_counter()
    prompt = mg.build_prompt(target, obj)

    reply = state.systems.sistema_rag_avanzado(prompt)

    refused = valve_fired(reply)
    passed, _detail = passes_production_filter(
        reply, target, obj, state.catalog_index, state.full_index)

    media_path, record = (None, None)
    if not refused and passed:
        media_path, record = resolve_media(reply, state.full_index, DATASET_DIR)
        display_text = reply
    else:
        # Rechazo del modelo O respuesta no verificada: mostramos la válvula y
        # NINGUNA imagen. Es la mitigación ética de la sección 7.
        display_text = VALVE_MESSAGE

    has_media = media_path is not None
    comps = components_used(state.techniques, reply, has_media)
    latency_ms = (time.perf_counter() - start) * 1000.0

    record_json = L.build_record(
        latency_ms=latency_ms,
        muscle=target,
        object_category_value=mg.equipment_for_object(obj) or "unknown",
        components_used=comps,
        valve_fired=refused,
        passed_filter=passed,
        recommended=has_media,
    )
    L.log_record(record_json)
    observer.log_request(latency_ms, comps, refused, passed, has_media)

    return display_text, media_path


# --------------------------------------------------------------------------
# La UI de Gradio (import perezoso de gradio)
# --------------------------------------------------------------------------

def build_interface(state=None, observer=None):
    """Construye los Blocks de Gradio. Importa gradio aquí, no al cargar el módulo."""
    import gradio as gr

    import health as health_mod
    import wandb_utils as W

    state = state or SystemState()
    observer = observer or W.init_observer()

    def _ensure_loaded():
        if not state.ready:
            state.load()

    def on_submit(target, obj):
        _ensure_loaded()
        text, media = handle_request(state, observer, target, obj)
        return text, media

    def on_health():
        import json as _json

        return _json.dumps(health_mod.health_check(state), ensure_ascii=False, indent=2)

    with gr.Blocks(title="MacGyver Gym Rat (M5)") as demo:
        gr.Markdown("# MacGyver Gym Rat\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column():
                target = gr.Textbox(label="Músculo objetivo (p. ej. biceps, lats)",
                                    value="biceps")
                obj = gr.Dropdown(label="Objeto casero disponible",
                                  choices=DEFAULT_OBJECTS,
                                  value=DEFAULT_OBJECTS[0], allow_custom_value=True)
                submit = gr.Button("Recomendar", variant="primary")
            with gr.Column():
                out_text = gr.Textbox(label="Recomendación", lines=10)
                out_media = gr.Image(label="Ejercicio del catálogo (imagen/GIF, no generada)")
        with gr.Accordion("Health check (liveness, sin consulta real)", open=False):
            health_btn = gr.Button("Ver estado")
            health_out = gr.Code(label="/health", language="json")
            health_btn.click(on_health, inputs=None, outputs=health_out)

        submit.click(on_submit, inputs=[target, obj], outputs=[out_text, out_media])

    return demo, state, observer


def main():  # pragma: no cover - solo en el Space/Colab con deps completas
    demo, _state, _observer = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":  # pragma: no cover
    main()
