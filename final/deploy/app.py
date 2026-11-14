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

Dónde corre (el mismo archivo en los tres casos):
  - Space con ZeroGPU: `SPACES_ZERO_GPU` está definida. El generador se carga en
    `cuda` al arrancar (fuera de `@spaces.GPU`, como pide ZeroGPU) y la GPU solo
    se pide dentro de `run_advanced`. El encoder de consultas y el reranker
    también van en `cuda`; los embeddings del catálogo se calculan en un proceso
    hijo (`embed_docs.py`). Ver `build_zerogpu_retriever`.
  - Colab o un computador con GPU: `SHARE=1` imprime un enlace público de Gradio.
    `QUANTIZED=1` carga el generador int4 (la técnica de eficiencia medida).
  - Sin el paquete `spaces`, `@spaces.GPU` se reemplaza por un decorador que no
    hace nada, así que el self-test y Colab importan este archivo sin cambios.
"""

import os
import sys
import time
from pathlib import Path

try:
    import spaces
except ImportError:  # Colab, local y el self-test: el decorador no hace nada.
    spaces = None

# ROOT = raíz del repo (…/JoseJimenez-...). final/deploy/app.py -> parents[2].
ROOT = Path(__file__).resolve().parents[2]
# `final/` va en sys.path para que `from efficiency import efficiency` resuelva.
for _sub in ("scripts", "eval", "m3", "final"):
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

# ZeroGPU define esta variable en el Space. Decide dónde van los modelos.
ON_ZEROGPU = bool(os.environ.get("SPACES_ZERO_GPU"))

# Tope de GPU por petición en ZeroGPU. ZeroGPU compara lo que la petición
# RESERVA (este tope, más un margen suyo) con la cuota diaria que le queda a
# quien usa la demo, no lo que de verdad usa: con 60 s una cuenta gratuita
# (5 min al día) pedía 90 s y se quedaba sin cuota tras pocas peticiones. El
# RAG avanzado son dos generaciones (borrador de HyDE + respuesta): mediana
# 15.2 s y máximo 31.6 s en un T4 (reports/latency_m3.csv), y la GPU de ZeroGPU
# es varias veces más rápida.
GPU_SECONDS = 30

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
    "Esta herramienta **sugiere**; la decisión es tuya y no sustituye a un "
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
# Generador para ZeroGPU
# --------------------------------------------------------------------------

def load_zerogpu_generator(adapter):
    """El generador M1 (Qwen3-1.7B + LoRA fusionado) en bf16, puesto en `cuda`.

    Es la misma carga que `mg.load_model` sin cuantizar, con el dispositivo y el
    dtype fijos. `mg.pick_device_dtype` pregunta la capacidad de la GPU, y al
    arrancar un Space de ZeroGPU todavía no hay una GPU real a la cual
    preguntarle. bf16 porque la GPU de ZeroGPU (RTX Pro 6000, Blackwell) lo tiene
    en hardware; en el T4 de las mediciones fue fp16, y el reporte lo dice.

    Sin `set_seed` ni `torch.set_grad_enabled` aquí: en el proceso principal de
    ZeroGPU, una llamada de CUDA que la emulación no intercepta deja a los
    workers sin GPU ("No CUDA GPUs are available" en `worker_init`), y
    `set_seed` llama a `torch.cuda.manual_seed_all`. No cambia ninguna
    respuesta: la decodificación es greedy (sin muestreo) y
    `harness.make_hf_system` ya genera bajo `torch.no_grad()`.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from efficiency import efficiency as E

    tok = mg.load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(mg.BASE_MODEL, dtype=torch.bfloat16)
    # torch_device="cpu": al arrancar, ZeroGPU hace que torch.cuda.is_available()
    # diga True, y PEFT intentaría leer los pesos del adaptador directo a cuda,
    # donde todavía no hay GPU real ("No CUDA GPUs are available"). Se fusiona
    # en CPU y solo el modelo ya fusionado se pone en cuda.
    model = PeftModel.from_pretrained(model, adapter, torch_device="cpu").merge_and_unload()
    model = model.to("cuda").eval()
    return E.make_generator(model, tok)


class GpuDenseIndex:
    """Búsqueda densa exacta en la GPU, para el Space de ZeroGPU.

    Hace lo mismo que `R.DenseIndex.search` (similitud coseno entre la consulta y
    los 1324 documentos, con los mismos vectores normalizados de bge-small) como
    un producto de matrices en `cuda`, en vez de consultar Chroma. Chroma 1.x
    corre sobre un pool de hilos de Rust que no existe en el worker de ZeroGPU,
    que es un fork del proceso principal: la primera consulta dentro del worker
    se quedaba esperando para siempre. El HNSW de Chroma aproxima justo esta
    búsqueda exacta, así que el resultado solo cambia en el ruido que ya
    reportó M3 (dos construcciones del índice de Chroma tampoco coinciden
    siempre).
    """

    def __init__(self, docs, embeddings, model_id, query_prefix):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.query_prefix = query_prefix
        self.ids = [d["id"] for d in docs]
        # Al arrancar solo se COLOCA en cuda (la emulación de ZeroGPU lo
        # permite); todo cálculo ocurre dentro de `run_advanced`.
        self.matrix = torch.from_numpy(embeddings).to("cuda")
        self.encoder = SentenceTransformer(model_id, device="cuda")

    def count(self):
        """Cuántos vectores tiene el índice. Lo usa el health check: no calcula."""
        return len(self.ids)

    def search(self, query, n):
        import torch

        q = self.encoder.encode([self.query_prefix + query], normalize_embeddings=True,
                                convert_to_tensor=True)
        scores = self.matrix @ q[0].to(self.matrix.dtype)
        return [self.ids[i] for i in torch.topk(scores, n).indices.tolist()]


def build_zerogpu_retriever(docs):
    """El retriever de M3 para ZeroGPU: mismos vectores, mismo top 8, todo en GPU.

    Tres cosas lo separan de `E.build_retriever`, las tres medidas en el Space:

    - Los embeddings de los documentos se calculan en un proceso hijo
      (`embed_docs.py`): calcularlos con torch en el proceso principal dejaba
      a todos los workers sin GPU.
    - La búsqueda densa es `GpuDenseIndex`, no Chroma, que se cuelga dentro del
      worker.
    - El encoder de consultas y el reranker van en `cuda`: el worker hereda el
      pool de hilos de CPU de torch roto por el fork, así que nada calcula en
      CPU dentro de `run_advanced`.
    """
    import subprocess
    import tempfile

    import numpy as np
    import rag as R

    from efficiency import efficiency as E

    out = Path(tempfile.mkdtemp()) / "doc_embeddings.npy"
    subprocess.run([sys.executable, str(Path(__file__).with_name("embed_docs.py")), str(out)],
                   check=True)
    dense = GpuDenseIndex(docs, np.load(out), E.EMBED_MODEL,
                          R.EMBEDDING_CANDIDATES[E.EMBED_MODEL]["query_prefix"])
    return R.Retriever(docs, dense, reranker=R.load_reranker(device="cuda"),
                       k=E.TOP_K, pool=E.POOL)


# --------------------------------------------------------------------------
# Estado del sistema (modelos + índice)
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

        # En ZeroGPU no se fija la semilla (ver `load_zerogpu_generator`).
        if not ON_ZEROGPU:
            set_seed(H.SEED)

        self.docs = C.build_documents()
        self.catalog_index = mg.build_name_index(H.reachable_catalog())
        self.full_index = mg.build_name_index(mg.load_catalog())
        self.techniques = R.TECHNIQUES  # hybrid + rerank + hyde (avanzado)

        # Generador: int4 si QUANTIZED=1, fp16 en caso contrario. Ambas ramas
        # reutilizan los atajos de final/efficiency/efficiency.py, así el
        # generador desplegado y la fila medida en la tabla antes/después salen
        # del MISMO código. En ZeroGPU va su propia carga (ver
        # `load_zerogpu_generator`). La ruta del adaptador es absoluta porque en
        # el Space el directorio de trabajo no es la raíz del repo.
        from efficiency import efficiency as E

        adapter = str(ROOT / E.ADAPTER)
        if self.quantized:
            llm = E.load_int4_generator(adapter=adapter)
        elif ON_ZEROGPU:
            llm = load_zerogpu_generator(adapter)
        else:
            llm = E.load_fp16_generator(adapter=adapter)

        # El mismo retriever que midió M3 (bge-small, rerank sobre 20, top 8).
        # En ZeroGPU los embeddings de los documentos se calculan en otro
        # proceso (ver `build_zerogpu_retriever`).
        if ON_ZEROGPU:
            self.retriever = build_zerogpu_retriever(self.docs)
        else:
            self.retriever = E.build_retriever(self.docs)
        S.bind(self.retriever, llm, self.docs, techniques=self.techniques)
        self.systems = S
        self.ready = True
        return self


# --------------------------------------------------------------------------
# La parte que usa la GPU
# --------------------------------------------------------------------------

# El estado que sirve la app. Lo asigna `main()` al arrancar; `run_advanced` lo
# lee desde aquí porque ZeroGPU ejecuta la función decorada en otro proceso, que
# hereda las variables globales pero no los argumentos que no se pueden copiar.
STATE = None


def _gpu(fn):
    """`@spaces.GPU(duration=GPU_SECONDS)` en un Space; sin `spaces`, nada."""
    if spaces is None:
        return fn
    return spaces.GPU(duration=GPU_SECONDS)(fn)


@_gpu
def run_advanced(prompt):
    """El RAG avanzado de M3 (HyDE + híbrido + rerank + generación) sobre STATE.

    Es lo único que necesita GPU: el borrador de HyDE y la respuesta final son
    dos generaciones. En ZeroGPU la GPU se asigna al entrar y se libera al salir.
    """
    return STATE.systems.sistema_rag_avanzado(prompt)


# --------------------------------------------------------------------------
# El manejador de una petición (texto + imagen + log + observabilidad)
# --------------------------------------------------------------------------

def handle_request(state, observer, target, obj, run=None):
    """Procesa una petición de punta a punta y devuelve (texto, ruta_media).

    Pasos: construir el prompt (misma definición que M1/M2/M3), correr el sistema
    avanzado, aplicar el FILTRO de producción, resolver la imagen si procede,
    registrar el log JSON sin PII y la observabilidad W&B. `run` es quien genera
    la respuesta; por defecto, el sistema avanzado de `state` sin decorar.
    """
    start = time.perf_counter()
    prompt = mg.build_prompt(target, obj)

    reply = (run or state.systems.sistema_rag_avanzado)(prompt)

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
    observer.log_request(latency_ms, comps, refused, passed, has_media,
                         muscle=record_json["muscle"],
                         object_category=record_json["object_category"])

    return display_text, media_path


# --------------------------------------------------------------------------
# La UI de Gradio (import perezoso de gradio)
# --------------------------------------------------------------------------

def catalog_muscles():
    """Los músculos del catálogo alcanzable, con la grafía que usa el filtro."""
    return sorted({record["target"] for record in H.reachable_catalog()})


def health_json(state):
    """El health check como texto JSON, para la UI y para la API `/health`."""
    import json

    import health as health_mod

    return json.dumps(health_mod.health_check(state), ensure_ascii=False, indent=2)


def build_interface(state=None, observer=None, run=None):
    """Construye los Blocks de Gradio. Importa gradio aquí, no al cargar el módulo.

    El estado ya viene cargado (`main()` lo carga al arrancar). Cargar en el
    primer clic hacía que esa petición tardara minutos, y ZeroGPU necesita los
    modelos en `cuda` antes de la primera llamada a `@spaces.GPU`.
    """
    import gradio as gr

    import wandb_utils as W

    state = state or SystemState()
    observer = observer or W.init_observer()

    def on_submit(target, obj):
        return handle_request(state, observer, target, obj, run=run)

    def on_health():
        return health_json(state)

    with gr.Blocks(title="MacGyver Gym Rat (M5)") as demo:
        gr.Markdown("# MacGyver Gym Rat\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column():
                # Lista cerrada con el vocabulario del catálogo: el filtro de
                # producción compara el músculo pedido con el del ejercicio, y
                # con texto libre "tricep" (el catálogo dice "triceps") rechazaba
                # una respuesta correcta. Lo vimos en el Space.
                target = gr.Dropdown(label="Músculo objetivo",
                                     choices=catalog_muscles(), value="biceps")
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
            # api_name="health" lo publica en la API de Gradio: se llama sin
            # abrir la página (ver final/deploy/README.md, "Health check").
            health_btn.click(on_health, inputs=None, outputs=health_out,
                             api_name="health")

        submit.click(on_submit, inputs=[target, obj], outputs=[out_text, out_media],
                     api_name="recommend")

    return demo, state, observer


def main():  # pragma: no cover - solo en el Space/Colab con deps completas
    global STATE

    # Cargar ANTES de lanzar: en ZeroGPU es obligatorio y en Colab evita que la
    # primera petición de la sustentación pague la carga.
    STATE = SystemState().load()
    demo, _state, _observer = build_interface(STATE, run=run_advanced)
    # En el Space, Gradio elige host y puerto. En Colab o un computador,
    # SHARE=1 imprime un enlace público https://xxxx.gradio.live.
    share = os.environ.get("SHARE", "0") == "1"
    # `allowed_paths` deja que Gradio sirva los GIF del catálogo, que viven fuera
    # del directorio de trabajo cuando el repo se clona dentro del Space.
    demo.launch(share=share, allowed_paths=[str(DATASET_DIR)])


if __name__ == "__main__":  # pragma: no cover
    main()
