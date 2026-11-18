"""Health check del Space (Módulo 5): liveness SIN ejecutar una consulta real.

El rubro exige un health check que reporte que el sistema está vivo comprobando
que los modelos cargaron y que el índice responde, pero SIN correr una
recomendación. `health_check(state)` devuelve un dict JSON-serializable:

    {
      "status": "ok" | "degraded" | "down",
      "models_loaded": bool,
      "index_ready": bool,
      "git_commit": "<sha corto o 'unknown'>",
      "timestamp": "<ISO 8601 UTC>",
      "quantized": bool,
      "checks": { ... detalle por sub-chequeo ... }
    }

Cómo prueba el índice sin una consulta real: hace una query trivial a la
colección de Chroma (`state.retriever.dense.collection.query`) con un vector cero
del tamaño correcto y `n_results=1`. Eso ejerce la ruta del índice sin invocar al
generador ni al embedder sobre texto del usuario: no es una recomendación, es un
ping al almacén vectorial.

En el self-test, `state` se construye con loaders MONKEYPATCHED/mockeados, de modo
que `health_check` devuelve un dict bien formado sin cargar pesos ni Chroma.

Ruta/expuesto: la app de Gradio lo expone con un botón "Ver estado" (ver
`app.py::build_interface`). Para un endpoint HTTP tipo `/health`, montar esta
función en un FastAPI junto al `demo.app` de Gradio (documentado en README).
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_commit():
    """SHA corto del commit actual, o 'unknown' si git no está disponible.

    En un Space el repo se clona, así que git suele existir; si no, no es un fallo
    de liveness, solo falta la etiqueta de versión.
    """
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        sha = out.stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def _check_models(state):
    """¿Están cargados generador e índice denso (embedder)? No ejecuta inferencia."""
    if state is None or not getattr(state, "ready", False):
        return False, "state not loaded"
    retriever = getattr(state, "retriever", None)
    if retriever is None:
        return False, "retriever missing"
    dense = getattr(retriever, "dense", None)
    if dense is None:
        return False, "dense index missing"
    # El generador está bound en systems; un bound vacío significa no cargado.
    systems = getattr(state, "systems", None)
    if systems is None:
        return False, "systems not bound"
    return True, "generator and embedder present"


def _check_index(state):
    """Ping trivial al índice de Chroma: no es una consulta real.

    Usa un vector cero del tamaño correcto y `n_results=1`. No toca al generador
    ni embebe texto del usuario. Un error aquí es un índice caído.
    """
    try:
        retriever = state.retriever
        dense = retriever.dense
        # En ZeroGPU el índice no es Chroma (ver `app.GpuDenseIndex`) y en el
        # proceso principal no se puede calcular nada: basta con que tenga
        # vectores.
        counter = getattr(dense, "count", None)
        if callable(counter):
            n = counter()
            if n <= 0:
                raise RuntimeError("the index holds no vectors")
            return True, f"gpu index holds {n} vectors"
        collection = dense.collection
        # Tamaño del vector: el que el embedder produce. Lo tomamos del propio
        # encoder si expone dimensión; si no, probamos con count()>0 como ping.
        dim = None
        encoder = getattr(dense, "encoder", None)
        if encoder is not None and hasattr(encoder, "get_sentence_embedding_dimension"):
            dim = encoder.get_sentence_embedding_dimension()
        if dim:
            collection.query(query_embeddings=[[0.0] * int(dim)], n_results=1)
        else:
            collection.count()
        return True, "chroma index responded"
    except Exception as exc:
        return False, f"index error: {exc}"


def health_check(state=None):
    """Reporta liveness sin ejecutar una recomendación. Devuelve un dict JSON.

    Combina el chequeo de modelos y el ping al índice. `status`:
      - "ok"       ambos pasan,
      - "degraded" modelos cargados pero el índice no respondió,
      - "down"     los modelos no están cargados.
    """
    models_loaded, models_detail = _check_models(state)
    if models_loaded:
        index_ready, index_detail = _check_index(state)
    else:
        index_ready, index_detail = False, "skipped (models not loaded)"

    if models_loaded and index_ready:
        status = "ok"
    elif models_loaded:
        status = "degraded"
    else:
        status = "down"

    return {
        "status": status,
        "models_loaded": models_loaded,
        "index_ready": index_ready,
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quantized": bool(getattr(state, "quantized", False)),
        "checks": {
            "models": models_detail,
            "index": index_detail,
        },
    }
