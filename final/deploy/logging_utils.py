"""Logging estructurado JSON por petición para el Space desplegado (Módulo 5).

El rubro de despliegue exige un log estructurado por petición con, al menos:
timestamp, latencia, qué componentes se usaron (RAG/tool/vision), si la válvula
de escape disparó, y SIN PII. Este módulo es puro (solo stdlib) para que el
self-test lo ejerza sin cargar gradio ni modelos.

## Decisión de NO PII (requisito mínimo del rubro)

El texto libre del usuario NO se registra. Un prompt libre podría llevar un
nombre, una ubicación o una condición médica, y guardarlo convertiría el log en
un almacén de datos personales. En su lugar registramos solo dos campos
derivados y acotados:

  - `muscle`: el músculo objetivo, normalizado a minúsculas y recortado a 40
    caracteres. Es vocabulario del catálogo (p. ej. "biceps", "lats"), no texto
    libre identificable.
  - `object_category`: la CLASE de equipo que el objeto representa
    (dumbbell / kettlebell / weighted / body weight / unknown), nunca la frase
    original del objeto. "dos botellas de agua" se registra como `dumbbell`.

Cualquier campo que reciba texto de entrada pasa por `redact`, que recorta a un
límite duro y colapsa saltos de línea, de modo que ni siquiera un uso futuro
accidental pueda volcar un prompt completo al log.

Una línea JSON por petición (JSON Lines), apta para `tail -f` en el Space y para
reenviarse a cualquier colector sin reparsear.
"""

import json
import sys
from datetime import datetime, timezone

# Los únicos componentes que un registro puede declarar como usados. Coinciden
# con las técnicas reales del pipeline M3 (rag.TECHNIQUES = hybrid/rerank/hyde)
# más 'tool' (el agente de m3/agent.py) y 'vision' (mostrar el GIF del catálogo).
# Se mantiene como tupla ordenada para que el esquema sea estable y auditable.
COMPONENT_VALUES = ("rag", "rerank", "hyde", "tool", "vision")

# Longitud máxima de cualquier campo derivado de texto. 40 caracteres cubren
# todos los nombres de músculo del catálogo con holgura y hacen imposible volcar
# un prompt completo.
MAX_FIELD_CHARS = 40

# Clases de equipo válidas como categoría de objeto. Reflejan macgyver.OBJECT_MAP
# (dumbbell/kettlebell/weighted/body weight); 'unknown' cubre un objeto que el
# mapeo no reconoce, sin registrar nunca la frase original.
OBJECT_CATEGORIES = ("dumbbell", "kettlebell", "weighted", "body weight", "unknown")

# Las claves obligatorias de cada registro y el tipo que se espera en cada una.
# El self-test valida una muestra contra este esquema; mantener el esquema aquí
# (no en un comentario) es lo que hace comprobable la afirmación "sin PII".
SCHEMA = {
    "timestamp": str,       # ISO 8601 UTC
    "latency_ms": (int, float),
    "muscle": str,          # acotado, vocabulario de catálogo
    "object_category": str,  # una de OBJECT_CATEGORIES, nunca la frase cruda
    "components_used": list,  # subconjunto de COMPONENT_VALUES
    "valve_fired": bool,     # ¿disparó la válvula de escape / rechazo?
    "passed_filter": bool,   # ¿pasó el filtro de satisfacción de restricciones?
    "recommended": bool,     # ¿se mostró una recomendación (hubo imagen)?
}


def redact(text, limit=MAX_FIELD_CHARS):
    """Recorta y aplana cualquier texto antes de que pueda tocar el log.

    Colapsa saltos de línea y espacios repetidos y recorta a `limit`. Es la
    última barrera: aunque un campo reciba texto del usuario por error, no puede
    volcar un prompt completo ni conservar su estructura.
    """
    if not text:
        return ""
    flat = " ".join(str(text).split())
    return flat[:limit]


def normalise_muscle(muscle):
    """El músculo objetivo, en minúsculas y acotado. No es texto libre del usuario."""
    return redact((muscle or "").lower().strip())


def object_category(category):
    """Devuelve una categoría de equipo válida o 'unknown'. Nunca la frase cruda."""
    category = (category or "").lower().strip()
    return category if category in OBJECT_CATEGORIES else "unknown"


def build_record(latency_ms, muscle, object_category_value, components_used,
                 valve_fired, passed_filter, recommended):
    """Arma un registro conforme al esquema, saneando cada campo.

    `components_used` se filtra contra COMPONENT_VALUES y se ordena, de modo que
    un valor inventado no entre al log y el orden sea determinista.
    """
    components = [c for c in COMPONENT_VALUES if c in set(components_used or ())]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round(float(latency_ms), 1),
        "muscle": normalise_muscle(muscle),
        "object_category": object_category(object_category_value),
        "components_used": components,
        "valve_fired": bool(valve_fired),
        "passed_filter": bool(passed_filter),
        "recommended": bool(recommended),
    }


def validate_record(record):
    """Valida un registro contra SCHEMA. Devuelve lista de problemas (vacía = ok).

    Además de los tipos, verifica las invariantes del rubro: sin PII (los campos
    de texto están acotados), componentes dentro del conjunto permitido, y la
    categoría de objeto dentro del enum (nunca una frase libre).
    """
    problems = []
    for key, expected in SCHEMA.items():
        if key not in record:
            problems.append(f"missing field {key!r}")
            continue
        if not isinstance(record[key], expected):
            problems.append(f"{key!r} is {type(record[key]).__name__}, expected {expected}")

    comps = record.get("components_used", [])
    if isinstance(comps, list):
        unknown = set(comps) - set(COMPONENT_VALUES)
        if unknown:
            problems.append(f"components_used has unknown values {sorted(unknown)}")

    cat = record.get("object_category")
    if cat not in OBJECT_CATEGORIES:
        problems.append(f"object_category {cat!r} is not one of {OBJECT_CATEGORIES}")

    # Guardarraíl anti-PII: ningún campo de texto puede superar el límite duro.
    for key in ("muscle", "object_category"):
        value = record.get(key, "")
        if isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
            problems.append(f"{key!r} exceeds {MAX_FIELD_CHARS} chars (possible PII leak)")

    return problems


def log_record(record, stream=None):
    """Emite el registro como una línea JSON (JSON Lines) a stdout (o `stream`).

    En un Space de Hugging Face stdout queda capturado en los logs del contenedor,
    así que no hace falta un archivo ni un servicio externo. `ensure_ascii=False`
    conserva cualquier acento del vocabulario de músculos.
    """
    stream = stream or sys.stdout
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()
    return record
