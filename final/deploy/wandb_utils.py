"""Observabilidad con Weights & Biases para el Space (Módulo 5), con no-op elegante.

El rubro de despliegue pide observabilidad W&B: latencia por petición y al menos
una métrica de uso. El requisito mínimo también exige que el sistema corra SIN
claves de pago obligatorias. Este módulo concilia ambos:

  - Si `wandb` no está instalado o `WANDB_API_KEY` no está en el entorno, todo
    es un no-op silencioso (salvo un aviso una sola vez). La app corre igual y
    el logging JSON local sigue siendo la fuente de verdad.
  - Si están disponibles, se inicializa un run (proyecto configurable por
    `WANDB_PROJECT`) y cada petición registra su latencia más contadores de uso.

Métricas de uso registradas (>=1 exigida; registramos varias):
  - `recommendations` vs `refusals`: acumulados, para ver cuántas veces el
    sistema recomienda frente a cuántas aplica la válvula / el filtro.
  - `filter_blocks`: cuántas veces el filtro de producción reemplazó una
    respuesta no verificada (la mitigación ética de la sección 7).
  - frecuencia de cada componente en `components_used` (rag/rerank/hyde/tool/vision).

Toda la carga de `wandb` es perezosa: importar este archivo NO importa wandb, de
modo que `ast.parse`/`py_compile` y el self-test quedan limpios sin la librería.
El estado vive en un objeto `WandbObserver`; una instancia nula (`disabled`) es
la que se usa cuando falta la clave, y expone la misma interfaz para que la app
no tenga ramas condicionales.

En un Space de ZeroGPU el run de W&B vive en un proceso aparte
(`wandb_logger.py`), y la app le manda cada petición por un pipe. ZeroGPU crea
cada worker de GPU con un fork del proceso principal, y un fork hecho después de
`wandb.init` deja colgado el run del padre: en el Space quedó congelado a los
2 min 45 s sin ninguna métrica, y en local el mismo fork colgó el proceso.
"""

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

# Columnas de la tabla de peticiones. Solo vocabulario del catálogo, sin texto
# libre del usuario (requisito de no guardar datos personales).
REQUEST_COLUMNS = ["request", "muscle", "object_category", "outcome", "latency_ms"]


def _wandb_available():
    """¿Está importable wandb? Import perezoso para no exigirlo en el self-test."""
    try:
        import wandb  # noqa: F401
    except Exception:
        return False
    return True


class _RunWriter:
    """Escribe en un run de W&B ya iniciado: las métricas y la tabla de peticiones.

    Lo usan los dos caminos (en el mismo proceso, o dentro de `wandb_logger.py`),
    así que lo que llega a W&B es igual en los dos.
    """

    def __init__(self):
        self.rows = []

    def write(self, step, payload, row):
        import wandb

        record = dict(payload)
        if row is not None:
            self.rows.append(row)
            # La tabla completa en cada petición: W&B muestra la última versión.
            record["requests"] = wandb.Table(columns=REQUEST_COLUMNS, data=list(self.rows))
        wandb.log(record, step=step)


class _PipeSink:
    """Manda cada petición a `wandb_logger.py`, que es el dueño del run."""

    def __init__(self, project):
        logger = Path(__file__).with_name("wandb_logger.py")
        self._proc = subprocess.Popen([sys.executable, str(logger), project],
                                      stdin=subprocess.PIPE, text=True)

    def write(self, step, payload, row):
        line = json.dumps({"step": step, "payload": payload, "row": row})
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def close(self):
        self._proc.stdin.close()
        self._proc.wait(timeout=60)


class WandbObserver:
    """Envoltura sobre un run de W&B. Si está deshabilitada, cada método es no-op.

    `enabled` es False cuando falta `wandb` o `WANDB_API_KEY`; entonces ningún
    método toca la red. Los contadores de uso se mantienen igual en memoria para
    poder reportarlos aunque W&B esté ausente (útil en pruebas locales).
    """

    def __init__(self, enabled=False, project=None, run=None, sink=None):
        self.enabled = enabled
        self.project = project
        self._run = run
        # `sink` recibe (step, payload, row). Sin sink, se escribe en `run` aquí.
        self._sink = sink or (_RunWriter() if run is not None else None)
        self._step = 0
        self._latencies = []
        # Contadores de uso acumulados (también útiles sin W&B).
        self.counters = {"recommendations": 0, "refusals": 0, "filter_blocks": 0}
        self.component_counts = {"rag": 0, "rerank": 0, "hyde": 0, "tool": 0, "vision": 0}

    def log_request(self, latency_ms, components_used, valve_fired,
                    passed_filter, recommended, muscle=None, object_category=None):
        """Registra una petición: latencia + métricas de uso acumuladas.

        Actualiza contadores en memoria SIEMPRE; solo envía a W&B si está activo.
        """
        if recommended:
            self.counters["recommendations"] += 1
        if valve_fired:
            self.counters["refusals"] += 1
        if not passed_filter:
            self.counters["filter_blocks"] += 1
        for component in components_used or ():
            if component in self.component_counts:
                self.component_counts[component] += 1

        self._latencies.append(float(latency_ms))

        if not self.enabled or self._sink is None:
            return

        self._step += 1
        if recommended:
            outcome = "recommended"
        elif valve_fired:
            outcome = "refused (valve)"
        else:
            outcome = "refused (filter)"
        payload = {
            "latency_ms": round(float(latency_ms), 1),
            "latency_p50_ms": round(statistics.median(self._latencies), 1),
            "valve_fired": int(bool(valve_fired)),
            "passed_filter": int(bool(passed_filter)),
            "recommended": int(bool(recommended)),
            "cumulative/recommendations": self.counters["recommendations"],
            "cumulative/refusals": self.counters["refusals"],
            "cumulative/filter_blocks": self.counters["filter_blocks"],
        }
        for component, count in self.component_counts.items():
            payload[f"component_use/{component}"] = count
        row = [self._step, muscle or "", object_category or "", outcome,
               round(float(latency_ms), 1)]
        try:
            self._sink.write(self._step, payload, row)
        except Exception as exc:  # pragma: no cover - red o proceso caído
            # La observabilidad nunca tumba una petición: el log JSON sigue.
            print(f"[wandb] no se pudo registrar la petición ({exc}).")

    def finish(self):
        if not self.enabled:
            return
        if isinstance(self._sink, _PipeSink):
            self._sink.close()
        elif self._run is not None:
            import wandb

            wandb.finish()
        self._run = None
        self._sink = None
        self.enabled = False


def init_observer(project=None, run_name=None):
    """Crea un `WandbObserver`. Devuelve uno deshabilitado si falta wandb o la clave.

    El proyecto se toma de `WANDB_PROJECT` (o del argumento). La clave se lee de
    `WANDB_API_KEY`, que en el Space se configura como *secret*, nunca en el repo.
    Un fallo de inicialización NO rompe la app: se degrada a no-op con un aviso.
    """
    project = project or os.environ.get("WANDB_PROJECT", "macgyver-gym-rat-m5")

    if not os.environ.get("WANDB_API_KEY"):
        print("[wandb] WANDB_API_KEY ausente: observabilidad W&B deshabilitada (no-op).")
        return WandbObserver(enabled=False, project=project)

    if not _wandb_available():
        print("[wandb] wandb no instalado: observabilidad W&B deshabilitada (no-op).")
        return WandbObserver(enabled=False, project=project)

    try:
        if os.environ.get("SPACES_ZERO_GPU"):
            sink = _PipeSink(project)
            print(f"[wandb] run en un proceso aparte, proyecto {project!r} (ZeroGPU).")
            return WandbObserver(enabled=True, project=project, sink=sink)

        run = start_run(project, run_name)
        print(f"[wandb] run iniciado en el proyecto {project!r}.")
        return WandbObserver(enabled=True, project=project, run=run)
    except Exception as exc:  # pragma: no cover - depende de la red/credenciales
        print(f"[wandb] no se pudo iniciar W&B ({exc}); se continúa sin observabilidad.")
        return WandbObserver(enabled=False, project=project)


def start_run(project, run_name=None):
    """`wandb.init` con la misma configuración en los dos caminos."""
    import wandb

    return wandb.init(project=project, name=run_name,
                      config={"module": "M5-deploy", "seed": 42})
