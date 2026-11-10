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
"""

import os


def _wandb_available():
    """¿Está importable wandb? Import perezoso para no exigirlo en el self-test."""
    try:
        import wandb  # noqa: F401
    except Exception:
        return False
    return True


class WandbObserver:
    """Envoltura sobre un run de W&B. Si está deshabilitada, cada método es no-op.

    `enabled` es False cuando falta `wandb` o `WANDB_API_KEY`; entonces ningún
    método toca la red. Los contadores de uso se mantienen igual en memoria para
    poder reportarlos aunque W&B esté ausente (útil en pruebas locales).
    """

    def __init__(self, enabled=False, project=None, run=None):
        self.enabled = enabled
        self.project = project
        self._run = run
        self._step = 0
        # Contadores de uso acumulados (también útiles sin W&B).
        self.counters = {"recommendations": 0, "refusals": 0, "filter_blocks": 0}
        self.component_counts = {"rag": 0, "rerank": 0, "hyde": 0, "tool": 0, "vision": 0}

    def log_request(self, latency_ms, components_used, valve_fired,
                    passed_filter, recommended):
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

        if not self.enabled or self._run is None:
            return

        self._step += 1
        payload = {
            "latency_ms": round(float(latency_ms), 1),
            "valve_fired": int(bool(valve_fired)),
            "passed_filter": int(bool(passed_filter)),
            "recommended": int(bool(recommended)),
            "cumulative/recommendations": self.counters["recommendations"],
            "cumulative/refusals": self.counters["refusals"],
            "cumulative/filter_blocks": self.counters["filter_blocks"],
        }
        for component, count in self.component_counts.items():
            payload[f"component_use/{component}"] = count
        # Import perezoso: solo aquí, nunca al cargar el módulo.
        import wandb

        wandb.log(payload, step=self._step)

    def finish(self):
        if self.enabled and self._run is not None:
            import wandb

            wandb.finish()
            self._run = None
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
        import wandb

        run = wandb.init(project=project, name=run_name,
                         config={"module": "M5-deploy", "seed": 42},
                         reinit=True)
        print(f"[wandb] run iniciado en el proyecto {project!r}.")
        return WandbObserver(enabled=True, project=project, run=run)
    except Exception as exc:  # pragma: no cover - depende de la red/credenciales
        print(f"[wandb] no se pudo iniciar W&B ({exc}); se continúa sin observabilidad.")
        return WandbObserver(enabled=False, project=project)
