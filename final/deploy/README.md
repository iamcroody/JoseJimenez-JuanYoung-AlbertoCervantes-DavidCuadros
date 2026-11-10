# Módulo 5: Despliegue (Hugging Face Spaces, Gradio)

App Gradio que corre el sistema **MacGyver Gym Rat** de punta a punta: recibe un
músculo objetivo y un objeto casero, devuelve una recomendación de ejercicio del
catálogo **o un rechazo honesto**, y muestra la **imagen/GIF del catálogo** del
ejercicio recomendado (el componente visual que pide M4). Incluye las tres piezas
de producción que verifica la rúbrica (health check, logging JSON por petición y
observabilidad W&B) y el **filtro de satisfacción de restricciones** de la
sección 7 de `docs/PROJECT_DEFINITION.md`.

No reimplementa nada: reutiliza el pipeline M3 (`m3/rag.py`, `m3/systems.py`,
`m3/corpus.py`, `m3/agent.py`), la dimensión 3 del harness (`eval/harness.py`)
para el filtro, y el generador int4 de la feature de eficiencia
(`final/efficiency/efficiency.py`) por la ruta cuantizada.

## Enlaces (los llena el equipo tras el primer deploy)

- **Space en vivo:** TODO: pegar aquí la URL pública tras desplegar
  (`https://huggingface.co/spaces/<org>/macgyver-gym-rat`).
- **Proyecto W&B:** TODO: pegar aquí el enlace del proyecto tras la primera
  corrida con `WANDB_API_KEY` configurada
  (`https://wandb.ai/<entidad>/macgyver-gym-rat-m5`).

Son **placeholders** a propósito: el código los lee del entorno/secrets y no
hay ninguna URL inventada.

## Archivos

| Archivo | Qué hace |
| --- | --- |
| `app.py` | Blocks de Gradio, el manejador de petición de punta a punta, el filtro de producción y la resolución de imagen/GIF. Imports perezosos (no importa gradio/torch al cargarse). |
| `health.py` | `health_check(state)` → liveness sin consulta real (modelos cargados + ping al índice Chroma). |
| `logging_utils.py` | Logger JSON estructurado por petición, sin PII. Esquema + validación. |
| `wandb_utils.py` | Observabilidad W&B con no-op elegante si falta la clave. |
| `selftest.py` | Self-test sin GPU/red (solo stdlib + numpy), con modelos mockeados. |
| `requirements.txt` | Pines exactos para el Space. |

## Correr localmente (con deps completas y GPU/Colab)

```sh
pip install -r final/deploy/requirements.txt
python final/deploy/app.py        # abre Gradio en http://0.0.0.0:7860
```

Rutas estilo Colab: el repo se clona en
`/content/JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros`. `app.py` agrega
`scripts/`, `eval/`, `m3/` y `final/efficiency/` a `sys.path` igual que los
notebooks, así el traslado a Colab o al Space es directo.

## Self-test (sin GPU, sin red, sin gradio/torch/wandb/chromadb)

```sh
python final/deploy/selftest.py
# y la verificación de sintaxis:
python -c "import ast; [ast.parse(open('final/deploy/'+f).read()) for f in \
  ['app.py','health.py','logging_utils.py','wandb_utils.py']]; print('syntax ok')"
```

El self-test valida el esquema del log (y la ausencia de PII), la derivación de
`components_used`/`valve_fired`, el `health_check` con loaders mockeados, y el
filtro de producción + la resolución de imagen sobre un catálogo/dataset falsos.

## Desplegar en Hugging Face Spaces

1. **SDK del Space:** `gradio`. Punto de entrada: `app.py` en la raíz del repo del
   Space (ver "Layout" abajo).
2. **Dataset / catálogo:** el Space necesita `data/exercises-dataset/` (el
   submódulo pineado al commit `8872272`), que contiene `data/exercises.json` y
   las carpetas `images/` y `videos/`. Igual que en `notebooks/M3_rag.ipynb`, se
   obtiene con:
   ```sh
   git clone --recurse-submodules <repo>
   # o, si ya está clonado:
   git submodule update --init --recursive
   ```
   Si el Space no permite submódulos, copiar `data/exercises-dataset/` al repo del
   Space (incluyendo `images/` y `videos/`, necesarias para el componente visual).
3. **Secrets (Settings → Repository secrets):**
   - `WANDB_API_KEY` (opcional). Sin ella, la observabilidad W&B hace no-op y la
     app corre igual (requisito: cero claves de pago obligatorias). **Nunca** se
     commitea en el repo.
   - `WANDB_PROJECT` (opcional), nombre del proyecto (por defecto
     `macgyver-gym-rat-m5`).
   - `QUANTIZED`: `1` para cargar el generador en int4 (nf4) de FEAT-002; `0`
     (por defecto) para fp16.

## Health check

- **Función:** `health.health_check(state)`.
- **Expuesto en la UI:** acordeón "Health check (liveness, sin consulta real)" →
  botón "Ver estado", que imprime el JSON del estado.
- **Endpoint HTTP `/health`:** para un endpoint dedicado, montar `health_check`
  en un FastAPI junto a `demo.app` de Gradio, p. ej.:
  ```python
  demo, state, _ = build_interface()
  app_fastapi = demo.app                      # Gradio monta FastAPI por dentro
  @app_fastapi.get("/health")
  def _health():
      return health_check(state)
  ```
- **Qué devuelve:** `{status, models_loaded, index_ready, git_commit, timestamp,
  quantized, checks}`. Comprueba que el generador y el embedder cargaron y que un
  ping trivial a Chroma responde (**sin** ejecutar una recomendación). `status` es
  `ok` / `degraded` (índice caído) / `down` (modelos no cargados).

## Formato del logging

Una línea **JSON Lines** por petición a stdout (capturado por los logs del
contenedor del Space). Campos:

```json
{
  "timestamp": "2025-01-01T12:00:00.000000+00:00",
  "latency_ms": 1234.5,
  "muscle": "biceps",
  "object_category": "dumbbell",
  "components_used": ["rag", "rerank", "hyde", "vision"],
  "valve_fired": false,
  "passed_filter": true,
  "recommended": true
}
```

**Sin PII (requisito mínimo del rubro):** no se registra texto libre del usuario.
Solo el músculo (vocabulario del catálogo, acotado a 40 caracteres) y la **clase
de equipo** del objeto (`dumbbell`/`kettlebell`/`weighted`/`body weight`/
`unknown`), nunca la frase original del objeto. `logging_utils.redact` recorta y
aplana cualquier texto como última barrera. El detalle está documentado en la
cabecera de `logging_utils.py`.

## Observabilidad W&B

`wandb_utils.init_observer()` inicia un run (proyecto por `WANDB_PROJECT`) y por
cada petición registra **latencia** más métricas de **uso**: acumulados de
recomendaciones vs rechazos, bloqueos del filtro de producción, y frecuencia de
cada componente (`rag/rerank/hyde/tool/vision`). Si falta `wandb` o
`WANDB_API_KEY`, todo es un no-op silencioso y el log JSON local sigue siendo la
fuente de verdad.

## Filtro de producción (mitigación ética, sección 7)

Tras la respuesta del modelo, `app.passes_production_filter` resuelve el ejercicio
nombrado contra el catálogo alcanzable **con la misma lógica de la dimensión 3 del
harness** (`harness.domain_check`, importada, no duplicada): verifica que existe,
entrena el músculo pedido y usa la clase de equipo del objeto. Si no pasa, la app
muestra un mensaje de **"no safe option"** en vez de una recomendación no
verificada. Las imágenes provienen del **catálogo** (no son generadas por IA) y la
UI deja claro que el sistema **sugiere**; la persona decide.

## Si no cabe en el hardware gratuito del Space

El generador fp16 (Qwen3-1.7B + LoRA) más el índice denso, el reranker y HyDE
pueden no caber en el hardware gratuito de Spaces. En ese caso se despliega la
versión **cuantizada** (`QUANTIZED=1`), que carga el generador en **int4 (nf4)**
(la técnica de eficiencia de FEAT-002) reduciendo la VRAM del modelo base sin
tocar el pipeline de RAG. La tabla antes/después y el presupuesto de pérdida
aceptable están en `final/efficiency/`. Lo que se recorta en la versión reducida
(p. ej. desactivar HyDE, que es una generación extra, o reducir el pool del
reranker) se documenta aquí y en `final/reporte.md` para que la degradación sea
explícita, como pide el PDF.

## Layout del Space (recomendado)

HF Spaces espera `app.py` en la **raíz** del repo del Space. Opciones:

- **Preferida (y permitida por el PDF):** crear un **repo de Space aparte** cuyo
  `app.py` haga `sys.path`-insert del repo principal clonado (submódulo o
  subcarpeta) y llame a `final.deploy.app.build_interface()`. Así el repo del
  proyecto queda limpio y el Space es desplegable independientemente.
- **Alternativa:** configurar el repo principal como Space y añadir un `app.py`
  en la raíz que reexporte `final/deploy/app.py`.

Este README documenta ambas; **no** se crean repos externos ni se hace push desde
aquí (eso lo hace el equipo al desplegar).
