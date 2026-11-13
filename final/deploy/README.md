# Módulo 5: Despliegue (Space de ZeroGPU, con respaldo en Colab)

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
| `health.py` | `health_check(state)` → liveness sin consulta real (modelos cargados + ping al índice Chroma, o en ZeroGPU que el índice tenga vectores). |
| `embed_docs.py` | Calcula los embeddings del catálogo en un proceso aparte; solo lo usa el Space de ZeroGPU. |
| `logging_utils.py` | Logger JSON estructurado por petición, sin PII. Esquema + validación. |
| `wandb_utils.py` | Observabilidad W&B con no-op elegante si falta la clave. |
| `selftest.py` | Self-test sin GPU/red (solo stdlib + numpy), con modelos mockeados. |
| `requirements.txt` | Pines exactos para correr `app.py` en Colab o en un computador (respaldo). |
| `space/` | Los tres archivos del repo del Space de ZeroGPU: `app.py` (clona este repo y corre `final/deploy/app.py`), `requirements.txt` y `README.md` (la configuración del Space). |

## Dónde corre

Hugging Face ya no deja crear Spaces de Gradio en CPU con cuentas gratuitas. El
profesor permitió tres salidas: un Space con **ZeroGPU**, correr `app.py` en Colab
o en un computador con un **enlace público**, u otro servicio gratuito avisándole.
Usamos las dos primeras con el mismo archivo `app.py`; solo cambia cómo se lanza.
El health check, el log JSON y W&B funcionan igual en las dos.

### Principal: Space de ZeroGPU

ZeroGPU presta una GPU solo mientras corre una petición. La cuenta que crea el
Space debe tener más de 30 días y el correo verificado. La GPU la paga, de su
cuota diaria, quien usa la demo (5 minutos al día con cuenta gratuita, 2 sin
iniciar sesión); una petición son segundos.

1. En huggingface.co: **New Space**, SDK **Gradio**, hardware **ZeroGPU**.
2. Subir al repo del Space los tres archivos de `final/deploy/space/`
   (`app.py`, `requirements.txt`, `README.md`) en su raíz.
3. **Settings > Variables:** `REPO_REF` con la rama, tag o commit a servir. Para
   la entrega, el commit que se presenta, así el Space no cambia solo.
4. **Settings > Secrets:** `WANDB_API_KEY` (opcional; sin ella W&B hace no-op) y,
   si se quiere otro nombre, `WANDB_PROJECT`. Nunca van en el repo.
5. Al arrancar, `space/app.py` clona este repo en `REPO_REF` con el submódulo del
   dataset (los repos de Spaces no siguen submódulos, y el catálogo con sus GIF
   vive ahí) y corre `final/deploy/app.py`.

Cómo usa la GPU (`app.py`): el generador se carga en `cuda` al arrancar, fuera de
`@spaces.GPU`, como pide ZeroGPU. La GPU solo se pide dentro de `run_advanced`
(`@spaces.GPU(duration=30)`), que corre el RAG avanzado de M3. Dos cosas que
encontramos al probar el Space, y que el código ya resuelve:

- **Nada de cálculo de torch en el proceso principal.** Codificar los 1324
  documentos del catálogo ahí dejaba a todos los workers sin GPU ("No CUDA GPUs
  are available" en `worker_init`). En ZeroGPU esos embeddings se calculan en un
  proceso hijo (`embed_docs.py`). Los vectores son idénticos bit a bit a los del
  proceso principal.
- **Nada de Chroma dentro del worker.** Chroma 1.x corre sobre un pool de hilos
  de Rust que no existe en el fork, y la consulta se quedaba esperando. En
  ZeroGPU la búsqueda densa es exacta, un producto de matrices en la GPU sobre
  los mismos vectores (`GpuDenseIndex`); el HNSW de Chroma aproxima esa misma
  búsqueda.
- **Nada de cálculo en CPU dentro del worker.** El worker es un fork del proceso
  principal, y el pool de hilos de CPU de torch no sobrevive al fork: la
  petición se quedaba colgada. El encoder de consultas y el reranker van en
  `cuda`, como el generador.

Tampoco se llama a `set_seed` en ZeroGPU: la decodificación es greedy, así que la
semilla no cambia ninguna respuesta.

### Respaldo: Colab o un computador con enlace público

```sh
# Colab (torch ya viene instalado):
pip install -r final/deploy/requirements.txt
SHARE=1 python final/deploy/app.py              # imprime https://xxxx.gradio.live
SHARE=1 QUANTIZED=1 python final/deploy/app.py  # igual, con el generador int4

# Un computador con el repo y uv (torch sale de uv.lock):
SHARE=1 uv run --with-requirements final/deploy/requirements.txt python final/deploy/app.py
```

El enlace funciona mientras el proceso siga vivo, así que hay que lanzarlo antes
de la sustentación y abrirlo desde otro dispositivo para comprobarlo. Cambia en
cada corrida. Sin `SHARE=1`, la app queda solo en `http://127.0.0.1:7860`, y un
túnel como `cloudflared tunnel --url http://localhost:7860` también la publica.

### Qué corre en cada lugar

| | Space de ZeroGPU | Colab T4 / computador |
| --- | --- | --- |
| Sistema | RAG avanzado de M3 (HyDE + híbrido + rerank), bge-small, top 8, pool 20 | el mismo |
| Generador | Qwen3-1.7B + LoRA de M1, fusionado, **bf16** | fp16, o **int4** con `QUANTIZED=1` |
| Encoder de consultas y reranker | GPU | GPU |
| Embeddings del catálogo | proceso hijo, CPU | proceso principal, GPU |
| Búsqueda densa | exacta, producto de matrices en GPU | Chroma (HNSW), como M3 |

El retriever sale de `final/efficiency/efficiency.py::build_retriever`, el mismo
que usa la fila "antes" de `notebooks/M5_efficiency.ipynb`, con las decisiones de
la corrida final de M3. Las tablas del reporte se miden en el T4; el Space no se
usa para medir. bf16 en el Space porque su GPU (RTX Pro 6000, Blackwell) lo tiene
en hardware y le sobra memoria, y porque cuantizar con bitsandbytes sobre la CUDA
emulada del arranque de ZeroGPU es la parte con más riesgo. Por eso el Space
**no** corre la técnica de eficiencia: la técnica se mide en el T4, donde sí ahorra
memoria, y se puede mostrar en vivo con el respaldo `QUANTIZED=1`.

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

## Health check

- **Función:** `health.health_check(state)`.
- **Expuesto en la UI:** acordeón "Health check (liveness, sin consulta real)" →
  botón "Ver estado", que imprime el JSON del estado.
- **Endpoint de API `health`:** el mismo botón se publica con
  `api_name="health"`, así que se consulta sin abrir la página ni hacer una
  pregunta real:
  ```sh
  URL=https://<usuario>-<space>.hf.space   # o el enlace gradio.live del respaldo
  ID=$(curl -s -X POST $URL/gradio_api/call/health \
       -H "Content-Type: application/json" -d '{"data": []}' \
       | python -c "import json,sys; print(json.load(sys.stdin)['event_id'])")
  curl -sN $URL/gradio_api/call/health/$ID
  ```
  o, con `gradio_client`: `Client(URL).predict(api_name="/health")`.
- **Qué devuelve:** `{status, models_loaded, index_ready, git_commit, timestamp,
  quantized, checks}`. Comprueba que el generador y el embedder cargaron y que un
  ping trivial a Chroma responde (**sin** ejecutar una recomendación). En ZeroGPU,
  donde el proceso principal no puede calcular nada, comprueba que el índice en GPU
  tenga sus vectores. `status` es
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
