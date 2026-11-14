# final/ (Integrador del Módulo 5)

Esta carpeta reúne la entrega final del proyecto: la técnica de eficiencia,
el despliegue, el reporte y el material de sustentación. El contenido se
llena en las features posteriores del Módulo 5; aquí vive el índice.

## Índice

| Carpeta / archivo | Qué contiene | Estado |
| --- | --- | --- |
| `reporte.md` | Reporte final con todas las secciones que pide la asignación (problema, decisiones, tabla evolutiva, ablación, análisis de errores, limitaciones, ética, quién hizo qué, uso de IA). | Hecho (cifras T4 como placeholders) |
| `efficiency/` | Técnica de eficiencia elegida (int4 nf4 QLoRA-style sobre el generador, el componente más lento) con tabla antes/después, presupuesto de pérdida aceptable fijado *antes* de medir, y las pruebas no-exactitud (invariancia, válvula, adversarial) repetidas sobre la versión cuantizada. Ver `efficiency/README.md`. Números en Colab T4. | Código + notebook listos; tabla se llena en T4 |
| `deploy/` | Código de despliegue (Gradio en HF Spaces): app end-to-end con componente visual (imagen/GIF del catálogo), health check, logging JSON estructurado por petición (sin PII), observabilidad en W&B (no-op sin clave) y el filtro de satisfacción de restricciones de la sección 7. Ver `deploy/README.md`. Self-test local sin GPU/red; URL del Space y enlace W&B son placeholders que el equipo llena tras el deploy. | Código + self-test listos; deploy y URLs en el Space |
| `ci/harness_smoke.py` | Humo determinista del harness para la GitHub Action: corre el harness sobre 5 casos contra un sistema *stub* y falla si la dimensión 3 cae por debajo de una línea base comiteada. Realiza, factible en CPU, el *"3 a 5 casos … falle si el puntaje cae"* del PDF. | Hecho |
| `REPRODUCIBILITY.md` | Pegamento de reproducibilidad: clonar con submódulo, ids/versiones de modelos pineados, semilla 42, decodificación greedy, y el orden de corrida de notebooks (M1 → M2 → M3 → M5 eficiencia → deploy) mapeado a cada `reports/*.csv`. | Hecho |
| `ENVIRONMENT_NOTES.md` | Restricciones del entorno local (sin red, sin GPU) y qué self-tests corren localmente vs cuáles se verifican en Colab T4. | Hecho |
| `slides/` | Diapositivas de la sustentación en open-slide (`slides/sustentacion-final/index.tsx`), diez páginas con notas de presentador. `npm install && npm run dev` y abrir `/s/sustentacion-final`. | Hechas; la página de eficiencia espera la corrida T4 |

## Dónde se ejecuta

Todo lo de esta carpeta debe seguir corriendo en un Colab T4 gratuito
(~15 GB VRAM, pip en tiempo de ejecución, repo clonado con submódulos) y en
un HF Space público. Las rutas se simulan como en Colab
(`/content/JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros`) para que el
traslado sea directo. Ver `ENVIRONMENT_NOTES.md`.

## Integración continua

`.github/workflows/harness.yml` (en la raíz del repo) corre en cada push y pull
request, en Ubuntu con Python 3.12 y solo `numpy + rouge_score` (sin GPU, sin
descargar modelos): reconstruye los splits desde el submódulo, corre los cinco
self-tests sin GPU (scorer, harness, M3, eficiencia, despliegue), el humo
determinista del harness (`ci/harness_smoke.py`) y la verificación de que el
eval set es byte-idéntico al regenerado. Es la GitHub Action opcional-y-valorada
del PDF. El guardián de calidad sobre el modelo real vive en los notebooks de
Colab T4 (ver `REPRODUCIBILITY.md`).
