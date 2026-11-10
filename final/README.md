# final/ — Integrador del Módulo 5

Esta carpeta reúne la entrega final del proyecto: la técnica de eficiencia,
el despliegue, el reporte y el material de sustentación. El contenido se
llena en las features posteriores del Módulo 5; aquí vive el índice.

## Índice

| Carpeta / archivo | Qué contiene | Estado |
| --- | --- | --- |
| `reporte.md` | Reporte final con todas las secciones que pide la asignación (problema, decisiones, tabla evolutiva, ablación, análisis de errores, limitaciones, ética, quién hizo qué, uso de IA). | Esqueleto (TODO) |
| `efficiency/` | Técnica de eficiencia elegida (cuantización / destilación / LoRA en modelo menor) sobre el componente más pesado del scorecard, con tabla antes/después y presupuesto de pérdida aceptable fijado *antes* de medir. | Pendiente |
| `deploy/` | Código de despliegue (Gradio / Streamlit en HF Spaces) o enlace al repo del Space: health check, logging JSON estructurado por petición, observabilidad en W&B. | Pendiente |
| `ENVIRONMENT_NOTES.md` | Restricciones del entorno local (sin red, sin GPU) y qué self-tests corren localmente vs cuáles se verifican en Colab T4. | Hecho |

## Dónde se ejecuta

Todo lo de esta carpeta debe seguir corriendo en un Colab T4 gratuito
(~15 GB VRAM, pip en tiempo de ejecución, repo clonado con submódulos) y en
un HF Space público. Las rutas se simulan como en Colab
(`/content/JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros`) para que el
traslado sea directo. Ver `ENVIRONMENT_NOTES.md`.
