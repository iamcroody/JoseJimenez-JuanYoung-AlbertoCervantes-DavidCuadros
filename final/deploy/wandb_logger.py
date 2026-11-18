"""Proceso dueño del run de W&B cuando la app corre en un Space de ZeroGPU.

    python final/deploy/wandb_logger.py <proyecto>

Lo arranca `wandb_utils._PipeSink`. Lee una línea JSON por petición desde stdin
({"step", "payload", "row"}) y la escribe en el run con el mismo `_RunWriter`
que usa el camino normal. Al cerrarse stdin (la app terminó) cierra el run.
Por qué existe: ver la cabecera de `wandb_utils.py`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wandb_utils as W  # noqa: E402


def main(project):
    W.start_run(project)
    writer = W._RunWriter()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            writer.write(message["step"], message["payload"], message.get("row"))
        except Exception as exc:
            print(f"[wandb_logger] petición sin registrar ({exc}).", flush=True)
    import wandb

    wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1])
