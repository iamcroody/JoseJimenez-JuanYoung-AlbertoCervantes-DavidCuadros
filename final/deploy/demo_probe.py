"""Sondeo del sistema desplegado sobre una grilla de músculo x objeto.

    python final/deploy/demo_probe.py reports/demo_probe_final.csv

Corre `app.handle_request` (el mismo camino que una petición del Space: RAG
avanzado de M3, filtro de producción e imagen) sobre 12 músculos y 5 objetos, y
escribe por combinación si se mostró un ejercicio verificado o la válvula. No es
el eval set ni el harness: sirve para elegir los casos de la demo y para medir
cuánto rechaza el sistema fuera de los 14 casos. Necesita GPU.
"""

import contextlib
import csv
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app as A  # noqa: E402
import macgyver as mg  # noqa: E402

MUSCLES = ["biceps", "triceps", "delts", "pectorals", "abs", "quads", "glutes",
           "hamstrings", "calves", "upper back", "lats", "forearms"]
OBJECTS = ["two filled water bottles", "a bucket with a handle",
           "a backpack loaded with books", "nothing at all", "two full cans of food"]


class _NoObserver:
    def log_request(self, *args, **kwargs):
        pass


def main(out_path):
    state = A.SystemState().load()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["muscle", "object", "equipment_class", "outcome", "exercise",
                         "latency_s"])
        for muscle in MUSCLES:
            for obj in OBJECTS:
                start = time.perf_counter()
                # El log JSON de cada petición va a stdout; aquí sobra.
                with contextlib.redirect_stdout(io.StringIO()):
                    text, media = A.handle_request(state, _NoObserver(), muscle, obj)
                latency = round(time.perf_counter() - start, 1)
                exercise = mg.parse_reply(text)["exercise"] if media else ""
                writer.writerow([muscle, obj, mg.equipment_for_object(obj),
                                 "recommended" if media else "refused", exercise, latency])
                print(f"{muscle:11} | {obj:30} | {'OK ' if media else 'NO '} {exercise}",
                      flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
