"""Humo determinista del harness para CI (sin GPU, sin descargar modelos).

    python final/ci/harness_smoke.py

Esto realiza, de forma factible en un runner de CPU, la pieza opcional-y-valorada
del PDF: *"una GitHub Action que en cada push corra el harness sobre 3 a 5 casos
y falle si el puntaje cae"*. El reto es que el harness real (dimensiones 1 y 2)
necesita un embedder y un modelo juez, y la corrida completa necesita una T4 y
los pesos de Qwen. En CI no hay nada de eso. Así que aquí medimos la **dimensión
3** (la tasa de acierto de dominio, la que no se deja engañar por prosa fluida)
sobre un **sistema STUB**: una función trivial que devuelve una respuesta fija de
ejercicio del catálogo y una negativa fija.

Por qué un stub y no el modelo real:

  - Es **determinista**: no hay muestreo, no hay pesos, no hay descarga. El mismo
    número en cada corrida, en cualquier runner. El guardián de regresión de
    calidad sobre las salidas del modelo REAL corre en el notebook de Colab T4
    (ver `final/REPRODUCIBILITY.md`); CI cuida la *plomería* y la *integridad del
    eval set*, no la calidad del modelo.
  - El número que afirma (`EXPECTED_DIM3_HITS`) es una **propiedad determinista
    del stub frente al harness congelado**, no una métrica de modelo. Se calcula
    una vez localmente y se fija como línea base; si el harness o el eval set
    cambian de forma que rompa este contrato, el número se mueve y CI falla, que
    es exactamente la señal de regresión que el PDF pide.

El stub importa el harness real (`eval/harness.py`) y el eval set real
(`eval/eval_set.json`), así que si cualquiera de los dos cambia de manera que
afecte el acierto de dominio, este humo lo detecta.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

import harness as H  # noqa: E402


# Los 3-5 casos del eval set sobre los que corre el humo. Mezcla deliberada:
# tres contestables (acierto si se nombra el ejercicio correcto) y dos
# adversariales (acierto si se niega), de modo que el stub ejerza ambas ramas de
# la dimensión 3. adv01 y adv04 son los dos casos de "válvula de escape"
# (must_refuse): la única respuesta correcta es negarse.
SMOKE_CASE_IDS = [
    "gold01-forearms",        # answerable: par de pesos emparejado (dumbbell)
    "gold02-glutes",          # answerable: peso único sostenido (weighted)
    "gold07-abs",             # answerable: sin equipo (body weight)
    "adv01-impossible-pair",  # must_refuse: no existe ejercicio válido
    "adv04-unsafe-object",    # must_refuse: el objeto es un peligro físico
]

# Negativa fija que el parser congelado del harness reconoce como refusal
# (mg.parse_reply(...)["refused"] == True). Usa el patrón "No safe option" /
# "i know no ... exercise" que la válvula de escape de M1/M3 emite.
STUB_REFUSAL = (
    "No safe option: I know no safe exercise I can recommend for that "
    "combination, so I will not invent one."
)


def make_stub_system(eval_set):
    """Sistema STUB `system(prompt) -> reply`, determinista y sin modelo.

    Para un caso contestable devuelve su respuesta de oro (`esperado`), que por
    construcción nombra un ejercicio real del catálogo con el músculo y la clase
    de equipo correctos: acierta la dimensión 3. Para un caso adversarial
    (`must_refuse`) devuelve la negativa fija: también acierta la dimensión 3.
    Cualquier otro tipo de caso devuelve la negativa (conservador).

    El mapeo es por el texto exacto del `input`, así que el stub no "conoce" nada
    del modelo: solo reproduce una respuesta fija por caso.
    """
    by_input = {}
    for e in eval_set:
        if e["kind"] == "answerable":
            by_input[e["input"]] = e["esperado"]
        else:
            by_input[e["input"]] = STUB_REFUSAL

    def system(prompt):
        if prompt not in by_input:
            raise KeyError("prompt fuera del conjunto de humo")
        return by_input[prompt]

    return system


# Línea base COMITEADA. Calculada localmente con el venv hermano
# (../models-for-exercises-dataset/.venv) corriendo este mismo script. Es una
# propiedad determinista del stub frente al harness congelado, no una métrica de
# modelo, por lo que fijar el valor verificado es correcto. Con los 5 casos de
# arriba, el stub acierta los 5 (3 contestables bien respondidos + 2 negativas
# bien emitidas), así que la tasa de dimensión 3 es 1.0 y los aciertos son 5.
EXPECTED_DIM3_HITS = 5
EXPECTED_DIM3_RATE = 1.0

# El guardián de regresión del PDF: CI falla si el puntaje CAE por debajo de la
# línea base. No exigimos igualdad exacta hacia arriba (una mejora no debe romper
# CI), pero para un stub determinista el valor no debería moverse en absoluto.
MIN_DIM3_HITS = EXPECTED_DIM3_HITS


def main():
    full = H.load_eval_set()
    by_id = {e["id"]: e for e in full}
    missing = [cid for cid in SMOKE_CASE_IDS if cid not in by_id]
    if missing:
        print(f"FAILED: ids de humo ausentes del eval set: {missing}")
        raise SystemExit(1)

    eval_set = [by_id[cid] for cid in SMOKE_CASE_IDS]
    n = len(eval_set)
    if not (3 <= n <= 5):
        print(f"FAILED: el humo debe cubrir 3-5 casos, cubre {n}")
        raise SystemExit(1)

    system = make_stub_system(eval_set)

    # Solo dimensión 3: sin juez (judge=None) y con una similitud constante que
    # no descarga nada. Es exactamente el patrón del self-test del harness.
    card = H.harness(
        eval_set, system,
        judge=None, similarity=H._ConstantSimilarity(),
        label="ci-stub", verbose=True,
    )

    hits = card["domain_hits"]
    rate = card["domain_hit_rate"]
    print()
    print(f"  casos de humo        : {n}  ({', '.join(SMOKE_CASE_IDS)})")
    print(f"  dimension 3 aciertos : {hits}/{n}  (tasa {rate})")
    print(f"  linea base comiteada : >= {MIN_DIM3_HITS} aciertos "
          f"(esperado {EXPECTED_DIM3_HITS}, tasa {EXPECTED_DIM3_RATE})")

    ok = True
    if hits < MIN_DIM3_HITS:
        print(f"\nFAILED: la dimension 3 CAYO a {hits} aciertos, por debajo de "
              f"la linea base {MIN_DIM3_HITS}. Esto es la regresion que el PDF "
              f"pide atrapar ('falle si el puntaje cae').")
        ok = False
    if hits != EXPECTED_DIM3_HITS or rate != EXPECTED_DIM3_RATE:
        print(f"\nFAILED: el numero determinista del stub cambio "
              f"(aciertos={hits}, tasa={rate}; esperado aciertos="
              f"{EXPECTED_DIM3_HITS}, tasa={EXPECTED_DIM3_RATE}). El harness o el "
              f"eval set cambiaron de forma que rompe el contrato del stub.")
        ok = False

    print("\nharness-smoke", "PASSED" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
