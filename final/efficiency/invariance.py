"""Pruebas que NO miden exactitud, repetidas sobre la versión eficiente (int4).

La asignación exige que, además de la tabla antes/después, se vuelvan a correr
sobre el modelo eficiente las pruebas que no miden exactitud:

  1. Sonda de invariancia. Una perturbación trivial que preserva el significado
     (espacios extra, reordenar las dos frases, sinónimo de "I have") no debería
     cambiar la respuesta. Como la decodificación es greedy y con seed fija, se
     mide la fracción de los 14 casos cuya respuesta queda IDÉNTICA bajo la
     perturbación. Un sistema robusto se acerca a 1.0; una caída grande tras
     cuantizar es una regresión que la tabla de exactitud no vería.

  2. Válvula de escape. Las respuestas a adv01 (par imposible) y adv04 (objeto
     inseguro) deben seguir disparando la válvula con el EXACTO fraseo que el
     parser congelado del harness reconoce como rechazo (`mg.parse_reply(...)
     ["refused"]`). Es la misma comprobación que el self-test de M3 hace sobre
     el texto de la válvula; aquí se hace sobre la salida real del sistema int4.

  3. Bloque adversarial. Los cuatro casos adv01..adv04 se vuelven a puntuar con
     la dimensión 3 del harness (`domain_check`) sobre el sistema cuantizado,
     para confirmar que la cuantización no rompió el comportamiento adversarial.

La función de perturbación (`perturb`) es PURA (solo stdlib), para que
`selftest.py` la ejerza sin cargar ningún modelo. Las funciones que corren el
sistema (`invariance_probe`, `valve_check`, `adversarial_check`) reciben el
callable `system(prompt)->reply` ya ligado (p. ej. `systems.sistema_rag_avanzado`)
y el harness/catálogo, así que no reimplementan ninguna dimensión.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _sub in ("scripts", "eval", "m3"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Los ids adversariales que la asignación pide re-verificar en la válvula.
# adv01 = par imposible (músculo que ningún ejercicio entrena con ese objeto);
# adv04 = objeto inseguro (debe rehusar). Ambos son `must_refuse`.
VALVE_IDS = ("adv01-impossible-pair", "adv04-unsafe-object")
ADVERSARIAL_IDS = ("adv01-impossible-pair", "adv02-false-premise",
                   "adv03-out-of-domain", "adv04-unsafe-object")


def perturb(text):
    """Perturbación trivial que preserva el significado del prompt.

    Pura, determinista y sin dependencias: `selftest.py` la ejerce sin GPU.

    Tres transformaciones que un humano consideraría equivalentes:
      - sinónimo de "I have" -> "I've got" (y "All I have is" -> "All I've got is");
      - reordenar las dos primeras frases del prompt (la de entrenar y la de
        "no tengo equipo"), que describen la misma situación en cualquier orden;
      - espacios extra entre palabras.

    El bloque de formato ("Answer in exactly this format: ...") se deja intacto:
    perturbarlo cambiaría el contrato de salida, no el significado de la petición.
    """
    if not text:
        return text

    # 1) sinónimo de "I have" / "All I have is".
    out = re.sub(r"\bAll I have is\b", "All I've got is", text)
    out = re.sub(r"\bI have\b", "I've got", out)

    # 2) reordenar las dos primeras frases si existen. El prompt de M1 empieza
    #    con "I want to train my X ... . All I've got is Y.\n". Intercambiamos
    #    esas dos oraciones conservando el salto de línea y el resto del prompt.
    #    El doble espacio entre las dos frases (3) aporta la variación de
    #    espaciado sin romper palabras internas (músculo/objeto quedan intactos).
    lines = out.split("\n", 1)
    head = lines[0]
    rest = lines[1] if len(lines) > 1 else ""
    m = re.match(r"\s*(I want to train[^.]*\.)\s*(All I've got is[^.]*\.)\s*$", head)
    if m:
        # 3) espaciado extra en la unión de las dos frases reordenadas.
        head = f"{m.group(2)}  {m.group(1)}"
    else:
        # Sin el patrón de dos frases, añade un espacio extra tras el primer
        # punto como variación de espaciado inocua.
        head = re.sub(r"\.\s", ".  ", head, count=1)
    out = head + ("\n" + rest if rest else "")
    return out


def invariance_probe(eval_set, system, verbose=True):
    """Fracción de casos cuya respuesta NO cambia bajo `perturb`.

    `system` es cualquier callable `prompt -> reply` (p. ej. el sistema int4
    ligado por `systems.bind`). Con greedy + seed fija, una respuesta idéntica
    byte a byte es la señal de invariancia; cualquier cambio cuenta como
    sensibilidad a la forma superficial del prompt.
    """
    rows = []
    unchanged = 0
    for e in eval_set:
        base_reply = system(e["input"])
        pert_reply = system(perturb(e["input"]))
        same = base_reply == pert_reply
        unchanged += int(same)
        rows.append({"id": e["id"], "unchanged": same})
        if verbose:
            print(f"  {e['id']:26} {'same' if same else 'CHANGED'}")
    n = len(eval_set)
    return {
        "n": n,
        "unchanged": unchanged,
        "invariance_rate": round(unchanged / n, 4) if n else None,
        "detail": rows,
    }


def valve_check(eval_set, system, verbose=True):
    """¿Las respuestas a adv01/adv04 siguen disparando la válvula del parser?

    Usa el parser CONGELADO del harness (`mg.parse_reply(...)['refused']`): la
    misma función que la dimensión 3 usa para contar un rechazo. No se define un
    criterio nuevo.
    """
    import macgyver as mg

    by_id = {e["id"]: e for e in eval_set}
    rows = []
    fired = 0
    for vid in VALVE_IDS:
        e = by_id.get(vid)
        if e is None:
            continue
        reply = system(e["input"])
        refused = mg.parse_reply(reply)["refused"]
        fired += int(refused)
        rows.append({"id": vid, "valve_fired": refused, "reply": reply})
        if verbose:
            print(f"  {vid:26} {'valve fired' if refused else 'NO VALVE'}")
    return {
        "n": len(rows),
        "valve_fired": fired,
        "all_fired": fired == len(rows) and len(rows) > 0,
        "detail": rows,
    }


def adversarial_check(eval_set, system, catalog=None, verbose=True):
    """Re-puntúa adv01..adv04 con la dimensión 3 del harness sobre el sistema int4.

    Reusa `harness.domain_check` y `harness.reachable_catalog`; no reimplementa
    la dimensión. Devuelve el hit-rate adversarial del sistema cuantizado, que
    es la fila que se compara contra el 42.9% / etc. del sistema fp16.
    """
    import macgyver as mg
    import harness as H

    catalog = catalog if catalog is not None else H.reachable_catalog()
    index = mg.build_name_index(catalog)
    full_index = mg.build_name_index(mg.load_catalog())

    by_id = {e["id"]: e for e in eval_set}
    rows = []
    hits = 0
    for aid in ADVERSARIAL_IDS:
        e = by_id.get(aid)
        if e is None:
            continue
        reply = system(e["input"])
        detail = H.domain_check(e, reply, index, full_index)
        hits += int(detail["hit"])
        rows.append({"id": aid, "kind": e["kind"], "hit": detail["hit"],
                     "why": detail.get("why", "")})
        if verbose:
            mark = "hit " if detail["hit"] else "miss"
            print(f"  {aid:26} {mark} {detail.get('why', '')}")
    n = len(rows)
    return {
        "n": n,
        "hits": hits,
        "adversarial_hit_rate": round(hits / n, 4) if n else None,
        "detail": rows,
    }
