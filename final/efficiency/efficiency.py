"""Módulo 5, técnica de eficiencia: cargar el generador M1 en int4 (nf4).

El componente que domina la latencia del sistema es el generador Qwen3-1.7B con
el adaptador LoRA de M1. La ablación de M3 lo deja claro: HyDE es una generación
completa extra y el agente llega a 38.9 s por consulta, casi todo gasto del
generador. Por eso la técnica de eficiencia se aplica aquí y no en la
recuperación: cuantizar el generador a 4 bits (QLoRA-style) reduce la VRAM del
modelo base sin tocar el pipeline de RAG.

Qué hace este módulo y qué NO:

  - Carga `Qwen/Qwen3-1.7B` con `BitsAndBytesConfig(load_in_4bit=True, nf4,
    double_quant, compute dtype fp16)` y le adjunta el adaptador LoRA
    `models/r16-all-linear` con peft. No fusiona el adaptador: un base de 4 bits
    no puede absorberlo (`merge_and_unload` falla), igual que ya documenta
    `scripts/macgyver.py::load_model`.
  - Expone `generate(prompt) -> reply`, EXACTAMENTE la misma firma que
    `m3/systems.bind` espera para `llm`, para que el MISMO pipeline de M3
    (`sistema_rag_avanzado` / `sistema_final`) corra sin cambios sobre el
    generador cuantizado. No reinventa la generación: reusa
    `harness.make_hf_system`, que ya fija greedy, seed y `enable_thinking=False`.

Determinismo: seed 42 (SEED del harness) y decodificación greedy, los mismos de
M1/M2/M3, para que la tabla antes/después compare el mismo generador medido de
dos formas y no dos configuraciones de muestreo distintas.

Este archivo importa torch/transformers/peft; NO se ejecuta localmente sin GPU.
La verificación local se hace con `selftest.py`, que no toca ningún modelo.

    # en Colab T4, con el repo clonado y los deps instalados:
    import sys
    for p in ("scripts", "eval", "m3"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from final.efficiency import efficiency as E   # o importa el .py directo
    gen_int4 = E.load_int4_generator()              # llm callable
    import systems as S
    S.bind(retriever, gen_int4, docs)
    reply = S.sistema_rag_avanzado(prompt)
"""

import sys
from pathlib import Path

# Rutas estilo Colab: el repo se clona en
# /content/JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros, y scripts/, eval/
# y m3/ se agregan a sys.path igual que en notebooks/M3_rag.ipynb. Replicamos esa
# resolución para que `import macgyver`/`import harness` funcione igual aquí.
ROOT = Path(__file__).resolve().parents[2]
for _sub in ("scripts", "eval", "m3"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import macgyver as mg  # noqa: E402
import harness as H  # noqa: E402

# El adaptador LoRA de M1 ya está comiteado en el repo, así los resultados
# reproducen sin reentrenar. Es el mismo que usan scripts/03_eval_zeroshot.py y
# la celda de M3_rag.ipynb.
ADAPTER = "models/r16-all-linear"


def build_4bit_config(compute_dtype=None):
    """La config nf4 QLoRA-style: 4 bits, nf4, doble cuantización, cómputo fp16.

    Idéntica a la rama `four_bit` de `scripts/macgyver.py::load_model`, extraída
    aquí para poder documentarla y medir su huella. `compute_dtype` por defecto
    es el dtype que `pick_device_dtype` elige para la GPU (fp16 en una T4).
    """
    import torch
    from transformers import BitsAndBytesConfig

    if compute_dtype is None:
        _device, compute_dtype = mg.pick_device_dtype()
        # En una T4 pick_device_dtype devuelve fp16; forzamos fp16 también si por
        # alguna razón llegara bf16, porque el cómputo de bnb en T4 es fp16.
        if compute_dtype is torch.bfloat16:
            compute_dtype = torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_int4_model(model_id=mg.BASE_MODEL, adapter=ADAPTER):
    """Carga el base en int4 (nf4) y le adjunta el adaptador LoRA de M1.

    Reusa `mg.load_model(..., four_bit=True)`, que ya hace exactamente esto:
    arma el `BitsAndBytesConfig` nf4, usa `device_map="auto"` y adjunta el
    adaptador SIN fusionarlo (un base de 4 bits no puede absorber los pesos
    LoRA). Devuelve el modelo listo para inferencia (`eval`, sin gradientes).
    """
    return mg.load_model(model_id, adapter=adapter, four_bit=True)


def load_fp16_model(model_id=mg.BASE_MODEL, adapter=ADAPTER):
    """La línea base 'antes': el mismo generador en fp16 con el adaptador fusionado.

    Es el generador que M3 usó en el scorecard (celda 17 de M3_rag.ipynb). Se
    expone aquí para que la celda del notebook arme la fila 'antes' con el mismo
    código que la fila 'después' y no haya deriva entre ambas.
    """
    return mg.load_model(model_id, adapter=adapter, four_bit=False)


def make_generator(model, tok, max_new_tokens=None):
    """Envuelve (model, tok) como `generate(prompt) -> reply`.

    Reusa `harness.make_hf_system`, el MISMO envoltorio que M2 y M3 usan para
    el generador: greedy, `enable_thinking=False`, sin muestreo. Así el callable
    que recibe `systems.bind` es indistinguible del de M3 salvo por los pesos
    cuantizados. No se reimplementa la generación.
    """
    return H.make_hf_system(model, tok, max_new_tokens=max_new_tokens)


def load_int4_generator(model_id=mg.BASE_MODEL, adapter=ADAPTER, tok=None,
                        max_new_tokens=None):
    """Atajo: devuelve el callable `generate(prompt)->reply` del modelo int4.

    Fija la seed del harness (42) antes de cargar, como hacen los scripts de M1.
    """
    from transformers import set_seed

    set_seed(H.SEED)
    tok = tok or mg.load_tokenizer(model_id)
    model = load_int4_model(model_id, adapter)
    return make_generator(model, tok, max_new_tokens=max_new_tokens)


def load_fp16_generator(model_id=mg.BASE_MODEL, adapter=ADAPTER, tok=None,
                        max_new_tokens=None):
    """Atajo equivalente para la línea base fp16 ('antes')."""
    from transformers import set_seed

    set_seed(H.SEED)
    tok = tok or mg.load_tokenizer(model_id)
    model = load_fp16_model(model_id, adapter)
    return make_generator(model, tok, max_new_tokens=max_new_tokens)


def model_footprint_mb(model):
    """Huella en memoria del modelo, en MB.

    Para un modelo int4 los pesos base son uint8 (nf4 empaquetado), así que
    `element_size()` cuenta los bytes reales de cada tensor y la suma refleja la
    reducción frente a fp16. Es la misma métrica para 'antes' y 'después', que es
    lo que exige la tabla. `torch.cuda.max_memory_allocated` (pico de VRAM) lo
    mide `measure.py` aparte, porque incluye activaciones, no solo pesos.
    """
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return round(total_bytes / (1024 * 1024), 2)
