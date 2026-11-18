"""La cara de la app de Gradio: tema, CSS y el HTML de cada estado de la respuesta.

`app.py` decide qué responder; este módulo decide cómo se ve. Todo es puro (solo
stdlib más el catálogo): el self-test lo ejercita sin gradio.

Estilo: "achromatic ledger" del skill de diseño. El cromo va en tinta, sin color
de marca; el único tono (#1d5fbf) marca el dato principal de la pantalla, que
aquí es la comprobación contra el catálogo: lo que hace que una recomendación sea
de fiar. El fondo es piedra (#f4f3f1), el piso de una casa.
"""

import html
from collections import Counter

import harness as H
import macgyver as mg

REPO_URL = "https://github.com/iamcroody/JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros"
WANDB_URL = "https://wandb.ai/jayoungh-me/macgyver-gym-rat-m5"

# Las 4 clases de equipo con sustituto casero, como las nombra la pantalla.
CLASS_NAMES = {
    "dumbbell": "mancuerna",
    "kettlebell": "pesa rusa",
    "weighted": "peso añadido",
    "body weight": "peso corporal",
}

HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;700'
    '&display=swap" rel="stylesheet">'
)

# La marca: una mancuerna dibujada, el objeto que todo lo demás sustituye.
MARK_SVG = (
    '<svg class="mg-mark" viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">'
    '<rect x="2" y="7" width="4" height="10" rx="1.5" fill="currentColor"/>'
    '<rect x="6" y="9" width="2.5" height="6" rx="1" fill="currentColor"/>'
    '<rect x="8.5" y="11" width="7" height="2" fill="currentColor"/>'
    '<rect x="15.5" y="9" width="2.5" height="6" rx="1" fill="currentColor"/>'
    '<rect x="18" y="7" width="4" height="10" rx="1.5" fill="currentColor"/>'
    '</svg>'
)
_CHECK = ('<svg viewBox="0 0 12 12" width="12" height="12" aria-hidden="true">'
          '<path d="M2.5 6.2l2.3 2.3 4.7-5" fill="none" stroke="currentColor" '
          'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>')
_CROSS = ('<svg viewBox="0 0 12 12" width="12" height="12" aria-hidden="true">'
          '<path d="M3 3l6 6M9 3l-6 6" fill="none" stroke="currentColor" '
          'stroke-width="1.8" stroke-linecap="round"/></svg>')

CSS = """
:root, .gradio-container {
  --mg-ground: #f4f3f1; --mg-surface: #ffffff; --mg-raised: #f7f6f4;
  --mg-border: #e6e6e6; --mg-rule: #f0f0f0; --mg-control: #d4d4d4;
  --mg-text: #0a0a0a; --mg-muted: #6b6b6b; --mg-primary: #141414;
  --mg-hue: #1d5fbf; --mg-negative: #c4242a;
  --mg-gap: 12px; --mg-pad: 24px; --mg-radius: 8px;
}
body, .gradio-container { background: var(--mg-ground) !important; }
.gradio-container { box-sizing: border-box !important; width: 100% !important;
  max-width: 1152px !important; margin: 0 auto !important;
  padding: 0 24px 48px !important; font-variant-numeric: tabular-nums; }
.gradio-container .main.app { padding: 8px 0 0 !important; }
.mg-html .html-container { padding: 0 !important; }
#mg-bar .form, #mg-answer .form { background: transparent !important; border: 0 !important;
  gap: 16px !important; box-shadow: none !important; }

/* Barra superior: marca y nombre sobre el borde izquierdo de la columna. */
.mg-top { display: flex; align-items: center; justify-content: space-between;
  height: 48px; margin: 0; color: var(--mg-text); }
.mg-brand { display: flex; align-items: center; gap: 10px; font-size: 15px;
  font-weight: 700; color: var(--mg-text); letter-spacing: -0.01em; white-space: nowrap; }
.mg-links { display: flex; gap: 20px; }
.mg-links a, .mg-links a span { font-size: 14px; font-weight: 500; color: var(--mg-muted);
  text-decoration: none; white-space: nowrap; }
.mg-links a:hover, .mg-links a:focus-visible { color: var(--mg-text); }

/* Fila del título. */
.mg-title { padding: 20px 0 8px; }
.mg-title h1 { font-size: 24px; line-height: 32px; font-weight: 700; margin: 0;
  color: var(--mg-text); letter-spacing: -0.01em; }
.mg-title p { font-size: 15px; line-height: 24px; color: var(--mg-muted);
  margin: 6px 0 0; max-width: 680px; }

/* Paneles: blanco sobre piedra, 1px de borde, sin sombra. */
#mg-bar, #mg-answer { background: var(--mg-surface) !important;
  border: 1px solid var(--mg-border) !important; border-radius: var(--mg-radius) !important;
  box-shadow: none !important; padding: var(--mg-pad) !important; }
#mg-bar { padding: 20px var(--mg-pad) 16px !important; }
#mg-bar-row { align-items: flex-end !important; gap: 16px !important; }
#mg-bar button.primary { min-height: 42px; }
#mg-bar .block, #mg-answer .block { border: 0 !important; box-shadow: none !important;
  padding: 0 !important; background: transparent !important; }

.mg-note { font-size: 13px; line-height: 20px; color: var(--mg-muted); margin: 0; }
.mg-note b { color: var(--mg-text); font-weight: 500; }

/* Lo que cubre el catálogo: un pozo de clave y valor. */
.mg-scope { margin-top: 4px; background: var(--mg-raised); border-radius: 6px;
  padding: 12px 14px; }
.mg-scope h3 { font-size: 13px; font-weight: 500; margin: 0 0 6px; color: var(--mg-text); }
.mg-scope dl { margin: 0; display: grid; grid-template-columns: 1fr auto; row-gap: 4px; }
.mg-scope dt { font-size: 13px; color: var(--mg-muted); margin: 0; }
.mg-scope dd { font-size: 13px; color: var(--mg-text); margin: 0; text-align: right; }

/* La respuesta. */
.mg-answer h2 { font-size: 28px; line-height: 34px; font-weight: 700; margin: 0;
  color: var(--mg-text); letter-spacing: -0.015em; }
.mg-answer p { font-size: 15px; line-height: 24px; color: var(--mg-text); margin: 10px 0 0;
  max-width: 720px; }
.mg-answer .mg-sub { color: var(--mg-muted); }
.mg-checks { list-style: none; display: flex; flex-wrap: wrap; gap: 6px 18px;
  padding: 0; margin: 12px 0 0; }
.mg-checks li { display: flex; align-items: center; gap: 6px; font-size: 13px;
  color: var(--mg-text); margin: 0; }
.mg-checks li.ok svg { color: var(--mg-hue); }
/* Gradio pinta cada elemento de un bloque HTML con su color de texto, también el
   <path> de un SVG, y `currentColor` tomaba ese negro en vez del color del ícono. */
.mg-checks svg *, .mg-brand svg * { color: inherit; }
.mg-checks li.no svg { color: var(--mg-negative); }
.mg-checks li.no { color: var(--mg-muted); }
.mg-checks li.wait { color: var(--mg-muted); }
.mg-checks li.wait svg { color: var(--mg-control); }
.mg-empty { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
  gap: 24px 40px; align-items: start; }
.mg-empty .mg-checks { flex-direction: column; gap: 8px; margin-top: 16px; }
.mg-fields { display: grid; grid-template-columns: 150px 1fr; gap: 6px 16px;
  margin: 20px 0 0; padding: 16px 0 0; border-top: 1px solid var(--mg-rule); }
.mg-fields dt { font-size: 13px; color: var(--mg-muted); margin: 0; }
.mg-fields dd { font-size: 14px; line-height: 21px; color: var(--mg-text); margin: 0; }
.mg-answer h3 { font-size: 16px; font-weight: 500; margin: 20px 0 8px; color: var(--mg-text); }
.mg-steps { margin: 0; padding-left: 22px; }
.mg-steps li { font-size: 14px; line-height: 22px; color: var(--mg-text); margin: 0 0 6px;
  padding-left: 4px; }
.mg-steps li::marker { color: var(--mg-muted); }
.mg-safety { font-size: 14px !important; border-left: 2px solid var(--mg-text);
  padding-left: 12px; margin-top: 16px !important; }
.mg-foot { font-size: 12px !important; line-height: 18px !important;
  color: var(--mg-muted) !important; margin-top: 20px !important; }

#mg-media img { border-radius: 6px; background: var(--mg-raised); }

/* Estado del sistema: una lectura para personas, el JSON debajo. */
.mg-health { display: grid; grid-template-columns: 160px 1fr; gap: 6px 16px; margin: 0 0 4px; }
.mg-health dt { font-size: 13px; color: var(--mg-muted); margin: 0; }
.mg-health dd { font-size: 14px; color: var(--mg-text); margin: 0; }
.mg-pill { display: inline-block; font-size: 12px; font-weight: 500; line-height: 20px;
  padding: 0 8px; border-radius: 4px; white-space: nowrap; }
.mg-pill.ok { background: #dcfae5; color: #04250e; }
.mg-pill.warn { background: #fffbeb; color: #92400e; }
.mg-pill.down { background: #ffedea; color: #831c1e; }
#mg-health-row { align-items: center !important; }
#mg-health-row button { flex: none !important; width: auto !important; min-width: 0 !important; }

/* Pie. */
.mg-footer { font-size: 12px; line-height: 18px; color: var(--mg-muted);
  padding: 20px 0 0; margin: 0; }
.mg-footer a { color: var(--mg-muted); }

@media (max-width: 640px) {
  .gradio-container { padding: 0 16px 32px !important; }
  .mg-links { gap: 14px; }
  .mg-links a { font-size: 13px; }
  .mg-wide { display: none; }
  .mg-brand { font-size: 14px; gap: 8px; }
  .mg-title { padding: 16px 0 14px; }
  .mg-title h1 { font-size: 22px; line-height: 28px; }
  #mg-controls, #mg-answer { padding: 16px !important; }
  .mg-answer h2 { font-size: 24px; line-height: 30px; }
  .mg-fields { grid-template-columns: 1fr; gap: 2px; }
  .mg-empty { grid-template-columns: 1fr; }
  #mg-bar { padding: 16px !important; }
  .mg-fields dd { margin-bottom: 8px; }
}
"""


# Los valores del tema. Gradio cambia a su tema oscuro si el navegador está en modo
# oscuro; la pantalla es solo clara, así que cada valor se repite en su variante
# `_dark` (lo vimos en un Chrome en modo oscuro: selectores grises y botón azul).
THEME_VALUES = {
    "body_background_fill": "#f4f3f1",
    "body_text_color": "#0a0a0a",
    "body_text_color_subdued": "#6b6b6b",
    "background_fill_primary": "#ffffff",
    "background_fill_secondary": "#f7f6f4",
    "block_background_fill": "#ffffff",
    "block_border_color": "#e6e6e6",
    "block_border_width": "1px",
    "block_radius": "8px",
    "block_shadow": "none",
    "block_label_text_color": "#3d3d3d",
    "block_label_background_fill": "transparent",
    "block_title_text_color": "#0a0a0a",
    "block_title_text_weight": "500",
    "block_info_text_color": "#6b6b6b",
    "border_color_primary": "#e6e6e6",
    "input_background_fill": "#ffffff",
    "input_background_fill_focus": "#ffffff",
    "input_border_color": "#d4d4d4",
    "input_border_color_focus": "#141414",
    "input_border_width": "1px",
    "input_placeholder_color": "#6b6b6b",
    "input_radius": "6px",
    "input_shadow": "none",
    "input_shadow_focus": "none",
    "button_primary_background_fill": "#141414",
    "button_primary_background_fill_hover": "#2b2b2b",
    "button_primary_text_color": "#ffffff",
    "button_primary_border_color": "#141414",
    "button_primary_shadow": "none",
    "button_secondary_background_fill": "#eeeeee",
    "button_secondary_background_fill_hover": "#e2e2e2",
    "button_secondary_text_color": "#0a0a0a",
    "button_secondary_border_color": "#e6e6e6",
    "button_large_radius": "6px",
    "button_small_radius": "6px",
    "color_accent": "#141414",
    "color_accent_soft": "#eeeeee",
    "code_background_fill": "#f7f6f4",
    "layout_gap": "12px",
    "panel_background_fill": "#ffffff",
    "panel_border_color": "#e6e6e6",
}


def _scoped(css):
    """Antepone el contenedor de Gradio a cada selector propio (.mg-*, #mg-*).

    Gradio pinta el texto de sus bloques HTML con reglas de "prose" que ganaban a
    las clases sueltas: en el Space los enlaces salían azules y subrayados y el
    texto gris salía negro. Con la clase del contenedor delante, las reglas de
    esta pantalla pesan más que las de Gradio.
    """
    import re

    def fix(match):
        selectors = [s.strip() for s in match.group(1).split(",")]
        # La clase doble suma un nivel más: `.gradio-container-6-30-0 .prose a`
        # empataba con `.gradio-container .mg-links a`, y ganaba por cargar después.
        scoped = [f".gradio-container.gradio-container {s}"
                  if s.startswith((".mg", "#mg")) else s for s in selectors]
        return ", ".join(scoped) + " {"

    return re.sub(r"(?m)^([^@{}\n][^{}\n]*)\{", fix, css)


def theme():
    """El tema de Gradio con los tokens del estilo (import perezoso de gradio)."""
    import gradio as gr

    base = gr.themes.Base(
        font=[gr.themes.GoogleFont("Geist"), "Inter", "system-ui", "sans-serif"],
        font_mono=["ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_md,
    )
    values = {k: v for k, v in THEME_VALUES.items() if hasattr(base, k)}
    values.update({f"{k}_dark": v for k, v in values.items() if hasattr(base, f"{k}_dark")})
    return base.set(**values)


def scope_counts():
    """Cuántos ejercicios del catálogo alcanzable hay por clase de equipo."""
    counts = Counter(record["equipment"] for record in H.reachable_catalog())
    return [(CLASS_NAMES[eq], counts.get(eq, 0)) for eq in CLASS_NAMES]


def top_bar_html():
    return (
        '<header class="mg-top">'
        f'<div class="mg-brand">{MARK_SVG}<span>MacGyver Gym Rat</span></div>'
        '<nav class="mg-links" aria-label="Enlaces del proyecto">'
        f'<a href="{WANDB_URL}" target="_blank" rel="noopener">Métricas</a>'
        f'<a href="{REPO_URL}" target="_blank" rel="noopener">Código<span class="mg-wide">'
        ' y reporte</span></a>'
        '</nav></header>'
    )


def title_html():
    total = f"{len(mg.load_catalog()):,}".replace(",", ".")
    return (
        '<div class="mg-title"><h1>Recomendación de ejercicio</h1>'
        f'<p>Un ejercicio real de un catálogo de {total} para el músculo que quieres, '
        'con lo que tienes en casa, o un "no" honesto cuando no hay uno seguro.</p></div>'
    )


def class_note_html(obj):
    """Qué clase de equipo sustituye el objeto elegido, en palabras."""
    equipment = mg.equipment_for_object(obj or "")
    if equipment is None:
        return ('<p class="mg-note">Este objeto no está en la tabla de sustitutos: '
                'el sistema no sabrá qué equipo reemplaza.</p>')
    return (f'<p class="mg-note">Para el catálogo cuenta como '
            f'<b>{html.escape(CLASS_NAMES[equipment])}</b> ({html.escape(equipment)}). '
            'Tarda unos segundos: busca en el catálogo y verifica la respuesta.</p>')


def scope_html():
    rows = "".join(f"<dt>{html.escape(name)}</dt><dd>{count}</dd>"
                   for name, count in scope_counts())
    return ('<div class="mg-scope"><h3>Ejercicios por clase de equipo</h3>'
            f'<dl>{rows}</dl></div>')


def _checks(detail, target, obj):
    """La comprobación contra el catálogo: tres hechos, cada uno sí o no."""
    equipment = mg.equipment_for_object(obj or "")
    cls = CLASS_NAMES.get(equipment, equipment or "este objeto")
    items = [
        (detail.get("exercise_real"), "Existe en el catálogo"),
        (detail.get("target_match"), f"Entrena {target}"),
        (detail.get("equipment_match"), f"Se hace con {cls}"),
    ]
    lis = "".join(
        f'<li class="{"ok" if ok else "no"}">{_CHECK if ok else _CROSS}'
        f'<span>{html.escape(text)}</span></li>' for ok, text in items)
    return f'<ul class="mg-checks" aria-label="Comprobación en el catálogo">{lis}</ul>'


FOOT = ('<p class="mg-foot">Sugiere; la decisión es tuya. No sustituye a un entrenador ni a '
        'un profesional de la salud. La imagen es del catálogo, no generada por IA.</p>')


def empty_html():
    """El marco de la respuesta antes de pedirla: lo que se va a comprobar y el catálogo."""
    pending = "".join(f'<li class="wait">{_CHECK}<span>{text}</span></li>' for text in (
        "Que el ejercicio exista en el catálogo",
        "Que entrene el músculo que elegiste",
        "Que se pueda hacer con tu objeto"))
    return ('<div class="mg-answer"><div class="mg-empty"><div>'
            '<h2>Todavía no hay recomendación</h2>'
            '<p class="mg-sub">Elige un músculo y lo que tienes en casa, y pulsa Recomendar. '
            'Antes de mostrar un ejercicio, el sistema comprueba en el catálogo:</p>'
            f'<ul class="mg-checks" aria-label="Comprobaciones pendientes">{pending}</ul>'
            '<p class="mg-sub">Si alguna falla, la respuesta lo dice en vez de inventar.</p>'
            + FOOT + '</div>' + scope_html() + '</div></div>')


def answer_html(result, target, obj):
    """El HTML de la respuesta para un resultado de `app.serve_request`."""
    parsed = mg.parse_reply(result["reply"])
    detail = result.get("detail") or {}
    target_txt = html.escape(target or "")
    obj_txt = html.escape(obj or "")

    if result["media_path"] is None or result["refused"] or not result["passed"]:
        if result["refused"]:
            reason = ("El modelo respondió que no conoce un ejercicio del catálogo que "
                      "funcione con eso. Lo dice en vez de inventar uno.")
            checks = ""
        else:
            proposed = html.escape(parsed.get("exercise") or "una respuesta sin formato")
            reason = (f"El modelo propuso <b>{proposed}</b>, pero no pasó la comprobación "
                      "del catálogo, así que no se muestra:")
            checks = _checks(detail, target, obj)
        return ('<div class="mg-answer"><h2>No hay una opción segura</h2>'
                f'<p class="mg-sub">Para <b>{target_txt}</b> con <b>{obj_txt}</b>.</p>'
                f'<p>{reason}</p>{checks}'
                '<p class="mg-sub">El sistema cubre 4 clases de equipo (mancuerna, pesa rusa, '
                'peso añadido y peso corporal). Prueba con otro objeto o con otro músculo.</p>'
                + FOOT + '</div>')

    steps = "".join(f"<li>{html.escape(step)}</li>" for step in parsed["steps"])
    fields = [("Equivalente de gimnasio", parsed.get("equipment")),
              ("Adaptación", parsed.get("adaptation"))]
    rows = "".join(f"<dt>{label}</dt><dd>{html.escape(value)}</dd>"
                   for label, value in fields if value)
    safety = parsed.get("safety")
    return ('<div class="mg-answer">'
            f'<h2>{html.escape(parsed["exercise"] or "")}</h2>'
            + _checks(detail, target, obj)
            + f'<dl class="mg-fields">{rows}</dl>'
            + (f'<h3>Pasos</h3><ol class="mg-steps">{steps}</ol>' if steps else "")
            + (f'<p class="mg-safety">{html.escape(safety)}</p>' if safety else "")
            + FOOT + '</div>')


HEALTH_LABELS = {"ok": ("ok", "Listo"), "degraded": ("warn", "Índice sin responder"),
                 "down": ("down", "Caído")}


def health_html(data):
    """El health check en palabras. `data` es el dict de `health.health_check`."""
    css, label = HEALTH_LABELS.get(data.get("status"), ("warn", data.get("status", "?")))
    checks = data.get("checks", {})
    stamp = (data.get("timestamp") or "")[:19].replace("T", " ")
    rows = [
        ("Estado", f'<span class="mg-pill {css}">{html.escape(label)}</span>'),
        ("Modelos", html.escape(checks.get("models", "-"))),
        ("Índice", html.escape(checks.get("index", "-"))),
        ("Generador", "int4 (QUANTIZED=1)" if data.get("quantized") else "bf16 o fp16"),
        ("Versión del código", html.escape(data.get("git_commit", "-"))),
        ("Revisado", f"{html.escape(stamp)} UTC" if stamp else "-"),
    ]
    body = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    return f'<dl class="mg-health" aria-label="Estado del sistema">{body}</dl>'


def footer_html():
    return ('<p class="mg-footer">Catálogo: exercises-dataset (metadatos e instrucciones, '
            'licencia MIT). Imágenes y GIF © Gym visual. Generador: Qwen3-1.7B con el LoRA del '
            f'Módulo 1. <a href="{REPO_URL}" target="_blank" rel="noopener">Reporte final</a>.</p>')
