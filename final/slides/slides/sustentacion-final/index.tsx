import { Step, Steps, useSlidePageNumber } from '@open-slide/core';
import type { DesignSystem, Page, SlideMeta, SlideTransition } from '@open-slide/core';

// Los GIF se leen del fork del catálogo en GitHub, fijado al mismo commit que el
// submódulo, sin copiarlos a este repo: Gym visual permite redistribuirlos allí,
// a 180x180 y con atribución (data/exercises-dataset/NOTICE.md). El servidor de
// las diapositivas no deja importar archivos de fuera de su carpeta.
const CATALOG_MEDIA =
  'https://raw.githubusercontent.com/jayounghoyos/exercises-dataset/8872272c9f78cd30889523b79e34e95cca8c5637/videos';
const gifHammerCurl = `${CATALOG_MEDIA}/0312-2NpxjC1.gif`;

// El mismo lenguaje que el Space: fondo piedra, tinta, Geist y un solo azul
// para lo que pasó la comprobación en el catálogo.
export const design: DesignSystem = {
  palette: { bg: '#f4f3f1', text: '#0a0a0a', accent: '#1d5fbf' },
  fonts: {
    display: '"Geist", Inter, system-ui, sans-serif',
    body: '"Geist", Inter, system-ui, sans-serif',
  },
  typeScale: { hero: 150, body: 40 },
  radius: 12,
};

const muted = '#6b6b6b';
const surface = '#ffffff';
const border = '#e6e6e6';
const negative = '#c4242a';
const PAD = 120;
const ATTRIBUTION = 'GIF del catálogo © Gym visual, https://gymvisual.com/';

const FONT_HREF = 'https://fonts.googleapis.com/css2?family=Geist:wght@400;500;700;800&display=swap';
const FONT_LINK_ID = 'osd-webfont-sustentacion-final';
if (typeof document !== 'undefined') {
  let link = document.getElementById(FONT_LINK_ID) as HTMLLinkElement | null;
  if (!link) {
    link = document.createElement('link');
    link.id = FONT_LINK_ID;
    link.rel = 'stylesheet';
    document.head.appendChild(link);
  }
  if (link.href !== FONT_HREF) link.href = FONT_HREF;
}

// Transiciones: una sola familia (RISE en la casa, SETTLE en la portada).
const EASE_OUT = 'cubic-bezier(0, 0, 0.2, 1)';
const EASE_IN = 'cubic-bezier(0.4, 0, 1, 1)';
const HOLD: Keyframe[] = [{ opacity: 1 }, { opacity: 1 }];

export const transition: SlideTransition = {
  duration: 260,
  exit: { duration: 260, easing: EASE_IN, keyframes: HOLD },
  enter: {
    duration: 260,
    easing: EASE_OUT,
    keyframes: [
      { opacity: 0, transform: 'translateY(6px)' },
      { opacity: 1, transform: 'translateY(0)' },
    ],
  },
};

const fill = {
  width: '100%',
  height: '100%',
  boxSizing: 'border-box',
  background: 'var(--osd-bg)',
  color: 'var(--osd-text)',
  fontFamily: 'var(--osd-font-body)',
  position: 'relative',
} as const;

const pageStyle = { ...fill, padding: `${PAD}px ${PAD + 40}px` } as const;

const h2 = {
  fontFamily: 'var(--osd-font-display)',
  fontSize: 72,
  lineHeight: 1.15,
  fontWeight: 800,
  letterSpacing: '-0.02em',
  margin: 0,
} as const;

const lead = {
  fontSize: 'var(--osd-size-body)',
  lineHeight: 1.5,
  color: muted,
  margin: '28px 0 0',
  maxWidth: 1300,
} as const;

// ---------------------------------------------------------------- piezas

const Footer = () => {
  const { current, total } = useSlidePageNumber();
  return (
    <div
      style={{
        position: 'absolute',
        left: PAD + 40,
        right: PAD + 40,
        bottom: 56,
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: 22,
        color: muted,
      }}
    >
      <span>MacGyver Gym Rat · SI4006</span>
      <span>
        {String(current).padStart(2, '0')} / {String(total).padStart(2, '0')}
      </span>
    </div>
  );
};

const CheckMark = ({ ok = true, size = 40 }: { ok?: boolean; size?: number }) => (
  <svg viewBox="0 0 12 12" width={size} height={size} aria-hidden="true" style={{ flex: 'none' }}>
    {ok ? (
      <path d="M2.5 6.2l2.3 2.3 4.7-5" fill="none" stroke="var(--osd-accent)" strokeWidth={1.8}
        strokeLinecap="round" strokeLinejoin="round" />
    ) : (
      <path d="M3 3l6 6M9 3l-6 6" fill="none" stroke={negative} strokeWidth={1.8}
        strokeLinecap="round" />
    )}
  </svg>
);

const CheckRow = ({ text }: { text: string }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 28, fontSize: 48, lineHeight: 1.3 }}>
    <CheckMark size={52} />
    <span>{text}</span>
  </div>
);

const GifCard = ({ src, caption, size = 360 }: { src: string; caption: string; size?: number }) => (
  <figure style={{ margin: 0, display: 'flex', flexDirection: 'column', gap: 20, width: size + 48 }}>
    <div
      style={{
        background: surface,
        border: `2px solid ${border}`,
        borderRadius: 'var(--osd-radius)',
        padding: 24,
      }}
    >
      <img src={src} alt={caption} style={{ width: size, height: size, display: 'block' }} />
    </div>
    <figcaption style={{ fontSize: 24, lineHeight: 1.4, color: muted }}>{caption}</figcaption>
  </figure>
);

const FlowNode = ({ title, sub, tag }: { title: string; sub: string; tag: string }) => (
  <div
    style={{
      background: surface,
      border: `2px solid ${border}`,
      borderRadius: 'var(--osd-radius)',
      padding: '32px 28px',
      width: 300,
      height: 300,
      boxSizing: 'border-box',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
    }}
  >
    <div>
      <div style={{ fontSize: 36, fontWeight: 700, lineHeight: 1.2 }}>{title}</div>
      <div style={{ fontSize: 26, lineHeight: 1.4, color: muted, marginTop: 14 }}>{sub}</div>
    </div>
    <div style={{ fontSize: 24, color: 'var(--osd-accent)', fontWeight: 500 }}>{tag}</div>
  </div>
);

const Arrow = () => (
  <svg viewBox="0 0 40 24" width={44} height={26} aria-hidden="true" style={{ flex: 'none' }}>
    <path d="M2 12h32M26 4l8 8-8 8" fill="none" stroke={muted} strokeWidth={2.5}
      strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const Outcome = ({ ok, title, text }: { ok: boolean; title: string; text: string }) => (
  <div
    style={{
      flex: 1,
      background: surface,
      border: `2px solid ${border}`,
      borderRadius: 'var(--osd-radius)',
      padding: '32px 36px',
      display: 'flex',
      alignItems: 'center',
      gap: 28,
    }}
  >
    <CheckMark ok={ok} size={56} />
    <div>
      <div style={{ fontSize: 36, fontWeight: 700 }}>{title}</div>
      <div style={{ fontSize: 30, lineHeight: 1.4, color: muted, marginTop: 6 }}>{text}</div>
    </div>
  </div>
);

const CaseCard = ({ kind, input, expect }: { kind: string; input: string; expect: string }) => (
  <div
    style={{
      background: surface,
      border: `2px solid ${border}`,
      borderRadius: 'var(--osd-radius)',
      padding: '40px 40px',
      width: 500,
      height: 400,
      boxSizing: 'border-box',
      display: 'flex',
      flexDirection: 'column',
      gap: 22,
    }}
  >
    <div style={{ fontSize: 30, fontWeight: 700, color: 'var(--osd-accent)' }}>{kind}</div>
    <div style={{ fontSize: 40, fontWeight: 700, lineHeight: 1.25 }}>{input}</div>
    <div style={{ fontSize: 30, lineHeight: 1.45, color: muted }}>{expect}</div>
  </div>
);

const Bar = ({ label, value, max, hue = false }: { label: string; value: number; max: number; hue?: boolean }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '440px 1fr 200px', alignItems: 'center', gap: 32 }}>
    <div style={{ fontSize: 34, lineHeight: 1.3 }}>{label}</div>
    <div style={{ height: 48, background: '#e9e7e3', borderRadius: 6 }}>
      <div
        style={{
          width: `${(value / max) * 100}%`,
          height: '100%',
          background: hue ? 'var(--osd-accent)' : 'var(--osd-text)',
          borderRadius: 6,
        }}
      />
    </div>
    <div style={{ fontSize: 48, fontWeight: 800, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
      {value.toFixed(1).replace('.', ',')} %
    </div>
  </div>
);

const Figure = ({ value, label, color }: { value: string; label: string; color?: string }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
    <div style={{ fontSize: 96, fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1,
      fontVariantNumeric: 'tabular-nums', color }}>{value}</div>
    <div style={{ fontSize: 30, lineHeight: 1.4, color: muted, maxWidth: 460 }}>{label}</div>
  </div>
);

const Cell = ({ children, head = false, pending = false }:
  { children: string; head?: boolean; pending?: boolean }) => (
  <div
    style={{
      fontSize: head ? 26 : 32,
      color: head ? muted : pending ? muted : 'var(--osd-text)',
      fontStyle: pending ? 'italic' : 'normal',
      padding: '22px 0',
      borderBottom: `2px solid ${head ? '#d4d4d4' : border}`,
    }}
  >
    {children}
  </div>
);

// ---------------------------------------------------------------- páginas

const Cover: Page = () => (
  <div style={{ ...fill, padding: `${PAD}px ${PAD + 40}px`, display: 'flex', alignItems: 'center', gap: 120 }}>
    <div style={{ flex: 1 }}>
      <h1
        style={{
          fontFamily: 'var(--osd-font-display)',
          fontSize: 'var(--osd-size-hero)',
          fontWeight: 800,
          letterSpacing: '-0.035em',
          lineHeight: 1,
          margin: 0,
        }}
      >
        MacGyver
        <br />
        Gym Rat
      </h1>
      <p style={{ ...lead, color: 'var(--osd-text)', fontSize: 44, marginTop: 44, maxWidth: 1000 }}>
        Un ejercicio real para el músculo que quieres, con lo que tienes en casa.
        O un "no" honesto cuando no hay uno seguro.
      </p>
      <p style={{ fontSize: 28, color: muted, marginTop: 64, lineHeight: 1.5 }}>
        Juan Andrés Young · José Alejandro Jiménez · David Cuadros · Alberto Cervantes
        <br />
        SI4006 · Universidad EAFIT · noviembre de 2026
      </p>
    </div>
    <GifCard src={gifHammerCurl} caption={`dumbbell hammer curl v. 2. ${ATTRIBUTION}`} />
  </div>
);
Cover.transition = {
  duration: 280,
  exit: { duration: 280, easing: EASE_IN, keyframes: HOLD },
  enter: {
    duration: 280,
    easing: EASE_OUT,
    keyframes: [
      { opacity: 0, transform: 'translateY(12px)', filter: 'blur(4px)' },
      { opacity: 1, transform: 'translateY(0)', filter: 'blur(0)' },
    ],
  },
};

const Problem: Page = () => (
  <div style={pageStyle}>
    <Steps>
      <h2 style={{ ...h2, fontSize: 96, maxWidth: 1500, marginTop: 60 }}>
        Hoy toca bíceps. No hay gimnasio. Solo hay dos botellas de agua.
      </h2>
      <Step>
        <p style={{ ...lead, fontSize: 44, marginTop: 80, maxWidth: 1400 }}>
          Hoy esa persona se salta el músculo, hace algo genérico o copia algo de internet
          sin saber si sirve ni si es seguro.
        </p>
      </Step>
    </Steps>
    <Footer />
  </div>
);

const GoodAnswer: Page = () => (
  <div style={pageStyle}>
    <h2 style={h2}>Una buena respuesta pasa tres comprobaciones</h2>
    <Steps>
      <Step><div style={{ marginTop: 72 }}><CheckRow text="El ejercicio existe en un catálogo de 1.324" /></div></Step>
      <Step><div style={{ marginTop: 36 }}><CheckRow text="Entrena el músculo que pediste" /></div></Step>
      <Step><div style={{ marginTop: 36 }}><CheckRow text="Se puede hacer con tu objeto" /></div></Step>
      <Step>
        <p style={{ ...lead, marginTop: 72 }}>
          Si ninguno pasa, la buena respuesta es decir que no. Esa es la métrica que defendemos.
        </p>
      </Step>
    </Steps>
    <Footer />
  </div>
);

const System: Page = () => (
  <div style={pageStyle}>
    <h2 style={h2}>El sistema, de tu pedido a la respuesta</h2>
    <div style={{ display: 'flex', alignItems: 'center', gap: 22, marginTop: 88 }}>
      <Steps>
        <FlowNode title="Tu pedido" sub="Un músculo y un objeto de tu casa" tag="La pantalla" />
        <Step>
          <div style={{ display: 'flex', alignItems: 'center', gap: 22 }}>
            <Arrow />
            <FlowNode title="Busca en el catálogo" sub="Los 8 ejercicios más parecidos" tag="Módulo 3" />
          </div>
        </Step>
        <Step>
          <div style={{ display: 'flex', alignItems: 'center', gap: 22 }}>
            <Arrow />
            <FlowNode title="El modelo escribe" sub="Qwen3-1.7B ajustado a la tarea" tag="Módulo 1" />
          </div>
        </Step>
        <Step>
          <div style={{ display: 'flex', alignItems: 'center', gap: 22 }}>
            <Arrow />
            <FlowNode title="Se comprueba" sub="Las tres comprobaciones" tag="Módulo 2 y final" />
          </div>
        </Step>
      </Steps>
    </div>
    {/* 4 cajas de 300 y 3 flechas de 88: el ancho del flujo de arriba. */}
    <div style={{ display: 'flex', gap: 40, marginTop: 64, width: 4 * 300 + 3 * 88 }}>
      <Outcome ok title="Pasa" text="El ejercicio, con sus pasos reales y su GIF del catálogo" />
      <Outcome ok={false} title="No pasa" text={'"No hay una opción segura", sin imagen'} />
    </div>
    <Footer />
  </div>
);

const Demo: Page = () => (
  <div style={pageStyle}>
    <h2 style={h2}>Demo en vivo: tres casos</h2>
    <div style={{ display: 'flex', gap: 40, marginTop: 72 }}>
      <Steps>
        <Step>
          <CaseCard kind="Fácil" input="Bíceps con dos botellas de agua" expect="Un curl con mancuernas, sus pasos y su GIF." />
        </Step>
        <Step>
          <CaseCard kind="Difícil" input="Espalda alta con dos latas de comida" expect='Las latas nunca dicen "mancuerna", y aun así llega a un remo.' />
        </Step>
        <Step>
          <CaseCard kind="Trampa" input="Dorsales con dos latas de comida" expect="El catálogo no tiene ninguno: tiene que decir que no." />
        </Step>
      </Steps>
    </div>
    <p style={{ fontSize: 30, color: muted, marginTop: 56 }}>
      huggingface.co/spaces/jayoungh/macgyver-gym-rat
    </p>
    <Footer />
  </div>
);

const Result: Page = () => (
  <div style={pageStyle}>
    <h2 style={h2}>Buscar en el catálogo duplicó los aciertos</h2>
    <p style={{ ...lead, fontSize: 30, marginTop: 20 }}>
      Las mismas 14 preguntas y la misma vara para todos: las tres comprobaciones del catálogo.
    </p>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 40, marginTop: 64 }}>
      <Steps>
        <Step><Bar label="Sin ajustar el modelo" value={0} max={50} /></Step>
        <Step><Bar label="Modelo ajustado, sin buscar" value={21.4} max={50} /></Step>
        <Step><Bar label="Modelo ajustado que busca" value={42.9} max={50} hue /></Step>
      </Steps>
    </div>
    <div style={{ display: 'flex', gap: 120, marginTop: 72 }}>
      <Figure value="0 → 3 de 4" label="casos trampa resueltos al buscar" />
      <Figure value="0,55 → 1,00" label="pasos iguales a los del catálogo (ROUGE-L)" />
    </div>
    <Footer />
  </div>
);

const Cost: Page = () => (
  <div style={pageStyle}>
    <h2 style={h2}>Lo que costó</h2>
    <div style={{ display: 'flex', gap: 140, marginTop: 96 }}>
      <Figure value="7,9 s → 15,2 s" label="por pregunta al buscar en el catálogo (GPU T4, mediana)" />
      <Figure value="38,9 s" label="con el agente y su herramienta, que bajaba los aciertos a 28,6 %" />
    </div>
    <Steps>
      <Step>
        <p style={{ ...lead, color: 'var(--osd-text)', marginTop: 96, maxWidth: 1500 }}>
          El doble de tiempo compra ejercicios reales con sus pasos reales. El agente no lo
          compensaba: lo medimos y lo dejamos fuera del despliegue.
        </p>
      </Step>
    </Steps>
    <Footer />
  </div>
);

const Efficiency: Page = () => (
  <div style={pageStyle}>
    <h2 style={h2}>Eficiencia: el modelo a 4 bits</h2>
    <p style={{ ...lead, fontSize: 32, marginTop: 20 }}>
      Antes de medir: aceptamos perder hasta 5 puntos si baja la memoria y no sube la latencia.
    </p>
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '360px repeat(4, 1fr)',
        columnGap: 32,
        marginTop: 64,
      }}
    >
      <Cell head>Versión</Cell>
      <Cell head>Aciertos</Cell>
      <Cell head>Latencia p50</Cell>
      <Cell head>Latencia p95</Cell>
      <Cell head>Memoria</Cell>
      <Cell>Antes (16 bits)</Cell>
      <Cell pending>pendiente</Cell>
      <Cell pending>pendiente</Cell>
      <Cell pending>pendiente</Cell>
      <Cell pending>pendiente</Cell>
      <Cell>Después (4 bits)</Cell>
      <Cell pending>pendiente</Cell>
      <Cell pending>pendiente</Cell>
      <Cell pending>pendiente</Cell>
      <Cell pending>pendiente</Cell>
    </div>
    <p style={{ fontSize: 28, color: muted, marginTop: 48 }}>
      Las cifras salen de la corrida en Colab T4 (notebooks/M5_efficiency.ipynb).
    </p>
    <Footer />
  </div>
);

const Fails: Page = () => (
  <div style={pageStyle}>
    <h2 style={h2}>Lo que falla: prefiere callar a inventar</h2>
    <div style={{ display: 'flex', gap: 120, marginTop: 88 }}>
      <Steps>
        <Step><Figure value="7 de 49" label="combinaciones con respuesta en el catálogo que sí respondió" color={negative} /></Step>
        <Step><Figure value="11 de 11" label="combinaciones sin respuesta en las que dijo que no" /></Step>
        <Step><Figure value="0" label="ejercicios inventados o que no aplican, mostrados al usuario" /></Step>
      </Steps>
    </div>
    <p style={{ ...lead, fontSize: 34, marginTop: 96, maxWidth: 1500 }}>
      Arreglo: buscar solo en la clase de equipo del objeto, sin preguntarle al modelo.
    </p>
    <Footer />
  </div>
);

const Protection = ({ text }: { text: string }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 22, fontSize: 36, lineHeight: 1.3 }}>
    <CheckMark size={40} />
    <span>{text}</span>
  </div>
);

const Ethics: Page = () => (
  <div style={pageStyle}>
    <h2 style={h2}>El peor error posible</h2>
    <p style={{ fontSize: 56, fontWeight: 700, lineHeight: 1.25, margin: '48px 0 0' }}>
      "Cuélgate de la barra de la cortina."
    </p>
    <p style={{ ...lead, fontSize: 32, marginTop: 16 }}>
      Lo dijo el modelo sin buscar, y el juez automático le puso 5 de 5.
    </p>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 22, marginTop: 52 }}>
      <Steps>
        <Step><Protection text="La válvula: dice que no en vez de inventar" /></Step>
        <Step><Protection text="La comprobación en el catálogo antes de mostrar" /></Step>
        <Step><Protection text="El aviso: el sistema sugiere, tú decides" /></Step>
        <Step><Protection text="Sin datos personales en los registros" /></Step>
      </Steps>
    </div>
    <div style={{ position: 'absolute', left: PAD + 40, bottom: 110, fontSize: 26, lineHeight: 1.6, color: muted }}>
      Demo: huggingface.co/spaces/jayoungh/macgyver-gym-rat
      <br />
      Métricas: wandb.ai/jayoungh-me/macgyver-gym-rat-m5
    </div>
    <Footer />
  </div>
);

export const meta: SlideMeta = {
  title: 'MacGyver Gym Rat: sustentación final',
  createdAt: '2026-11-14T15:21:08.667Z',
};

export default [
  Cover,
  Problem,
  GoodAnswer,
  System,
  Demo,
  Result,
  Cost,
  Efficiency,
  Fails,
  Ethics,
] satisfies Page[];

export const notes: (string | undefined)[] = [
  `Presentar el proyecto en una frase: le damos a una persona un ejercicio real para el músculo que quiere,
con lo que tiene en casa, o le decimos que no cuando no hay uno seguro. Presentar al equipo.
(30 segundos)`,
  `El usuario: alguien que ya tenía planeado entrenar un músculo y se queda sin el equipo.
Decisión que toma: qué ejercicio hacer con lo que tiene.
Hoy: se lo salta o improvisa sin saber si funciona o si es seguro. (40 segundos)`,
  `Esta es la métrica que importa para esa decisión, no para el modelo.
Leer las tres comprobaciones. Recalcar que decir que no también es una buena respuesta.
Es la dimensión 3 del harness. (45 segundos)`,
  `Recorrer el dibujo de izquierda a derecha, revelando cada caja.
El modelo de M1 nunca habla solo: antes busca en el catálogo (M3) y después lo que dice se comprueba
contra el catálogo (M2 y el final). Corre en un Space con GPU prestada y cada petición queda en W&B.
(1 minuto)`,
  `Cambiar al Space. Hacer los tres casos en este orden: fácil, difícil, trampa.
Cada petición tarda unos 10 a 15 segundos. Si el Space falla, usar el video de respaldo.
Al final abrir "Estado del sistema" para mostrar el health check. (2 minutos)`,
  `Mismas 14 preguntas y misma vara para todas las filas.
Sin ajustar: 0 por ciento. Ajustado: 21,4. Ajustado y buscando: 42,9.
Los casos trampa pasan de 0 a 3 de 4, y los pasos ahora son los del catálogo. (1 minuto)`,
  `El costo: buscar duplica el tiempo, de 7,9 a 15,2 segundos en un T4.
El agente con herramienta tardaba 38,9 segundos y acertaba menos, por eso no está en el despliegue.
(40 segundos)`,
  `La técnica de eficiencia: cuantizar el generador a 4 bits. El presupuesto lo fijamos antes de medir.
Leer la tabla con los números de la corrida T4 y decir si se cumplió el presupuesto. (50 segundos)`,
  `Lo que falla, medido en 60 combinaciones fuera del eval set.
Responde poco: 7 de 49. Pero nunca inventa: 11 de 11 rechazos correctos y 0 ejercicios inventados.
El arreglo es filtrar la búsqueda por la clase de equipo. (50 segundos)`,
  `El riesgo ético principal es una lesión. Contar el caso de la barra de cortina y que el juez le puso 5.
Las cuatro protecciones. Cerrar con los enlaces del Space y de W&B. (50 segundos)`,
];
