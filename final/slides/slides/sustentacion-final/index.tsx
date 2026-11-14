import { Step, Steps, useSlidePageNumber } from '@open-slide/core';
import type { DesignSystem, Page, SlideMeta, SlideTransition } from '@open-slide/core';
import type { CSSProperties, ReactNode } from 'react';
import crocCool from './assets/croc-cool.png';
import crocIcon from './assets/croc-icon.png';
import crocMuscle from './assets/croc-muscle.png';
import crocTeam from './assets/croc-team.png';
import crocThinking from './assets/croc-thinking.png';
import crocWizard from './assets/croc-wizard.png';

// Los GIF se leen del fork del catálogo en GitHub, fijado al mismo commit que el
// submódulo, sin copiarlos a este repo: Gym visual permite redistribuirlos allí,
// a 180x180 y con atribución (data/exercises-dataset/NOTICE.md). El servidor de
// las diapositivas no deja importar archivos de fuera de su carpeta.
const CATALOG_MEDIA =
  'https://raw.githubusercontent.com/jayounghoyos/exercises-dataset/8872272c9f78cd30889523b79e34e95cca8c5637/videos';
const gifHammerCurl = `${CATALOG_MEDIA}/0312-2NpxjC1.gif`;

// El lenguaje de la presentación del M1 (Croody): verde casi negro con anillos,
// páginas crema alternadas, tarjetas oscuras, un dorado y etiquetas en mono.
export const design: DesignSystem = {
  palette: { bg: '#030a02', text: '#f3f0e6', accent: '#d3b46a' },
  fonts: {
    display: '"Inter", "Helvetica Neue", Arial, sans-serif',
    body: '"Inter", "Helvetica Neue", Arial, sans-serif',
  },
  typeScale: { hero: 150, body: 40 },
  radius: 4,
};

const MONO = '"IBM Plex Mono", "Courier New", monospace';
const CREAM = '#f3f0e6';
const INK = '#13140f';
const CARD = '#13140f';
const CARD_GOLD = '#241f0e';
const GOLD_DEEP = '#9a7a32';
const MUTED_DARK = '#a3a897';
const MUTED_CREAM = '#6d6c65';
const LINE_DARK = 'rgba(243, 240, 230, 0.12)';
const LINE_GOLD = 'rgba(211, 180, 106, 0.55)';
const NEGATIVE = '#e0785f';
const PAD_X = 150;
const PAD_TOP = 104;

type Tone = 'dark' | 'cream';

// ---------------------------------------------------------------- fuentes y movimiento

const FONT_HREF =
  'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap';
const FONT_LINK_ID = 'osd-webfont-sustentacion-final';
const STYLE_ID = 'osd-styles-sustentacion-final';
const CSS = `
@keyframes mgr-ripple { 0%, 100% { stroke-opacity: 0.035 } 50% { stroke-opacity: 0.16 } }
@keyframes mgr-glow { 0%, 100% { opacity: 0.75 } 50% { opacity: 1 } }
@keyframes mgr-float { 0%, 100% { transform: translateY(0) rotate(0deg) } 50% { transform: translateY(-14px) rotate(-1.2deg) } }
@keyframes mgr-pop { from { opacity: 0; transform: scale(0.9) } to { opacity: 1; transform: scale(1) } }
@keyframes mgr-grow { from { transform: scaleX(0) } to { transform: scaleX(1) } }
@keyframes mgr-fade { from { opacity: 0; transform: translateY(8px) } to { opacity: 1; transform: translateY(0) } }
@keyframes mgr-blink { 0%, 49% { opacity: 1 } 50%, 100% { opacity: 0 } }
@media (prefers-reduced-motion: reduce) { .mgr-anim, .mgr-anim * { animation: none !important } }
`;
if (typeof document !== 'undefined') {
  let link = document.getElementById(FONT_LINK_ID) as HTMLLinkElement | null;
  if (!link) {
    link = document.createElement('link');
    link.id = FONT_LINK_ID;
    link.rel = 'stylesheet';
    document.head.appendChild(link);
  }
  if (link.href !== FONT_HREF) link.href = FONT_HREF;
  let style = document.getElementById(STYLE_ID);
  if (!style) {
    style = document.createElement('style');
    style.id = STYLE_ID;
    document.head.appendChild(style);
  }
  if (style.textContent !== CSS) style.textContent = CSS;
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

// ---------------------------------------------------------------- fondo y marco

const Rings = ({ cx = 960, cy = 420 }: { cx?: number; cy?: number }) => (
  <svg
    className="mgr-anim"
    viewBox="0 0 1920 1080"
    width={1920}
    height={1080}
    aria-hidden="true"
    style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
  >
    {Array.from({ length: 10 }, (_, i) => (
      <circle
        key={i}
        cx={cx}
        cy={cy}
        r={150 * (i + 1)}
        fill="none"
        stroke="#8fc46f"
        strokeOpacity={0.05}
        strokeWidth={2}
        style={{ animation: `mgr-ripple 7s ease-in-out ${i * 0.55}s infinite` }}
      />
    ))}
  </svg>
);

const Background = ({ tone, cx, cy }: { tone: Tone; cx?: number; cy?: number }) =>
  tone === 'cream' ? null : (
    <>
      <div
        className="mgr-anim"
        style={{
          position: 'absolute',
          inset: 0,
          background: `radial-gradient(circle at ${((cx ?? 960) / 1920) * 100}% ${((cy ?? 420) / 1080) * 100}%, #0d2609 0%, #061805 30%, #030a02 62%, #010401 100%)`,
          animation: 'mgr-glow 9s ease-in-out infinite',
        }}
      />
      <Rings cx={cx} cy={cy} />
    </>
  );

const PageCount = () => {
  const { current, total } = useSlidePageNumber();
  return (
    <span>
      [ {String(current).padStart(2, '0')} / {String(total).padStart(2, '0')} ]
    </span>
  );
};

const Footer = ({ tone }: { tone: Tone }) => {
  const color = tone === 'dark' ? 'rgba(243, 240, 230, 0.38)' : '#8d8b82';
  return (
    <div
      style={{
        position: 'absolute',
        left: PAD_X,
        right: PAD_X,
        bottom: 48,
        display: 'flex',
        justifyContent: 'space-between',
        fontFamily: MONO,
        fontSize: 20,
        letterSpacing: '0.18em',
        color,
      }}
    >
      <span>MACGYVER GYM RAT</span>
      <PageCount />
    </div>
  );
};

const Label = ({ n, text, tone }: { n: string; text: string; tone: Tone }) => (
  <div
    style={{
      fontFamily: MONO,
      fontSize: 24,
      fontWeight: 500,
      letterSpacing: '0.24em',
      textTransform: 'uppercase',
      color: tone === 'dark' ? 'var(--osd-accent)' : GOLD_DEEP,
      marginBottom: 36,
    }}
  >
    {n} · {text}
  </div>
);

const Shell = ({
  tone,
  n,
  label,
  children,
  cx,
  cy,
}: {
  tone: Tone;
  n: string;
  label: string;
  children: ReactNode;
  cx?: number;
  cy?: number;
}) => (
  <div
    style={{
      width: '100%',
      height: '100%',
      boxSizing: 'border-box',
      position: 'relative',
      overflow: 'hidden',
      background: tone === 'dark' ? 'var(--osd-bg)' : CREAM,
      color: tone === 'dark' ? 'var(--osd-text)' : INK,
      fontFamily: 'var(--osd-font-body)',
    }}
  >
    <Background tone={tone} cx={cx} cy={cy} />
    <div style={{ position: 'relative', padding: `${PAD_TOP}px ${PAD_X}px 0` }}>
      <Label n={n} text={label} tone={tone} />
      {children}
    </div>
    <Footer tone={tone} />
  </div>
);

// ---------------------------------------------------------------- piezas

const h2: CSSProperties = {
  fontFamily: 'var(--osd-font-display)',
  fontSize: 72,
  lineHeight: 1.12,
  fontWeight: 700,
  letterSpacing: '-0.02em',
  margin: 0,
};

const muted = (tone: Tone) => (tone === 'dark' ? MUTED_DARK : MUTED_CREAM);

const lead = (tone: Tone): CSSProperties => ({
  fontSize: 'var(--osd-size-body)',
  lineHeight: 1.5,
  color: muted(tone),
  margin: '28px 0 0',
  maxWidth: 1300,
});

const Card = ({
  children,
  gold = false,
  style,
}: {
  children: ReactNode;
  gold?: boolean;
  style?: CSSProperties;
}) => (
  <div
    style={{
      background: gold ? CARD_GOLD : CARD,
      border: `2px solid ${gold ? LINE_GOLD : LINE_DARK}`,
      borderRadius: 'var(--osd-radius)',
      color: CREAM,
      boxSizing: 'border-box',
      ...style,
    }}
  >
    {children}
  </div>
);

const Mascot = ({
  src,
  alt,
  width,
  delay = 0,
  style,
}: {
  src: string;
  alt: string;
  width: number;
  delay?: number;
  style?: CSSProperties;
}) => (
  <div
    className="mgr-anim"
    style={{
      position: 'absolute',
      animation: `mgr-pop 600ms cubic-bezier(0.2, 0.8, 0.2, 1) ${delay}ms both`,
      ...style,
    }}
  >
    <div style={{ animation: `mgr-float 6s ease-in-out ${delay + 600}ms infinite` }}>
      <div
        style={{
          background: '#030a02',
          border: `2px solid ${LINE_GOLD}`,
          borderRadius: 'var(--osd-radius)',
          overflow: 'hidden',
          width,
          lineHeight: 0,
        }}
      >
        <img src={src} alt={alt} style={{ width: '100%', display: 'block' }} />
      </div>
    </div>
  </div>
);

const CheckMark = ({ ok = true, size = 40, color }: { ok?: boolean; size?: number; color?: string }) => (
  <svg viewBox="0 0 12 12" width={size} height={size} aria-hidden="true" style={{ flex: 'none' }}>
    {ok ? (
      <path d="M2.5 6.2l2.3 2.3 4.7-5" fill="none" stroke={color ?? 'var(--osd-accent)'} strokeWidth={1.8}
        strokeLinecap="round" strokeLinejoin="round" />
    ) : (
      <path d="M3 3l6 6M9 3l-6 6" fill="none" stroke={NEGATIVE} strokeWidth={1.8}
        strokeLinecap="round" />
    )}
  </svg>
);

const Arrow = ({ color }: { color: string }) => (
  <svg viewBox="0 0 40 24" width={44} height={26} aria-hidden="true" style={{ flex: 'none' }}>
    <path d="M2 12h32M26 4l8 8-8 8" fill="none" stroke={color} strokeWidth={2.5}
      strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const Tag = ({ children, color = 'var(--osd-accent)', tight = false }:
  { children: ReactNode; color?: string; tight?: boolean }) => (
  <div style={{ fontFamily: MONO, fontSize: tight ? 20 : 22, letterSpacing: tight ? '0.08em' : '0.16em',
    textTransform: 'uppercase', whiteSpace: 'nowrap', color }}>
    {children}
  </div>
);

const Terminal = ({ rows, style }: { rows: [string, string, boolean?][]; style?: CSSProperties }) => (
  <Card style={{ padding: '28px 36px', fontFamily: MONO, ...style }}>
    {rows.map(([k, v, hi], i) => (
      <div key={k} style={{ display: 'flex', gap: 28, fontSize: 24, lineHeight: 1.75 }}>
        <span style={{ color: MUTED_DARK, width: i === 0 ? 'auto' : 180, flex: 'none' }}>{k}</span>
        <span style={{ color: hi ? 'var(--osd-accent)' : CREAM, fontWeight: hi ? 500 : 400 }}>{v}</span>
      </div>
    ))}
  </Card>
);

const Figure = ({
  value,
  label,
  color,
  size = 88,
}: {
  value: string;
  label: string;
  color?: string;
  size?: number;
}) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
    <div style={{ fontSize: size, fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1,
      fontVariantNumeric: 'tabular-nums', color }}>{value}</div>
    <div style={{ fontSize: 28, lineHeight: 1.4, color: MUTED_DARK, maxWidth: 460 }}>{label}</div>
  </div>
);

// ---------------------------------------------------------------- páginas

const Cover: Page = () => (
  <div
    style={{
      width: '100%',
      height: '100%',
      position: 'relative',
      overflow: 'hidden',
      background: 'var(--osd-bg)',
      color: 'var(--osd-text)',
      fontFamily: 'var(--osd-font-body)',
    }}
  >
    <Background tone="dark" cx={1480} cy={400} />
    <div style={{ position: 'relative', padding: `80px ${PAD_X}px 0` }}>
      <div style={{ fontFamily: MONO, fontSize: 22, letterSpacing: '0.24em', color: 'var(--osd-accent)',
        marginTop: 40 }}>
        SI4006 · TÓPICOS ESPECIALES EN IA · SUSTENTACIÓN FINAL
      </div>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginTop: 64 }}>
        <div>
          <h1 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 132, fontWeight: 800,
            letterSpacing: '-0.035em', lineHeight: 1, margin: 0 }}>
            MacGyver
            <br />
            Gym Rat
          </h1>
          <p style={{ ...lead('dark'), fontSize: 32, marginTop: 30, maxWidth: 1040 }}>
            Un ejercicio real para el músculo que quieres, con lo que tienes en casa.
            O un "no" honesto cuando no hay uno seguro.
          </p>
        </div>
        <figure className="mgr-anim" style={{ margin: '-100px 40px 0 0', width: 320,
          animation: 'mgr-pop 700ms cubic-bezier(0.2, 0.8, 0.2, 1) 250ms both' }}>
          <div style={{ background: CREAM, border: `2px solid ${LINE_GOLD}`, borderRadius: 'var(--osd-radius)',
            padding: 20 }}>
            <img src={gifHammerCurl} alt="dumbbell hammer curl v. 2" style={{ width: 276, height: 276, display: 'block' }} />
          </div>
          <figcaption style={{ fontFamily: MONO, fontSize: 18, lineHeight: 1.5, color: MUTED_DARK, marginTop: 14 }}>
            dumbbell hammer curl v. 2
            <br />
            GIF del catálogo © Gym visual
            <br />
            https://gymvisual.com/
          </figcaption>
        </figure>
      </div>
    </div>
    <div style={{ position: 'absolute', left: PAD_X, right: PAD_X, bottom: 100, display: 'flex', gap: 40 }}>
      <Terminal
        style={{ width: 700, padding: '22px 32px' }}
        rows={[
          ['> final · despliegue', ''],
          ['modelo', 'Qwen3-1.7B + LoRA'],
          ['sistema', 'RAG + comprobación'],
          ['estado', 'EN LÍNEA', true],
        ]}
      />
      <Card gold style={{ flex: 1, padding: 28, display: 'flex', alignItems: 'center', gap: 32 }}>
        <img src={crocTeam} alt="Croody en Medellín" style={{ width: 150, height: 150, borderRadius: 2, flex: 'none' }} />
        <div>
          <div style={{ fontFamily: MONO, fontSize: 22, color: 'var(--osd-accent)', letterSpacing: '0.08em' }}>
            MacGyver Gym Rat · Equipo
          </div>
          <div style={{ fontSize: 25, lineHeight: 1.55, marginTop: 14 }}>
            Juan Andrés Young · José Alejandro Jiménez
            <br />
            David Cuadros · Alberto Cervantes
          </div>
          <div style={{ fontSize: 22, color: MUTED_DARK, marginTop: 8 }}>Universidad EAFIT · noviembre de 2026</div>
        </div>
      </Card>
    </div>
    <div style={{ position: 'absolute', right: PAD_X, bottom: 48, fontFamily: MONO, fontSize: 20,
      letterSpacing: '0.18em', color: 'rgba(243, 240, 230, 0.38)' }}>
      <PageCount />
    </div>
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
  <Shell tone="cream" n="01" label="El problema">
    <div style={{ display: 'flex', gap: 96, alignItems: 'flex-start' }}>
      <div style={{ flex: 1 }}>
        <Steps>
          <h2 style={{ ...h2, fontSize: 92, marginTop: 24 }}>
            Hoy toca bíceps. No hay gimnasio. Solo hay dos botellas de agua.
          </h2>
          <Step>
            <p style={{ ...lead('cream'), fontSize: 40, marginTop: 64 }}>
              Hoy esa persona se salta el músculo, hace algo genérico o copia algo de internet
              sin saber si sirve ni si es seguro.
            </p>
          </Step>
        </Steps>
      </div>
      <Card style={{ width: 560, height: 640, flex: 'none', padding: '40px 44px', display: 'flex',
        flexDirection: 'column', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div style={{ fontSize: 120, lineHeight: 0.8, fontWeight: 800, color: 'var(--osd-accent)' }}>"</div>
          <img src={crocIcon} alt="" style={{ width: 96, height: 96, mixBlendMode: 'lighten' }} />
        </div>
        <div style={{ fontFamily: MONO, fontSize: 26, lineHeight: 1.7 }}>
          <div style={{ color: 'var(--osd-accent)' }}>$ pedido</div>
          <div>quiero entrenar bíceps</div>
          <div>tengo: dos botellas de agua</div>
          <div style={{ color: 'var(--osd-accent)', marginTop: 28 }}>$ respuesta</div>
          <div>
            ?
            <span className="mgr-anim" style={{ animation: 'mgr-blink 1.1s steps(1) infinite' }}>_</span>
          </div>
        </div>
        <div style={{ fontSize: 22, lineHeight: 1.5, color: MUTED_DARK }}>
          Croody construye productos para vivir mejor fuera de ellos, no para quedarse dentro.
        </div>
      </Card>
    </div>
  </Shell>
);

const CheckRow = ({ text }: { text: string }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 28, borderLeft: '6px solid var(--osd-accent)',
    padding: '18px 0 18px 32px', fontSize: 46, lineHeight: 1.25, fontWeight: 500 }}>
    <CheckMark size={50} />
    <span>{text}</span>
  </div>
);

const GoodAnswer: Page = () => (
  <Shell tone="dark" n="02" label="La métrica" cx={1560} cy={560}>
    <h2 style={h2}>Una buena respuesta pasa tres comprobaciones</h2>
    <Steps>
      <Step><div style={{ marginTop: 64 }}><CheckRow text="El ejercicio existe en un catálogo de 1.324" /></div></Step>
      <Step><div style={{ marginTop: 28 }}><CheckRow text="Entrena el músculo que pediste" /></div></Step>
      <Step><div style={{ marginTop: 28 }}><CheckRow text="Se puede hacer con tu objeto" /></div></Step>
      <Step>
        <Card gold style={{ marginTop: 64, padding: '28px 36px', maxWidth: 1400 }}>
          <p style={{ fontSize: 36, lineHeight: 1.45, margin: 0 }}>
            Si ninguno pasa, la buena respuesta es decir que no. Esa es la métrica que defendemos.
          </p>
        </Card>
      </Step>
    </Steps>
  </Shell>
);

const FlowNode = ({ title, sub, tag }: { title: string; sub: string; tag: string }) => (
  <Card style={{ padding: '32px 28px', width: 300, height: 300, display: 'flex', flexDirection: 'column',
    justifyContent: 'space-between' }}>
    <div>
      <div style={{ fontSize: 34, fontWeight: 700, lineHeight: 1.2 }}>{title}</div>
      <div style={{ fontSize: 25, lineHeight: 1.4, color: MUTED_DARK, marginTop: 14 }}>{sub}</div>
    </div>
    <Tag tight>{tag}</Tag>
  </Card>
);

const Outcome = ({ ok, title, text }: { ok: boolean; title: string; text: string }) => (
  <Card gold={ok} style={{ flex: 1, padding: '28px 36px', display: 'flex', alignItems: 'center', gap: 28 }}>
    <CheckMark ok={ok} size={56} />
    <div>
      <div style={{ fontSize: 34, fontWeight: 700 }}>{title}</div>
      <div style={{ fontSize: 28, lineHeight: 1.4, color: MUTED_DARK, marginTop: 6 }}>{text}</div>
    </div>
  </Card>
);

const System: Page = () => (
  <Shell tone="cream" n="03" label="El sistema">
    <Mascot src={crocThinking} alt="Croody pensando" width={200} delay={150} style={{ right: PAD_X, top: 56 }} />
    <h2 style={h2}>El sistema, de tu pedido a la respuesta</h2>
    <div style={{ display: 'flex', alignItems: 'center', gap: 22, marginTop: 80 }}>
      <Steps>
        <FlowNode title="Tu pedido" sub="Un músculo y un objeto de tu casa" tag="La pantalla" />
        <Step>
          <div style={{ display: 'flex', alignItems: 'center', gap: 22 }}>
            <Arrow color={INK} />
            <FlowNode title="Busca en el catálogo" sub="Los 8 ejercicios más parecidos" tag="Módulo 3" />
          </div>
        </Step>
        <Step>
          <div style={{ display: 'flex', alignItems: 'center', gap: 22 }}>
            <Arrow color={INK} />
            <FlowNode title="El modelo escribe" sub="Qwen3-1.7B ajustado a la tarea" tag="Módulo 1" />
          </div>
        </Step>
        <Step>
          <div style={{ display: 'flex', alignItems: 'center', gap: 22 }}>
            <Arrow color={INK} />
            <FlowNode title="Se comprueba" sub="Las tres comprobaciones" tag="Módulo 2 y final" />
          </div>
        </Step>
      </Steps>
    </div>
    {/* 4 cajas de 300 y 3 flechas de 88: el ancho del flujo de arriba. */}
    <div style={{ display: 'flex', gap: 40, marginTop: 56, width: 4 * 300 + 3 * 88 }}>
      <Outcome ok title="Pasa" text="El ejercicio, con sus pasos reales y su GIF del catálogo" />
      <Outcome ok={false} title="No pasa" text={'"No hay una opción segura", sin imagen'} />
    </div>
  </Shell>
);

const CaseCard = ({ kind, input, expect, gold = false }:
  { kind: string; input: string; expect: string; gold?: boolean }) => (
  <Card gold={gold} style={{ padding: '40px 40px', width: 500, height: 420, display: 'flex',
    flexDirection: 'column', gap: 24 }}>
    <Tag>{kind}</Tag>
    <div style={{ fontSize: 40, fontWeight: 700, lineHeight: 1.25 }}>{input}</div>
    <div style={{ fontSize: 29, lineHeight: 1.45, color: MUTED_DARK }}>{expect}</div>
  </Card>
);

const Demo: Page = () => (
  <Shell tone="dark" n="04" label="Demo en vivo" cx={960} cy={980}>
    <h2 style={h2}>Demo en vivo: tres casos</h2>
    <div style={{ display: 'flex', gap: 40, marginTop: 72 }}>
      <Steps>
        <Step>
          <CaseCard kind="01 · Fácil" input="Bíceps con dos botellas de agua"
            expect="Un curl con mancuernas, sus pasos y su GIF." />
        </Step>
        <Step>
          <CaseCard kind="02 · Difícil" input="Espalda alta con dos latas de comida"
            expect='Las latas nunca dicen "mancuerna", y aun así llega a un remo.' />
        </Step>
        <Step>
          <CaseCard gold kind="03 · Trampa" input="Dorsales con dos latas de comida"
            expect="El catálogo no tiene ninguno: tiene que decir que no." />
        </Step>
      </Steps>
    </div>
    <div style={{ fontFamily: MONO, fontSize: 28, marginTop: 56, color: MUTED_DARK }}>
      <span style={{ color: 'var(--osd-accent)' }}>&gt; </span>
      huggingface.co/spaces/jayoungh/macgyver-gym-rat
    </div>
  </Shell>
);

const Bar = ({ label, value, max, delay, hi = false }:
  { label: string; value: number; max: number; delay: number; hi?: boolean }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '440px 1fr 200px', alignItems: 'center', gap: 32 }}>
    <div style={{ fontSize: 32, lineHeight: 1.3, fontWeight: hi ? 700 : 400 }}>{label}</div>
    <div style={{ height: 44, background: '#e2ddcc', borderRadius: 2 }}>
      <div
        className="mgr-anim"
        style={{
          width: `${(value / max) * 100}%`,
          height: '100%',
          background: hi ? GOLD_DEEP : INK,
          borderRadius: 2,
          transformOrigin: 'left center',
          animation: `mgr-grow 1000ms cubic-bezier(0.2, 0.7, 0.2, 1) ${delay}ms both`,
        }}
      />
    </div>
    <div
      className="mgr-anim"
      style={{
        fontFamily: MONO,
        fontSize: 44,
        fontWeight: 500,
        textAlign: 'right',
        color: hi ? GOLD_DEEP : INK,
        animation: `mgr-fade 500ms ease-out ${delay + 700}ms both`,
      }}
    >
      {value.toFixed(1).replace('.', ',')} %
    </div>
  </div>
);

const Result: Page = () => (
  <Shell tone="cream" n="05" label="Resultados">
    <Mascot src={crocMuscle} alt="Croody fuerte" width={190} delay={1800} style={{ right: PAD_X, top: 56 }} />
    <h2 style={h2}>Buscar en el catálogo duplicó los aciertos</h2>
    <p style={{ ...lead('cream'), fontFamily: MONO, fontSize: 24, marginTop: 20 }}>
      Las mismas 14 preguntas y la misma vara para todos: las tres comprobaciones del catálogo.
    </p>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 36, marginTop: 64 }}>
      <Bar label="Sin ajustar el modelo" value={0} max={50} delay={200} />
      <Bar label="Modelo ajustado, sin buscar" value={21.4} max={50} delay={600} />
      <Bar label="Modelo ajustado que busca" value={42.9} max={50} delay={1000} hi />
    </div>
    <div className="mgr-anim" style={{ display: 'flex', gap: 40, marginTop: 64,
      animation: 'mgr-fade 600ms ease-out 1900ms both' }}>
      <Card style={{ padding: '32px 40px', width: 620 }}>
        <Figure size={72} value="0 → 3 de 4" label="casos trampa resueltos al buscar" color="var(--osd-accent)" />
      </Card>
      <Card style={{ padding: '32px 40px', width: 620 }}>
        <Figure size={72} value="0,55 → 1,00" label="pasos iguales a los del catálogo (ROUGE-L)" />
      </Card>
    </div>
  </Shell>
);

const Cost: Page = () => (
  <Shell tone="dark" n="06" label="El costo" cx={420} cy={860}>
    <h2 style={h2}>Lo que costó</h2>
    <div style={{ display: 'flex', gap: 40, marginTop: 80 }}>
      <Card style={{ padding: '44px 48px', width: 760 }}>
        <Figure value="7,9 s → 15,2 s" label="por pregunta al buscar en el catálogo (GPU T4, mediana)" />
      </Card>
      <Card gold style={{ padding: '44px 48px', width: 760 }}>
        <Figure value="38,9 s" color="var(--osd-accent)"
          label="con el agente y su herramienta, que bajaba los aciertos a 28,6 %" />
      </Card>
    </div>
    <Steps>
      <Step>
        <p style={{ ...lead('dark'), color: 'var(--osd-text)', marginTop: 80, maxWidth: 1500 }}>
          El doble de tiempo compra ejercicios reales con sus pasos reales. El agente no lo
          compensaba: lo medimos y lo dejamos fuera del despliegue.
        </p>
      </Step>
    </Steps>
  </Shell>
);

const Cell = ({ children, head = false, pending = false, first = false }:
  { children: string; head?: boolean; pending?: boolean; first?: boolean }) => (
  <div
    style={{
      fontFamily: first && !head ? 'var(--osd-font-body)' : MONO,
      fontSize: head ? 20 : first ? 30 : 28,
      letterSpacing: head ? '0.16em' : 0,
      textTransform: head ? 'uppercase' : 'none',
      color: head ? 'var(--osd-accent)' : pending ? MUTED_DARK : CREAM,
      fontStyle: pending ? 'italic' : 'normal',
      fontWeight: first && !head ? 700 : 400,
      padding: '26px 0',
      borderBottom: `2px solid ${LINE_DARK}`,
    }}
  >
    {children}
  </div>
);

const Efficiency: Page = () => (
  <Shell tone="cream" n="07" label="Eficiencia">
    <h2 style={h2}>Eficiencia: el modelo a 4 bits</h2>
    <p style={{ ...lead('cream'), fontSize: 32, marginTop: 20 }}>
      Antes de medir: aceptamos perder hasta 5 puntos si baja la memoria y no sube la latencia.
    </p>
    <Card style={{ marginTop: 56, padding: '28px 48px 36px' }}>
      <Tag>Corrida en GPU T4 · antes y después</Tag>
      <div style={{ display: 'grid', gridTemplateColumns: '340px repeat(4, 1fr)', columnGap: 32, marginTop: 12 }}>
        <Cell head>Versión</Cell>
        <Cell head>Aciertos</Cell>
        <Cell head>Latencia p50</Cell>
        <Cell head>Latencia p95</Cell>
        <Cell head>Memoria</Cell>
        <Cell first>Antes (16 bits)</Cell>
        <Cell pending>pendiente</Cell>
        <Cell pending>pendiente</Cell>
        <Cell pending>pendiente</Cell>
        <Cell pending>pendiente</Cell>
        <Cell first>Después (4 bits)</Cell>
        <Cell pending>pendiente</Cell>
        <Cell pending>pendiente</Cell>
        <Cell pending>pendiente</Cell>
        <Cell pending>pendiente</Cell>
      </div>
    </Card>
    <p style={{ fontFamily: MONO, fontSize: 22, color: MUTED_CREAM, marginTop: 36 }}>
      Las cifras salen de la corrida en Colab T4 (notebooks/M5_efficiency.ipynb).
    </p>
  </Shell>
);

const Fails: Page = () => (
  <Shell tone="dark" n="08" label="Lo que falla" cx={1620} cy={300}>
    <Mascot src={crocWizard} alt="Croody mago" width={210} delay={150} style={{ right: PAD_X, top: 56 }} />
    <h2 style={{ ...h2, maxWidth: 1300 }}>Lo que falla: prefiere callar a inventar</h2>
    <div style={{ display: 'flex', gap: 40, marginTop: 80 }}>
      <Steps>
        <Step>
          <Card style={{ padding: '40px 44px', width: 500, height: 290, borderLeft: `6px solid ${NEGATIVE}` }}>
            <Figure value="7 de 49" label="combinaciones con respuesta en el catálogo que sí respondió" color={NEGATIVE} />
          </Card>
        </Step>
        <Step>
          <Card style={{ padding: '40px 44px', width: 500, height: 290 }}>
            <Figure value="11 de 11" label="combinaciones sin respuesta en las que dijo que no" />
          </Card>
        </Step>
        <Step>
          <Card style={{ padding: '40px 44px', width: 500, height: 290 }}>
            <Figure value="0" label="ejercicios inventados o que no aplican, mostrados al usuario" />
          </Card>
        </Step>
      </Steps>
    </div>
    <Card gold style={{ marginTop: 56, padding: '28px 36px', display: 'flex', gap: 28, alignItems: 'baseline' }}>
      <Tag>Arreglo</Tag>
      <span style={{ fontSize: 32, lineHeight: 1.4 }}>
        Buscar solo en la clase de equipo del objeto, sin preguntarle al modelo.
      </span>
    </Card>
  </Shell>
);

const Protection = ({ text }: { text: string }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 22, fontSize: 34, lineHeight: 1.3 }}>
    <CheckMark size={38} />
    <span>{text}</span>
  </div>
);

const Ethics: Page = () => (
  <Shell tone="dark" n="09" label="Ética" cx={1500} cy={560}>
    <h2 style={h2}>El peor error posible</h2>
    <div style={{ display: 'flex', gap: 64, alignItems: 'flex-start', marginTop: 64 }}>
      <div style={{ flex: 1 }}>
        <Tag>Las protecciones</Tag>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 30, marginTop: 36 }}>
          <Steps>
            <Step><Protection text="La válvula: dice que no en vez de inventar" /></Step>
            <Step><Protection text="La comprobación en el catálogo antes de mostrar" /></Step>
            <Step><Protection text="El aviso: el sistema sugiere, tú decides" /></Step>
            <Step><Protection text="Sin datos personales en los registros" /></Step>
          </Steps>
        </div>
      </div>
      <Card gold style={{ width: 660, flex: 'none', padding: '44px 52px' }}>
        <div style={{ fontSize: 120, lineHeight: 0.7, fontWeight: 800, color: 'var(--osd-accent)' }}>"</div>
        <p style={{ fontSize: 50, fontWeight: 700, lineHeight: 1.2, margin: '28px 0 0' }}>
          Cuélgate de la barra de la cortina.
        </p>
        <p style={{ fontSize: 28, lineHeight: 1.45, color: MUTED_DARK, margin: '24px 0 0' }}>
          Lo dijo el modelo sin buscar, y el juez automático le puso
        </p>
        <div style={{ fontFamily: MONO, fontSize: 64, fontWeight: 500, color: NEGATIVE, marginTop: 12 }}>5 / 5</div>
      </Card>
    </div>
  </Shell>
);

const Thanks: Page = () => (
  <Shell tone="dark" n="10" label="Gracias" cx={1460} cy={520}>
    <Mascot src={crocCool} alt="Croody con gafas" width={520} delay={250} style={{ right: PAD_X, top: 220 }} />
    <h1 style={{ fontFamily: 'var(--osd-font-display)', fontSize: 150, fontWeight: 800, letterSpacing: '-0.035em',
      lineHeight: 1, margin: '36px 0 0' }}>
      Gracias
      <br />
      por el curso.
    </h1>
    <div style={{ fontFamily: MONO, fontSize: 36, color: 'var(--osd-accent)', marginTop: 44 }}>
      ¿Preguntas?
      <span className="mgr-anim" style={{ animation: 'mgr-blink 1.1s steps(1) infinite' }}>_</span>
    </div>
    <p style={{ ...lead('dark'), fontSize: 28, marginTop: 20 }}>
      SI4006 · Tópicos especiales en IA · Universidad EAFIT
    </p>
    <Terminal
      style={{ marginTop: 56, padding: '22px 32px' }}
      rows={[
        ['> enlaces', ''],
        ['demo', 'huggingface.co/spaces/jayoungh/macgyver-gym-rat', true],
        ['métricas', 'wandb.ai/jayoungh-me/macgyver-gym-rat-m5', true],
        ['código', 'github.com/iamcroody/JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros', true],
      ]}
    />
  </Shell>
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
  Thanks,
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
  `Mismas 14 preguntas y misma vara para todas las filas. Las barras crecen solas al entrar.
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
Las cuatro protecciones, una por una. (50 segundos)`,
  `Agradecer al profesor por el curso. Dejar en pantalla los enlaces del Space, de W&B y del repositorio,
y abrir las preguntas. (20 segundos)`,
];
