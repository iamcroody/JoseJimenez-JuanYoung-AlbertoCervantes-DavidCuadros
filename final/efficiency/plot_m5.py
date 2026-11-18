import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1], encoding="utf-8")); out = sys.argv[2]
b, a = d
C = ["#4C72B0", "#DD8452"]; L = ["Antes (fp16)", "Después (int4 nf4)"]
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
def bars(axis, vals, title, fmt, ylabel):
    r = axis.bar(L, vals, color=C); axis.set_title(title); axis.set_ylabel(ylabel)
    axis.bar_label(r, labels=[fmt.format(v) for v in vals]); axis.spines[["top","right"]].set_visible(False)
q = lambda r, k: r["quality"][k] * 100
x = range(3); w = 0.38
for i, r in enumerate((b, a)):
    vals = [q(r,"domain_hit_rate"), q(r,"domain_hit_rate_gold"), q(r,"domain_hit_rate_adversarial")]
    bb = ax[0,0].bar([j + (i-0.5)*w for j in x], vals, w, color=C[i], label=L[i])
    ax[0,0].bar_label(bb, labels=[f"{v:.1f}%" for v in vals], fontsize=8)
ax[0,0].axhline(37.9, ls="--", color="gray", lw=1); ax[0,0].text(2.45, 38.5, "piso del presupuesto 37.9%", ha="right", fontsize=8, color="gray")
ax[0,0].set_xticks(list(x), ["Total", "Answerable", "Adversarial"]); ax[0,0].set_ylabel("% acierto"); ax[0,0].set_ylim(0, 100)
ax[0,0].set_title("Calidad: acierto de dominio (dimensión 3)"); ax[0,0].legend(fontsize=8); ax[0,0].spines[["top","right"]].set_visible(False)
for i, r in enumerate((b, a)):
    ax[0,1].boxplot(r["timings_s"], positions=[i], widths=0.5, patch_artist=True, boxprops=dict(facecolor=C[i], alpha=.5))
    ax[0,1].scatter([i]*14, r["timings_s"], color=C[i], s=12, zorder=3)
    ax[0,1].text(i+0.3, r["latency_p50_s"], f"p50 {r['latency_p50_s']:.1f}s\np95 {r['latency_p95_s']:.1f}s", fontsize=8, va="center")
ax[0,1].set_xticks([0,1], L); ax[0,1].set_ylabel("segundos por consulta"); ax[0,1].set_title("Latencia (14 consultas, misma máquina)")
ax[0,1].spines[["top","right"]].set_visible(False); ax[0,1].set_xlim(-0.6, 1.9)
xs = range(2)
for i, (k, lab) in enumerate((("peak_vram_mb","Pico VRAM"), ("footprint_mb","Pesos del generador"))):
    vals = [b[k], a[k]]
    bb = ax[1,0].bar([j + (i-0.5)*w for j in xs], vals, w, color=["#8172B3","#64B5CD"][i], label=lab)
    ax[1,0].bar_label(bb, labels=[f"{v:,.0f}" for v in vals], fontsize=8)
ax[1,0].set_xticks(list(xs), L); ax[1,0].set_ylabel("MB"); ax[1,0].set_title("Memoria"); ax[1,0].legend(fontsize=8); ax[1,0].spines[["top","right"]].set_visible(False)
na = lambda r: r["non_accuracy"]
labels = ["Invarianza\n(resp. idénticas)", "Válvula dispara\n(adv01 + adv04)", "Adversarial\n(dim 3)"]
for i, r in enumerate((b, a)):
    vals = [na(r)["invariance"]["invariance_rate"]*100, 100 if na(r)["valve"]["all_fired"] else 0, na(r)["adversarial"]["adversarial_hit_rate"]*100]
    bb = ax[1,1].bar([j + (i-0.5)*w for j in range(3)], vals, w, color=C[i], label=L[i])
    ax[1,1].bar_label(bb, labels=[f"{v:.0f}%" for v in vals], fontsize=8)
ax[1,1].set_xticks(range(3), labels, fontsize=8); ax[1,1].set_ylim(0, 110); ax[1,1].set_ylabel("%")
ax[1,1].set_title("Pruebas que no miden exactitud"); ax[1,1].legend(fontsize=8); ax[1,1].spines[["top","right"]].set_visible(False)
fig.suptitle("M5 · int4 (nf4) vs fp16 · RAG avanzado · 14 casos · RTX 4060 Laptop", fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=150)
