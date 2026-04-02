import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

# ── Palette ──────────────────────────────────────────────────────
BG       = "#0C0E14"
PANEL    = "#13161F"
BORDER   = "#252A38"
SAFFRON  = "#FF9933"
GOLD     = "#E8C547"
TEAL     = "#3ECFB2"
CRIMSON  = "#E05555"
LAVENDER = "#A78BFA"
GREEN    = "#5BDB8A"
TEXT_HI  = "#EDE8D8"
TEXT_LO  = "#656D87"

CAT_COLORS = {
    "philosophical":     SAFFRON,
    "enumeration":       TEAL,
    "poetic_instruction":GOLD,
    "chapter_reference": LAVENDER,
    "ood_stress":        CRIMSON,
}

SCORE_CMAP = LinearSegmentedColormap.from_list(
    "sc", ["#C0392B","#E8C547","#2ECC71"], N=256)

def load(path):
    return json.loads(open(path, encoding="utf-8").read())

def cat_avg(results, key="composite"):
    v = [r[key] for r in results if r.get(key) is not None]
    return round(sum(v)/len(v), 2) if v else 0.0

def flat(data):
    return [r for cat in data.values() for r in cat]

def sax(ax, title=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER); sp.set_linewidth(0.7)
    ax.tick_params(colors=TEXT_LO, labelsize=7.5)
    if title:
        ax.set_title(title, color=TEXT_HI, fontsize=8.8,
                     pad=7, loc="left", fontweight="bold")

# ── 1. Category bars ──────────────────────────────────────────────
def plot_cat_bars(ax, data):
    sax(ax, "① Category Avg Score  /10")
    cats  = list(data.keys())
    avgs  = [cat_avg(data[c]) for c in cats]
    cols  = [CAT_COLORS.get(c, GOLD) for c in cats]
    y     = np.arange(len(cats))
    bars  = ax.barh(y, avgs, color=cols, height=0.52,
                    edgecolor=BG, linewidth=0.4)
    for bar, val in zip(bars, avgs):
        ax.text(val+0.12, bar.get_y()+bar.get_height()/2,
                f"{val:.2f}", va="center", color=TEXT_HI, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_"," ").title() for c in cats],
                       color=TEXT_HI, fontsize=8.2)
    ax.set_xlim(0, 11)
    ax.axvline(7.55, color=GOLD, lw=1, ls="--", alpha=0.6)
    ax.text(7.57, len(cats)-0.3, "μ=7.55", color=GOLD, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Score", color=TEXT_LO, fontsize=7.5)

# ── 2. Per-prompt line+scatter ────────────────────────────────────
def plot_prompts(ax, data):
    sax(ax, "② Per-Prompt Composite Scores")
    x, y_vals, cols = [], [], []
    labels = []
    i = 0
    for cat, results in data.items():
        col = CAT_COLORS.get(cat, GOLD)
        for r in results:
            x.append(i); y_vals.append(r["composite"])
            cols.append(col)
            labels.append(r.get("prompt","")[:12])
            i += 1
    ax.plot(x, y_vals, color=BORDER, lw=0.9, zorder=2)
    ax.scatter(x, y_vals, c=cols, s=65, zorder=4,
               edgecolors=BG, linewidths=0.5)
    mean = sum(y_vals)/len(y_vals)
    ax.axhline(mean, color=GOLD, lw=1.2, ls="--", alpha=0.8)
    ax.text(len(x)-0.4, mean+0.18, f"μ={mean:.2f}",
            color=GOLD, fontsize=7.5, ha="right")
    # highlight best/worst
    best_i = y_vals.index(max(y_vals))
    worst_i = y_vals.index(min(y_vals))
    ax.annotate(f"▲{y_vals[best_i]}", (x[best_i], y_vals[best_i]),
                xytext=(0,8), textcoords="offset points",
                color=GREEN, fontsize=7, ha="center")
    ax.annotate(f"▼{y_vals[worst_i]}", (x[worst_i], y_vals[worst_i]),
                xytext=(0,-12), textcoords="offset points",
                color=CRIMSON, fontsize=7, ha="center")
    ax.set_ylim(5, 10)
    ax.set_xlim(-0.5, len(x)-0.5)
    ax.set_xlabel("Prompt Index", color=TEXT_LO, fontsize=7.5)
    ax.set_ylabel("Score /10", color=TEXT_LO, fontsize=7.5)
    patches = [mpatches.Patch(color=v, label=k.replace("_"," "))
               for k,v in CAT_COLORS.items() if k in data]
    ax.legend(handles=patches, fontsize=6.5, facecolor=PANEL,
              edgecolor=BORDER, labelcolor=TEXT_LO,
              loc="lower right", framealpha=0.85)

# ── 3. Radar ──────────────────────────────────────────────────────
def plot_radar(ax, data):
    metrics = ["repetition","ttr","devanagari_purity",
               "sentence_complete","length_score",
               "prompt_absorption","relevance"]
    labels  = ["Repet.","TTR","Deva%","Sent.","Len","Absorb","Relev."]
    N = len(metrics)
    fl = flat(data)
    avgs = []
    for m in metrics:
        v = [r[m] for r in fl if r.get(m) is not None]
        # clip devanagari_purity which can exceed 1
        val = sum(v)/len(v) if v else 0
        avgs.append(min(1.0, val))
    angles = [n/N*2*math.pi for n in range(N)] + [0]
    avgs_c = avgs + [avgs[0]]
    ax.set_facecolor(PANEL)
    ax.set_theta_offset(math.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT_HI, fontsize=8.2)
    ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["","0.5","","1.0"], color=TEXT_LO, fontsize=6)
    ax.set_ylim(0,1)
    ax.spines["polar"].set_edgecolor(BORDER)
    ax.grid(color=BORDER, lw=0.6)
    ax.plot(angles, avgs_c, color=SAFFRON, lw=2.2, zorder=3)
    ax.fill(angles, avgs_c, color=SAFFRON, alpha=0.15)
    # per-category overlay
    for cat, results in data.items():
        col = CAT_COLORS.get(cat, GOLD)
        cv = []
        for m in metrics:
            v = [r[m] for r in results if r.get(m) is not None]
            cv.append(min(1.0, sum(v)/len(v)) if v else 0)
        cv_c = cv + [cv[0]]
        ax.plot(angles, cv_c, color=col, lw=0.8, alpha=0.5, ls="--")
    ax.set_title("③ Metric Radar", color=TEXT_HI,
                 fontsize=8.8, pad=14, loc="left", fontweight="bold")

# ── 4. Perplexity bar ─────────────────────────────────────────────
def plot_ppl(ax, data):
    sax(ax, "④ Avg Perplexity  (↓ better)")
    cats = list(data.keys())
    ppls = [cat_avg(data[c], "perplexity") for c in cats]
    cols = [CAT_COLORS.get(c, GOLD) for c in cats]
    x    = np.arange(len(cats))
    bars = ax.bar(x, ppls, color=cols, width=0.55,
                  edgecolor=BG, linewidth=0.4)
    for bar, val in zip(bars, ppls):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.03,
                f"{val:.2f}", ha="center", color=TEXT_HI, fontsize=7.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_","\n") for c in cats],
                       color=TEXT_HI, fontsize=7)
    ax.set_ylabel("Perplexity", color=TEXT_LO, fontsize=7.5)
    ax.axhline(2.68, color=GOLD, lw=1, ls="--", alpha=0.7)
    ax.text(len(cats)-0.4, 2.75, "μ=2.68", color=GOLD, fontsize=7, ha="right")

# ── 5. OOD vs Sanskrit ───────────────────────────────────────────
def plot_ood(ax, data):
    sax(ax, "⑤ OOD vs Sanskrit")
    ood = [r["composite"] for r in data.get("ood_stress",[])]
    san = [r["composite"] for cat,res in data.items()
           if cat!="ood_stress" for r in res]
    for p,(label,vals,col) in enumerate(
            [("Sanskrit",san,TEAL),("OOD",ood,CRIMSON)]):
        if not vals: continue
        jitter = np.random.uniform(-0.14,0.14,len(vals))
        ax.scatter([p+j for j in jitter], vals,
                   color=col, s=60, alpha=0.88,
                   edgecolors=BG, lw=0.4, zorder=3)
        m = sum(vals)/len(vals)
        ax.hlines(m, p-0.28, p+0.28, colors=col, lw=2.5, zorder=4)
        ax.text(p, m+0.25, f"{m:.2f}", ha="center",
                color=col, fontsize=9, fontweight="bold")
    ax.set_xticks([0,1])
    ax.set_xticklabels(["Sanskrit\nPrompts","OOD\nPrompts"],
                       color=TEXT_HI, fontsize=8.5)
    ax.set_ylim(4, 11)
    ax.set_ylabel("Score /10", color=TEXT_LO, fontsize=7.5)

# ── 6. Metric heatmap ────────────────────────────────────────────
def plot_heatmap(ax, data):
    metrics = ["composite","devanagari_purity","repetition","ttr",
               "sentence_complete","length_score","prompt_absorption","relevance"]
    mlabels = ["★Score","Deva%","Repet.","TTR","Sent.","Len","Absorb","Relev."]
    fl = flat(data)
    n, m = len(fl), len(metrics)
    mat = np.zeros((m, n))
    for j, r in enumerate(fl):
        for i, key in enumerate(metrics):
            val = r.get(key) or 0.0
            mat[i,j] = min(1.0, val/10.0 if key=="composite" else val)
    im = ax.imshow(mat, aspect="auto", cmap=SCORE_CMAP,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(m))
    ax.set_yticklabels(mlabels, color=TEXT_HI, fontsize=8)
    ax.set_xticks(range(n))
    xlabels = [f"P{j+1}" for j in range(n)]
    ax.set_xticklabels(xlabels, color=TEXT_LO, fontsize=7.5)
    # value annotations
    for i in range(m):
        for j in range(n):
            v = mat[i,j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.5,
                    color="black" if v > 0.6 else TEXT_HI)
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.ax.tick_params(colors=TEXT_LO, labelsize=6.5)
    cbar.outline.set_edgecolor(BORDER)
    ax.set_title("⑥ Full Metric Heatmap  (all 20 prompts × 8 metrics)",
                 color=TEXT_HI, fontsize=8.8, pad=7, loc="left", fontweight="bold")
    # category separator lines
    idx = 0
    for cat, results in data.items():
        idx += len(results)
        if idx < n:
            ax.axvline(idx-0.5, color=BORDER, lw=1.5)

# ── 7. Score distribution violin-style (manual) ──────────────────
def plot_dist(ax, data):
    sax(ax, "⑦ Score Distribution per Category")
    cats = list(data.keys())
    for i, cat in enumerate(cats):
        scores = [r["composite"] for r in data[cat]]
        col = CAT_COLORS.get(cat, GOLD)
        jitter = np.random.uniform(-0.18, 0.18, len(scores))
        ax.scatter([i+j for j in jitter], scores,
                   color=col, s=55, alpha=0.85,
                   edgecolors=BG, lw=0.4, zorder=3)
        m = sum(scores)/len(scores)
        ax.hlines(m, i-0.3, i+0.3, colors=col, lw=2, zorder=4)
        mn, mx = min(scores), max(scores)
        ax.vlines(i, mn, mx, colors=col, lw=0.8, alpha=0.5)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c.replace("_","\n") for c in cats],
                       color=TEXT_HI, fontsize=7)
    ax.set_ylim(4, 11)
    ax.set_ylabel("Score /10", color=TEXT_LO, fontsize=7.5)
    ax.axhline(7.55, color=GOLD, lw=1, ls="--", alpha=0.6)

# ── MAIN FIGURE ───────────────────────────────────────────────────
def build(data):
    fig = plt.figure(figsize=(22, 15), facecolor=BG)

    # Header
    fig.text(0.5, 0.975, "Sanskrit LLM — GPT-2 Small  |  Evaluation Dashboard",
             ha="center", color=SAFFRON, fontsize=17, fontweight="bold")
    fig.text(0.5, 0.958,
             "Checkpoint 83000  ·  Sangraha Corpus  ·  20 Prompts  ·  5 Categories  ·  10 Metrics",
             ha="center", color=TEXT_LO, fontsize=8.5)
    # Grand score
    fig.text(0.5, 0.942,
             "Overall Score:  7.55 / 10   ·   Best: 8.33   ·   Worst: 6.30   ·   Avg PPL: 2.68",
             ha="center", color=TEAL, fontsize=10.5, fontweight="bold")

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           left=0.05, right=0.97,
                           top=0.93, bottom=0.07,
                           hspace=0.48, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:3])
    ax3 = fig.add_subplot(gs[0, 3], polar=True)
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2:])
    ax7 = fig.add_subplot(gs[2, :])

    plot_cat_bars(ax1, data)
    plot_prompts(ax2, data)
    plot_radar(ax3, data)
    plot_ppl(ax4, data)
    plot_ood(ax5, data)
    plot_dist(ax6, data)
    plot_heatmap(ax7, data)

    return fig

if __name__ == "__main__":
    data = load(r"M:\research papers\sanskrit paper work\SANKSRIT__LLM GPT2\eval_report_20260326_084445.json")
    fig  = build(data)
    out  = r"M:\research papers\sanskrit paper work\SANKSRIT__LLM GPT2\evavisual.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"Saved → {out}")