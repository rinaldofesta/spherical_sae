"""Generate the explanatory figure for the Spherical SAE idea.

Palette: black / white / red / blue only.
Writes ``assets/spherical_sae.png`` — a two-part schematic on a single
equal-aspect canvas (so the spheres render round, not squished).

  Top    -- Classical SAE vs Spherical SAE side by side, so the *only* added
            step (normalising the sparse code before decoding) stands out.
  Bottom -- the geometry: concepts are directions on a sphere, and the
            normalised code keeps only the *proportion* of concepts.

    python scripts/make_figure.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Arc, Rectangle

# ---- palette: black / white / red / blue --------------------------------
WHITE = "#FFFFFF"
INK = "#0F1216"                     # near-black: text, arrows, axes
BLUE = "#1B4DD8"                    # standard SAE machinery
BLUE_F = "#E7EEFC"
RED = "#D21F3C"                     # the spherical step and its outputs
RED_F = "#FBE5EA"
LINE = (0.06, 0.07, 0.09, 0.22)     # ink @ low alpha — sphere wireframe
MUTE = (0.06, 0.07, 0.09, 0.32)     # ink @ low alpha — dormant items
SUB = (0.06, 0.07, 0.09, 0.60)      # ink @ medium alpha — captions

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})


# ---- primitives ----------------------------------------------------------
def box(ax, cx, cy, w, h, text, *, fill=BLUE_F, edge=BLUE, fs=11,
        weight="bold", tcolor=INK):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2.0, facecolor=fill, edgecolor=edge, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=tcolor, weight=weight, zorder=4)


def arrow(ax, x0, x1, y, label=None, *, color=INK, lw=2.0, fs=9.0, dy=0.26):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=13,
        linewidth=lw, color=color, zorder=2))
    if label:
        ax.text((x0 + x1) / 2, y + dy, label, ha="center", va="bottom",
                fontsize=fs, color=SUB, style="italic")


def sphere(ax, cx, cy, r, *, vec_deg=None, vec_color=INK, label=None):
    """A round globe: circular outline + elliptical equator/meridian."""
    ax.add_patch(Circle((cx, cy), r, facecolor=WHITE, edgecolor=LINE,
                         linewidth=1.5, zorder=2))
    ax.add_patch(Arc((cx, cy), 2 * r, 0.78 * r, theta1=0, theta2=360,
                     edgecolor=LINE, lw=0.9, zorder=2))
    ax.add_patch(Arc((cx, cy), 0.78 * r, 2 * r, theta1=0, theta2=360,
                     edgecolor=LINE, lw=0.9, zorder=2))
    if vec_deg is not None:
        a = np.deg2rad(vec_deg)
        px, py = cx + r * np.cos(a), cy + r * np.sin(a)
        ax.add_patch(FancyArrowPatch((cx, cy), (px, py), arrowstyle="-|>",
                     mutation_scale=10, color=vec_color, lw=2.2, zorder=4))
        ax.add_patch(Circle((px, py), 0.07, facecolor=vec_color,
                     edgecolor=WHITE, linewidth=1.0, zorder=5))
    if label:
        ax.text(cx, cy - r - 0.34, label, ha="center", va="top",
                fontsize=12, color=INK, weight="bold")


def bars(ax, cx, cy, heights, *, w=1.30, h=0.95, color=BLUE):
    n = len(heights)
    slot = w / n
    bw = slot * 0.60
    x0 = cx - w / 2
    maxh = max(heights) if max(heights) > 0 else 1.0
    for i, v in enumerate(heights):
        bx = x0 + (i + 0.5) * slot
        if v > 0:
            ax.add_patch(Rectangle((bx - bw / 2, cy - h / 2), bw,
                         (v / maxh) * h, facecolor=color, edgecolor="none",
                         zorder=3))
        else:
            ax.add_patch(Rectangle((bx - bw / 2, cy - h / 2), bw, 0.035,
                         facecolor=MUTE, edgecolor="none", zorder=3))
    ax.plot([x0, x0 + w], [cy - h / 2, cy - h / 2], color=INK, lw=1.0,
            zorder=3)


# ---- one pipeline row -----------------------------------------------------
def pipeline(ax, y, raw, *, spherical):
    sphere(ax, 1.30, y, 0.55, vec_deg=58, vec_color=INK, label=r"$x$")
    arrow(ax, 1.95, 3.00, y, "encoder · TopK", fs=8.5)
    box(ax, 3.70, y, 1.05, 0.88, r"$h$", fs=12)
    bars(ax, 5.05, y, raw, color=INK)

    if spherical:
        arrow(ax, 5.78, 6.62, y)
        box(ax, 7.42, y, 1.58, 0.98, r"$\tilde h=\frac{h}{\|h\|}$",
            fill=RED_F, edge=RED, fs=12)
        arrow(ax, 8.22, 8.80, y)
        norm = raw / (np.linalg.norm(raw) + 1e-8)
        bars(ax, 9.55, y, norm, color=RED)
        arrow(ax, 10.25, 11.30, y, "decoder")
        sphere(ax, 11.95, y, 0.55, vec_deg=58, vec_color=RED, label=r"$\hat x$")
    else:
        arrow(ax, 5.78, 11.30, y, "decoder")
        sphere(ax, 11.95, y, 0.55, vec_deg=58, vec_color=BLUE, label=r"$\hat x$")


# ---- figure ---------------------------------------------------------------
def main() -> None:
    # single equal-aspect canvas; data range ~12.9 x 10.9 -> figsize matched
    fig, ax = plt.subplots(figsize=(12.0, 10.1), dpi=200)
    ax.set_xlim(0, 12.9)
    ax.set_ylim(0, 10.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.012, right=0.988, top=0.992, bottom=0.008)

    raw = np.array([0, 1.00, 0, 0.55, 0, 0, 0.80, 0], dtype=float)
    OFF = 4.95  # vertical offset lifting the pipeline panel above the geometry

    # ===================== TOP : the pipeline ============================
    ax.text(0.18, 5.75 + OFF, "1", fontsize=12, weight="bold", color=WHITE,
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.30", fc=INK, ec="none"))
    ax.text(0.58, 5.75 + OFF, "One extra step in the pipeline", fontsize=13,
            weight="bold", color=INK, va="center")

    ax.text(0.22, 4.85 + OFF, "Classical", fontsize=11.5, weight="bold",
            color=BLUE, rotation=90, ha="center", va="center")
    pipeline(ax, 4.75 + OFF, raw, spherical=False)

    ax.text(0.22, 1.85 + OFF, "Spherical", fontsize=11.5, weight="bold",
            color=RED, rotation=90, ha="center", va="center")
    pipeline(ax, 1.75 + OFF, raw, spherical=True)

    # highlight the one added step + callout
    ax.add_patch(FancyBboxPatch((6.55, 1.14 + OFF), 1.74, 1.24,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 facecolor="none", edgecolor=RED, lw=1.4,
                 linestyle=(0, (4, 3)), zorder=1))
    ax.annotate("normalise the code\nbefore decoding",
                xy=(7.42, 2.38 + OFF), xytext=(7.42, 3.55 + OFF), ha="center",
                fontsize=10.5, color=RED, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))

    # ===================== BOTTOM : the geometry =========================
    ax.text(0.18, 3.95, "2", fontsize=12, weight="bold", color=WHITE,
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.30", fc=INK, ec="none"))
    ax.text(0.58, 3.95, "What the code becomes", fontsize=13,
            weight="bold", color=INK, va="center")

    # -- sphere of concept directions (round, equal aspect) --
    cx, cy, R = 3.15, 2.00, 1.32
    ax.add_patch(Circle((cx, cy), R, facecolor=WHITE, edgecolor=LINE,
                 lw=1.6, zorder=1))
    ax.add_patch(Arc((cx, cy), 2 * R, 0.78 * R, theta1=0, theta2=360,
                 edgecolor=LINE, lw=0.9, zorder=1))
    ax.add_patch(Arc((cx, cy), 0.78 * R, 2 * R, theta1=0, theta2=360,
                 edgecolor=LINE, lw=0.9, zorder=1))

    for ang in (15, 175, 215, 300):
        a = np.deg2rad(ang)
        px, py = cx + R * np.cos(a), cy + R * np.sin(a)
        ax.add_patch(FancyArrowPatch((cx, cy), (px, py), arrowstyle="-|>",
                     mutation_scale=7, color=MUTE, lw=1.1, zorder=2))
        ax.add_patch(Circle((px, py), 0.05, facecolor=MUTE, edgecolor="none",
                     zorder=2))

    for ang, name in ((133, "A"), (45, "B"), (255, "C")):
        a = np.deg2rad(ang)
        px, py = cx + R * np.cos(a), cy + R * np.sin(a)
        ax.add_patch(FancyArrowPatch((cx, cy), (px, py), arrowstyle="-|>",
                     mutation_scale=11, color=BLUE, lw=2.3, zorder=3))
        ax.add_patch(Circle((px, py), 0.075, facecolor=BLUE, edgecolor=WHITE,
                     lw=1.0, zorder=4))
        ax.text(px + 0.22 * np.cos(a), py + 0.26 * np.sin(a), name,
                fontsize=11, color=BLUE, weight="bold", ha="center",
                va="center")

    a = np.deg2rad(85)
    px, py = cx + R * np.cos(a), cy + R * np.sin(a)
    ax.add_patch(FancyArrowPatch((cx, cy), (px, py), arrowstyle="-|>",
                 mutation_scale=12, color=RED, lw=2.8, zorder=5))
    ax.add_patch(Circle((px, py), 0.085, facecolor=RED, edgecolor=WHITE,
                 lw=1.0, zorder=6))
    ax.text(px + 0.05, py + 0.24, r"$\tilde h$", fontsize=12, color=RED,
            weight="bold", ha="center", va="bottom")

    ax.text(cx, cy - R - 0.34, "concepts = directions on the sphere",
            ha="center", va="top", fontsize=10.5, color=INK, weight="bold")
    ax.text(cx, cy - R - 0.64, "blue = active   ·   grey = dormant",
            ha="center", va="top", fontsize=9.0, color=SUB)

    # -- the distribution over active concepts --
    bx = 6.95
    ax.text(bx, 3.42, "the code is purely distributional", fontsize=12,
            weight="bold", color=INK)
    ax.text(bx, 3.05, "which concepts, and in what proportion — "
            "not how strongly", fontsize=9.5, color=SUB)

    by, scale = 2.42, 4.6
    for name, w in (("A", 0.45), ("B", 0.20), ("C", 0.35)):
        ax.text(bx, by, name, fontsize=11, weight="bold", color=BLUE,
                va="center")
        ax.add_patch(Rectangle((bx + 0.45, by - 0.15), scale * w, 0.30,
                     facecolor=RED, edgecolor="none", zorder=3))
        ax.text(bx + 0.45 + scale * w + 0.16, by, f"{w:.2f}", fontsize=10,
                color=SUB, va="center")
        by -= 0.58
    ax.plot([bx + 0.45, bx + 0.45], [0.52, 2.62], color=MUTE, lw=1.0,
            zorder=2)
    ax.text(bx, 0.42, r"$\tilde h$ on the sphere ($\ell_2$) or simplex "
            r"($\ell_1$, softmax)   $\Rightarrow$   the proportions are all "
            r"that remain", fontsize=9.3, color=SUB, va="top")

    out = Path(__file__).resolve().parent.parent / "assets" / "spherical_sae.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=WHITE)
    print("wrote", out)


if __name__ == "__main__":
    main()