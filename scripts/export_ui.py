"""Export a self-contained black-and-white HTML UI to browse SAE features.

Builds a single static file (data embedded, no server needed):
  - Features tab: every alive feature, its firing density, and the abstracts
    that activate it most (search by feature id or title text).
  - Documents tab: sample abstracts decomposed as a mixture of concepts.

    python scripts/export_ui.py --mode l2
    # then open ui/index.html in a browser
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for `interpret`
sys.path.insert(0, str(ROOT))
from interpret import feature_top_docs, load_model
from spherical_sae.data import load_embeddings

UI_DIR = ROOT / "ui"


def clean(t: str) -> str:
    return " ".join(str(t).split())


def build_data(mode, tag, top_n, n_docs, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = load_embeddings(ROOT / "data" / "embeddings.npy").to(device)
    titles = [clean(t) for t in pd.read_parquet(ROOT / "data" / "papers.parquet")["title"].tolist()]
    model = load_model(mode, tag, x.shape[1], device)

    top_v, top_i, density, fired, mean_act = feature_top_docs(model, x, top_n=top_n)
    label = {f: titles[int(top_i[f, 0])] for f in range(model.n_latents)}

    alive = torch.where(fired > 0)[0]
    order = alive[torch.argsort(density[alive], descending=True)].tolist()
    features = []
    for f in order:
        tops = []
        for r in range(top_n):
            di = int(top_i[f, r])
            if di < 0:
                break
            tops.append([titles[di], round(float(top_v[f, r]), 3)])
        features.append({
            "id": int(f),
            "density": round(float(density[f]) * 100, 3),
            "firings": int(fired[f]),
            "mean_act": round(float(mean_act[f]), 3),
            "top": tops,
        })

    rng = np.random.default_rng(seed)
    doc_ids = [0] + sorted(set(int(i) for i in rng.integers(0, x.shape[0], size=n_docs)))
    docs = []
    with torch.no_grad():
        for d in doc_ids:
            h = model.encode(x[d:d + 1])[0]
            act = torch.where(h > 0)[0]
            vals = h[act]
            prop = (vals / vals.sum())
            srt = torch.argsort(prop, descending=True)
            mix = [[round(float(prop[j]) * 100, 2), int(act[j]), label[int(act[j])]]
                   for j in srt.tolist()]
            docs.append({"id": d, "title": titles[d], "n_active": int(len(act)), "mix": mix})

    return {
        "mode": mode, "n_features": model.n_latents, "n_alive": len(order),
        "n_docs": int(x.shape[0]), "k": model.k, "top_n": top_n,
        "features": features, "docs": docs,
    }


HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spherical SAE — feature browser</title>
<style>
  :root { --line:#000; }
  * { box-sizing: border-box; }
  body { margin:0; background:#fff; color:#000;
         font:14px/1.5 ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  header { border-bottom:2px solid var(--line); padding:14px 18px; }
  h1 { margin:0; font-size:16px; letter-spacing:.5px; text-transform:uppercase; }
  .meta { margin-top:4px; font-size:12px; }
  .tabs { display:flex; gap:0; border-bottom:2px solid var(--line); }
  .tab { padding:8px 16px; cursor:pointer; border-right:1px solid var(--line);
         user-select:none; text-transform:uppercase; font-size:12px; letter-spacing:.5px; }
  .tab.on { background:#000; color:#fff; }
  .wrap { padding:14px 18px; }
  input { width:100%; padding:8px 10px; border:1px solid var(--line); background:#fff;
          color:#000; font:inherit; margin-bottom:12px; }
  .count { font-size:12px; margin-bottom:10px; opacity:.7; }
  .feat { border:1px solid var(--line); margin-bottom:-1px; }
  .feat > .hd { padding:8px 10px; cursor:pointer; display:flex; gap:14px; align-items:baseline; }
  .feat > .hd:hover { background:#000; color:#fff; }
  .fid { font-weight:bold; min-width:90px; }
  .dens { min-width:120px; font-size:12px; }
  .prev { opacity:.75; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .body { display:none; padding:6px 10px 12px 10px; border-top:1px dashed var(--line); }
  .feat.open .body { display:block; }
  .row { display:flex; gap:10px; padding:2px 0; }
  .act { min-width:54px; text-align:right; opacity:.6; }
  .docpick { border:1px solid var(--line); padding:6px 10px; margin-bottom:-1px; cursor:pointer; }
  .docpick:hover { background:#000; color:#fff; }
  .docview { border:2px solid var(--line); padding:14px; margin-top:12px; }
  .bar { height:12px; background:#000; }
  .mixrow { display:flex; gap:10px; align-items:center; padding:3px 0; }
  .pct { min-width:56px; text-align:right; }
  .barcell { width:160px; border:1px solid var(--line); }
  .flabel { opacity:.8; }
  .hidden { display:none; }
  a { color:#000; }
</style></head><body>
<header>
  <h1>Spherical SAE · feature browser</h1>
  <div class="meta" id="meta"></div>
</header>
<div class="tabs">
  <div class="tab on" data-t="features">Features</div>
  <div class="tab" data-t="documents">Documents</div>
</div>

<div class="wrap" id="features-tab">
  <input id="q" placeholder="search by feature id or title text…">
  <div class="count" id="fcount"></div>
  <div id="flist"></div>
</div>

<div class="wrap hidden" id="documents-tab">
  <div class="count">Click a document to see its concept mixture.</div>
  <div id="doclist"></div>
  <div id="docview"></div>
</div>

<script>
const DATA = __DATA__;
const MAX = 250;
document.getElementById('meta').textContent =
  `latent_norm=${DATA.mode} · ${DATA.n_alive}/${DATA.n_features} alive features · ` +
  `k=${DATA.k} · ${DATA.n_docs.toLocaleString()} abstracts`;

// ---- tabs ----
document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('on'));
  t.classList.add('on');
  document.getElementById('features-tab').classList.toggle('hidden', t.dataset.t !== 'features');
  document.getElementById('documents-tab').classList.toggle('hidden', t.dataset.t !== 'documents');
});

// ---- features ----
const flist = document.getElementById('flist'), fcount = document.getElementById('fcount');
function renderFeatures(query) {
  query = (query || '').toLowerCase().trim();
  let items = DATA.features;
  if (query) {
    items = items.filter(f =>
      ('feature ' + f.id).includes(query) || String(f.id) === query ||
      f.top.some(t => t[0].toLowerCase().includes(query)));
  }
  fcount.textContent = `${items.length} feature(s)` + (items.length > MAX ? ` — showing first ${MAX}` : '');
  flist.innerHTML = '';
  items.slice(0, MAX).forEach(f => {
    const el = document.createElement('div'); el.className = 'feat';
    const top = f.top.map(t =>
      `<div class="row"><span class="act">${t[1].toFixed(3)}</span><span>${esc(t[0])}</span></div>`).join('');
    el.innerHTML =
      `<div class="hd"><span class="fid">feat ${f.id}</span>` +
      `<span class="dens">${f.density.toFixed(2)}% · ${f.firings}</span>` +
      `<span class="prev">${esc(f.top[0] ? f.top[0][0] : '')}</span></div>` +
      `<div class="body">${top}</div>`;
    el.querySelector('.hd').onclick = () => el.classList.toggle('open');
    flist.appendChild(el);
  });
}
document.getElementById('q').addEventListener('input', e => renderFeatures(e.target.value));
renderFeatures('');

// ---- documents ----
const doclist = document.getElementById('doclist'), docview = document.getElementById('docview');
DATA.docs.forEach(d => {
  const el = document.createElement('div'); el.className = 'docpick';
  el.innerHTML = `<b>doc ${d.id}</b> · ${esc(d.title)}`;
  el.onclick = () => showDoc(d);
  doclist.appendChild(el);
});
function showDoc(d) {
  const rows = d.mix.slice(0, 14).map(m =>
    `<div class="mixrow"><span class="pct">${m[0].toFixed(1)}%</span>` +
    `<span class="barcell"><div class="bar" style="width:${Math.min(100, m[0]*3)}%"></div></span>` +
    `<span>feat ${m[1]}</span><span class="flabel">· ${esc(m[2])}</span></div>`).join('');
  docview.innerHTML =
    `<div class="docview"><b>doc ${d.id}:</b> ${esc(d.title)}<br>` +
    `<span class="meta">${d.n_active} active concepts (top 14 shown)</span><div style="margin-top:8px">${rows}</div></div>`;
  docview.scrollIntoView({behavior:'smooth', block:'nearest'});
}
function esc(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="l2", choices=["none", "l2", "l1", "softmax"])
    ap.add_argument("--tag", type=str, default="full")
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--n-docs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = build_data(args.mode, args.tag, args.top_n, args.n_docs, args.seed)
    UI_DIR.mkdir(exist_ok=True)
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    out = UI_DIR / f"{args.mode}.html"
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"wrote {out} ({kb:.0f} KB) — {data['n_alive']} features, {len(data['docs'])} sample docs")
    print(f"open it with:  xdg-open {out}")


if __name__ == "__main__":
    main()
