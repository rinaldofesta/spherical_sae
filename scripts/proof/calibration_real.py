"""M2+M3 -- CALIBRATED proportional control on the real arXiv embeddings (PR #2).

The falsifiable claim (PIANO_CONTROLLO_CALIBRATO.md): on simplex codes (l1/softmax)
ONE shared monotone map g: p_requested -> knob, fitted once on CALIBRATION concepts
and applied UNCHANGED to HELD-OUT concepts, makes the realised mix in the top-K
track p (MACE <= 0.10 on the pre-registered domain p in [0.10, 0.35]) -- while the
same symmetric treatment cannot do it for the standard SAE steelman, nor for the
no-SAE interpolation control (which, if it matches, kills the SAE's raison d'etre).

Design (all pre-registered):
  - ONE neutral frozen panel: concepts = title keywords + their doc sets, defined
    outside any model (results/proof/concept_panel.json), density-stratified
    concept-level split ~10 calibration / ~5 held-out, IDENTICAL for every arm.
  - Identical query sets across arms (kw-based exclusion, seeded), so arms differ
    only in the operator, never in the data.
  - SYMMETRIC map fitting: every arm gets the same isotonic (PAV) shared map from
    pooled calibration measurements; nobody rides a free identity map.
  - Arms: l1, softmax (simplex); none (STEELMAN exact-share operator); l2 (sphere,
    isolates "normalisation in general"); interp (no-SAE kill control).
  - Metrics: MACE_kw primary (model-independent), MACE_fit vs MACE_holdout
    (cross-concept transfer), slope/R^2, Spearman, monotonicity violations,
    per-concept oracle maps (how much per-concept tuning would buy; for `none`
    this doubles as the plan's "per-concept-rescaled steelman", strictly
    stronger than a 95th-pct rescale), MACE_sae gated on kw/sae membership
    Jaccard >= 0.6, off-target leak (raw + geometric-subtracted residual at the
    retrieval level), 2-concept mix error, continuous centroid-cosine.
  - >= 5 seeds, 95% CIs, automatic check of the pre-registered GO / SOFT-GO /
    NO-GO conditions (plan section 8), incl. the retrieval guardrail read from
    results/proof/retrieval_real.json.

DEFERRED (explicitly out of scope here, per plan M6): the code-level
identifiability chain (atom-Gram off-diagonal -> Spearman leak coupling) on a
non-simplex synthetic generator. The leak reported here is the retrieval-level
analogue; its geometric component is subtracted in closed form (see
_calibmap.leak_metrics) but no GO may cite the M6 chain from this script.

    python scripts/proof/calibration_real.py                  # full (5 seeds)
    python scripts/proof/calibration_real.py --seeds 0        # 1-seed sanity
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
PROOF = Path(__file__).resolve().parent
for p in (str(ROOT), str(PROOF)):
    if p not in sys.path:
        sys.path.insert(0, p)

import _calib  # noqa: E402
import _calibmap  # noqa: E402
import _panel  # noqa: E402
from _common import ci95  # noqa: E402
from spherical_sae.data import load_embeddings  # noqa: E402
from steering_real import STOP, encode_all, pick_device, retrieve, train_one  # noqa: E402

SAE_ARMS = ["none", "l2", "l1", "softmax"]
ALL_ARMS = SAE_ARMS + ["interp"]
# knob operator used by each SAE arm (l2 has no native share operator: it gets the
# exact-implied-share rewrite and its own training-time L2 re-normalisation).
OPERATOR = {"none": "none", "l2": "none", "l1": "l1", "softmax": "softmax"}


# --------------------------------------------------------------------- panel
def build_or_load_panel(path, titles_low, title_toksets, n_panel, n_holdout,
                        lo, hi, Xn):
    """Frozen neutral panel: keyword concepts + density-stratified split."""
    if path.exists():
        panel = json.loads(path.read_text())
        print(f"panel: loaded frozen {path.name} "
              f"({len(panel['calibration'])} calib / {len(panel['holdout'])} holdout)")
        return panel
    n_docs = len(titles_low)
    df = _panel.token_doc_freq(titles_low, STOP)
    cands = _panel.band_tokens(df, n_docs, lo, hi)
    # doc sets via prefix match (so 'graph' covers 'graphs'), coherence-ranked
    docsets, coh = {}, {}
    rng = np.random.default_rng(0)
    for t in cands[:120]:
        ds = {i for i, ts in enumerate(title_toksets) if any(w.startswith(t) for w in ts)}
        if len(ds) < 30:
            continue
        docsets[t] = ds
        sample = rng.choice(sorted(ds), size=min(200, len(ds)), replace=False)
        E = Xn[sample]
        G = E @ E.t()
        m = G.shape[0]
        coh[t] = float((G.sum() - m) / (m * (m - 1)))
    ranked = sorted(docsets, key=lambda t: -coh[t])
    kept = _panel.dedup_by_jaccard(ranked, docsets, max_jaccard=0.5)[:n_panel]
    freqs = {t: len(docsets[t]) for t in kept}
    calib, hold = _panel.split_panel(kept, freqs, n_holdout=n_holdout, seed=0)
    panel = {
        "tokens": kept,
        "doc_freq": freqs,
        "coherence": {t: round(coh[t], 4) for t in kept},
        "calibration": calib,
        "holdout": hold,
        "note": "frozen BEFORE any calibration result; identical for every arm/seed",
    }
    path.write_text(json.dumps(panel, indent=2))
    print(f"panel: built and FROZEN -> {path.name}  calib={calib}  holdout={hold}")
    return panel


def kw_masks_and_centroids(panel, title_toksets, Xn):
    masks, cents = {}, {}
    for t in panel["tokens"]:
        m = torch.tensor([any(w.startswith(t) for w in ts) for ts in title_toksets])
        masks[t] = m
        cents[t] = F.normalize(Xn[m].mean(0), dim=-1)
    return masks, cents


# ----------------------------------------------------------------- measuring
class Arm:
    """Everything needed to turn the knob for one arm: query codes/embeddings,
    the knob operator, and the decode path back to embedding space."""

    def __init__(self, name, model=None, latmap=None, active=None):
        self.name = name
        self.model = model
        self.latmap = latmap or {}
        self.active = active

    def usable(self, token):
        return self.name == "interp" or token in self.latmap

    def query_vectors(self, X, qsel, token, knob, centroid, device):
        """Apply the arm's knob at strength `knob` and return query vectors."""
        if self.name == "interp":
            return _calibmap.interpolate_toward(X[qsel].to(device),
                                                centroid.to(device),
                                                0.0 if knob is None else knob)
        m = self.model
        hq = m.encode(X[qsel].to(device))
        if knob is None:                       # control: the model's own recon
            return m.decode(hq)
        s = _calib.set_share(hq, self.latmap[token], knob, OPERATOR[self.name])
        if self.name == "l2":                  # sphere arm re-normalises as trained
            return m.decode(s)
        return m.decode_from_shares(s)

    def mix_vectors(self, X, qsel, tok_a, tok_b, ka, kb, cents, device):
        if self.name == "interp":
            q = X[qsel].to(device)
            mixed = (1.0 - ka - kb) * q + ka * cents[tok_a].to(device) \
                + kb * cents[tok_b].to(device)
            return F.normalize(mixed, dim=-1)
        if self.name not in ("none", "l1"):    # set_mix exists for l1/none only
            return None
        m = self.model
        hq = m.encode(X[qsel].to(device))
        s = _calib.set_mix(hq, {self.latmap[tok_a]: ka, self.latmap[tok_b]: kb},
                           OPERATOR[self.name])
        return m.decode_from_shares(s)


def measure(arm, X, qsel, token, knob, kwmask, doc_cos, centroid, corpus_n,
            topk, device):
    """One point of the dose-response: realised kw / sae / continuous-cos share.

    Also returns the retrieved indices so the caller can score OFF-target
    panel shares on the same result set (leak metric).
    """
    qv = arm.query_vectors(X, qsel, token, knob, centroid, device)
    idx = retrieve(qv, corpus_n, topk, qsel.to(device)).cpu()
    out = {"kw": kwmask[idx].float().mean().item(),
           "cos": doc_cos[idx].mean().item()}
    if arm.active is not None and token in arm.latmap:
        out["sae"] = arm.active[:, arm.latmap[token]][idx].float().mean().item()
    return out, idx


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default=str(ROOT / "data" / "embeddings.npy"))
    ap.add_argument("--papers", type=str, default=str(ROOT / "data" / "papers.parquet"))
    ap.add_argument("--arms", type=str, nargs="+", default=ALL_ARMS)
    ap.add_argument("--subsample", type=int, default=40000)
    ap.add_argument("--n-latents", type=int, default=2048)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-panel", type=int, default=15)
    ap.add_argument("--n-holdout", type=int, default=5)
    ap.add_argument("--queries-per-concept", type=int, default=40)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--fine-grid", type=float, nargs="+",
                    default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18,
                             0.21, 0.25, 0.30, 0.40, 0.50])
    # interpolation may need a stronger knob to reach the same realised share
    ap.add_argument("--interp-extra", type=float, nargs="+", default=[0.65, 0.80, 0.90])
    ap.add_argument("--eval-grid", type=float, nargs="+",
                    default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    ap.add_argument("--mix-targets", type=float, nargs=2, default=[0.30, 0.15])
    ap.add_argument("--dens-lo", type=float, default=0.004)
    ap.add_argument("--dens-hi", type=float, default=0.08)
    ap.add_argument("--f1-floor", type=float, default=0.15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--panel-json", type=str,
                    default=str(ROOT / "results" / "proof" / "concept_panel.json"))
    args = ap.parse_args()

    if not Path(args.embeddings).exists():
        print(f"!! {args.embeddings} not found. Run scripts/build_dataset.py first.")
        return
    device = pick_device(args.device)

    # ---- fixed corpus (generator seed 0) so the panel is frozen once ----
    X = load_embeddings(args.embeddings)
    titles_all = pd.read_parquet(args.papers)["title"].astype(str).tolist()
    g = torch.Generator().manual_seed(0)
    if X.shape[0] > args.subsample:
        keep = torch.randperm(X.shape[0], generator=g)[:args.subsample]
        X = X[keep]
        titles = [titles_all[i] for i in keep.tolist()]
    else:
        titles = titles_all[:X.shape[0]]
    titles_low = [t.lower() for t in titles]
    title_toksets = [set(_panel.TOKEN_RE.findall(t)) for t in titles_low]
    N = X.shape[0]
    Xn = F.normalize(X, dim=-1)
    corpus_n = Xn.to(device)
    print(f"== Calibrated control (M2+M3) ==  device={device}  N={N}  "
          f"arms={args.arms}  seeds={args.seeds}\n", flush=True)

    panel = build_or_load_panel(Path(args.panel_json), titles_low, title_toksets,
                                args.n_panel, args.n_holdout,
                                args.dens_lo, args.dens_hi, Xn)
    kwmasks, centroids = kw_masks_and_centroids(panel, title_toksets, Xn)
    doc_cos = {t: Xn @ centroids[t] for t in panel["tokens"]}
    calib_tokens, hold_tokens = panel["calibration"], panel["holdout"]

    report = {"config": vars(args), "panel": panel, "seeds": {}}
    dest = ROOT / "results" / "proof" / "calibration_real.json"
    t0 = time.time()

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        # identical query sets for every arm (kw-based exclusion, model-free)
        qsel = {}
        for t in panel["tokens"]:
            pool = (~kwmasks[t]).nonzero().squeeze(1).numpy()
            qsel[t] = torch.from_numpy(
                rng.choice(pool, size=min(args.queries_per_concept, len(pool)),
                           replace=False))

        # ---- build arms: train SAEs, assign panel concepts to latents ----
        # ONE-TO-ONE assignment (Hungarian): independent argmax can hand the
        # same latent to two concepts, silently collapsing set_mix's targets
        # and letting a "held-out" knob reuse a calibration axis.
        arms, drops, jac = {}, {}, {}
        for name in args.arms:
            if name == "interp":
                arms[name] = Arm("interp")
                continue
            model = train_one(X.to(device), args.n_latents, args.k, name,
                              args.steps, device, seed=seed)
            model.eval()
            with torch.no_grad():
                h = encode_all(model, X, device)
            active = h > 0
            del h
            assigned = _panel.assign_latents(
                active, {t: kwmasks[t] for t in panel["tokens"]},
                floor=args.f1_floor)
            latmap = {t: idx for t, (idx, _) in assigned.items()}
            f1s = {t: round(f1, 3) for t, (_, f1) in assigned.items()}
            jac[name] = {t: round(_panel.jaccard(kwmasks[t], active[:, latmap[t]]), 3)
                         for t in latmap}
            drops[name] = [t for t in panel["tokens"] if t not in latmap]
            arms[name] = Arm(name, model, latmap, active)
            print(f"[seed {seed}][{name}] trained; latent match F1={f1s}; "
                  f"dropped={drops[name]}  ({time.time()-t0:.0f}s)", flush=True)
            if device == "mps":
                torch.mps.empty_cache()

        # concepts usable by EVERY arm this seed (panel identical across arms)
        usable = [t for t in panel["tokens"]
                  if all(arms[a].usable(t) for a in args.arms)]
        u_calib = [t for t in calib_tokens if t in usable]
        u_hold = [t for t in hold_tokens if t in usable]
        print(f"[seed {seed}] usable: {len(u_calib)} calib / {len(u_hold)} holdout",
              flush=True)
        if len(u_calib) < 3 or len(u_hold) < 2:
            print(f"[seed {seed}] !! too few usable concepts, seed SKIPPED")
            report["seeds"][str(seed)] = {"skipped": True, "dropped": drops,
                                          "usable_calib": u_calib,
                                          "usable_holdout": u_hold}
            continue

        seed_rep = {"dropped": drops, "usable_calib": u_calib,
                    "usable_holdout": u_hold, "arms": {}}

        # pre-draw mix pairs + queries ONCE per seed, so every arm sees the
        # exact same inputs (drawing inside the arm loop would desync the rng)
        mix_pairs = list(itertools.combinations(u_hold, 2))[:6]
        mix_qsel = {}
        for ta, tb in mix_pairs:
            excl = (~kwmasks[ta] & ~kwmasks[tb]).nonzero().squeeze(1).numpy()
            mix_qsel[(ta, tb)] = torch.from_numpy(
                rng.choice(excl, size=min(args.queries_per_concept, len(excl)),
                           replace=False))

        with torch.no_grad():
            for name in args.arms:
                arm = arms[name]
                grid = list(args.fine_grid) + (list(args.interp_extra)
                                               if name == "interp" else [])

                # -- fine dose-response sweep on ALL usable concepts --
                fine = {t: {} for t in u_calib + u_hold}
                control, ctrl_idx = {}, {}
                for t in u_calib + u_hold:
                    control[t], ctrl_idx[t] = measure(
                        arm, X, qsel[t], t, None, kwmasks[t], doc_cos[t],
                        centroids[t], corpus_n, args.topk, device)
                    for q in grid:
                        fine[t][q], _ = measure(
                            arm, X, qsel[t], t, q, kwmasks[t], doc_cos[t],
                            centroids[t], corpus_n, args.topk, device)

                # -- ONE shared map from pooled CALIBRATION measurements --
                kn = np.array([q for t in u_calib for q in grid])
                rl = np.array([fine[t][q]["kw"] for t in u_calib for q in grid])
                curve = _calibmap.fit_shared_map(kn, rl)
                gmap = {p: _calibmap.map_command(curve, p) for p in args.eval_grid}

                # -- apply the SAME map to holdout (transfer) and calib (fit);
                #    on holdout also score OFF-target panel shares (leak) --
                def eval_concepts(tokens, collect_leak=False):
                    per, leaks = {}, []
                    for t in tokens:
                        others = [j for j in u_calib + u_hold if j != t]
                        off_ctrl = np.array(
                            [kwmasks[j][ctrl_idx[t]].float().mean().item()
                             for j in others])
                        per[t] = {}
                        for p in args.eval_grid:
                            out, idx = measure(
                                arm, X, qsel[t], t, gmap[p], kwmasks[t],
                                doc_cos[t], centroids[t], corpus_n,
                                args.topk, device)
                            per[t][p] = out
                            if collect_leak:
                                off_after = np.array(
                                    [kwmasks[j][idx].float().mean().item()
                                     for j in others])
                                raw, resid = _calibmap.leak_metrics(
                                    control[t]["kw"], out["kw"],
                                    off_ctrl, off_after)
                                if raw is not None:
                                    leaks.append((raw, resid))
                    return per, leaks
                hold_real, leaks = eval_concepts(u_hold, collect_leak=True)
                fit_real, _ = eval_concepts(u_calib)

                # -- per-concept ORACLE maps on holdout (tuning headroom; for
                #    `none` this is the plan's per-concept-rescaled steelman) --
                oracle_err = []
                for t in u_hold:
                    oc = _calibmap.fit_shared_map(
                        np.array(grid), np.array([fine[t][q]["kw"] for q in grid]))
                    for p in args.eval_grid:
                        r, _ = measure(arm, X, qsel[t], t,
                                       _calibmap.map_command(oc, p), kwmasks[t],
                                       doc_cos[t], centroids[t], corpus_n,
                                       args.topk, device)
                        oracle_err.append(abs(r["kw"] - p))

                # -- metrics --
                ps = np.array(args.eval_grid)
                def arm_metrics(per):
                    rk = np.array([[per[t][p]["kw"] for p in args.eval_grid]
                                   for t in per])           # (C, P)
                    m = _calibmap.mace(rk, np.tile(ps, (rk.shape[0], 1)))
                    slopes, r2s, monos, rhos = [], [], [], []
                    for row in rk:
                        s, _, r2 = _calibmap.ols_slope_r2(ps, row)
                        slopes.append(s); r2s.append(r2)
                        monos.append(_calibmap.monotonicity_violation_rate(row))
                        rho = spearmanr(ps, row).statistic
                        rhos.append(0.0 if np.isnan(rho) else float(rho))
                    return m, slopes, r2s, monos, rhos
                mace_h, slopes, r2s, monos, rhos = arm_metrics(hold_real)
                mace_f = arm_metrics(fit_real)[0]

                # -- MACE_sae cross-check, gated on kw/sae membership Jaccard --
                mace_sae = gate_rate = None
                if name != "interp":
                    gated = [t for t in u_hold if jac[name].get(t, 0.0) >= 0.6]
                    gate_rate = round(len(gated) / len(u_hold), 3)
                    errs = [abs(hold_real[t][p]["sae"] - p)
                            for t in gated for p in args.eval_grid
                            if "sae" in hold_real[t][p]]
                    mace_sae = round(float(np.mean(errs)), 4) if errs else None
                leak_raw = (round(float(np.mean([l[0] for l in leaks])), 4)
                            if leaks else None)
                leak_resid = (round(float(np.mean([l[1] for l in leaks])), 4)
                              if leaks else None)

                # -- 2-concept mix on holdout pairs (l1 / none / interp) --
                mix = None
                pa, pb = args.mix_targets
                if mix_pairs and (name in ("none", "l1", "interp")):
                    errs = []
                    for ta, tb in mix_pairs:
                        ka = _calibmap.map_command(curve, pa)
                        kb = _calibmap.map_command(curve, pb)
                        if ka + kb >= 0.95:
                            sc = 0.95 / (ka + kb)
                            ka, kb = ka * sc, kb * sc
                        qs = mix_qsel[(ta, tb)]
                        qv = arm.mix_vectors(X, qs, ta, tb, ka, kb, centroids, device)
                        if qv is None:
                            continue
                        idx = retrieve(qv, corpus_n, args.topk, qs.to(device)).cpu()
                        ra = kwmasks[ta][idx].float().mean().item()
                        rb = kwmasks[tb][idx].float().mean().item()
                        errs.append(abs(ra - pa) + abs(rb - pb))
                    mix = round(float(np.mean(errs)), 4) if errs else None

                seed_rep["arms"][name] = {
                    "control_kw": round(float(np.mean(
                        [control[t]["kw"] for t in u_hold])), 4),
                    "shared_map": {str(p): round(gmap[p], 4) for p in args.eval_grid},
                    "latent_jaccard": jac.get(name),
                    "fine": {t: {str(q): fine[t][q] for q in grid}
                             for t in fine},
                    "holdout_realized": {t: {str(p): hold_real[t][p]
                                             for p in args.eval_grid}
                                         for t in hold_real},
                    "mace_holdout": round(mace_h, 4),
                    "mace_fit": round(mace_f, 4),
                    "transfer": round(mace_h - mace_f, 4),
                    "mace_oracle": round(float(np.mean(oracle_err)), 4),
                    "mace_sae": mace_sae,
                    "jaccard_gate_rate": gate_rate,
                    "slope_mean": round(float(np.mean(slopes)), 4),
                    "slope_std": round(float(np.std(slopes)), 4),
                    "r2_mean": round(float(np.mean(r2s)), 4),
                    "spearman_mean": round(float(np.mean(rhos)), 4),
                    "mono_viol": round(float(np.mean(monos)), 4),
                    "leak_raw": leak_raw,
                    "leak_resid": leak_resid,
                    "mix_l1_error": mix,
                }
                print(f"[seed {seed}][{name}] MACE hold={mace_h:.3f} "
                      f"fit={mace_f:.3f} slope={np.mean(slopes):.2f} "
                      f"R2={np.mean(r2s):.2f} leak={leak_resid} mix={mix}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

        report["seeds"][str(seed)] = seed_rep
        dest.write_text(json.dumps(report, indent=2))   # incremental checkpoint
        for a in arms.values():
            a.model = a.active = None
        if device == "mps":
            torch.mps.empty_cache()

    # ------------------------------------------------ aggregate + verdict
    metrics = ["mace_holdout", "mace_fit", "transfer", "mace_oracle", "mace_sae",
               "jaccard_gate_rate", "slope_mean", "slope_std", "r2_mean",
               "spearman_mean", "mono_viol", "leak_raw", "leak_resid",
               "mix_l1_error"]
    agg = {}
    for name in args.arms:
        agg[name] = {}
        for met in metrics:
            vals = []
            for s in args.seeds:
                v = report["seeds"][str(s)].get("arms", {}).get(name, {}).get(met)
                if v is not None:
                    vals.append(v)
            if vals:
                m, h = ci95(vals)
                agg[name][met] = {"mean": round(m, 4), "ci95": round(h, 4)}
    report["aggregate"] = agg

    def no_overlap(a, b, met_a="mace_holdout", met_b="mace_holdout"):
        A, B = agg[a][met_a], agg[b][met_b]
        return abs(A["mean"] - B["mean"]) > A["ci95"] + B["ci95"]

    # retrieval guardrail (plan section 8, condition 6): knob-OFF reconstruction
    # must not destroy retrieval -- read from the dedicated experiment's output.
    guard = None
    gpath = ROOT / "results" / "proof" / "retrieval_real.json"
    if gpath.exists():
        gr = json.loads(gpath.read_text())["results"]
        base = gr["recon_none"]["recall@10_of_raw"]
        guard = all(gr[f"recon_{m}"]["recall@10_of_raw"] >= base - 0.05
                    for m in ("l1", "softmax", "l2") if f"recon_{m}" in gr)

    simplex = min((a for a in ("l1", "softmax") if a in agg),
                  key=lambda a: agg[a]["mace_holdout"]["mean"], default=None)
    verdict = {}
    if simplex:
        sm = agg[simplex]
        others = [a for a in args.arms if a in agg and a != simplex]
        verdict = {
            "best_simplex": simplex,
            # CI-aware: the whole 95% interval must sit at or below the bar
            "C1_simplex_mace_ci<=0.10":
                sm["mace_holdout"]["mean"] + sm["mace_holdout"]["ci95"] <= 0.10,
            "C2_beats_none_steelman_no_CI_overlap":
                "none" in agg and sm["mace_holdout"]["mean"]
                < agg["none"]["mace_holdout"]["mean"]
                and no_overlap(simplex, "none"),
            # plan: the separation must hold even vs none WITH per-concept
            # tuning (our per-concept oracle maps, stronger than 95th-pct rescale)
            "C2b_beats_none_perconcept_tuned_no_CI_overlap":
                "none" in agg and sm["mace_holdout"]["mean"]
                < agg["none"]["mace_oracle"]["mean"]
                and no_overlap(simplex, "none", "mace_holdout", "mace_oracle"),
            "C3_slope_in_[0.8,1.2]": 0.8 <= sm["slope_mean"]["mean"] <= 1.2,
            "C3_r2>=0.85": sm["r2_mean"]["mean"] >= 0.85,
            "C3_spearman>=0.85": sm["spearman_mean"]["mean"] >= 0.85,
            "C3_mono_viol<=0.05": sm["mono_viol"]["mean"] <= 0.05,
            "C4_transfer<=0.05": sm["transfer"]["mean"] <= 0.05,
            "C4b_slope_spread_strictly_smallest": all(
                sm["slope_std"]["mean"] < agg[b]["slope_std"]["mean"]
                for b in others if "slope_std" in agg[b]),
            "C5_beats_interp_no_CI_overlap":
                "interp" in agg and sm["mace_holdout"]["mean"]
                < agg["interp"]["mace_holdout"]["mean"]
                and no_overlap(simplex, "interp"),
            "C6_retrieval_guardrail": guard,
            "C7_leak_residual<=0.3":
                sm["leak_resid"]["mean"] <= 0.3 if "leak_resid" in sm else None,
        }
        core = [v for k, v in verdict.items() if k.startswith("C")]
        verdict["GO"] = all(bool(v) for v in core)
        # SOFT-GO (plan section 8): zero-shot simplex wins the symmetric
        # shared-map comparison AND the kill control, but per-concept-tuned
        # none matches it -> restricted claim "calibrated zero-shot without
        # per-concept tuning"; the multi-concept mix becomes the deliverable.
        verdict["SOFT_GO"] = (
            not verdict["GO"]
            and bool(verdict["C2_beats_none_steelman_no_CI_overlap"])
            and bool(verdict["C5_beats_interp_no_CI_overlap"])
            and bool(verdict["C4_transfer<=0.05"])
            and not verdict["C2b_beats_none_perconcept_tuned_no_CI_overlap"])
        verdict["NO_GO_mace>0.15"] = sm["mace_holdout"]["mean"] > 0.15
        verdict["NO_GO_leak_residual>0.3"] = (
            sm["leak_resid"]["mean"] > 0.3 if "leak_resid" in sm else None)
        # binary switch: even per-concept tuning cannot track p for ANY arm
        verdict["NO_GO_binary_switch_all_arms"] = all(
            agg[a]["mace_oracle"]["mean"] > 0.15
            for a in agg if "mace_oracle" in agg[a])
    report["verdict"] = verdict

    # ---------------------------------------------------------- ASCII report
    print("\n" + "=" * 100)
    print(f"{'arm':>8} | {'MACE hold':>12} | {'MACE fit':>9} | {'transfer':>9} | "
          f"{'oracle':>7} | {'slope':>11} | {'R2':>5} | {'mono':>5} | {'mix':>6}")
    print("-" * 100)
    for name in args.arms:
        a = agg[name]
        mh = a["mace_holdout"]
        mixs = a.get("mix_l1_error")
        print(f"{name:>8} | {mh['mean']:.3f}±{mh['ci95']:.3f} | "
              f"{a['mace_fit']['mean']:>9.3f} | {a['transfer']['mean']:>9.3f} | "
              f"{a['mace_oracle']['mean']:>7.3f} | "
              f"{a['slope_mean']['mean']:.2f}±{a['slope_std']['mean']:.2f} | "
              f"{a['r2_mean']['mean']:>5.2f} | {a['mono_viol']['mean']:>5.2f} | "
              f"{(f'{mixs['mean']:.3f}' if mixs else '--'):>6}")
    print("-" * 100)
    for name in args.arms:
        a = agg[name]
        cells = []
        for met, lab in (("spearman_mean", "rho"), ("leak_raw", "leak"),
                         ("leak_resid", "leak-resid"), ("mace_sae", "MACEsae"),
                         ("jaccard_gate_rate", "jacc-gate")):
            if met in a:
                cells.append(f"{lab}={a[met]['mean']:.3f}")
        print(f"{name:>8} | " + "  ".join(cells))
    print("=" * 100)
    for k, v in verdict.items():
        print(f"  {k}: {v}")

    dest.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {dest}")


if __name__ == "__main__":
    main()
