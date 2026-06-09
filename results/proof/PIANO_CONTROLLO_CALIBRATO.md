# Piano operativo — Controllo calibrato/proporzionale del retrieval

> Prodotto da un workflow multi-agente (4 bozze indipendenti → 3 critici avversariali → sintesi),
> con verifica diretta sul codice del repo. Le correzioni in §2/§3/§10 nascono da fatti misurati,
> non da assunzioni. Questo è il piano canonico, decision-ready.

## 0. Tesi in una riga

Su un codice SAE normalizzato sul **simplesso** (l1/softmax) ogni concetto è una **proporzione vincolata che somma a 1**, quindi "imposta il concetto C al p%" è un comando *bounded* e confrontabile tra concetti; testiamo se questo produce una **singola mappa comando→mix-recuperato condivisa tra concetti** che lo SAE standard a magnitudo non può eguagliare — e lo testiamo nella sola regione dove la manopola si muove davvero (saturazione a parte), con baseline steelman e un controllo senza-SAE che può uccidere la claim.

## 1. La claim falsificabile (e il null onesto)

**CLAIM (da disprovare).** Per i modi simplesso (l1/softmax) esiste **una sola** mappa monotona `g: p_richiesto → manopola`, **stimata una volta su un set di concetti di calibrazione e applicata invariata a concetti held-out**, tale che la **frazione di mix realizzata** nel top-K traccia `p` con:
- MACE (errore assoluto medio di calibrazione) **≤ 0.10** sul dominio utile pre-registrato `p ∈ [0.10, 0.35]`,
- pendenza ∈ [0.8, 1.2], R² ≥ 0.85, tasso di violazioni di monotonicità ≤ 0.05,
- **errore di transfer cross-concetto** (MACE held-out − MACE fit) ≤ 0.05 e **spread per-concetto** (std della pendenza) strettamente minore dei baseline;

mentre **nessuna** mappa condivisa porta lo SAE standard `none` — *anche dopo* steelman (operatore per-query a share-esatta + riscalatura per-concetto al 95° percentile) — sotto la stessa soglia con **CI 95% non sovrapposti** su ≥5 seed.

**Cosa la DISPROVA (uno qualsiasi basta):**
- (a) `none` steelman eguaglia il simplesso (CI sovrapposti su MACE keyword), oppure
- (b) il simplesso non raggiunge MACE ≤ 0.10 nemmeno su `[0.10, 0.35]`, oppure
- (c) il **controllo senza-SAE** (interpolazione `q' = normalize((1-p)·q + p·centroid_C)`) calibra altrettanto bene → lo SAE è superfluo, oppure
- (d) il transfer cross-concetto del simplesso non batte `none`.

**Il NULL ONESTO (pre-registrato come esito pubblicabile).** Dal pilota già misurato la dose-risposta è un **sigmoide saturante**: per l1, density ~0.02, `p=0.10→0.08`, `0.15→0.30`, `0.20→0.64`, `0.30→0.99`, `≥0.40→1.00`; softmax satura ancora prima. Se nemmeno su `[0.10,0.35]` si traccia `p`, l'esito onesto è: *"la manopola è un interruttore binario, non un quadrante; il controllo proporzionale non è ottenibile a questa scala di dizionario con nessun modo"* — e questo **ritira la scommessa strategica**, in modo pulito.

## 2. Fondamento teorico: disentanglement → calibrabilità

**Catena logica.** La calibrazione è a valle dell'identificabilità: una manopola sull'asse C ha effetto isolato e prevedibile **solo se** l'asse appreso È l'asse vero (altrimenti la manopola "sanguina" sui concetti vicini e il mix realizzato diventa funzione rumorosa e concetto-dipendente del comando).

**Correzione critica (premessa non gonfiata).** **NON** citiamo più MMCS-recall (l2=0.973 vs none=0.953) come prova di identificabilità: il file `identifiability.json` mostra che il conteggio **Hungarian `matched@0.9` PAREGGIA** (none ~246, l2 ~247, l1 ~246, softmax ~245 su 256, CI sovrapposti). Il gap MMCS è solo *vicinanza media soft-cosine* di ~0.02, non "gli assi sono gli assi veri uno-a-uno". La premessa di identificabilità va **ri-derivata** dalla **massa off-diagonale della Gram degli atomi** `G_c = max_{j≠c} |⟨atom̂_c, atom_j_vero⟩|` (ottenibile dalla matrice `C` già calcolata in `recovery_metrics`): bisogna **provarla prima** su sintetico — se anch'essa pareggia, l'esito predetto è *nessun* vantaggio di calibrazione, ed è pre-registrato.

**Correzione critica (l'intervento NON è isolato — lo è per geometria, non per entanglement).** Il decode è affine nel codice normalizzato (`x_hat = s @ W_dec.T + b_dec`, verificato). L'operatore set-share del simplesso (azzera C, rinormalizza gli altri ~k coordinate attive a `1−p`, riscrivi `p`) **modifica tutte le ~32 coordinate attive**, non solo C: parte della massa dello shift di reconstruction cade su atomi ≠ C **anche a identificabilità perfetta**. Quindi "muove solo l'asse C" è **falso per geometria**. Conseguenza operativa:
- modelliamo il leak come somma di una **componente geometrica** in forma chiusa (leak su `j ∝ s_j · p/(1−p_old)`, dalla rinormalizzazione) **+** una componente di **entanglement** (off-diagonale Gram);
- **sottraiamo la componente geometrica** prima di attribuire qualunque residuo all'identificabilità. Il test Spearman leak↔Gram è la verifica del solo residuo.

**Confrontabilità: separare definitorio da sostanziale.** Che il codice l1 abbia 0.3 su C *per costruzione*, identico per ogni concetto, è **definitorio e gratuito (non è il contributo)**. Il contributo testabile è la **confrontabilità in OUTPUT**: una mappa unica comando→mix-recuperato che generalizza tra concetti. È questo che misuriamo.

## 3. Il motore: obiettivo vMF (definizione + come si implementa)

**Definizione.** Modelliamo la ricostruzione unit-norm come osservazione von Mises–Fisher con direzione media `μ = x_hat/‖x_hat‖` e concentrazione `κ`:
`log p_vMF(x|μ,κ) = log C_d(κ) + κ·⟨x,μ⟩`, con `log C_d(κ) = (d/2−1)log κ − (d/2)log(2π) − log I_{d/2−1}(κ)`.
La loss è la NLL: `L = −(κ·cos(x,x̂) + log C_d(κ))`. Identità chiave (verificata in `train.py`): la loss corrente `obj = (1−cos).mean()` **è esattamente vMF NLL a κ fisso col normalizzatore eliminato**. vMF è quindi una generalizzazione stretta: l'onere è mostrare che il normalizzatore/concentrazione compra qualcosa di **misurabile**.

**Correzione critica (la griglia κ proposta era numericamente morta a d=384).** A `d=384` → `ν = d/2−1 = 191`. Le griglie `{2,5,10,20,40,80}` dei draft sono nel regime **quasi-uniforme** (mean cosine 0.03–0.20) mentre il modello opera a `cos ~0.85` (sintetico) / `~0.85–0.95` (reale), che richiede `κ ~ 1000–4000`; peggio, `log I_191(κ)` **underflow a −inf per κ≤2**. Correzioni:
- griglia ri-centrata sul punto operativo reale: **`κ ∈ {500, 1000, 2000, 4000, 8000}`** (sintetico più piccolo: `{50,100,200,400,800}` se `d_in=64`);
- `κ` come scalare positivo via softplus **+ offset ≥ ν**; `log I_ν(κ)` calcolato su **CPU** con `scipy.special.ive` (scalato: `log I_ν(κ)=log ive(ν,κ)+κ`), costo trascurabile (~0.002 s/4000 step, verificato);
- **unit test obbligatorio**: `log C_d` finito su ogni punto di griglia a d=384; e `vmf == cos·κ + cost` a κ fisso.

**Scopo e perimetro.** vMF è un **singolo braccio di robustness ablation DENTRO la famiglia simplesso** (mai un fattore che distingue simplesso da `none`, per non dargli capacità extra). κ fisso = loss coseno attuale (nota-buona) resta il braccio di default. Se calibrazione/leak sono **piatti** sulla griglia corretta (probabile a d alto), **vMF si droppa dall'headline come null** — non si tiene viva una claim infalsificabile.

**Connessione al simplesso: analogia, non teorema.** Temperatura softmax `τ` e concentrazione `κ` concentrano **spazi diversi** (τ sul simplesso-codice in `n_latents`; κ sulla sfera-output in `d`): **NON** `τ = 1/κ`. Lo dichiariamo analogia e testiamo empiricamente se sweep di `τ ∈ {0.25,0.5,1,2}` e di `κ` hanno effetti correlati sul leak.

## 4. Il meccanismo di controllo (come si gira la manopola all'inferenza)

Si interviene sul **codice sparso**, si **ri-decodifica**, si **ri-recupera**; mai sul testo. Nessun retraining per richiesta.

1. Encode query → `h = model.encode(x)` (top-k, pre-normalizzazione).
2. **Simplesso (l1/softmax):** per target `p` su concetto C: l1 → `set_share` (azzera C, rinormalizza il resto a `1−p`, scrivi `p`); softmax → shift di logit in forma chiusa `δ = logit(p) − logit_corrente_C` data la somma degli altri logit, poi re-softmax. Verificato live: hit esatto del code-share (`p=0.1→0.100`, `0.5→0.500`, `0.9→0.900`).
3. Decode esplicito dal codice già sul simplesso: nuovo helper **`model.decode_from_shares(s) = s @ W_dec.T + b_dec`** (bypassa una seconda normalizzazione).
4. Re-rank del corpus per coseno (riuso di `retrieve()`).
5. **`none` (baseline) — STEELMAN, non strawman.** Si dà a `none` l'**identico operatore per-query a share-esatta** (risolvi la magnitudo affinché l'implied-L1-share `= p` per ogni query, in forma chiusa `h[c] = p/(1−p)·somma_altri`, verificato hit esatto). La vittoria del simplesso va mostrata al **livello di mix-recuperato**, dopo che `none` ha già parità di share a livello di codice. Si dà inoltre a `none` la sua mappa isotona comando→mix e la riscalatura per-concetto al 95° percentile.

**Simmetria delle mappe (correzione fairness).** Qualunque fitting di mappa si applica **simmetricamente**: o ogni modo (simplesso incluso) riceve una mappa isotona su split di concetti held-out, o nessuno. Vietato far usare al simplesso una mappa-identità gratis mentre `none` passa per una mappa fittata.

## 5. Il benchmark: dati, backbone, task

- **Corpus:** `data/embeddings.npy` (117.592 × 384, embedding all-MiniLM-L6-v2 già salvati) + `data/papers.parquet`. Nessun encoder a inference, nessun download.
- **Config compute (verificata MPS):** `n_latents=2048, k=32, steps=4000, subsample=40000`. ~6.4 s/train misurati; ~30 train + sweep di decode (matmul) → **ore, non giorni**. **Cap a 2048** (OOM noto a 4096); fallback subsample 20k se OOM.
- **Pannello concetti (congelato).** **UN solo** pannello, density-stratificato (raro+denso) nella banda `[0.004, 0.08]`, selezionato su riferimento neutro/intersezione tra modi e **identico per tutti i modi** (elimina il confound `select_concepts` per-modello). Split **a livello di concetto**: ~10 calibrazione / ~5 held-out. Congelato a JSON prima di vedere i risultati.
- **Sintetico = solo mechanism check, NON headline** (il generatore disegna pesi sul simplesso `w/=w.sum()`, verificato → pre-favorisce l1/softmax). Se usato, aggiungere un **generatore NON-simplesso** (pesi gaussiani/sparse-magnitude o one-hot+rumore) per la catena identificabilità→leak.
- **Task:** per ogni query held-out, comanda una frazione `p` per un concetto held-out, ri-decodifica, ri-recupera top-K, misura il mix realizzato.

## 6. Metriche (la calibration error come primaria)

| Metrica | Definizione | Perché |
|---|---|---|
| **MACE_kw (PRIMARIA, non circolare)** | media su (query×concetto held-out) di \|p − share_realizzata\|, share = frazione del top-K che esprime C via **keyword/centroid sul titolo** (indipendente dallo SAE), su griglia `p ∈ [0.10,0.35]` | È la claim letterale, misurata fuori dal modello che fa l'intervento |
| **MACE_sae (secondaria)** | idem ma share = `active[:,C]` (SAE fires) | Sanity model-grounded; circolare → solo cross-check. Richiesto Jaccard ≥ 0.6 tra le due membership prima di fidarsi |
| **Share realizzata CONTINUA** | attivazione media per-doc di C (o centroid-cosine) sul result set, non hit-count | L'hit-count a K piccolo **quantizza E satura**; il segnale continuo distingue "lineare-recalibrabile" da "interruttore binario" — la distinzione su cui ruota tutto |
| **Pendenza & R²** | OLS share~p, per-concetto poi aggregata | Separa bias-ma-lineare (recalibrabile) da non-lineare/saturante |
| **Violazioni di monotonicità** | frazione di passi `p` adiacenti con share decrescente | Una manopola usabile deve essere monotona |
| **Leak off-target (geom-sottratto)** | Σ_{j≠C}\|Δshare_j\| / Δshare_C, **meno** la componente geometrica in forma chiusa | Faccia operativa dell'entanglement; il residuo testa l'identificabilità |
| **Transfer cross-concetto** | MACE held-out − MACE fit; e std-per-concetto della pendenza | IL numero che prova/uccide "confrontabile tra concetti" |
| **Mix multi-concetto** | target `{A:0.4, B:0.2}` → errore L1 del vettore realizzato | Capacità genuinamente nuova del budget unitario condiviso |
| **Guardrail retrieval** | Recall@10 / rank-fidelity a manopola OFF (`retrieval_real.py`) | La calibrazione non deve distruggere il retrieval (deve restare nella banda TIE ~0.53) |
| **κ→leak coupling (sintetico)** | Spearman tra leak residuo e `G_c` Gram off-diag | Test diretto del meccanismo identificabilità→calibrazione |

Nota: K=20–50 (non 10) per granularità di share più fine del bar di pass/fail; entrambe le metriche (kw e sae) saturano → il segnale **continuo** è la primaria di risoluzione.

## 7. Baseline da battere (forti e oneste)

| Baseline | Come | Perché è giusta |
|---|---|---|
| **`none` steelman (per-query share-esatta)** | operatore in forma chiusa `h[c]=p/(1−p)·Σaltri` per ogni query, share-parità a livello di codice; **+** mappa isotona comando→mix | Confronto apples-to-apples al **livello di mix-recuperato**; se la vittoria evapora qui, la claim è morta |
| **`none` + riscalatura per-concetto 95° pct** | magnitudo riscalata per la scala di attivazione di ogni concetto | **First-class**, non oracolo: se pareggia il simplesso, il contributo si restringe a "calibrazione zero-shot senza tuning per-concetto" |
| **Interpolazione embedding SENZA SAE** | `q' = normalize((1-p)·q + p·centroid_C)`, retrieve | **Obbligatoria, può uccidere la claim**: se l'interpolazione naive calibra altrettanto, lo SAE è superfluo |
| **l2 (sfera, non simplesso)** | imposta C poi re-L2-normalizza | Isola "normalizzazione in generale" da "simplesso (budget vincolato) specifico" |
| **Iniezione keyword nella query** | append keyword concetto, re-embed | Il baseline pratico più economico; on/off, testa se serve controllo latente |
| **vMF κ-fisso vs κ-appreso (ablation, solo simplesso)** | stessa famiglia simplesso, loss coseno vs NLL κ-appreso (griglia corretta §3) | Isola il contributo del motore vMF, mai usato per distinguere da `none` |

Escluse: win-rate (satura a 100%, non informa sulla calibrazione), nDCG/recall come metriche di vittoria (questione TIE già risolta, qui solo guardrail). Cross-encoder fuori scope (controlla precisione, non frazione di mix).

## 8. Criterio go/no-go (numeri concreti)

**GO** (tutto, CI 95% su ≥5 seed, sul dominio pre-registrato `[0.10,0.35]`):
1. simplesso (best di l1/softmax) **MACE_kw ≤ 0.10** E CI **non sovrapposto** a `none` steelman (incl. riscalato per-concetto);
2. `none` steelman MACE_kw materialmente peggiore (CI non sovrapposti);
3. pendenza ∈ [0.8,1.2], R² ≥ 0.85, violazioni monotonicità ≤ 0.05, Spearman(comando, realizzato) ≥ 0.85;
4. **transfer cross-concetto ≤ 0.05** per il simplesso e std-per-concetto strettamente minore dei baseline;
5. **interpolazione senza-SAE NON eguaglia** il simplesso (CI non sovrapposti);
6. guardrail retrieval entro la banda TIE.

**SOFT-GO** (pubblicabile, claim ristretta): il simplesso vince su MACE e transfer ma `none` riscalato per-concetto lo eguaglia → riformula come **"calibrato zero-shot senza tuning per-concetto, con controllo multi-concetto simultaneo sotto budget unitario condiviso"** (il test di mix multi-concetto diventa il deliverable centrale).

**NO-GO / null onesto:** MACE_kw simplesso > 0.15 anche su `[0.10,0.35]`; OPPURE CI sovrapposto a `none`/interpolazione; OPPURE leak residuo (post-sottrazione geometrica) > 0.3; OPPURE la dose-risposta resta interruttore binario per **tutti** i modi → si riporta il null e si ritira la scommessa.

## 9. Milestone / timeline (giorni di effort)

| # | Titolo | Dettaglio | Effort |
|---|---|---|---|
| **M0** | **Pilota dose-risposta (PRIMA di tutto)** | 1 l1 + 1 none + 1 softmax, plot share-realizzata-vs-`p` su 3–5 concetti. La curva misurata **fissa la griglia `p` e la soglia GO**. NON hard-codare MACE≤0.10 su `{0.1..0.9}` (già falsificato dal pilota) | 0.5 g |
| **M1** | `decode_from_shares` + utils intervento | `model.decode_from_shares(s)`; `scripts/proof/_calib.py` con `set_share`/`set_mix` (simplesso) e l'operatore per-query share-esatta per `none`. Unit test: somma=1, hit target esatto | 1 g |
| **M2** | `scripts/proof/calibration_real.py` (sweep singolo concetto) | Fork di `steering_real.py`; pannello concetti congelato e condiviso; split a livello di concetto; sweep `p` su dominio del pilota; mappa isotona **simmetrica**; share **continua** + kw + sae; MACE, pendenza/R², monotonicità. 1 seed prima | 2 g |
| **M3** | Baseline kill + leak + multi-concetto | Interpolazione senza-SAE; `none` riscalato 95° pct; leak off-target con **sottrazione geometrica**; mix 2-concetti | 2 g |
| **M4** | vMF (ablation robustness, solo simplesso) | branch `loss='vmf'` in `train.py`, κ softplus+offset, `log C_d` via `ive` su CPU; griglia **`{500,1000,2000,4000,8000}`**; unit test finitezza/identità. Sweep τ softmax | 1.5 g |
| **M5** | Multi-seed + statistica + write-up | 5 seed, `ci95()`; figura calibrazione (realizzato vs comandato: simplesso vs `none`-steelman vs interpolazione, con spread per-concetto); verdetto GO/SOFT-GO/NO-GO in `RISULTATI.md` con CI | 2 g |
| **M6 (stretch, fase 2)** | Catena identificabilità→leak su sintetico | Gram off-diagonale + Spearman su generatore **NON-simplesso**; differire se M2 non mostra un quadrante reale | 2 g |

Totale core (M0–M5) ~9 giorni → rientra in 1–2 settimane; M6 solo se il core dà GO.

## 10. Rischi & mitigazioni

| Rischio | Mitigazione |
|---|---|
| **Manopola = sigmoide saturante** (pilota: l1 `p=0.30→0.99`, ≥0.40→1.00; softmax peggio; persiste a K=10/50/100) | Dominio pre-registrato `[0.10,0.35]`; null onesto "interruttore binario" come esito di prima classe; M0 fissa griglia e soglie **prima** di costruire l'harness |
| **Metrica circolare** (stesso SAE setta e valuta) | Primaria = keyword/centroid indipendente; sae solo cross-check; gate Jaccard ≥ 0.6 |
| **Strawman su `none`** (la "impossibilità strutturale" è falsa: `none` può colpire share-esatta per-query) | `none` steelman con operatore per-query share-esatta + isotona + riscalatura 95° pct; vittoria mostrata al livello di mix-recuperato |
| **Asimmetria mappe** (simplesso usa identità gratis) | Fitting **simmetrico**: o tutti i modi o nessuno |
| **Leak attribuito a entanglement ma è geometrico** (verificato: l'operatore tocca tutte le ~32 coord) | Sottrarre la componente geometrica in forma chiusa (`∝ s_j·p/(1−p_old)`) prima di Spearman leak↔Gram |
| **Sintetico pre-favorisce simplesso** (`w/=w.sum()`) | Sintetico = solo mechanism; headline su arXiv reale; generatore NON-simplesso per la catena identificabilità |
| **Premessa identificabilità gonfiata** (MMCS soft, `matched@0.9` pareggia) | Non citare MMCS-recall; usare Gram off-diagonale e provarla con CI prima di asserire calibrazione |
| **vMF κ numericamente morto / infalsificabile a d=384** | Griglia ri-centrata `{500..8000}`, softplus+offset≥ν, unit test finitezza; se piatto → droppato come null |
| **Controllo senza-SAE vince → SAE superfluo** | Implementarlo presto (M3); se vince, ritirare/riformulare onestamente |
| **MPS OOM / nondeterminismo** | Cap `n_latents=2048`, `torch.mps.empty_cache()` tra modi, fallback subsample 20k, seed fissati, riportare CI |
| **Quantizzazione/saturazione anche del kw** (kw `none`: 0.06→0.62→0.86) | Segnale **continuo** (attivazione media per-doc / centroid-cosine) come primaria di risoluzione |
| **Novelty vs prior art** (Simplex AE 2301.06489, Archetypal 2502.12892, SAE-retrieval 2506.00041) | Concedere l'architettura come prior art; differenziare sul **task di controllo + mappa di calibrazione condivisa cross-concetto a inference-time (no retraining del decoder)**; leggere 2506.00041 per confermare che fa steering on/off, non mix proporzionale; valutare un braccio decoder-convex-hull |

## 11. Cosa costruire per prima (la prima PR concreta)

**PR #1 — "Pilota dose-risposta + utils di intervento calibrato" (M0 + M1, ~1.5 giorni).** È il check che decide se la scommessa è viva prima di investire nell'harness completo.

Contenuto:
1. `spherical_sae/model.py`: aggiungere `decode_from_shares(s) → s @ W_dec.T + b_dec` (nessuna seconda normalizzazione).
2. `scripts/proof/_calib.py` (nuovo): `set_share(h, c, p)` e `set_mix(h, {c:p})` per l1; shift-di-logit in forma chiusa per softmax; operatore per-query share-esatta per `none` (`h[c]=p/(1−p)·Σaltri`). Unit test: somma=1 sul simplesso; hit esatto dei target (`0.1/0.3/0.5/0.9` e `{0.4,0.2}`).
3. `scripts/proof/pilot_calibration.py` (nuovo, usa-e-getta): allena 1 modello per modo {none, l1, softmax} alla config verificata (2048/32/4000/40k), su 3–5 concetti del pannello congelato, e produce **la curva share-realizzata-vs-`p`** (continua + kw) con `p` su griglia fine in `[0.05, 0.6]`.

**Output che sblocca tutto:** la forma misurata della dose-risposta. Se conferma il sigmoide saturante (atteso), fissiamo il dominio `[0.10,0.35]` e le soglie GO; se per qualche modo la curva è lineare oltre 0.35, allarghiamo il dominio. Solo dopo si costruisce `calibration_real.py` (PR #2 = M2).
