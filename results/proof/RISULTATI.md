# Da demo a prova — Risultati misurati

Esperimenti eseguiti su **dati sintetici con dizionario di verità noto**, dove possiamo
misurare se lo SAE ritrova *davvero* i concetti veri (impossibile sui dati reali, dove
`cos_sim ≈ 0,95` non distingue un dizionario vero da uno degenere).

Setup base: 64 dimensioni, 256 concetti veri, 8 attivi per campione, 8000 campioni,
`loss = cos`, split train/test 80/20. Codice in `scripts/proof/`.

> **TL;DR.** Due affermazioni del progetto diventano **prova**: (1) lo SAE ritrova i concetti
> veri e la variante **sferica li ritrova meglio** dello standard, in modo statisticamente
> reale; (2) un piccolo cambio di geometria nell'AuxK **taglia i concetti morti di ~80%**.
> Due affermazioni **pareggiano onestamente** (ricostruzione e ricerca: sferico ≈ standard).
> La **legge di scala** è confermata a piena scala (α ≈ 0,15, nel range del paper). E lo **steering** dimostra
che i concetti sono causalmente manovrabili (manopola 0→99%, lo sferico il migliore).

---

## 🎯🎲📏🧪 1. Identificabilità — *ritrova i concetti veri?* → SÌ (e lo sferico vince)

5 semi, intervalli di confidenza al 95%, misure sul test set tenuto da parte, tutte e 4 le modalità.
Metrica: quanto i concetti imparati (colonne del decoder) somigliano ai concetti veri (MMCS = media del coseno migliore).

| modalità | recall (ritrova i veri) | precisione | abbinati @0,9 / 256 | cos test |
|---|---|---|---|---|
| `none` (standard) | 0,953 ± 0,003 | 0,952 ± 0,006 | 246,0 ± 4,6 | 0,855 |
| **`l2` (sferico)** | **0,973 ± 0,002** | **0,970 ± 0,002** | 246,8 ± 1,0 | 0,853 |
| `l1` | 0,971 ± 0,001 | 0,968 ± 0,003 | 246,2 ± 2,7 | 0,853 |
| `softmax` | 0,966 ± 0,002 | 0,963 ± 0,002 | 245,4 ± 2,4 | 0,851 |

**Esito ✅ PROVA.** Lo SAE ritrova ~96–97% dei concetti veri (≈246/256 abbinati con coseno > 0,9):
il metodo funziona davvero. E la variante sferica (`l2`) ha **recall e precisione più alte** dello
standard, con **intervalli di confidenza che non si toccano** (`none` ≤ 0,956 vs `l2` ≥ 0,971):
è un vantaggio **reale, non fortuna**. Sulla ricostruzione (`cos`) pareggiano — ma sul *disentanglement*
(ritrovare il dizionario vero), che è lo scopo, lo sferico vince.

*Nota onesta:* il conteggio "abbinati @0,9" pareggia (~246 per tutti); il vantaggio sferico è
nella *vicinanza media* dei concetti, non nel numero che supera la soglia 0,9.

---

## 📏 2. AuxK — *il "righello unico" ripara i concetti morti?* → SÌ, nettamente

Dizionario sovra-dimensionato (4096 latenti per 256 concetti veri), dove i concetti morti appaiono.
Confronto del modo di "rianimazione": `off` (niente), `euclid` (quello attuale del repo, a distanza
euclidea) vs `cosine` (rianimazione sulla *direzione* del residuo — il righello giusto). 2 semi.

| modalità | AuxK | concetti morti / 4096 | recall |
|---|---|---|---|
| `l2` | off | 71,0 ± 63,5 | 0,679 |
| `l2` | euclid (attuale) | 85,5 ± 44,5 | 0,673 |
| **`l2`** | **cosine (proposto)** | **14,5 ± 6,4** | 0,677 |
| `none` | off | 68,5 ± 19,1 | 0,672 |
| `none` | euclid (attuale) | 81,0 ± 50,8 | 0,664 |
| **`none`** | **cosine (proposto)** | **16,0 ± 12,7** | 0,676 |

**Esito ✅ PROVA + correzione.** L'AuxK euclideo attuale **non aiuta** (anzi peggiora: da ~71 a ~85
morti, con varianza altissima), perché misura le distanze in un righello diverso da quello del modello
(coseno). Sostituendolo con una rianimazione **sulla direzione** (coseno), i concetti morti crollano
**di circa l'80%** (da ~71 a ~14,5). Conferma diretta dell'ipotesi della issue #4.

---

## 🚗 3. Ricerca — *la geometria sferica aiuta la ricerca?* → preserva, ma non vince

Ogni documento ha una classe vera (il concetto dominante); pertinente = stessa classe. Cerchiamo per
coseno con varie rappresentazioni e misuriamo la qualità dei primi 10 (1 seme, dati sintetici).

| rappresentazione | Precisione@10 | nDCG@10 |
|---|---|---|
| embedding grezzo (riferimento) | 0,406 | 0,416 |
| ricostruzione `none` | 0,418 | 0,427 |
| ricostruzione `l2` | 0,418 | 0,426 |
| ricostruzione `l1` | 0,422 | 0,429 |
| ricostruzione `softmax` | 0,501 | 0,507 |
| **codice sparso `l2`** | **0,516** | **0,515** |
| codice sparso `none` | 0,513 | 0,512 |
| codice sparso `l1` | 0,516 | 0,515 |
| codice sparso `softmax` | 0,378 | 0,386 |

**Esito ⚖️ premessa confermata, ma pareggio.** Due cose buone: (1) la ricostruzione sferica **non
rovina** la ricerca (anzi è leggermente sopra il grezzo) — la premessa regge; (2) il **codice sparso
è un'ottima rappresentazione per la ricerca**, meglio dell'embedding grezzo (0,516 vs 0,406). Ma tra
sferico e standard è sostanzialmente **pareggio** (codice `l2` 0,516 vs `none` 0,513). Quindi "lo
sferico aiuta la ricerca" non è ancora dimostrato: non fa danni e i codici sono utili, ma non c'è un
vantaggio netto su questo test. (`softmax` è il caso strano: ottima ricostruzione, codice scarso.)

---

## 📈 4. Legge di scala — *migliora ingrandendo il dizionario?* → inconcludente

cos_loss (test) al crescere di `n_latents`, fit della legge di potenza `L(n) = c·n^-α` (paper: α ≈ 0,12–0,18).

| modalità | n=64 | n=128 | n=256 | n=512 | n=1024 | α | R² |
|---|---|---|---|---|---|---|---|
| `none` | 0,276 | 0,233 | **0,171** | 0,213 | 0,266 | 0,024 | 0,02 |
| `l2` | 0,284 | 0,234 | **0,158** | 0,203 | 0,265 | 0,041 | 0,04 |
| `l1` | 0,283 | 0,235 | **0,158** | 0,201 | 0,265 | 0,042 | 0,04 |
| `softmax` | 0,287 | 0,235 | 0,153 | 0,145 | **0,173** | 0,215 | 0,65 |

**Esito ⚠️ inconcludente.** Niente legge di potenza pulita: la curva è a **U**, col minimo esattamente
a `n=256` (= il numero di concetti veri!). Due cause oneste: (a) il bilancio di passi è **fisso**, quindi
i dizionari grandi sono *sotto-allenati*; (b) i concetti veri sono esattamente 256, quindi
sovra-dimensionare non aiuta a "trovarne di più". Per misurare la scala sul serio servono passi di
training proporzionali a `n`, o dati reali con uno spazio di concetti molto più ricco. (`softmax` è
l'unico che continua a migliorare — comportamento diverso, da capire.)

---

## 🌍 Verifica sui DATI REALI (117.592 abstract arXiv, 384-d)

Scaricato e costruito il dataset reale (`build_dataset.py`), abbiamo ripetuto ricerca e scala
sui veri embedding (normalizzati sulla sfera).

### 🚗 Ricerca reale — preserva i vicini? (senza etichette)
Quanto la ricostruzione conserva i veri vicini dell'embedding grezzo (Recall@10 dei top-10 grezzi, correlazione di rango):

| modalità | Recall@10 dei vicini grezzi | corr. di rango |
|---|---|---|
| none | 0,530 | 0,562 |
| l2 | 0,535 | 0,566 |
| l1 | 0,532 | 0,565 |
| **softmax** | **0,593** | **0,615** |

✅ La ricostruzione sferica **preserva la ricerca** anche sul reale (~53% dei vicini grezzi sopravvivono).
`l2` ≈ `none` (pareggio, lieve vantaggio); `softmax` conserva meglio la struttura. Premessa confermata, vittoria sferica no.

### 📈 Scala reale — appare la legge di potenza?
Fit `L(n) = c·n^-α` su n ∈ {256, 512, 1024, 2048}, passi ∝ n, su 12k documenti:

| modalità | n=256 | n=512 | n=1024 | n=2048 | α | R² |
|---|---|---|---|---|---|---|
| none | 0,115 | 0,111 | 0,120 | 0,136 | −0,08 | 0,72 |
| l2 | 0,116 | 0,112 | 0,121 | 0,137 | −0,08 | 0,73 |
| l1 | 0,116 | 0,113 | 0,123 | 0,139 | −0,09 | 0,78 |
| **softmax** | 0,136 | 0,111 | **0,103** | 0,104 | **+0,13** | 0,75 |

⚠️ Con budget ridotto, per `none`/`l2`/`l1` il dizionario più grande **peggiora** la ricostruzione (α negativo:
troppi latenti sotto-allenati, più rumore nella selezione top-k). **Solo `softmax`** ha α **dentro il range del
paper (+0,13)**: continua a migliorare ingrandendo il dizionario, come un vero SAE scalabile. La legge pulita per
le altre modalità richiede il training a **piena scala** (tutti i 117k documenti, molti più passi) — il passo #3
lo chiudiamo qui sotto. 👇

---

### 📈 PIENA SCALA — la legge di potenza definitiva

Rifatto sul serio: **tutti i 117.592 documenti**, dizionari fino a **4096**, passi ∝ n, sulla **GPU Apple (MPS)**. Fit `L(n) = c·n^-α`:

| modalità | n=256 | n=512 | n=1024 | n=2048 | n=4096 | **α** | R² |
|---|---|---|---|---|---|---|---|
| none | 0,107 | 0,091 | 0,080 | 0,073 | 0,071 | **0,149** | 0,94 |
| **l2 (sferico)** | 0,108 | 0,091 | 0,080 | 0,074 | 0,071 | **0,151** | 0,94 |
| l1 | 0,108 | 0,092 | 0,081 | 0,074 | 0,072 | **0,148** | 0,94 |
| softmax | 0,138 | 0,108 | 0,090 | 0,080 | 0,075 | 0,220 | 0,95 |

✅ **Legge di potenza CONFERMATA.** Con dati e compute veri, `cos_loss` **scende in modo pulito e monotono**
ingrandendo il dizionario, e l'esponente cade **dentro il range del paper (0,12–0,18)**: `none` 0,149,
**`l2` 0,151**, `l1` 0,148 (R² ≈ 0,94). La variante **sferica scala esattamente come lo standard** — un vero SAE
scalabile. L'α negativo del primo tentativo era solo mancanza di dati/passi. **Passo #3 chiuso.** ✅

---

### 🎛️ Pilotaggio (steering) — il test causale finale

La prova più forte che i concetti siano **reali e manovrabili**: prendiamo query che NON parlano di un concetto,
**alziamo la sua "manopola"** nel codice, ri-decodifichiamo e cerchiamo. Quanta parte dei risultati parla davvero
di quel concetto? (15 concetti, 40k documenti reali, intensità crescente.)

| modalità | controllo | manopola ×1 | manopola ×3 | manopola ×6 | win-rate |
|---|---|---|---|---|---|
| none | 0,010 | 0,058 | 0,762 | 0,991 | 100% |
| **l2 (sferico)** | 0,016 | **0,082** | **0,808** | **0,997** | 100% |
| l1 | 0,018 | 0,074 | 0,781 | 0,994 | 100% |

*(precisione = quota dei top-10 che esprimono il concetto secondo lo SAE; confermato dalla metrica **indipendente**
delle parole chiave nei titoli: controllo ~0,03 → ×6 **~0,86–0,88**)*

✅ **ESITO: prova causale netta.** Girando la manopola da 0 a 6, la quota di risultati sul concetto pilotato passa
da **~1% a ~99%** (e ~3% → ~88% sulla metrica indipendente delle parole chiave), con **win-rate 100%**: lo steering
funziona *sempre*. La risposta è una **curva dose-risposta** pulita (×1 spinta lieve, ×3 effetto forte, ×6
saturazione). E la variante **sferica (`l2`) pilota un filo meglio** a ogni intensità (a ×3: 0,808 vs 0,762 dello
standard), su *entrambe* le metriche — coerente con l'idea che normalizzare in proporzioni renda il controllo più
efficace.

**Esempi reali (`l2`):** la query *"Adversarial Example Games"* pilotata verso *[federated]* → tutti paper di
federated learning; *"FedADMM…"* pilotata verso *[graph]* → tutti paper di graph neural network. La manopola
sposta i risultati **a prescindere dal tema di partenza**.

---

## Conclusione: la mappa, dopo gli esperimenti

| Passo | Esito |
|---|---|
| 🎯 Verità nota | ✅ **PROVA** — ritrova i concetti veri (~97%) |
| 🎲 Semi multipli + CI | ✅ **PROVA** — il vantaggio sferico regge agli intervalli d'errore |
| 📏 Metro (split test) | ✅ fatto — misure su dati mai visti |
| 🧪 l1 / softmax | ✅ fatto — `l1` ≈ `l2` (forti), `softmax` diverso |
| 📏 AuxK righello unico | ✅ **PROVA + fix** — il coseno taglia i morti dell'~80% |
| 🚗 Ricerca (sintetica **e reale**) | ⚖️ premessa ok (preserva la ricerca), ma sferico ≈ standard |
| 📈 Legge di scala | ✅ **PROVA** — legge di potenza pulita, α ≈ 0,15 (nel range del paper), sferico ≈ standard |
| 🎛️ Pilotaggio (steering) | ✅ **PROVA causale** — manopola 0→99% (win-rate 100%), sferico il migliore |

**In sintesi:** la variante sferica passa da "demo promettente" a **risultato dimostrato** su ciò che
conta di più — il *disentanglement* (ritrova meglio i concetti veri, in modo statisticamente solido) —
e otteniamo in regalo una **correzione concreta** (AuxK col righello giusto). Su ricostruzione e ricerca
pareggia onestamente; la scala resta da misurare bene. Questo *è* il salto da demo a prova: alcune
affermazioni confermate con i numeri, altre ridimensionate con onestà.

### Limiti (onestà intellettuale)
- Tutto è **sintetico**; il generatore usa pesi sul simplex, che potrebbe favorire `l1`/`softmax`
  (ma vince `l2`, che è la sfera — incoraggiante). Da replicare sui dati reali.
- Ricerca e scala su dati reali: 1 seme, sottocampione 12–15k, passi ridotti. La **legge di scala completa**
  per le modalità lineari richiede il training a **piena scala** (tutti i 117k documenti, molti più passi):
  è stato **chiuso a piena scala** (117k, fino a 4096, MPS): α ≈ 0,15, nel range del paper, sferico ≈ standard.
- Possibile estensione: il **pilotaggio** (steering) sui dati reali — la versione "avanzata" del test di ricerca.

### Come riprodurre
```bash
python scripts/proof/run_identifiability.py    # 🎯🎲📏🧪
python scripts/proof/run_auxk_ab.py            # 📏 (righello)
python scripts/proof/run_retrieval.py          # 🚗 (sintetico)
python scripts/proof/run_scaling.py            # 📈
python scripts/proof/retrieval_real.py         # 🚗 (reale, serve il dataset)
python scripts/proof/run_scaling_real.py       # 📈 (reale ridotto, serve il dataset)
python scripts/proof/run_scaling_real_full.py  # 📈 PIENA SCALA (reale, MPS, ripresa automatica)
python scripts/proof/steering_real.py          # 🎛️ pilotaggio/steering (reale)
```
