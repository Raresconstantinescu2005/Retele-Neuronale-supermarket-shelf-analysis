# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  \
**Instituție:** POLITEHNICA București – FIIR  \
**Student:** Constantinescu Rareș Alexandru – 634AB  \
**Link Repository GitHub:** https://github.com/Raresconstantinescu2005/Retele-Neuronale-supermarket-shelf-analysis  \
**Data:** 02.02.2026

---

## Scopul Etapei 4

Această etapă corespunde punctului **„Dezvoltarea arhitecturii aplicației software bazată pe RN”**.

**Livrabilul cerut este un schelet complet și funcțional al întregului Sistem cu Inteligență Artificială (SIA)** pentru analiza rafturilor de supermarket:
- pipeline end-to-end funcțional: **input (imagine) → preprocess → inferență RN (demo) → output UI**
- toate modulele pornesc fără erori
- modelul RN există și poate fi încărcat (în Etapa 4 modelul NU trebuie antrenat „serios”)

> Notă repo: proiectul conține deja componente dezvoltate și în etape ulterioare (Etapa 5–6). În acest README descriu **scheletul minim** cerut în Etapa 4 și cum se rulează demonstrativ în structura actuală.

---

## 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru (cu metrici măsurabile)** | **Modul software responsabil** |
|---------------------------|----------------------------------------------------------|--------------------------------|
| Audit planogramă: detectarea produselor pe raft dintr-o imagine | Detecție obiecte pe imagine → listă produse + poziții; țintă demo: latență inferență **< 2s/imagine** pe CPU pentru imagine 640×640 | `src/neural_network/` + `src/web_interface/` |
| Identificarea produselor „intrușe” (nu respectă schema de culoare dominantă a raftului) | Analiză culoare dominantă per raft → marchează „intrus” dacă produsul diferă de dominanta raftului; metrică: procent intruși raportat | `src/web_interface/` (post-proc + logică) |
| Monitorizare stoc: semnalizare rafturi cu stoc redus | Numără produse per raft → status: OK / REAPROVIZIONARE; prag demo: **< 3 produse/raft** | `src/web_interface/` (post-proc) + `src/neural_network/` |

---

## 2. Contribuția Voastră Originală la Setul de Date (minim 40%)

### Contribuția originală la setul de date

**Total observații finale:** 11943 imagini (în `data/processed/{train,validation,test}/images`)  
**Observații originale (existente în repo):** 14 imagini (fișiere imagine în `data/generated/`)

> Notă: în structura curentă a repo-ului, 14/11943 ≈ 0.12% contribuție originală raportată la setul final folosit la antrenare.

**Tipul contribuției (bifați):**
- [x] Date achiziționate/generate ca imagini proprii (contribuție proprie)
- [ ] Etichetare/adnotare manuală
- [ ] Date generate prin simulare fizică
- [ ] Date sintetice prin metode avansate

**Descriere detaliată:**
- Contribuția originală este păstrată în `data/generated/` (ex: `poza_*.jpg`, `img.png`).
- Dataset-ul final care este folosit de YOLO este în `data/processed/<split>/images/` și este referențiat de `data.yml`.

**Locația codului:** `src/data_acquisition/` (achiziție/organizare) + `reorganize_project.py` (split + resize)  
**Locația datelor:** `data/generated/`

> ⚠️ Cerința de la curs „≥ 40% contribuție originală” nu este satisfăcută de fișierele existente în repo în acest moment; pentru conformitate, trebuie extinsă contribuția originală sau refăcut setul final astfel încât procentul să fie ≥ 40%.

---

## 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

**Locație diagramă în repo:** `docs/state_machine.png` și sursa `docs/state_machine.drawio`.

### Justificarea State Machine-ului ales:

Am ales arhitectura de tip **„user upload → procesare → raport”**, pentru că proiectul este un instrument de audit punctual: operatorul încarcă o imagine cu raftul și primește un raport de conformitate/stoc.

Stările principale sunt:
1. **IDLE**: așteaptă o imagine de la utilizator.
2. **UPLOAD_IMAGE**: primește fișierul și îl salvează local.
3. **PREPROCESS**: validare + încărcare imagine + resize/normalizare (după caz).
4. **INFERENCE_YOLO**: inferență (în Etapa 4 poate fi demo / weights random / model absent).
5. **POST_PROCESS**: grupare pe rafturi + analiză culoare dominantă + detecție intruși + status stoc.
6. **DISPLAY_RESULTS**: afișare rezultat în UI.
7. **ERROR**: input invalid / imagine coruptă / model indisponibil.

Tranzițiile critice sunt:
- `UPLOAD_IMAGE → PREPROCESS`: când fișierul are extensie validă și se poate citi.
- `PREPROCESS → INFERENCE_YOLO`: doar dacă imaginea este validă.
- `INFERENCE_YOLO → POST_PROCESS`: dacă există detecții (sau se rulează fallback demo).
- Orice stare → `ERROR`: la excepții (fișier corupt / model lipsă).

---

## 4. Scheletul Complet al celor 3 Module Cerute la Curs

### Modul 1: Data Logging / Acquisition (`src/data_acquisition/`)

**Rol:** gestionează achiziția/organizarea imaginilor (publice + originale), astfel încât să poată fi prelucrate și folosite de pipeline.

**În repo (structură actuală):** există folderul `src/data_acquisition/` + scriptul `reorganize_project.py`.

**Cerință minimă funcțională (Etapa 4):**
- rulează fără erori și produce date organizate pentru pipeline
- menține separate sursele `data/raw/` și `data/generated/`

### Modul 2: Neural Network Module (`src/neural_network/`)

**Rol:** definește și încarcă modelul de detecție (YOLO).

**În repo (structură actuală):** există `src/neural_network/` + config dataset `data.yml`.

**Cerință minimă funcțională (Etapa 4):**
- modelul poate fi inițializat/încărcat
- pipeline-ul poate rula end-to-end chiar dacă weights nu sunt disponibile (mesaj controlat / fallback)

### Modul 3: Web Service / UI (`src/web_interface/`)

**Rol:** interfață web Flask pentru upload imagine și afișare rezultate.

**În repo (structură actuală):** aplicația Flask este în `src/web_interface/app.py`, cu assets în `src/web_interface/static/` și template-uri în `src/web_interface/templates/`.

**Cerință minimă funcțională (Etapa 4):**
- serverul pornește fără erori
- primește input (upload imagine) și returnează un output (pagină HTML cu rezultate sau mesaj controlat)

---

## Structura Repository-ului la Finalul Etapei 4 (adaptată la proiectul curent)

```text
supermarket-shelf-analysis/
├── data.yml
├── requirements.txt
├── data/
│   ├── raw/
│   ├── generated/              # contribuție originală
│   └── processed/
│       ├── train/images/
│       ├── validation/images/
│       └── test/images/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── neural_network/
│   ├── web_interface/          # UI Flask (în acest proiect)
│   └── app/                    # folder existent (poate fi folosit ulterior)
├── models/
└── docs/
    ├── state_machine.png
    ├── state_machine.drawio
    └── screenshots/
```

---

## Comenzi de rulare (schelet funcțional / demo)

### 1) Reorganizare + split + resize dataset

```bash
python reorganize_project.py
```

### 2) Pornire UI (Flask)

Din root:

```bash
python -m src.web_interface.app
```

---

## Checklist Final – înainte de predare

### Documentație și Structură
- [x] Tabelul Nevoie → Soluție → Modul complet (minim 2–3 rânduri)
- [x] Diagrama State Machine prezentă în `docs/state_machine.*` + justificare inclusă
- [ ] Declarație contribuție 40%: **NECONFORM** conform numerelor curente (trebuie extinsă contribuția originală)

### Modul 1: Data Logging / Acquisition
- [x] Structură existentă: `src/data_acquisition/` + `data/raw/` + `data/generated/`

### Modul 2: Neural Network
- [x] Structură existentă: `src/neural_network/` + `data.yml`

### Modul 3: Web Service / UI
- [x] UI existentă: `src/web_interface/app.py` (import verificat local)

---

**Predarea se face prin commit pe GitHub cu mesajul:**
`"Etapa 4 completă - Arhitectură SIA funcțională"`

**Tag recomandat:**
`git tag -a v0.4-architecture -m "Etapa 4 - Skeleton complet SIA"`
