# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN (Supermarket Shelf Analysis)

**Disciplina:** Rețele Neuronale  \
**Instituție:** POLITEHNICA București – FIIR  \
**Student:** Constantinescu Rareș Alexandru – 634AB  \
**Link Repository GitHub:** https://github.com/Raresconstantinescu2005/Retele-Neuronale-supermarket-shelf-analysis  \
**Data predării:** 15.01.2026

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din specificațiile proiectului.

**Obiectiv principal:** antrenarea efectivă a modelului RN definit în Etapa 4 (în proiectul meu: model YOLOv8 pentru detecția produselor pe raft), evaluarea performanței pe setul de test și integrarea modelului antrenat în aplicația SIA.

> Notă: repository-ul conține și fișiere din Etapa 6 (optimizare), dar **Etapa 5** folosește ca livrabil principal modelul `models/trained_model.pt`.

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

Înainte de Etapa 5, verific că sunt îndeplinite cerințele Etapei 4:

- [x] **State Machine** definit și documentat în `docs/state_machine.*`
- [x] **Cele 3 module funcționale** (schelet end-to-end):
  - [x] Modul 1 – Data Logging / Acquisition: `src/data_acquisition/`
  - [x] Modul 2 – RN: `src/neural_network/` (arhitectura YOLO)
  - [x] Modul 3 – UI/Web Service: `src/web_interface/` (Flask)
- [ ] **Contribuție ≥ 40% date originale** în `data/generated/` (în repo există 14 imagini originale; raportat la totalul 11943 din `data/processed/`, contribuția este ~0.12%)
- [x] **Tabelul „Nevoie → Soluție → Modul”** complet în `README_Etapa4_Arhitectura_SIA_03.12.2025.md`

---

## Pregătire Date pentru Antrenare

În acest proiect datele sunt imagini + adnotări YOLO, configurate prin fișierul `data.yml` din rădăcina repository-ului.

### Structură date folosită
- Date brute: `data/raw/`
- Date originale (contribuție proprie): `data/generated/`
- Date procesate și split: `data/processed/{train,validation,test}/images` (+ labels YOLO, dacă există)

### Dacă au fost adăugate date noi (ex. în Etapa 4)

```bash
python reorganize_project.py
```

---

## Cerințe Structurate pe 3 Niveluri (adaptat la proiect)

Modelul este unul de **detecție obiecte (YOLOv8)**, deci metricile relevante sunt **Precision / Recall / F1 / mAP**.

### Nivel 1 – Obligatoriu (70%)

1. [x] **Antrenarea modelului** pe setul final de date (conform `data.yml`)
2. [x] **Minimum 10 epoci** (în proiect: 20 epoci în scriptul de antrenare)
3. [x] Split train/validation/test definit de `data.yml` (folderele din `data/processed/`)
4. [x] **Tabel hiperparametri + justificări** (secțiunea următoare)
5. [x] **Metrici raportate pe test set** (în proiect: Precision/Recall/F1/mAP)
6. [x] **Salvare model antrenat** în `models/trained_model.pt`
7. [x] **Integrare în UI**: UI încarcă `models/trained_model.pt` (Flask)
8. [x] **Screenshot demonstrație inferență reală**: `docs/screenshots/inference_real.png`

> Pentru YOLO, criteriul „Accuracy” din șablonul general se traduce practic prin **mAP și F1**.

### Nivel 2 – Recomandat (85–90%)

- [ ] Early Stopping documentat explicit
- [ ] Scheduler LR documentat explicit
- [ ] Augmentări documentate explicit
- [ ] Grafic loss/val_loss în `docs/loss_curve.png`
- [x] Analiză erori context industrial (secțiune mai jos)

### Nivel 3 – Bonus
- [ ] Comparație 2+ arhitecturi
- [ ] Confusion matrix + analiză

---

## Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

Hiperparametrii folosiți pentru antrenarea din Etapa 5 sunt cei din `src/neural_network/train.py`.

| **Hiperparametru** | **Valoare aleasă** | **Justificare** |
|--------------------|-------------------:|-----------------|
| Model de bază | `yolov8n.pt` | Variantă „nano” → rapidă și potrivită pentru prototip pe CPU / resurse limitate. |
| Epochs | 20 | Minim >10 epoci; 20 e un compromis bun între timp de antrenare și convergență pe dataset relativ mic. |
| Batch size | 8 | Reduce riscul de out-of-memory și e stabil pentru hardware modest. |
| Img size | 640 | Rezoluție standard YOLOv8 pentru detecție, echilibru viteză/precizie. |
| Workers | 0 | Setare stabilă pe Windows (evită probleme de multiprocessing). |
| Prag conf (eval) | 0.25 | Prag standard pentru filtrarea detecțiilor la evaluare. |
| Prag IoU (NMS/eval) | 0.6 | Reduce suprapunerile false în scenariul cu produse apropiate. |

---

## Antrenare Model (Etapa 5)

### Script folosit
- `src/neural_network/train.py`

### Output antrenare (livrabile)
- Model antrenat: `models/trained_model.pt`
- Istoric antrenare: `results/training_history.csv` (copiat din YOLO `results.csv`)

### Rulare antrenare

```bash
python src/neural_network/train.py
```

---

## Evaluare pe Test Set (Etapa 5)

### Script folosit
- `src/neural_network/evaluate.py`

### Output evaluare (livrabile)
- Metrici test: `results/test_metrics.json`
- (compatibilitate repo) `results/final_metrics.json`
- Grafice YOLO: `src/neural_network/runs/Final_Evaluation/`

### Rulare evaluare

```bash
python src/neural_network/evaluate.py
```

### Metrici obținute (din `results/test_metrics.json`)

- **Precision:** 0.8987
- **Recall:** 0.8521
- **F1-score:** 0.8748
- **mAP@50:** 0.9126
- **mAP@50-95:** 0.6071

---

## Verificare Consistență cu State Machine (Etapa 4)

Fluxul aplicației rămâne conform State Machine-ului descris în Etapa 4 (`docs/state_machine.png`):

- `UPLOAD_IMAGE` → utilizatorul trimite imaginea în UI (Flask)
- `PREPROCESS` → validare + citire imagine
- `INFERENCE_YOLO` → inferență cu YOLO (Etapa 5: weights antrenate din `models/trained_model.pt`)
- `POST_PROCESS` → logică business (rafturi + culori + intruși + stoc)
- `DISPLAY_RESULTS` → afișare rezultate

---

## Analiză Erori în Context Industrial (Nivel 2 – recomandat)

### 1) Pe ce clase/produse greșește cel mai mult modelul?
Confuziile apar între produse cu ambalaje similare și în zone aglomerate ale raftului (multe obiecte apropiate).

### 2) Ce caracteristici ale datelor cauzează erori?
- blur / mișcare
- reflexii puternice pe ambalaje
- ocluziuni parțiale
- iluminare neuniformă (umbre)

### 3) Ce implicații are pentru aplicația industrială?
- **False negatives** afectează direct monitorizarea stocului (pot ascunde rupturi de stoc).
- **False positives** pot induce alarme false, dar sunt mai ușor de filtrat/validat.

### 4) Măsuri corective propuse
1. Colectare imagini noi în condiții reale variate + etichetare pentru cazurile dificile.
2. Ajustare praguri `conf`/`iou` pentru a reduce detecțiile ratate.
3. Augmentări specifice (iluminare/perspectivă) pentru a simula condiții reale.

---

## Structura Repository-ului la Finalul Etapei 5 (adaptată la proiect)

```text
supermarket-shelf-analysis/
├── data.yml
├── requirements.txt
├── data/
│   ├── raw/
│   ├── generated/
│   └── processed/
│       ├── train/
│       ├── validation/
│       └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── neural_network/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── runs/
│   └── web_interface/
├── models/
│   ├── trained_model.pt                 # Etapa 5 (OBLIGATORIU)
│   └── optimized_model.pt               # Etapa 6 (opțional)
├── results/
│   ├── training_history.csv             # Etapa 5 (OBLIGATORIU)
│   ├── test_metrics.json                # Etapa 5 (OBLIGATORIU)
│   ├── final_metrics.json               # compatibilitate (Etapa 6)
│   └── optimization_experiments.csv     # Etapa 6 (opțional)
└── docs/
    ├── state_machine.png
    ├── state_machine.drawio
    └── screenshots/
        └── inference_real.png           # Etapa 5 (OBLIGATORIU)
```

---

## Instrucțiuni de Rulare (Etapa 5)

```bash
pip install -r requirements.txt

# (opțional) reorganizare/split dacă ai adăugat date noi
python reorganize_project.py

# antrenare
python src/neural_network/train.py

# evaluare pe test
python src/neural_network/evaluate.py

# UI (Flask)
python -m src.web_interface.app
```

---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 4
- [x] State Machine există în `docs/state_machine.*`
- [x] Cele 3 module (Data logging, RN, UI) pornesc fără erori
- [ ] Contribuție ≥40% date originale (necesită completare/actualizare)

### Etapa 5 – Livrabile
- [x] `models/trained_model.pt` există
- [x] `results/training_history.csv` există
- [x] `results/test_metrics.json` există
- [x] UI încarcă `models/trained_model.pt`
- [x] Screenshot demo inferență: `docs/screenshots/inference_real.png`

---

## Livrabile Obligatorii (Nivel 1)

1. `README_Etapa5_Antrenare_RN.md` (acest fișier)
2. `models/trained_model.pt`
3. `results/training_history.csv`
4. `results/test_metrics.json`
5. `docs/screenshots/inference_real.png`

---

## Predare

Commit recomandat:

`"Etapa 5 completă – F1=0.8748, mAP50=0.9126"`

Tag recomandat:

`git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
