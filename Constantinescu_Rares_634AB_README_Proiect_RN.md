# Constantinescu_Rares_634AB_README_Proiect_RN.md

> **Document principal (Livrabil 1 / Moodle)** – aplicație software completă + documentație finală (Etapa 6)

---

## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Constantinescu Rareș Alexandru |
| **Grupa / Specializare** | 634AB / SIA |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | https://github.com/Raresconstantinescu2005/Retele-Neuronale-supermarket-shelf-analysis |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python |
| **Domeniul Industrial de Interes (DII)** | Retail / Logistică (audit raft, disponibilitate stoc) |
| **Tip Rețea Neuronală** | CNN (YOLOv8 – detecție obiecte) |

### Rezultate cheie (Versiunea Etapa 6 vs Final)

> Notă: proiectul este **detecție obiecte** (YOLO), unde metricele standard sunt **Precision/Recall/F1/mAP**.

Surse:
- `results/final_metrics.json`
- `results/optimization_experiments.csv`

| Metric | Țintă minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| **F1-score** | ≥ 0.65 | 0.8733 | 0.8733 | (baseline→final: vezi tabel experimente) | ✓ |
| **Nr. experimente optimizare** | ≥ 4 | 5 (Baseline + Exp1..Exp4) | 5 | - | ✓ |
| **Contribuție date originale** | ≥ 40% | ~0.12% (14/11.943) | ~0.12% (14/11.943) | - | ✗ |
| **Demonstrație end-to-end** | obligatoriu | `docs/demo/` | `docs/demo/` | - | ✓ |

### Declarație de originalitate & Politica de utilizare AI
**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială este permisă ca unealtă de dezvoltare (explicații, idei, debugging, structurarea documentației).

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință | Confirmare |
|-----|---------|------------|
| 1 | Modelul RN a fost antrenat de la zero (weights inițializate random, NU model pre-antrenat descărcat) | [X] DA     |
| 2 | Minimum 40% din date sunt contribuție originală | [ ] DA     |
| 3 | Codul este propriu sau sursele externe sunt citate explicit în Bibliografie | [X] DA     |
| 4 | Arhitectura/codul/interpretarea rezultatelor reprezintă muncă proprie (AI folosit doar ca tool) | [X] DA     |
| 5 | Pot explica și justifica fiecare decizie importantă | [X] DA     |

**Semnătură student (prin completare):** [Constantinescu Rareș Alexandru]

---

## 2. Descrierea nevoii și soluția SIA

### 2.1 Nevoia reală / Studiul de caz
În retail, auditul rafturilor (conformitate cu planograma) și monitorizarea disponibilității stocului sunt activități repetitive și costisitoare, realizate manual. Erorile duc la pierderi (rupturi de stoc) sau la poziționări greșite ale produselor (impact direct asupra vânzărilor).

Proiectul propune un SIA care analizează imagini ale rafturilor și oferă automat: detecția produselor, o estimare a organizării pe rafturi, identificarea „intrușilor” (produse care nu respectă schema dominantă pe raft) și indicatori de stoc.

### 2.2 Beneficii măsurabile urmărite
1. Reducerea timpului de inspecție manuală (ex: >50%)
2. Detectarea produselor în imagini cu mAP@50 ridicat (target: >0.80)
3. Identificarea rafturilor cu stoc redus (<3 produse) automat
4. Evidențiere rapidă a intrușilor pe raft (regulă culoare dominantă)

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software
| Nevoie reală concretă | Cum o rezolvă SIA-ul | Modul software responsabil | Metric măsurabil |
|---|---|---|---|
| Detecția produselor pe raft | Detecție obiecte (YOLOv8) | `src/neural_network/` | mAP@50, Precision/Recall |
| Identificarea „intrușilor” pe raft | Analiză culoare dominantă + reguli | `src/web_interface/` (post-procesare) | % intruși/raft |
| Detectarea rafturilor cu stoc mic | Numărare obiecte/raft | `src/web_interface/` | # produse/raft |
| Generare raport și vizualizări | Heatmap + imagine cu bounding boxes | `src/web_interface/` | latență end-to-end |

---

## 3. Dataset și contribuție originală

### 3.1 Sursa și caracteristicile datelor
| Caracteristică | Valoare |
|---|---|
| Origine date | Mixt (public + original) |
| Număr total observații finale (N) | **11.943 imagini** |
| Tipuri de date | Imagini + adnotări YOLO |
| Format fișiere | JPG/PNG + TXT (labels) |
| Perioada colectării/generării | Decembrie 2025 |
| Config dataset YOLO | `data.yml` |

### 3.2 Contribuția originală (minim 40% – OBLIGATORIU)
| Câmp | Valoare |
|---|---|
| Total observații finale (N) | **11.943 imagini** |
| Observații originale (M) | **14 imagini** (`data/generated/`) |
| Procent contribuție originală | **~0.12%** (14/11.943) |
| Locație date originale | `data/generated/` |

> Notă: pentru a atinge pragul de 40% este necesară extinderea contribuției originale (mai multe imagini originale + etichete YOLO), apoi reconstruirea split-urilor în `data/processed/`.

### 3.3 Preprocesare și split date
| Set | Procent | Nr. imagini | Locație |
|---|---:|---:|---|
| Train | ~70.1% | 8.377 | `data/processed/train/` |
| Validation | ~5.4% | 641 | `data/processed/validation/` |
| Test | ~24.5% | 2.925 | `data/processed/test/` |
| TOTAL | 100% | 11.943 | - |

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 module software
| Modul | Tehnologie | Funcționalitate principală | Locație în repo |
|---|---|---|---|
| **Data Logging / Acquisition** | Python | Management/organizare + generare/achiziție date | `src/data_acquisition/` |
| **Neural Network** | PyTorch/Ultralytics (YOLOv8) | Detecție obiecte produse | `src/neural_network/` |
| **Web Service / UI** | Flask + OpenCV | Upload imagine, inferență, raport vizual | `src/web_interface/` |

### 4.2 State Machine
**Locație diagramă:** `docs/state_machine.png`

**Stări principale (implementare logică):**
| Stare | Descriere |
|---|---|
| `IDLE` | Așteptare input utilizator |
| `UPLOAD_IMAGE` | Utilizator încarcă imagine |
| `PREPROCESS` | Validare imagine, încărcare, pregătire |
| `INFERENCE_YOLO` | Detecție produse cu modelul YOLO |
| `POST_PROCESS_DATA` | Grupare pe rafturi, analiza intruși, numărare |
| `DISPLAY_RESULTS` | Generare rezultat + heatmap + raport |
| `ERROR` | Gestionare input invalid/lipsă rezultate |

**Justificare:** State Machine-ul crește predictibilitatea pipeline-ului end-to-end și permite debugging ușor (pas cu pas).

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura rețelei neuronale
Model YOLOv8 (CNN pentru detecție obiecte). În `data.yml` există o singură clasă (`nc: 1`, `names: ['product']`).

### 5.2 Hiperparametri finali (model optimizat – Etapa 6)
| Hiperparametru | Valoare finală | Justificare |
|---|---|---|
| Model | `models/optimized_model.pt` (YOLOv8) | Model final ales după optimizare |
| Learning rate | 0.001 (în Exp4) | Stabilitate + performanță mai bună |
| Epochs | 25 (în Exp4) | Compromis performanță/timp |
| Batch size | 32 (în Exp2/Exp4) | Utilizare resurse + stabilitate |

### 5.3 Experimente de optimizare (minim 4) – valori reale
Sursa: `results/optimization_experiments.csv`

| Exp# | Modificare față de Baseline | Precision | Recall | F1-score | Timp antrenare | Observații |
|---|---|---:|---:|---:|---:|---|
| Baseline | Configurație standard YOLOv8n | 0.8228 | 0.8072 | 0.8149 | 393.9 min | Referință |
| Exp1 | Learning Rate redus (0.001) | 0.8208 | 0.8092 | 0.8150 | 368.5 min | Mic câștig |
| Exp2 | Batch Size crescut (32) | 0.8250 | 0.8074 | 0.8161 | 537.4 min | Stabilitate OK |
| Exp3 | Număr epoci crescut (20) | 0.8365 | 0.8185 | 0.8274 | 915.7 min | Câștig semnificativ |
| **Exp4 (FINAL)** | Configurație optimizată (LR 0.001 + 25 epoci) | **0.8358** | **0.8236** | **0.8297** | 905.1 min | **Model folosit în aplicație** |

**Justificare alegere model final:** Exp4 are cel mai bun F1-score dintre experimente și păstrează modelul suficient de compact pentru inferență practică.

---

## 6. Performanță finală și analiză erori

### 6.1 Metrici pe test set (model optimizat)
Sursa: `results/final_metrics.json`

| Metric | Valoare | Target minim | Status |
|---|---:|---:|---|
| Precision | 0.8982 | - | - |
| Recall | 0.8498 | - | - |
| **F1-score** | **0.8733** | **≥ 0.65** | **✓** |
| mAP@50 | 0.9099 | - | - |
| mAP@50-95 | 0.6052 | - | - |

### 6.2 Confusion Matrix
**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare (rezumat):**
- Cu 1 singură clasă (`product`), matricea reflectă în principal confuzia între „product” și background (FN/FP).
- Erorile sunt influențate de ocluzii, iluminare, produse mici/dense.

### 6.3 Analiza Top 5 erori
**Template (de completat manual):**
| # | Input (descriere scurtă) | Predicție RN | Clasă reală | Cauză probabilă | Implicație industrială |
|---:|---|---|---|---|---|
| 1 | [ ] | [ ] | product | [ ] | [ ] |
| 2 | [ ] | [ ] | product | [ ] | [ ] |
| 3 | [ ] | [ ] | product | [ ] | [ ] |
| 4 | [ ] | [ ] | product | [ ] | [ ] |
| 5 | [ ] | [ ] | product | [ ] | [ ] |

---

## 7. Aplicația software finală

### 7.1 Modificări implementate în Etapa 6
| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|---|---|---|---|
| Model încărcat în UI | `models/trained_model.pt` | `models/optimized_model.pt` (fallback la trained) | UI folosește modelul final optimizat |

### 7.2 Screenshot UI cu model optimizat
**Locație:** `docs/screenshots/inference_optimized.png` (**existent în repo**) 

### 7.3 Demonstrație funcțională end-to-end (OBLIGATORIU)
**Locație dovadă:** `docs/demo/`
- input: `docs/demo/input/`
- output: `docs/demo/output/`

Script reproducibil:
```bash
python src/app/generate_demo_assets.py
```

---

## 8. Structura repository-ului final
Repo-ul respectă structura cerută în Etapa 6 (modele în `models/`, rezultate în `results/`, vizualizări în `docs/`, cod în `src/`).

---

## 9. Instrucțiuni de instalare și rulare

### 9.1 Cerințe preliminare
- Python 3.8+ (recomandat 3.10+)

### 9.2 Instalare
```bash
pip install -r requirements.txt
```

### 9.3 Rulare pipeline complet (minimal)
**Evaluare YOLO (fără a depinde de CLI `yolo`):**
```bash
python -c "from ultralytics import YOLO; m=YOLO('models/optimized_model.pt'); r=m.val(data='data.yml', imgsz=640); print(getattr(r,'results_dict',None))"
```

**Pornire UI (Flask):**
```bash
python src/web_interface/app.py
```

---

## 10. Concluzii și discuții
Proiectul demonstrează un pipeline complet (UI + inferență YOLO + post-procesare logic de raft) și atinge F1-score peste pragul minim. Limitările sunt legate în principal de variabilitatea iluminării și ocluziile produse.

---

## 11. Bibliografie
1. Ultralytics YOLOv8 Documentation: https://docs.ultralytics.com/
2. Redmon, J. et al., You Only Look Once: Unified, Real-Time Object Detection, 2016. https://arxiv.org/abs/1506.02640
3. SKU-110K Dataset: https://github.com/eg4000/SKU110K_CVPR19

---

## 12. Checklist final (auto-verificare înainte de predare)
- [x] Model optimizat disponibil: `models/optimized_model.pt`
- [x] Minimum 4 experimente optimizare: `results/optimization_experiments.csv`
- [x] Metrici finale disponibile: `results/final_metrics.json`
- [x] Confusion matrix: `docs/confusion_matrix_optimized.png`
- [x] Screenshot UI optimizat: `docs/screenshots/inference_optimized.png`
- [x] Completat câmpurile rămase marcate "[de completat]" (specializare, N total, etc.)

---

**Ultima actualizare:** 02.02.2026  
**Tag Git recomandat:** `v0.6-optimized-final`
