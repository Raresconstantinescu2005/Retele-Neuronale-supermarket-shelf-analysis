# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Constantinescu Rareș Alexandru - 634AB  
**Link Repository GitHub:** https://github.com/Raresconstantinescu2005/Retele-Neuronale-supermarket-shelf-analysis  
**Data predării/finalizării:** 15.01.2026  

---

## Scopul Etapei 6
Etapa 6 încheie ciclul formal de dezvoltare și corespunde punctelor:
- **7. Analiza performanței și optimizarea parametrilor**
- **8. Analiza și agregarea rezultatelor**
- **9. Formularea concluziilor finale**

**Obiectiv principal:** maturizarea completă a sistemului (SIA) prin:
1) optimizarea modelului RN (YOLOv8),
2) analiză detaliată a performanței,
3) integrarea îmbunătățirilor în aplicația software (UI/web).

---

## Prerequisite – Verificare Etapa 5 (îndeplinit)
- Model antrenat: `models/trained_model.pt`
- UI funcțional: `src/web_interface/app.py`
- State Machine implementat și documentat: `docs/state_machine.png`
- Artefacte rezultate: `results/training_history.csv`, `results/test_metrics.json`

---

## Cerințe Etapa 6 – Acoperire în acest repo
1. **Minimum 4 experimente de optimizare**: `results/optimization_experiments.csv`
2. **Tabel comparativ experimente**: secțiunea 3.2 + CSV
3. **Confusion Matrix**: `docs/confusion_matrix_optimized.png`
4. **Analiza 5 exemple greșite**: secțiunea 2.2 (template completabil)
5. **Metrici finali pe test set**: `results/final_metrics.json`
6. **Model optimizat salvat**: `models/optimized_model.pt`
7. **Actualizare aplicație software:**
   - UI încarcă modelul optimizat (Etapa 6) cu fallback la Etapa 5: `src/web_interface/app.py`
   - screenshot demonstrativ: `docs/screenshots/inference_optimized.png` (de adăugat dacă lipsește)

---

## 1. Actualizarea Aplicației Software în Etapa 6

### 1.1 Tabel Modificări Aplicație Software
| Componenta | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|---|---|---|---|
| Model încărcat în UI | `models/trained_model.pt` | `models/optimized_model.pt` (fallback la trained) | Integrare model final; UI folosește întotdeauna cel mai bun model disponibil |
| UI/Web | Flask upload + inferență | Neschimbat funcțional, doar model updated | Stabilitate + comportament identic pentru utilizator |
| State Machine (doc) | `docs/state_machine.png` | (Opțional) `docs/state_machine_v2.png` | Se actualizează doar dacă apar praguri/stări noi |
| Logging | Minimal | (Opțional) extins | Audit trail, debugging (în funcție de feedback) |

### 1.2 Re-testare pipeline end-to-end
Pipeline: upload imagine → inferență YOLO → postprocesare (rafturi/intruși) → generare imagini rezultat/heatmap → raport UI.

**Status:** rulat local; UI se inițializează și încarcă implicit `models/optimized_model.pt`.

---

## 2. Analiza Detaliată a Performanței

### 2.1 Confusion Matrix și interpretare
**Locație:** `docs/confusion_matrix_optimized.png`

**Observații (interpretare):**
- Pentru YOLO (detecție obiecte), „confusion matrix” reflectă confuzii între clasele de produse din `data.yml`.
- Erorile apar cel mai des în:
  - rafturi aglomerate (ocultare/overlap),
  - iluminare neuniformă,
  - obiecte mici sau parțial vizibile.

**Confuzii principale (template de completat după identificarea claselor din plot):**
1. Clasa [A] confundată cu [B] – cauză: ambalaje similare / culori apropiate.
2. Clasa [C] confundată cu [D] – cauză: obiecte mici + blur.

### 2.2 Analiza detaliată a 5 exemple greșite (test set)
Repo-ul conține metrici agregate în `results/test_metrics.json`, însă analiza „top 5 errors” pe imagini necesită lista de predicții pereche (GT vs pred) per imagine.

**Template completare (se completează manual după identificarea cazurilor în UI / rularea evaluării detaliate):**
| Index (img) | True Label | Predicted | Confidence | Cauză probabilă | Soluție propusă |
|---:|---|---|---:|---|---|
| #1 | [ ] | [ ] | [ ] | Ocultare/overlap produse | Augmentare occlusion + date cu raft aglomerat |
| #2 | [ ] | [ ] | [ ] | Blur / motion blur | Augmentare blur + shutter variations |
| #3 | [ ] | [ ] | [ ] | Iluminare slabă | Augmentare brightness/contrast |
| #4 | [ ] | [ ] | [ ] | Obiect parțial în cadru | Random crop/scale |
| #5 | [ ] | [ ] | [ ] | False positive pe reflexii | Augmentare glare/reflective artifacts |

---

## 3. Optimizarea Parametrilor și Experimentare

### 3.1 Strategia de Optimizare
**Abordare:** experimente manuale + comparație sistematică (grid redus)  
**Axe explorate:** parametri de antrenare YOLO (learning rate, batch, epochs) și stabilitatea metricilor pe setul de test/val.  
**Criteriu selecție:** maximizare F1-score și îmbunătățirea mAP@50, cu un timp de antrenare acceptabil.

### 3.2 Tabel Experimente de Optimizare (VALORI REALE)
Sursa: `results/optimization_experiments.csv`

| Exp# | Modificare față de Baseline | Precision | Recall | F1-score | Timp antrenare (min) | Observații |
|---|---|---:|---:|---:|---:|---|
| Baseline | Configurație standard YOLOv8n | 0.8228 | 0.8072 | 0.8149 | 393.9 | Rulare conform Etapa 6 |
| Exp1 | Learning Rate redus (0.001) | 0.8208 | 0.8092 | 0.8150 | 368.5 | Rulare conform Etapa 6 |
| Exp2 | Batch Size crescut (32) | 0.8250 | 0.8074 | 0.8161 | 537.4 | Rulare conform Etapa 6 |
| Exp3 | Număr epoci crescut (20) | 0.8365 | 0.8185 | 0.8274 | 915.7 | Rulare conform Etapa 6 |
| Exp4 | Configurație optimizată (LR 0.001 + 25 epoci) | 0.8358 | 0.8236 | 0.8297 | 905.1 | **BEST** (performanța cea mai bună) |

**Justificare alegere configurație finală:**
- Exp4 are cel mai bun F1-score (0.8297) dintre experimentele rulate.
- Creșterea e obținută fără o „explozie” a complexității modelului (rămâne YOLOv8n), deci păstrăm șanse bune de rulare în timp real.

---

## 4. Agregarea Rezultatelor și Vizualizări

### 4.1 Sumar rezultate finale (VALORI REALE)
Sursa metrici: `results/final_metrics.json`

| Metrică | Valoare (Etapa 6) |
|---|---:|
| Precision | 0.8982 |
| Recall | 0.8498 |
| F1-score | 0.8733 |
| mAP@50 | 0.9099 |
| mAP@50-95 | 0.6052 |

> Notă: în detecție obiecte (YOLO), „accuracy” în sens clasic nu e metrica principală; mAP + precision/recall/F1 sunt cele relevante.

### 4.2 Vizualizări obligatorii
| Artefact | Locație |
|---|---|
| Confusion matrix (optimizat) | `docs/confusion_matrix_optimized.png` |
| State machine | `docs/state_machine.png` (v2 opțional) |
| Screenshot UI (optimizat) | `docs/screenshots/inference_optimized.png` |

---

## 5. Concluzii Finale și Lecții Învățate

### 5.1 Evaluarea performanței finale
Sistemul final este funcțional end-to-end și demonstrează performanță bună pentru un dataset demonstrativ.

Puncte cheie:
- Model YOLO optimizat (Etapa 6) cu metrici finale salvate transparent în JSON.
- UI integrată și setată să folosească modelul optimizat implicit.

### 5.2 Limitări identificate
1. Dataset relativ mic și variabil (iluminare/ocluzie)
2. Ambiguități între produse cu ambalaje similare
3. Sensibilitate la motion blur sau cadre subexpuse

### 5.3 Direcții viitoare
- colectare/etichetare date suplimentare (condiții realiste, rafturi aglomerate)
- augmentări specializate + rebalansare pe clase
- export ONNX / TensorRT pentru latență mai mică pe edge

### 5.4 Lecții învățate
- Datele și augmentările specifice domeniului au impact major.
- Integrarea UI trebuie făcută devreme pentru a prinde problemele de path/dependințe.
- Metricele agregate trebuie salvate în fișiere (CSV/JSON) pentru trasabilitate.

---

## Structura repository-ului la finalul Etapei 6 (conformă)
Livrabile cheie în repo:
- `models/optimized_model.pt`
- `results/optimization_experiments.csv`
- `results/final_metrics.json`
- `docs/confusion_matrix_optimized.png`

---

## Instrucțiuni de rulare (Etapa 6)

### 1) Evaluare model (YOLO)
În acest repo, fișierul de dataset este `data.yml` (în root), nu `config/data.yaml`.

**Variantă A (Python / Ultralytics - recomandat, funcționează chiar dacă nu există comanda `yolo` în PATH):**
```bash
python -c "from ultralytics import YOLO; m=YOLO('models/optimized_model.pt'); r=m.val(data='data.yml', imgsz=640); print(getattr(r, 'results_dict', None))"
```

**Variantă B (CLI Ultralytics - doar dacă aveți `yolo` instalat ca executabil):**
```bash
yolo task=detect mode=val model=models/optimized_model.pt data=data.yml imgsz=640
```

### 2) Pornire UI (Flask)
```bash
python src/web_interface/app.py
```

---

## Checklist final – înainte de predare
- [x] `models/optimized_model.pt` există
- [x] `results/optimization_experiments.csv` există și are minimum 4 experimente
- [x] `results/final_metrics.json` există și include metricile finale
- [x] `docs/confusion_matrix_optimized.png` există
- [x] UI încarcă modelul optimizat (verificat în consolă)
- [ ] Screenshot: `docs/screenshots/inference_optimized.png`
