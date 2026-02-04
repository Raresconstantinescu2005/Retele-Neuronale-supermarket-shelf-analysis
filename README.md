# 📘 Supermarket Shelf Analysis - Sistem Inteligent de Analiză a Rafturilor

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Constantinescu Rareș Alexandru - 634AB  
**Link Repository GitHub:** [https://github.com/Raresconstantinescu2005/Retele-Neuronale-supermarket-shelf-analysis]  
**Data finalizare:** 15.01.2026  

---

## Documentație pe etape
- Etapa 3: `README – Etapa 3 -Analiza si Pregatirea Setului de Date pentru Retele Neuronale (3).md`
- Etapa 4: `README_Etapa4_Arhitectura_SIA_03.12.2025.md`
- Etapa 5: `README_Etapa5_Antrenare_RN.md`
- **Etapa 6 (final pre-examen): `README_Etapa6_Analiza_Performantei_Optimizare_Concluzii.md`**

> Notă: UI (Flask) încarcă implicit `models/optimized_model.pt` (Etapa 6) și folosește fallback la `models/trained_model.pt` dacă optimizatul lipsește.

---

## Descriere Proiect

Acest proiect implementează un sistem inteligent bazat pe rețele neuronale pentru analiza rafturilor de supermarket, cu două funcționalități principale:

1. **Auditarea automată a conformității planogramei** - detectează produse așezate greșit pe raft
2. **Monitorizarea disponibilității stocului** - identifică situațiile de rupturi de stoc

Sistemul utilizează un model YOLOv8 pentru a detecta produsele de pe raft și algoritmi de analiză a culorii pentru a identifica „intrușii” (produse care nu respectă schema de culori a raftului).

---

## Tehnologii Utilizate

- **Python 3.8+**
- **Ultralytics YOLOv8** pentru detecția obiectelor
- **Flask** pentru interfața web
- **OpenCV** pentru procesarea imaginilor

---

## Structura Proiectului

```text
supermarket-shelf-analysis/
├── README.md
├── requirements.txt
├── data.yml
├── data/
│   ├── raw/
│   ├── generated/
│   └── processed/
│       ├── train/
│       ├── validation/
│       └── test/
├── docs/
├── models/
└── results/
```

---

## Setul de Date (numere reale din repo)

Datele YOLO sunt definite în `data.yml` și se află în `data/processed/`:

- **Train:** 8377 imagini (`data/processed/train/images`)
- **Validation:** 641 imagini (`data/processed/validation/images`)
- **Test:** 2925 imagini (`data/processed/test/images`)
- **Total:** 11943 imagini

---

## Performanță (YOLO - metrice corecte)

În detecția de obiecte (YOLO), metricele relevante sunt **Precision / Recall / F1 / mAP**.

Pentru rezultatele oficiale ale proiectului, vezi:
- `results/final_metrics.json` (metrice finale)
- `results/optimization_experiments.csv` (experimente optimizare)

---

## Instalare și Utilizare

### Instalare
```bash
pip install -r requirements.txt
```

### Evaluare YOLO (dataset complet)
```bash
python -c "from ultralytics import YOLO; m=YOLO('models/optimized_model.pt'); r=m.val(data='data.yml', imgsz=640); print(getattr(r, 'results_dict', None))"
```

### Pornire UI (Flask)
```bash
python src/web_interface/app.py
```

---

## Document principal (Livrabil final)
Pentru livrabilul final consolidat (Etapa 6), folosește: `Constantinescu_Rares_634AB_README_Proiect_RN.md`.
