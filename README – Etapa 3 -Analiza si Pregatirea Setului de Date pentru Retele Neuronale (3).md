# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  \
**Instituție:** POLITEHNICA București – FIIR  \
**Student:** Constantinescu Rareș Alexandru – 634AB  \
**Data:** 15.01.2026

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului **„Supermarket Shelf Analysis”**.

Scopul etapei este pregătirea corectă a datelor pentru instruirea unui model de detecție de obiecte (YOLO), respectând bunele practici privind **calitatea**, **consistența** și **reproductibilitatea** datelor.

---

## 1. Structura Repository-ului GitHub (versiunea Etapei 3)

Structura relevantă pentru Etapa 3 (date + preprocesare) este:

```text
supermarket-shelf-analysis/
├── README.md
├── README – Etapa 3 -Analiza si Pregatirea Setului de Date pentru Retele Neuronale (3).md
├── data.yml                         # configurare dataset YOLO (train/val/test, nume clase)
├── requirements.txt
├── config/
├── data/
│   ├── raw/                         # imagini brute (subset public)
│   ├── generated/                   # imagini originale (contribuție proprie)
│   └── processed/
│       ├── train/
│       │   └── images/              # imagini preprocesate pentru train
│       ├── validation/
│       │   └── images/              # imagini preprocesate pentru validation
│       └── test/
│           └── images/              # imagini preprocesate pentru test
├── docs/
│   └── dataset/                     # materiale despre dataset (ex: docs/dataset/dataset.docx)
└── src/
    ├── preprocessing/               # preprocesare imagini (resize/organizare)
    └── data_acquisition/            # (dacă e cazul) achiziție/generare date
```

**Observații:**
- În acest repo, split-ul final este sub `data/processed/{train,validation,test}/images` (format YOLO standard).
- Fișierul de configurare YOLO este `data.yml` (nu `data.yaml`).

---

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor

- **Origine:** imagini de rafturi de supermarket
  - subset **public** (imagini în `data/raw/`)
  - subset **original** (imagini în `data/generated/`)
- **Modul de achiziție:** ☑ Fișier extern (imagini) / ☑ Generare programatică (organizare + resize)
- **Perioada / condițiile colectării:** proiect educațional (set mixt: public + contribuție proprie)

### 2.2 Caracteristicile dataset-ului

- **Tipuri de date:** ☑ Imagini
- **Format fișiere:** ☑ JPG / ☑ JPEG / ☑ PNG
- **Sarcina de învățare:** detecție de obiecte (YOLO)
- **Clase (conform `data.yml`):**
  - `nc: 1`
  - `names: ['product']`

> Important: acest repo conține imagini deja split-uite și resize-uite la 640×640 în `data/processed/...`.

### 2.3 Descrierea „caracteristicilor” (adaptat pentru dataset imagistic)

Pentru dataset-uri imagistice de detecție, „features” sunt în principal:

| Element | Tip | Unitate | Descriere | Domeniu valori |
|--------|-----|---------|-----------|----------------|
| Imagine | imagine (RGB) | pixeli | fotografie cu raft | 640×640 după preprocesare |
| Etichetă (dacă există) | fișier text (YOLO) | – | bounding boxes + class_id | coordonate normalizate în [0, 1] |
| Class ID | int | – | clasa obiectului | {0} pentru `product` |

**Fișier recomandat (opțional, pentru completare):** `data/README.md` (nu există încă în repo).

---

## 3. Analiza Exploratorie a Datelor (EDA) – Sintetic

În contextul unui dataset imagistic, EDA include verificări cantitative și vizuale:

### 3.1 Statistici descriptive aplicate

- număr de imagini pe split (train/validation/test)
- verificarea formatelor de fișier și dimensiunilor (în pixeli)
- (dacă există etichete) distribuția numărului de obiecte/imagine

**Sumar (calculat pe conținutul curent din `data/processed/`):**
- `train`: **8377** imagini
- `validation`: **641** imagini
- `test`: **2925** imagini
- dimensiune imagini (verificare pe eșantion): **640×640**

### 3.2 Analiza calității datelor

- detectarea fișierelor corupte / care nu se pot citi cu OpenCV/PIL
- detectarea duplicatelor (după nume/bytes – opțional)
- verificarea consistenței dimensiunii după preprocesare (toate 640×640)
- (dacă există etichete) sanity checks pentru bounding boxes:
  - coordonate normalizate în [0, 1]
  - x_center, y_center, width, height > 0

### 3.3 Probleme identificate (de completat)

- [ ] imagini care nu se pot deschide
- [ ] formate mixte / extensii neuniforme
- [ ] duplicări
- [ ] (dacă există etichete) bounding boxes invalide

---

## 4. Preprocesarea Datelor

### 4.1 Curățarea datelor

- eliminare (sau ignorare) fișiere care nu pot fi citite
- conversie în RGB (unde e necesar)
- redimensionare la **640×640** (standard YOLO)

Preprocesarea și split-ul sunt realizate în repo prin:
- `reorganize_project.py` (split 70/15/15 + resize la 640×640)
- `src/preprocessing/resize.py` (resize + salvare în `data/processed/` pe surse)

### 4.2 Transformarea caracteristicilor

- **Resize** la 640×640
- (opțional) normalizare pixeli (0–1) – relevantă la încărcarea în model, nu neapărat la salvarea pe disc
- (opțional) augmentări (flip/blur/brightness) – sunt specifice etapei de antrenare și pot fi aplicate online

### 4.3 Structurarea seturilor de date

**Împărțire recomandată (și folosită în script):**
- 70% – train
- 15% – validation
- 15% – test

**Principii respectate:**
- split determinist (seed) pentru reproductibilitate (`random.seed(42)` în `reorganize_project.py`)
- evitarea amestecării involuntare a fișierelor între split-uri

### 4.4 Salvarea rezultatelor preprocesării

- imagini preprocesate: `data/processed/<split>/images/`
- configurarea dataset-ului YOLO: `data.yml`

Conținut relevant din `data.yml`:
- `train: processed/train/images`
- `val: processed/validation/images`
- `test: processed/test/images`

---

## 5. Fișiere Generate în Această Etapă

- `data/processed/` – imaginile resize-uite și split-uite
- `data.yml` – configurația dataset-ului pentru YOLO (căi + clase)
- `reorganize_project.py` – script de reorganizare + split + resize
- `src/preprocessing/resize.py` – script de resize și salvare în `data/processed/`

---

## 6. Stare Etapă (de completat de student)

- [x] Structură repository configurată
- [ ] Dataset analizat (EDA realizată + sumar numeric)
- [x] Date preprocesate (resize 640×640)
- [x] Seturi train/val/test generate
- [ ] Documentație dataset detaliată în `data/README.md` (opțional)

---

### Note

- Detaliile despre arhitectură, antrenare, evaluare și optimizare sunt documentate în README-urile etapelor următoare (Etapa 4–6).
