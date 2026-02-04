# Demo end-to-end (Etapa 6)

Acest folder conține o demonstrație end-to-end a aplicației:

**Input → Preprocesare → Inferență YOLO → Post-procesare (rafturi/intruși/stoc) → Output UI/artefacte**

## Conținut
- `input/` – imagini de intrare folosite în demo
- `output/` – rezultate generate (imagini cu detecții / raport JSON etc.)

## Cum se generează demo-ul (reproductibil)
Rulare din rădăcina repo-ului:

```bash
python src/app/generate_demo_assets.py
```

Scriptul va:
1. lua imaginile din `docs/demo/input/`
2. rula inferența cu modelul din `models/optimized_model.pt` (fallback la `models/trained_model.pt`)
3. salva rezultatele în `docs/demo/output/`

## Observație
Dacă nu există imagini în `docs/demo/input/`, copiați manual 1-3 imagini (preferabil din `data/generated/`) în acest folder.
