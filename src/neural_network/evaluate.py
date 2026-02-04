import json
from pathlib import Path
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[2]


TRAINED_MODEL_PATH = BASE_DIR / "models" / "trained_model.pt"


OPTIMIZED_MODEL_PATH = BASE_DIR / "models" / "optimized_model.pt"

DATA_CONFIG = BASE_DIR / "data.yml"
RESULTS_DIR = BASE_DIR / "results"
RUNS_DIR = BASE_DIR / "src" / "neural_network" / "runs"


def run_final_evaluation(model_path: Path | None = None):
    print(" Pornire evaluare finală pe setul de DATE DE TEST...")


    if model_path is None:
        model_path = TRAINED_MODEL_PATH


    if not model_path.exists() and OPTIMIZED_MODEL_PATH.exists():
        print(f" Nu am găsit modelul Etapa 5 la {TRAINED_MODEL_PATH}.")
        print(f"️ Folosesc fallback: {OPTIMIZED_MODEL_PATH} (model asociat Etapa 6).")
        model_path = OPTIMIZED_MODEL_PATH


    if not model_path.exists():
        print(f"EROARE: Nu am găsit modelul la {model_path}.")
        print("Rulează antrenarea (train.py) sau copiază best.pt în /models/trained_model.pt.")
        return

    # 3. Încărcăm modelul
    model = YOLO(str(model_path))

    # 4. Rulăm validarea pe setul de TEST
    results = model.val(
        data=str(DATA_CONFIG),
        split='test',
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.6,
        project=str(RUNS_DIR),
        name="Final_Evaluation",
        exist_ok=True
    )

    # 5. Extragerea metricilor din dicționarul de rezultate

    stats = results.results_dict
    precision = stats.get('metrics/precision(B)', 0)
    recall = stats.get('metrics/recall(B)', 0)
    map50 = stats.get('metrics/mAP50(B)', 0)
    map95 = stats.get('metrics/mAP50-95(B)', 0)

    # Calculăm manual F1-Score (Media armonică)
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0


    print("\n" + "=" * 30)
    print(" REZULTATE FINALE (TEST SET)")
    print("=" * 30)
    print(f"Model evaluat: {model_path}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-Score:   {f1_score:.4f}")
    print(f"mAP@50:     {map50:.4f}")
    print(f"mAP@50-95:  {map95:.4f}")
    print("=" * 30)

    final_metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mAP50": map50,
        "mAP50_95": map95
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Etapa 5: fișier cerut de șablon
    with open(RESULTS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)

    # Compatibilitate: fișier existent în repo (folosit în Etapa 6)
    with open(RESULTS_DIR / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"\n Rezultatele (Etapa 5) au fost salvate în: {RESULTS_DIR / 'test_metrics.json'}")
    print(f" Rezultatele (compatibilitate) au fost salvate în: {RESULTS_DIR / 'final_metrics.json'}")
    print(f" Graficele au fost generate în: {RUNS_DIR / 'Final_Evaluation'}")


if __name__ == "__main__":
    run_final_evaluation()