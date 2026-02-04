from ultralytics import YOLO
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]  # .../supermarket-shelf-analysis
DATA_YAML = BASE_DIR / "data.yml"

RUNS_DIR = BASE_DIR / "src" / "neural_network" / "runs"
EXP_NAME = "supermarket_model"

TRAIN_RUN_DIR = RUNS_DIR / "detect" / EXP_NAME
BEST_WEIGHTS = TRAIN_RUN_DIR / "weights" / "best.pt"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

TRAINED_MODEL_OUT = MODELS_DIR / "trained_model.pt"  # Etapa 5
TRAINING_HISTORY_OUT = RESULTS_DIR / "training_history.csv"  # Etapa 5


def antreneaza_modelul():
    # Dacă livrabilul Etapa 5 există deja, nu mai antrenăm.
    if TRAINED_MODEL_OUT.exists():
        print("Modelul Etapa 5 există deja.")
        print(f" Îl găsești aici: {TRAINED_MODEL_OUT}")
        print(" Nu este nevoie să reiei antrenarea.")
        return

    if not DATA_YAML.exists():
        print(f" EROARE: Nu găsesc fișierul de configurare: {DATA_YAML}")
        return

    print(" Începem antrenarea... (Ctrl+C pentru oprire)")

    # Model YOLOv8 de bază
    model = YOLO("yolov8n.pt")

    # Antrenare
    results = model.train(
        data=str(DATA_YAML),
        epochs=20,
        imgsz=640,
        batch=8,
        workers=0,
        project=str(RUNS_DIR),
        name=EXP_NAME,
        exist_ok=True,
    )

    print("Antrenare finalizată cu succes!")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Copiem best.pt -> models/trained_model.pt
    # (folosim results.save_dir ca sursă 'sigură' în caz că YOLO schimbă structura)
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy(best_src, TRAINED_MODEL_OUT)
        print(f" Model antrenat salvat în: {TRAINED_MODEL_OUT}")
    elif BEST_WEIGHTS.exists():
        shutil.copy(BEST_WEIGHTS, TRAINED_MODEL_OUT)
        print(f" Model antrenat salvat în: {TRAINED_MODEL_OUT}")
    else:
        print("️ Nu am găsit best.pt pentru export (verifică folderul runs/).")

    # 2) Copiem results.csv -> results/training_history.csv
    history_src = Path(results.save_dir) / "results.csv"
    if history_src.exists():
        shutil.copy(history_src, TRAINING_HISTORY_OUT)
        print(f"Istoric antrenare salvat în: {TRAINING_HISTORY_OUT}")
    else:

        default_history = TRAIN_RUN_DIR / "results.csv"
        if default_history.exists():
            shutil.copy(default_history, TRAINING_HISTORY_OUT)
            print(f" Istoric antrenare salvat în: {TRAINING_HISTORY_OUT}")
        else:
            print("Nu am găsit results.csv pentru export (YOLO ar trebui să-l genereze).")


if __name__ == "__main__":
    antreneaza_modelul()