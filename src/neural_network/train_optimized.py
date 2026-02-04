import os
import csv
import time
import shutil
from pathlib import Path
from ultralytics import YOLO

# ===============================
# CONFIGURAȚIE PROIECT
# ===============================
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_CONFIG = ROOT_DIR / "data.yml"
RESULTS_CSV = ROOT_DIR / "results" / "optimization_experiments.csv"
MODELS_DIR = ROOT_DIR / "models"
RUNS_DIR = ROOT_DIR / "src" / "neural_network" / "runs"


# ===============================
# UTILITARE VERIFICARE EXISTENȚĂ
# ===============================

def is_experiment_done(exp_name):
    """Verifică dacă experimentul a fost deja salvat complet în CSV."""
    if not RESULTS_CSV.exists():
        return False

    try:
        with open(RESULTS_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == exp_name:
                    return True
    except Exception as e:
        print(f"⚠️ Eroare la citirea CSV-ului: {e}")

    return False


def log_experiment(exp_name, modification, metrics, duration_min):
    """Salvează rezultatele finale în results/optimization_experiments.csv"""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_CSV.exists()

    with open(RESULTS_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Exp#", "Modificare", "Precision", "Recall", "F1-score", "Timp antrenare", "Observații"])

        writer.writerow([
            exp_name, modification,
            f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}", f"{metrics['f1']:.4f}",
            f"{duration_min:.1f} min", "Rulare conform Etapa 6"
        ])



def run_experiment(exp_name, modification, epochs=10, lr=0.01, batch=16):
    if is_experiment_done(exp_name):
        print(f" SĂRIM PESTE {exp_name}: Rezultatele există deja în CSV.")
        return

    print(f"\n VERIFICARE EXPERIMENT {exp_name}")

    ckpt_path = RUNS_DIR / exp_name / "weights" / "last.pt"

    if ckpt_path.exists():
        print(f" Checkpoint găsit la: {ckpt_path}")
        print(f"   RELUĂM antrenarea pentru {exp_name}...")
        model = YOLO(str(ckpt_path))
        resume_flag = True
    else:
        print(f" Nu s-a găsit checkpoint. Pornim {exp_name} de la zero.")
        model = YOLO("yolov8n.pt")
        resume_flag = False

    if not DATA_CONFIG.exists():
        raise FileNotFoundError(f"Nu am găsit {DATA_CONFIG}!")

    start_time = time.time()

    results = model.train(
        data=str(DATA_CONFIG),
        epochs=epochs,
        lr0=lr,
        batch=batch,
        imgsz=640,
        name=exp_name,
        project=str(RUNS_DIR),
        exist_ok=True,
        resume=resume_flag,
        verbose=False
    )

    duration_min = (time.time() - start_time) / 60

    precision = results.results_dict.get("metrics/precision(B)", 0.0)
    recall = results.results_dict.get("metrics/recall(B)", 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {"precision": precision, "recall": recall, "f1": f1}
    log_experiment(exp_name, modification, metrics, duration_min)

    print(f" {exp_name} finalizat complet | F1={f1:.3f}")


    if exp_name == "Exp4":
        MODELS_DIR.mkdir(exist_ok=True)
        best_model_src = Path(results.save_dir) / "weights" / "best.pt"
        if best_model_src.exists():
            shutil.copy(best_model_src, MODELS_DIR / "optimized_model.pt")
            print(f"Model optimizat salvat în: models/optimized_model.pt")



if __name__ == "__main__":



    run_experiment("Baseline", "Configurație standard", epochs=10, lr=0.01, batch=16)
    run_experiment("Exp1", "Learning Rate redus (0.001)", epochs=10, lr=0.001, batch=16)
    run_experiment("Exp2", "Batch Size crescut (32)", epochs=10, lr=0.01, batch=32)


    run_experiment("Exp3", "Număr epoci crescut (20)", epochs=20, lr=0.01, batch=16)


    run_experiment("Exp4", "Configurație optimizată (LR 0.001 + 25 epoci)", epochs=25, lr=0.001, batch=16)