import os
import shutil
import random
from pathlib import Path
from PIL import Image

def reorganize():
    root = Path.cwd()
    print(f"Începere reorganizare în: {root}")

    # 1. Definire Structură de Foldere (Conform README Etapa 6)
    folders_to_create = [
        "models",
        "results",
        "docs/results",
        "docs/screenshots",
        "src/neural_network",
        "src/app",
        "data/processed/train/images",
        "data/processed/validation/images",
        "data/processed/test/images",
    ]

    for folder in folders_to_create:
        (root / folder).mkdir(parents=True, exist_ok=True)
    print("✓ Foldere structurale create.")

    # 2. Reorganizare Cod (Modularitate)
    # Mutăm model.py din data_acquisition/runs în neural_network
    src_model = root / "src/data_acquisition/runs/model.py"
    dest_model = root / "src/neural_network/model.py"
    if src_model.exists():
        shutil.move(str(src_model), str(dest_model))
        print("🚀 Mutat model.py -> src/neural_network/")

    # Redenumim web_interface în app pentru a respecta README-ul
    old_ui = root / "src/web_interface"
    new_ui = root / "src/app"
    if old_ui.exists() and not new_ui.exists():
        old_ui.rename(new_ui)
        print("📱 Redenumit web_interface -> src/app")

    # 3. Procesare Date și Eliminare Redundanțe (Split 70/15/15)
    # Colectăm toate imaginile din sursele brute
    raw_dir = root / "data/raw"
    gen_dir = root / "data/generated"

    # Includem și pozele rătăcite în folderele vechi de split dacă există
    old_splits = [root / "data/train", root / "data/validation", root / "data/test"]

    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    all_imgs = []
    for d in [raw_dir, gen_dir] + old_splits:
        if d.exists():
            for ext in image_extensions:
                all_imgs.extend(list(d.glob(ext)))

    if all_imgs:
        print(f"🔍 Găsite {len(all_imgs)} imagini. Se aplică split-ul 70/15/15...")
        random.seed(42) # Reproductibilitate Etapa 5
        random.shuffle(all_imgs)

        total = len(all_imgs)
        tr_end = int(0.7 * total)
        val_end = int(0.85 * total)

        splits = {
            "train": all_imgs[:tr_end],
            "validation": all_imgs[tr_end:val_end],
            "test": all_imgs[val_end:]
        }

        for name, files in splits.items():
            dest = root / f"data/processed/{name}/images"
            for f in files:
                try:
                    with Image.open(f) as img:
                        img = img.convert("RGB")
                        img = img.resize((640, 640)) # Standard YOLO
                        img.save(dest / f.name)
                except Exception as e:
                    print(f" Eroare la {f.name}: {e}")
            print(f"📦 Mutat {len(files)} imagini în data/processed/{name}")

    # 4. Curățenie Finală (Ștergere foldere goale/redundante)
    for old_folder in old_splits:
        if old_folder.exists():
            shutil.rmtree(old_folder)
            print(f"🗑 Șters folder redundant: {old_folder.name}")

    # Ștergem și folderele duplicate din data/processed dacă au rămas din screenshot
    redundant_processed = [root / "data/processed/raw", root / "data/processed/generated"]
    for rp in redundant_processed:
        if rp.exists():
            shutil.rmtree(rp)

    print("\n✨ REORGANIZARE COMPLETĂ!")
    print("Structura ta este acum conformă cu Etapa 6.")

if __name__ == "__main__":
    reorganize()