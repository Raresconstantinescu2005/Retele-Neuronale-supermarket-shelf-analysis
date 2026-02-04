import cv2
import os

# --- CONFIGURARE ---
# Definim perechi: (Sursa, Sub-folderul dorit în processed)
# Acum ia pozele din 'generated' și 'raw' și le pune în 'processed'
MAPPING_FOLDERE = {
    '../../data/generated': 'generated',  # <--- AICI SUNT POZELE TALE ORIGINALE
    '../../data/raw': 'raw'               # <--- AICI SUNT CELE PUBLICE (EȘANTIONUL)
}

OUTPUT_BASE = '../../data/processed'
NEW_SIZE = (640, 640)

def resize_si_salvare_in_processed():
    total_procesate = 0
    cwd = os.getcwd()
    print(f"📍 Scriptul rulează din: {cwd}")

    # Iterăm prin fiecare folder sursă
    for source_rel, dest_subfolder in MAPPING_FOLDERE.items():

        # 1. Calculăm căile absolute
        # Scriptul e în src/preprocessing, deci urcăm 2 nivele (../../) până la data
        source_abs = os.path.abspath(os.path.join(cwd, source_rel))
        dest_abs = os.path.abspath(os.path.join(cwd, OUTPUT_BASE, dest_subfolder))

        print(f"\n📂 Sursa: {source_abs}")

        # Verificăm dacă sursa există
        if not os.path.exists(source_abs):
            print(f"⚠️  ATENȚIE: Folderul sursă nu există: {source_abs}")
            continue

        # 2. Creăm folderul de destinație în processed (dacă nu există)
        if not os.path.exists(dest_abs):
            os.makedirs(dest_abs)
            print(f"🔨  Am creat folderul destinație: {dest_abs}")
        else:
            print(f"✅  Folderul destinație există: {dest_abs}")

        # 3. Procesăm fișierele
        files = os.listdir(source_abs)
        count_folder = 0

        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                input_path = os.path.join(source_abs, filename)
                output_path = os.path.join(dest_abs, filename)

                try:
                    img = cv2.imread(input_path)
                    if img is None: continue

                    # Resize la 640x640 (standard YOLO)
                    img_resized = cv2.resize(img, NEW_SIZE, interpolation=cv2.INTER_AREA)

                    # Salvare în noul folder (processed)
                    cv2.imwrite(output_path, img_resized)
                    count_folder += 1

                except Exception as e:
                    print(f"❌ Eroare la {filename}: {e}")

        print(f"🏁  Gata! {count_folder} imagini salvate în 'processed/{dest_subfolder}'")
        total_procesate += count_folder

    print(f"\n🚀 FINAL: Am procesat în total {total_procesate} imagini.")


if __name__ == "__main__":
    resize_si_salvare_in_processed()