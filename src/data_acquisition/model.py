from ultralytics import YOLO
import cv2
import os

# Calea către modelul antrenat
MODEL_PATH = '../neural_network/runs/detect/supermarket_model/weights/best.pt'
# Folderul cu poze de test (cele personale)
TEST_IMAGES = '../../data/processed/raw'  # Sau 'raw' direct, depinde unde le-ai pus

model = YOLO(MODEL_PATH)

# Verificăm performanța pe folderul de validare (metrici exacte)
metrics = model.val()
print(f"Precizie (mAP50): {metrics.box.map50:.2f}")
print(f"Precizie (mAP50-95): {metrics.box.map:.2f}")

# Facem și o predicție vizuală pe o poză
for img_file in os.listdir(TEST_IMAGES)[:3]:  # Luăm primele 3 poze
    if img_file.endswith('.jpg'):
        path = os.path.join(TEST_IMAGES, img_file)
        results = model(path)
        res_plotted = results[0].plot()

        # Afișăm
        cv2.imshow("Verificare", res_plotted)
        cv2.waitKey(0)  # Apasă o tastă pentru următoarea poză

cv2.destroyAllWindows()