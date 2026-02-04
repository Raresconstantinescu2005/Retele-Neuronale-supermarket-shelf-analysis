from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from dataclasses import dataclass

app = Flask(__name__)

# --- 1. CONFIGURARE ---
UPLOAD_FOLDER = 'static/uploads'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rădăcina proiectului (…/supermarket-shelf-analysis)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

# Etapa 6: încărcăm modelul OPTIMIZAT din /models, cu fallback la cel antrenat (Etapa 5)
OPTIMIZED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'optimized_model.pt')
TRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'trained_model.pt')

MODEL_PATH = OPTIMIZED_MODEL_PATH if os.path.exists(OPTIMIZED_MODEL_PATH) else TRAINED_MODEL_PATH

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, UPLOAD_FOLDER)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if os.path.exists(MODEL_PATH):
    print(f"🧠 Încărcare model din: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
else:
    print(f"⚠️ EROARE: Nu găsesc modelul la {MODEL_PATH}")
    print("Rulează antrenarea (python src/neural_network/train.py) sau copiază best.pt în models/trained_model.pt")
    model = None


# --- 2. FUNCȚII AUXILIARE ---

def get_hue_mediu(img, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    h_img, w_img, _ = img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0: return -1

    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue_mean = np.mean(hsv_crop[:, :, 0])
    return hue_mean


# --- MODIFICARE AICI: Am scos numele brandurilor ---
def get_nume_brand(hue_value):
    if hue_value == -1: return "Necunoscut"

    # Intervalele de culoare rămân aceleași, doar etichetele se schimbă
    if hue_value < 10 or hue_value > 170:
        return "Produs Roșu"
    elif 10 <= hue_value < 25:
        return "Produs Portocaliu"
    elif 25 <= hue_value < 35:
        return "Produs Galben"
    elif 35 <= hue_value < 85:
        return "Produs Verde"
    elif 85 <= hue_value < 130:
        return "Produs Albastru"
    elif 130 <= hue_value < 170:
        return "Produs Mov/Roz"
    else:
        return "Nedefinit"


# --- 3. LOGICĂ BUSINESS ---

def detecteaza_dezordine(boxes):
    if not boxes: return "N/A"
    y_centers = [box.xywh[0][1].item() for box in boxes]
    deviatie = np.std(y_centers)
    if deviatie > 25:
        return f"⚠️ DEZORDONAT (Scor: {int(deviatie)})"
    return "✅ ALINIAT"


def analizeaza_consistenta_raft(img_path, boxes):
    total_produse = len(boxes)
    if total_produse < 3: return "Prea puține produse", []

    img = cv2.imread(img_path)
    date_produse = []

    # 1. Colectare Date
    for box in boxes:
        w = box.xywh[0][2].item()
        h = box.xywh[0][3].item()
        hue = get_hue_mediu(img, box)
        brand = get_nume_brand(hue)  # Acum va returna "Produs Verde" etc.
        date_produse.append({'box': box, 'w': w, 'h': h, 'hue': hue, 'brand': brand})

    # Statistici Dimensiuni
    median_w = float(np.median([d['w'] for d in date_produse]))

    # 2. ANALIZA DOMINANȚEI (Regula de 60%)
    brands_valide = [d['brand'] for d in date_produse if d['brand'] != "Necunoscut"]

    dominant_brand = None
    este_raft_uniform = False

    if brands_valide:
        counter = Counter(brands_valide)
        top_brand, count = counter.most_common(1)[0]
        procentaj_dominanta = count / len(brands_valide)

        # Dacă un tip de culoare ocupă >= 60%
        if procentaj_dominanta >= 0.60:
            dominant_brand = top_brand
            este_raft_uniform = True

    intrusi = []

    # 3. Identificare Intruși
    for item in date_produse:
        is_intrus = False

        # CRITERIUL A: Dimensiune
        w_item = float(item['w'])
        if (w_item > float(median_w) * 2.5 or w_item < float(median_w) * 0.4):
            is_intrus = True

        # CRITERIUL B: Identitate Produs (Culoare)
        if este_raft_uniform:
            # Dacă nu e culoarea dominantă -> INTRUS
            if item['brand'] != "Necunoscut" and item['brand'] != dominant_brand:
                is_intrus = True
        else:
            # Fallback pentru raft mixt
            hues_valide = [d['hue'] for d in date_produse if d['hue'] != -1]
            std_dev = np.std(hues_valide) if hues_valide else 0

            # Dacă variația e mare (> 35), e raft asortat, nu marcăm intruși de culoare
            if std_dev > 35:
                pass
            else:
                median_hue = float(np.median(hues_valide)) if hues_valide else None
                if median_hue is not None and item['hue'] != -1:
                    diff = float(abs(float(item['hue']) - float(median_hue)))
                    diff = min(diff, 180 - diff)
                    if diff > 30:
                        is_intrus = True

        if is_intrus:
            intrusi.append(item['box'])

    # 4. Rezultat Final
    nr_intrusi = len(intrusi)

    if este_raft_uniform:
        descriere = f"Dominat de {dominant_brand}"
    else:
        descriere = "Raft Mixt"

    if nr_intrusi > 0:
        procent_intrusi = (nr_intrusi / total_produse) * 100
        return f"🚨 {nr_intrusi} ({int(procent_intrusi)}%) INTRUȘI - {descriere}", intrusi
    else:
        return f"✅ Uniform ({descriere})", []


def analizeaza_dominanta(img_path, boxes):
    if not boxes: return "-"
    img = cv2.imread(img_path)
    branduri = [get_nume_brand(get_hue_mediu(img, box)) for box in boxes]
    counter = Counter(branduri)

    if not counter: return "-"

    dominant, count = counter.most_common(1)[0]
    procent = (count / len(boxes)) * 100

    if procent > 50:
        return f"{dominant} ({int(procent)}%)"
    return "Mixt / Asortat"


@dataclass
class ProductFeatures:
    box: object
    cx: float
    cy: float
    w: float
    h: float
    area: float
    hue: float
    brand: str
    conf: float


def _safe_median(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else 0.0


def _circular_hue_distance(h1: float, h2: float) -> float:
    """Hue is circular in [0,180). Return shortest distance."""
    if h1 < 0 or h2 < 0:
        return 180.0
    diff = abs(h1 - h2)
    return float(min(diff, 180 - diff))


def extract_product_features(img_path, boxes):
    img = cv2.imread(img_path)
    feats = []
    for box in boxes:
        x, y, w, h = box.xywh[0]
        x = float(x.item()); y = float(y.item()); w = float(w.item()); h = float(h.item())
        conf = float(box.conf[0].item()) if hasattr(box, 'conf') else 0.0
        hue = get_hue_mediu(img, box)
        brand = get_nume_brand(hue)
        feats.append(ProductFeatures(
            box=box,
            cx=x,
            cy=y,
            w=w,
            h=h,
            area=w*h,
            hue=float(hue),
            brand=brand,
            conf=conf,
        ))
    return feats


def group_boxes_into_shelves(features, *, min_items_per_shelf=1):
    """Group detections into shelves using vertical proximity (cy).

    Deterministic output:
    - Sort by cy
    - Start a new shelf when cy jump exceeds adaptive threshold
    - Return shelves ordered top->bottom
    """
    if not features:
        return []

    features_sorted = sorted(features, key=lambda f: f.cy)
    median_h = _safe_median([f.h for f in features_sorted])
    threshold = max(20.0, 0.55 * median_h)  # adaptive but with a stable lower bound

    shelves = []
    current = [features_sorted[0]]

    for f in features_sorted[1:]:
        prev = current[-1]
        if (f.cy - prev.cy) <= threshold:
            current.append(f)
        else:
            shelves.append(current)
            current = [f]
    shelves.append(current)

    # Optional: merge tiny shelves if needed (guard for noisy splits)
    merged = []
    for s in shelves:
        if merged and len(s) < min_items_per_shelf:
            merged[-1].extend(s)
        else:
            merged.append(s)

    # Ensure top->bottom ordering by mean cy
    merged.sort(key=lambda shelf: float(np.mean([f.cy for f in shelf])))
    return merged


def intruder_scores_for_shelf(items, *, dominant_threshold=0.60):
    """Return (message, intruder_boxes) using a multi-factor score.

    Score components:
    - color_outlier: distance from shelf dominant hue
    - size_outlier: deviation in bbox area
    - pos_outlier: x-position deviation
    - low_conf_penalty: higher when YOLO confidence is small
    """
    if len(items) < 3:
        return "Prea puține produse", []

    # Dominant color (based on brand buckets) for human-readable explanation
    valid_brands = [it.brand for it in items if it.brand != "Necunoscut"]
    dominant_brand = None
    is_uniform = False
    if valid_brands:
        c = Counter(valid_brands)
        top_brand, count = c.most_common(1)[0]
        if (count / len(valid_brands)) >= dominant_threshold:
            dominant_brand = top_brand
            is_uniform = True

    # Robust stats
    areas = np.array([it.area for it in items], dtype=float)
    xs = np.array([it.cx for it in items], dtype=float)
    hues = [it.hue for it in items if it.hue != -1]

    area_med = float(np.median(areas))
    area_std = float(np.std(areas)) if len(areas) >= 2 else 0.0
    x_med = float(np.median(xs))
    x_std = float(np.std(xs)) if len(xs) >= 2 else 0.0
    hue_med = float(np.median(hues)) if hues else -1.0

    intruders = []
    intruders_details = []

    for it in items:
        # 1) color outlier
        color_dist = _circular_hue_distance(it.hue, hue_med)
        color_outlier = min(1.0, max(0.0, (color_dist - 20.0) / 40.0))  # 0 until ~20deg

        # If shelf is uniform, enforce also brand mismatch
        if is_uniform and it.brand != "Necunoscut" and it.brand != dominant_brand:
            color_outlier = max(color_outlier, 0.8)

        # 2) size outlier (area)
        if area_std > 1e-6:
            z_area = abs((it.area - area_med) / area_std)
        else:
            z_area = 0.0
        size_outlier = min(1.0, z_area / 3.0)

        # 3) position outlier (x)
        if x_std > 1e-6:
            z_x = abs((it.cx - x_med) / x_std)
        else:
            z_x = 0.0
        pos_outlier = min(1.0, z_x / 3.0)

        # 4) confidence penalty (low conf -> more penalty)
        low_conf = min(1.0, max(0.0, (0.60 - it.conf) / 0.60))

        intruder_score = 0.50 * color_outlier + 0.20 * size_outlier + 0.20 * pos_outlier + 0.10 * low_conf

        # Decision threshold
        if intruder_score >= 0.55:
            intruders.append(it.box)
            intruders_details.append({
                "label": it.brand,
                "score": float(intruder_score),
                "conf": float(it.conf),
            })

    descriere = f"Dominat de {dominant_brand}" if is_uniform else "Raft Mixt"
    nr_intrusi = len(intruders)

    if nr_intrusi > 0:
        procent_intrusi = (nr_intrusi / len(items)) * 100
        return f"🚨 {nr_intrusi} ({int(procent_intrusi)}%) INTRUȘI - {descriere}", intruders, intruders_details
    return f"✅ Uniform ({descriere})", [], []


# --- 4. SORTARE DINAMICĂ RAFTURI ---

def sorteaza_si_analizeaza(results, img_path):
    boxes = results[0].boxes
    if not boxes:
        return []

    features = extract_product_features(img_path, boxes)
    shelves = group_boxes_into_shelves(features)

    raport_final = []

    for i, shelf_items in enumerate(shelves):
        boxes_reale = [it.box for it in shelf_items]
        numar = len(boxes_reale)
        nume = f"Raftul {i + 1}"  # numerotare de sus în jos

        # intruși (logică îmbunătățită)
        msg_intrusi, lista_intrusi, intrusi_detalii = intruder_scores_for_shelf(shelf_items)

        # dominant brand (culoare) pentru UI
        info_dominant = analizeaza_dominanta(img_path, boxes_reale)

        # dezordine
        dezordine = detecteaza_dezordine(boxes_reale)

        if numar == 0:
            status = "🔴 GOL"
        elif numar < 3:
            status = "⚠️ REAPROVIZIONARE"
        else:
            status = "✅ OPTIM"

        raport_final.append({
            "nume": nume,
            "cantitate": numar,
            "status": status,
            "dezordine": dezordine,
            "intrusi": msg_intrusi,
            "dominant": info_dominant,
            "box_intrusi": lista_intrusi,
            "intrusi_detalii": intrusi_detalii,
        })

    return raport_final


# --- 5. RUTE & DESENARE ---

def genereaza_heatmap(results, filename):
    boxes = results[0].boxes
    if not boxes: return filename
    h, w = results[0].orig_shape[:2]
    heatmap = np.zeros((h, w))
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        heatmap[y1:y2, x1:x2] += 1

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap, cmap="hot", cbar=False, xticklabels=False, yticklabels=False)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return filename


def _bbox_xyxy(box):
    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
    return x1, y1, x2, y2


def _bbox_iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0


def _map_boxes_to_shelf_ids(features, shelves):
    shelf_id = {}
    for i, shelf_items in enumerate(shelves):
        sid = f"S{i + 1}"
        for it in shelf_items:
            shelf_id[it.box] = sid
    return shelf_id


def _shelf_color(sid: str):
    """Return BGR color for a shelf id (S1, S2, ...)."""
    palette = [
        (255, 99, 71),    # tomato
        (30, 144, 255),   # dodgerblue
        (50, 205, 50),    # limegreen
        (255, 215, 0),    # gold
        (138, 43, 226),   # blueviolet
        (0, 206, 209),    # darkturquoise
        (255, 105, 180),  # hotpink
        (255, 165, 0),    # orange
    ]
    try:
        idx = max(1, int(sid.replace('S', ''))) - 1
    except Exception:
        idx = 0
    return palette[idx % len(palette)]


def _draw_shelf_separators(img, shelves, *, color=(255, 255, 255)):
    """Draw horizontal separators between shelves, based on mean y of each shelf."""
    if not shelves or len(shelves) < 2:
        return
    # mean cy for each shelf
    means = [float(np.mean([it.cy for it in shelf])) for shelf in shelves]
    means.sort()
    # separators at midpoints
    for a, b in zip(means, means[1:]):
        y = int((a + b) / 2.0)
        cv2.line(img, (0, y), (img.shape[1], y), color, 2)


def _draw_shelf_legend(img, shelves):
    """Small legend top-left: S# + color."""
    x0, y0 = 15, 20
    for i in range(min(len(shelves), 8)):
        sid = f"S{i+1}"
        c = _shelf_color(sid)
        y = y0 + i * 22
        cv2.rectangle(img, (x0, y - 12), (x0 + 16, y + 4), c, -1)
        cv2.putText(img, sid, (x0 + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def deseneaza_rezultat_custom(img_path, results, raport_rafturi):
    img = cv2.imread(img_path)

    intruder_boxes = []
    intruder_score_by_box = []
    for r in raport_rafturi:
        boxes_r = r.get('box_intrusi', [])
        det_r = r.get('intrusi_detalii', [])
        intruder_boxes.extend(boxes_r)
        for i, b in enumerate(boxes_r):
            score = None
            if i < len(det_r):
                score = det_r[i].get('score')
            intruder_score_by_box.append((b, float(score) if score is not None else None))

    # Recompute shelves for stable shelf-id overlay
    features = extract_product_features(img_path, results[0].boxes)
    shelves = group_boxes_into_shelves(features)
    shelf_id_by_box = _map_boxes_to_shelf_ids(features, shelves)

    # Draw separators + legend
    _draw_shelf_separators(img, shelves)
    _draw_shelf_legend(img, shelves)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Determine shelf id using IoU matching between this box and grouped features
        sid = None
        best_iou = 0.0
        cur_xyxy = _bbox_xyxy(box)
        for it in features:
            iou = _bbox_iou_xyxy(cur_xyxy, _bbox_xyxy(it.box))
            if iou > best_iou:
                best_iou = iou
                sid = shelf_id_by_box.get(it.box)

        # Determine if intruder + best score (IoU match)
        is_intrus = False
        intr_score = None
        best_intr_iou = 0.0
        best_intr_score = None
        for b, sc in intruder_score_by_box:
            iou = _bbox_iou_xyxy(cur_xyxy, _bbox_xyxy(b))
            if iou > best_intr_iou:
                best_intr_iou = iou
                best_intr_score = sc
        if best_intr_iou >= 0.90:
            is_intrus = True
            intr_score = best_intr_score

        # Colors: intrus red, else shelf color
        shelf_color = _shelf_color(sid or 'S1')
        color = (0, 0, 255) if is_intrus else shelf_color

        conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0

        base_label = f"{sid or 'S?'} | {conf:.2f}"
        if is_intrus:
            if intr_score is not None:
                base_label = f"INTRUS {base_label} | score {intr_score:.2f}"
            else:
                base_label = f"INTRUS {base_label}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if not is_intrus else 4)

        # Draw label box
        (tw, th), _ = cv2.getTextSize(base_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, base_label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if is_intrus:
            cv2.putText(img, "!", (x1, max(15, y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    out_name = 'res_' + os.path.basename(img_path)
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
    # Guard: img can be None if file read failed
    if img is not None:
        cv2.imwrite(out_path, img)
    return out_name


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '' and model:
            filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            results = model(path)
            raport = sorteaza_si_analizeaza(results, path)
            res_img = deseneaza_rezultat_custom(path, results, raport)
            heat_img = genereaza_heatmap(results, 'heat_' + filename)

            return render_template('index.html', original=filename, result=res_img, heatmap=heat_img, raport=raport)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)