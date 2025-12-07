from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import re
import tempfile
import os

# -------------------------------
#  INITIALIZE MODELS ONCE
# -------------------------------
# YOLO model (trained)
yolo_model = YOLO("best.pt")   # best.pt must be in same folder as app.py

# EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

app = Flask(__name__)


# -------------------------------
#  UTIL FUNCTIONS (from cell 12)
# -------------------------------
def clean_text(t: str) -> str:
    d = re.sub(r"\D", "", t)
    return d[:2] if len(d) > 2 else d

def count_loops(bin_img):
    inv = 255 - bin_img
    conts, _ = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    loops = 0
    for c in conts:
        area = cv2.contourArea(c)
        if 30 < area < 5000:
            loops += 1
    return loops

def classify_digit_shape(bin_img):
    h, w = bin_img.shape
    ratio = h / max(w, 1)
    loops = count_loops(bin_img)

    # --- Loop-based rules ---
    if loops == 2:
        return 8
    if loops == 1:
        upper = bin_img[:h // 2, :]
        lower = bin_img[h // 2:, :]
        if count_loops(upper) == 1:
            return 9
        else:
            return 6

    # --- Stroke & structure rules ---
    edges = cv2.Canny(bin_img, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=20,
        minLineLength=10,
        maxLineGap=3
    )
    line_count = 0 if lines is None else len(lines)

    # 1 (thin & tall)
    if ratio > 1.8 and w < 40:
        return 1

    # 7 (horizontal + diagonal)
    if line_count >= 2 and loops == 0 and ratio < 1.5:
        return 7

    # 4 (crossing strokes)
    if line_count >= 3:
        return 4

    # Distinguish 2,3,5 (very rough)
    if ratio < 1.4:
        if np.sum(bin_img[:h // 2]) < np.sum(bin_img[h // 2:]):
            return 2
        else:
            return 3

    return None


def process_image(image_path):
    """
    Full pipeline:
    1) YOLO → ruler + digit boxes
    2) Distance between bottom green & bottom blue box
    3) EasyOCR + shape AI → digit values
    4) Interpolation + distance correction
    Returns: (final_water_level, distance_pixels)
    """

    # ---------------------------
    # 1. Read image
    # ---------------------------
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Failed to read image.")

    # ---------------------------
    # 2. YOLO detection
    # ---------------------------
    results = yolo_model.predict(
        source=image_path,
        conf=0.4,
        save=False
    )[0]

    ruler_box = None
    value_boxes = []

    for box in results.boxes:
        cls = int(box.cls[0])  # 0 = ruler, 1 = value
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:
            ruler_box = (x1, y1, x2, y2)
        elif cls == 1:
            value_boxes.append((x1, y1, x2, y2))

    if ruler_box is None:
        raise Exception("YOLO could not detect the ruler.")
    if len(value_boxes) == 0:
        raise Exception("YOLO did not detect any value boxes.")

    # ---------------------------
    # 3. Distance between bottom
    # ---------------------------
    bx1, by1, bx2, by2 = ruler_box
    blue_bottom = by2
    lowest_green_bottom = max(vy2 for (_, _, _, vy2) in value_boxes)
    distance_pixels = blue_bottom - lowest_green_bottom

    # ---------------------------
    # 4. OCR on each value box
    # ---------------------------
    digit_values = []
    digit_pixels = []

    for (vx1, vy1, vx2, vy2) in value_boxes:
        pad = 5
        y1c = max(0, vy1 - pad)
        y2c = min(img.shape[0], vy2 + pad)
        x1c = max(0, vx1 - pad)
        x2c = min(img.shape[1], vx2 + pad)
        crop = img[y1c:y2c, x1c:x2c]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, bin_img = cv2.threshold(
            big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        best = None
        best_prob = 0.0

        # EasyOCR
        easy = reader.readtext(cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB))
        for _, text, prob in easy:
            c = clean_text(text)
            if c.isdigit() and prob > best_prob:
                best = c
                best_prob = prob

        # Shape fallback
        if not best:
            predicted = classify_digit_shape(bin_img)
            if predicted is not None:
                best = str(predicted)

        if best and best.isdigit():
            val = int(best)
            cy = (vy1 + vy2) // 2
            digit_values.append(val)
            digit_pixels.append(cy)

    if len(digit_values) < 1:
        raise Exception("OCR/shape failed to read any digits.")

    # ---------------------------
    # 5. Waterline Y (lowest box)
    # ---------------------------
    last_box = max(value_boxes, key=lambda b: b[3])
    _, _, _, vy2 = last_box
    waterline_y = vy2

    # ---------------------------
    # 6. Pixel → real water level
    #     (your cell 14 logic)
    # ---------------------------
    vals = np.array(digit_values, dtype=float)
    pix = np.array(digit_pixels, dtype=float)

    if len(vals) < 2:
        # If only one digit, just use that value.
        water_level = float(vals[0])
        return water_level, float(distance_pixels)

    idx = np.argsort(pix)
    vals_sorted = vals[idx]
    pix_sorted = pix[idx]

    if waterline_y in pix_sorted:
        water_level = float(
            vals_sorted[np.where(pix_sorted == waterline_y)][0]
        )
    else:
        above_idx = np.where(pix_sorted < waterline_y)[0]
        below_idx = np.where(pix_sorted > waterline_y)[0]

        if len(above_idx) == 0 or len(below_idx) == 0:
            water_level = float(vals_sorted[-1])
        else:
            upper_i = above_idx[-1]
            lower_i = below_idx[0]

            upper_digit = vals_sorted[upper_i]
            lower_digit = vals_sorted[lower_i]

            upper_pix = pix_sorted[upper_i]
            lower_pix = pix_sorted[lower_i]

            ratio = (waterline_y - upper_pix) / (lower_pix - upper_pix)
            water_level = upper_digit + ratio * (lower_digit - upper_digit)

    # ---------------------------
    # 7. Distance-based correction
    # ---------------------------
    dp = distance_pixels
    correction = 0.0

    if dp > 50:
        if 51 <= dp <= 60:
            correction = 0.9
        elif 61 <= dp <= 70:
            correction = 0.8
        elif 71 <= dp <= 80:
            correction = 0.6
        elif 81 <= dp <= 90:
            correction = 0.4
        elif 91 <= dp <= 100:
            correction = 0.2
        water_level = water_level - correction

    return float(water_level), float(distance_pixels)


# -------------------------------
#  FLASK ENDPOINT
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects multipart/form-data with field name: 'image'
    Returns JSON: { "water_level": <float>, "distance_pixels": <float> }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file with key 'image'"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        file.save(tmp.name)
        tmp.close()

        water_level, dp = process_image(tmp.name)

        return jsonify({
            "water_level": water_level,
            "distance_pixels": dp
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.route("/", methods=["GET"])
def health():
    return "Gauge water level API is running", 200


if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=5000)
