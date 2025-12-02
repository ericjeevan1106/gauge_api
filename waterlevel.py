import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import easyocr
from PIL import Image


# ==========================
# Helper functions
# ==========================

def clean_text(s: str) -> str:
    """Keep only digits from text."""
    return "".join(ch for ch in s if ch.isdigit())


def count_loops(bin_img):
    """
    Rough shape feature: count 'loops' by finding contours.
    Used as fallback digit classifier if OCR fails.
    """
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    loops = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        # if contour has child, treat as loop
        loops += 1

    if loops == 0:
        return 0
    if loops == 1:
        return 1
    if loops == 2:
        return 2
    return 3


def classify_digit_shape(bin_img):
    """
    Extremely simple digit shape classifier – fallback if OCR is bad.

    Uses:
    - loops (0/1/2)
    - line structure via HoughLinesP
    """
    h, w = bin_img.shape

    loops = count_loops(bin_img)
    if loops >= 2:
        return 8
    if loops == 1:
        # could be 0, 6, 9
        # check upper vs lower half brightness
        upper = bin_img[: h // 2, :]
        lower = bin_img[h // 2 :, :]
        if np.sum(upper) < np.sum(lower):
            return 9
        else:
            return 6

    # loops == 0 → use edges & stroke count
    edges = cv2.Canny(bin_img, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=10,
        minLineLength=10, maxLineGap=3
    )
    line_count = 0 if lines is None else len(lines)

    # Very rough rules (good enough as backup)
    if line_count <= 2:
        return 1
    if 2 < line_count <= 4:
        return 7
    if 4 < line_count <= 7:
        return 4
    # fallback
    return None


def choose_best_digit(easy_text, tesser_text, shape_digit):
    """
    Decide final digit from:
      - EasyOCR text
      - Tesseract text
      - Shape-based digit
    Priority:
      1) EasyOCR (if clean and 1 digit)
      2) Tesseract
      3) Shape-based
    """
    easy_clean = clean_text(easy_text or "")
    tess_clean = clean_text(tesser_text or "")

    if len(easy_clean) == 1:
        return int(easy_clean)

    if len(tess_clean) == 1:
        return int(tess_clean)

    if shape_digit is not None:
        return int(shape_digit)

    # last fallback: if multiple digits, pick first
    if len(easy_clean) > 0:
        return int(easy_clean[0])
    if len(tess_clean) > 0:
        return int(tess_clean[0])

    return None


# ==========================
# Main Gauge Model
# ==========================

class GaugeModel:
    def __init__(self, weight_path: str):
        # YOLOv8 model for ruler + numbers
        self.yolo = YOLO(weight_path)

        # EasyOCR reader
        self.reader = easyocr.Reader(['en'], gpu=False)

        # Tesseract (optional – if present in system PATH)
        # On Windows you may set:
        # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        # On Linux (Render) usually no need to set manually.
        try:
            _ = pytesseract.get_tesseract_version()
            self.has_tesseract = True
        except Exception:
            self.has_tesseract = False

    # --------------------------
    # 1) YOLO detection
    # --------------------------
    def _detect_boxes(self, image_path):
        results = self.yolo.predict(
            source=image_path,
            save=False,
            conf=0.4
        )
        boxes = results[0].boxes

        ruler_box = None
        value_boxes = []

        for box in boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0:        # ruler
                ruler_box = (x1, y1, x2, y2)
            elif cls == 1:      # digit area
                value_boxes.append((x1, y1, x2, y2))

        return ruler_box, value_boxes

    # --------------------------
    # 2) OCR + shape based digits
    # --------------------------
    def _read_digit_box(self, img, box):
        vx1, vy1, vx2, vy2 = box
        crop = img[vy1:vy2, vx1:vx2]

        # Preprocess
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # For shape classifier
        _, bin_img = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # EasyOCR
        easy_result = self.reader.readtext(gray)
        easy_text = easy_result[0][-2] if len(easy_result) > 0 else ""

        # Tesseract (if available)
        tess_text = ""
        if self.has_tesseract:
            try:
                tess_text = pytesseract.image_to_string(
                    gray,
                    config='--psm 7 -c tessedit_char_whitelist=0123456789'
                )
            except Exception:
                tess_text = ""

        # Shape-based
        shape_digit = classify_digit_shape(bin_img)

        final_digit = choose_best_digit(easy_text, tess_text, shape_digit)
        center_y = (vy1 + vy2) // 2

        return final_digit, center_y

    # --------------------------
    # 3) Robust waterline detection
    # --------------------------
    def _detect_waterline(self, img, ruler_box, digit_pixels):
        """
        Use Canny + HoughLinesP inside bottom portion of ruler.
        Fallback: row-wise edge density scan.
        Returns water_y_global (pixel in full image coords) or None.
        """
        rx1, ry1, rx2, ry2 = ruler_box
        ruler_crop = img[ry1:ry2, rx1:rx2]
        H, W = ruler_crop.shape[:2]

        # use bottom ~45% of ruler for waterline
        cut_start = int(H * 0.55)
        bottom_region = ruler_crop[cut_start:, :]

        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)

        # Canny
        edges = cv2.Canny(blur, 30, 120)

        # little closing to connect line segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_closed = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        # HoughLinesP
        lines = cv2.HoughLinesP(
            edges_closed,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=50,
            maxLineGap=10
        )

        water_y_ruler = None
        candidates = []

        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                # horizontal check
                if abs(y1 - y2) > 5:
                    continue

                y = (y1 + y2) // 2

                # ignore very bottom few pixels (reflection)
                if y > bottom_region.shape[0] - 5:
                    continue

                candidates.append((x1, y, x2, y))

        if len(candidates) > 0:
            # choose line with largest y (closest to bottom of ruler)
            best = max(candidates, key=lambda c: c[1])
            water_y_ruler = best[1] + cut_start
        else:
            # --- Fallback: row-wise edge density scan ---
            row_strength = edges_closed.sum(axis=1)  # per-row edge intensity
            if np.max(row_strength) > 0:
                # ignore very top/bottom of bottom_region
                y_min = 5
                y_max = bottom_region.shape[0] - 5
                sub = row_strength[y_min:y_max]
                best_row_local = int(np.argmax(sub)) + y_min
                water_y_ruler = best_row_local + cut_start

        if water_y_ruler is None:
            return None

        # Convert to full image coordinate
        water_y_global = ry1 + water_y_ruler

        # Optional: clamp to digit region if any digits exist
        if len(digit_pixels) > 0:
            min_digit = min(digit_pixels)
            max_digit = max(digit_pixels)
            water_y_global = int(np.clip(water_y_global, min_digit, max_digit))

        return water_y_global

    # --------------------------
    # 4) Pixel → real water level
    # --------------------------
    def _pixel_to_level(self, water_y, digit_values, digit_pixels):
        """
        Map water_y pixel to a water level in cm using
        linear interpolation between nearest digit marks.
        """
        if water_y is None or len(digit_values) == 0:
            return None

        pix = np.array(digit_pixels, dtype=float)
        vals = np.array(digit_values, dtype=float)

        # sort from top (small y) to bottom (large y)
        order = np.argsort(pix)
        pix = pix[order]
        vals = vals[order]

        # exact match (within 2 px)
        diffs = np.abs(pix - water_y)
        if np.min(diffs) <= 2:
            return float(vals[np.argmin(diffs)])

        # find segment where water_y lies between two marks
        above = pix <= water_y
        if not np.any(above) or np.all(above):
            # outside digit region → clamp to nearest
            return float(vals[np.argmin(diffs)])

        idx_upper = np.max(np.where(above)[0])
        idx_lower = idx_upper + 1

        y1, y2 = pix[idx_upper], pix[idx_lower]
        v1, v2 = vals[idx_upper], vals[idx_lower]

        if y2 == y1:
            return float(v1)

        t = (water_y - y1) / (y2 - y1)
        level = v1 + t * (v2 - v1)
        return float(level)

    # --------------------------
    # Public API
    # --------------------------
    def process(self, image_path: str):
        """
        Full pipeline:
        YOLO → OCR → waterline → water level.
        Returns dictionary (JSON serialisable).
        """
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not load image"}

        # 1) YOLO detect
        ruler_box, value_boxes = self._detect_boxes(image_path)
        if ruler_box is None:
            return {"error": "Ruler not detected"}

        # 2) OCR / shape on each value box
        digit_values = []
        digit_pixels = []

        for box in value_boxes:
            digit, cy = self._read_digit_box(img, box)
            if digit is not None:
                digit_values.append(int(digit))
                digit_pixels.append(int(cy))

        if len(digit_values) == 0:
            return {"error": "No digits detected"}

        # 3) Waterline
        water_y = self._detect_waterline(img, ruler_box, digit_pixels)
        if water_y is None:
            return {
                "error": "Waterline not detected",
                "digit_values": digit_values,
                "digit_pixels": digit_pixels,
            }

        # 4) Pixel → level
        level_cm = self._pixel_to_level(
            water_y, digit_values, digit_pixels
        )
        if level_cm is None:
            return {
                "error": "Could not compute water level",
                "digit_values": digit_values,
                "digit_pixels": digit_pixels,
                "waterline_pixel": int(water_y),
            }

        return {
            "water_level_cm": level_cm,
            "digit_values": digit_values,
            "digit_pixels": digit_pixels,
            "waterline_pixel": int(water_y),
            "ruler_box": ruler_box,
            "value_boxes": value_boxes,
        }
