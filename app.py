import os
import io
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template, url_for, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.units import mm

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

IMG_SIZE = 224
CLASS_NAMES = ["Normal", "Foot Ulcer (DFU)", "Gangrene"]
CONFIDENCE_THRESHOLD = 55.0

RECOMMENDATIONS = {
    "Normal": {
        "urgency": "low",
        "urgency_label": "No Immediate Concern",
        "summary": "No signs of ulcer or gangrene detected. Continue preventive foot care.",
        "steps": [
            "Inspect your feet daily for cuts, blisters, redness, or swelling — especially if you have diabetes.",
            "Wash feet daily with lukewarm water and dry thoroughly, especially between toes.",
            "Moisturize dry skin, but avoid applying lotion between the toes.",
            "Wear well-fitting, protective footwear — avoid walking barefoot.",
            "Keep blood sugar levels well-controlled if you are diabetic.",
            "Schedule regular foot checkups with a doctor or podiatrist, particularly if diabetic.",
            "If you notice a small cut or scrape: clean it gently with mild soap and water, pat dry, apply an antiseptic, and cover with a clean bandage.",
            "Change the bandage daily and watch for signs of infection (increasing redness, swelling, warmth, or pus) — see a doctor if these appear.",
            "If diabetic, treat even minor cuts seriously and consult a doctor promptly, since healing is slower and infection risk is higher."
        ]
    },
    "Foot Ulcer (DFU)": {
        "urgency": "medium",
        "urgency_label": "Medical Attention Recommended",
        "summary": "Signs consistent with a foot ulcer were detected. Early care significantly reduces risk of complications.",
        "steps": [
            "See a doctor or podiatrist promptly — early treatment prevents progression to infection or gangrene.",
            "Keep the wound clean and covered with a sterile, non-adherent dressing.",
            "Avoid putting weight or pressure on the affected foot as much as possible (off-loading).",
            "Do not attempt to cut away dead skin or self-treat the wound at home.",
            "Watch for warning signs: increasing redness, swelling, warmth, odor, or fever — seek care immediately if these occur.",
            "If diabetic, tightly manage blood sugar levels, as poor control slows healing significantly.",
            "Avoid smoking, which impairs circulation and wound healing."
        ]
    },
    "Gangrene": {
        "urgency": "high",
        "urgency_label": "Urgent Medical Attention Needed",
        "summary": "Signs consistent with gangrene were detected. This is a serious condition that can progress rapidly.",
        "steps": [
            "Seek emergency medical care immediately — go to the nearest hospital or emergency room, do not wait.",
            "Do not attempt to treat, clean aggressively, or apply home remedies to the affected area.",
            "Keep the affected limb still and avoid putting weight on it while seeking care.",
            "Do not apply heat, ice, or tight bandages to the area.",
            "If you notice fever, rapid heartbeat, confusion, or spreading discoloration, this may indicate a spreading infection — treat as a medical emergency.",
            "Bring a list of current medications and medical history (especially diabetes) when seeking care, as this affects treatment decisions."
        ]
    }
}

WAGNER_STAGES = [
    {"grade": "0", "desc": "Pre-ulcerative lesion or healed ulcer; skin intact"},
    {"grade": "1", "desc": "Superficial ulcer, not extending into deeper tissue"},
    {"grade": "2", "desc": "Deeper ulcer reaching tendon, ligament, joint capsule, or fascia; no abscess or bone infection"},
    {"grade": "3", "desc": "Deep ulcer with abscess, osteomyelitis, or joint infection"},
    {"grade": "4", "desc": "Localized gangrene (e.g., confined to toe or forefoot)"},
    {"grade": "5", "desc": "Extensive gangrene involving the entire foot"},
]

GANGRENE_TYPES = [
    {"type": "Dry Gangrene", "desc": "Caused by reduced blood flow. Tissue becomes dry, shriveled, and dark brown/black. Typically progresses more slowly."},
    {"type": "Wet Gangrene", "desc": "Involves bacterial infection. Tissue appears swollen and moist, may have blisters or foul odor. Spreads rapidly — a medical emergency."},
    {"type": "Gas Gangrene", "desc": "Caused by Clostridium bacteria producing gas within tissue. Rapidly life-threatening; requires immediate emergency treatment."},
]

print("Loading model...")
model = load_model("model/final_model.keras")
print("Model loaded successfully.")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype="float32")
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def is_likely_skin_image(filepath, min_skin_ratio=0.15):
    """
    Quick heuristic check: does the image contain enough skin-toned pixels
    to plausibly be a photo of skin? Filters out obviously unrelated
    images (objects, scenery, flowers, etc.) before running the model.
    """
    img = cv2.imread(filepath)
    if img is None:
        return False

    img = cv2.resize(img, (300, 300))
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)

    skin_ratio = np.sum(skin_mask > 0) / (skin_mask.shape[0] * skin_mask.shape[1])
    return skin_ratio >= min_skin_ratio


def analyze_severity_heuristic(filepath, predicted_class):
    """
    Experimental, NON-CLINICAL heuristic based on simple color/darkness analysis
    of the image. This is NOT a validated staging tool -- it is a rough,
    transparent estimate meant for demonstration purposes only.
    """
    if predicted_class == "Normal":
        return None

    img = cv2.imread(filepath)
    img = cv2.resize(img, (300, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    dark_ratio = float(np.mean(v < 60))
    avg_saturation = float(np.mean(s))

    if dark_ratio < 0.10:
        severity = "Mild"
    elif dark_ratio < 0.25:
        severity = "Moderate"
    else:
        severity = "Severe"

    result = {
        "severity": severity,
        "dark_ratio_pct": round(dark_ratio * 100, 1),
    }

    if predicted_class == "Gangrene":
        if avg_saturation < 60:
            result["type_hint"] = "Color pattern leans closer to Dry Gangrene (darker, less moist appearance)"
        else:
            result["type_hint"] = "Color pattern leans closer to Wet Gangrene (more discoloration/moisture indicators)"

    return result


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file selected.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    if not allowed_file(file.filename):
        return render_template("index.html", error="Please upload a PNG or JPG image.")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Layer 1: reject images that don't look like skin at all (objects, scenery, etc.)
    if not is_likely_skin_image(filepath):
        return render_template(
            "index.html",
            error="This doesn't appear to be a photo of a foot or skin area. Please upload a clear image of the affected foot."
        )

    img = prepare_image(filepath)
    predictions = model.predict(img)[0]
    predicted_class = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[predicted_class]
    confidence = float(predictions[predicted_class]) * 100

    # Layer 2: reject low-confidence / ambiguous predictions
    if confidence < CONFIDENCE_THRESHOLD:
        return render_template(
            "index.html",
            error=f"The model isn't confident enough about this image ({confidence:.1f}% confidence). "
                  f"Please upload a clearer, well-lit photo of the affected foot area."
        )

    all_confidences = {
        CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    recommendation = RECOMMENDATIONS[predicted_label]
    severity_info = analyze_severity_heuristic(filepath, predicted_label)

    reference_stages = None
    if predicted_label == "Foot Ulcer (DFU)":
        reference_stages = {"title": "Wagner Ulcer Classification (Reference)", "items": WAGNER_STAGES, "key": "grade"}
    elif predicted_label == "Gangrene":
        reference_stages = {"title": "Common Gangrene Types (Reference)", "items": GANGRENE_TYPES, "key": "type"}

    return render_template(
        "result.html",
        prediction=predicted_label,
        confidence=round(confidence, 2),
        image_path=url_for("static", filename=f"uploads/{filename}"),
        image_filename=filename,
        all_confidences=all_confidences,
        timestamp=timestamp,
        recommendation=recommendation,
        severity_info=severity_info,
        reference_stages=reference_stages
    )


@app.route("/download_report", methods=["GET"])
def download_report():
    filename = request.args.get("filename", "")
    prediction = request.args.get("prediction", "")
    confidence = request.args.get("confidence", "")
    timestamp = request.args.get("timestamp", "")
    normal_pct = request.args.get("normal", "0")
    dfu_pct = request.args.get("dfu", "0")
    gangrene_pct = request.args.get("gangrene", "0")

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    buffer = io.BytesIO()
    c = pdfcanvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFillColorRGB(0.12, 0.24, 0.45)
    c.rect(0, height - 30 * mm, width, 30 * mm, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, height - 18 * mm, "Foot Ulcer & Gangrene Detection Report")
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, height - 25 * mm, "Generated by Deep Learning Diagnostic System (SE-ResNet50)")

    y = height - 45 * mm

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 11)
    c.drawString(20 * mm, y, f"Report Generated: {timestamp}")
    y -= 10 * mm

    if os.path.exists(filepath):
        try:
            img_display_size = 80 * mm
            c.drawImage(
                filepath, 20 * mm, y - img_display_size,
                width=img_display_size, height=img_display_size,
                preserveAspectRatio=True, anchor='n'
            )
            y -= (img_display_size + 10 * mm)
        except Exception:
            y -= 10 * mm

    c.setFillColorRGB(0.9, 0.95, 1.0)
    c.roundRect(20 * mm, y - 20 * mm, width - 40 * mm, 18 * mm, 3 * mm, fill=1, stroke=0)
    c.setFillColorRGB(0.12, 0.24, 0.45)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(25 * mm, y - 8 * mm, f"Prediction: {prediction}")
    c.setFont("Helvetica", 11)
    c.drawString(25 * mm, y - 15 * mm, f"Confidence: {confidence}%")
    y -= 30 * mm

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Confidence Breakdown:")
    y -= 8 * mm
    c.setFont("Helvetica", 11)
    c.drawString(25 * mm, y, f"Normal: {normal_pct}%")
    y -= 7 * mm
    c.drawString(25 * mm, y, f"Foot Ulcer (DFU): {dfu_pct}%")
    y -= 7 * mm
    c.drawString(25 * mm, y, f"Gangrene: {gangrene_pct}%")
    y -= 15 * mm

    c.setFont("Helvetica-Oblique", 9)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    disclaimer = (
        "Disclaimer: This report is generated by an academic project tool for "
        "demonstration purposes only and is not a certified medical diagnostic device. "
        "Always consult a qualified healthcare professional for medical diagnosis."
    )
    c.drawString(20 * mm, y, disclaimer[:95])
    y -= 5 * mm
    c.drawString(20 * mm, y, disclaimer[95:])

    c.showPage()
    c.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"foot_ulcer_report_{filename.rsplit('.', 1)[0]}.pdf",
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run(debug=True)