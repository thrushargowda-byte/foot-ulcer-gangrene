import os
import uuid
import textwrap
import numpy as np
import cv2
from datetime import datetime
from functools import wraps
from flask import Flask, request, render_template, redirect, url_for, flash, abort, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.units import mm

BASE = os.path.abspath(os.path.dirname(__file__))
IMG_DIR = os.path.join(BASE, "storage", "images")
PDF_DIR = os.path.join(BASE, "storage", "reports")

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "change-this-secret-key"),
    SQLALCHEMY_DATABASE_URI="sqlite:///mediscan.db",
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
)
DOCTOR_CODE = os.environ.get("DOCTOR_CODE", "DOC2026")

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to continue."
login_manager.login_message_category = "error"

IMG_SIZE = 224
CLASS_NAMES = ["Normal", "Foot Ulcer (DFU)", "Gangrene"]
DIAGNOSES = CLASS_NAMES + ["Inconclusive"]
ALLOWED = {"png", "jpg", "jpeg"}
CONFIDENCE_THRESHOLD = 55.0

# ------------------------------------------------------------------ DATABASE
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # "patient" or "doctor"
    specialization = db.Column(db.String(80))
    created_at = db.Column(db.DateTime, default=datetime.now)


class Case(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    image_file = db.Column(db.String(120), nullable=False)
    report_file = db.Column(db.String(120))
    prediction = db.Column(db.String(30), nullable=False)
    confidence = db.Column(db.Float)
    p_normal = db.Column(db.Float)
    p_dfu = db.Column(db.Float)
    p_gangrene = db.Column(db.Float)
    severity = db.Column(db.String(20))
    severity_hint = db.Column(db.String(200))
    patient_note = db.Column(db.Text)
    status = db.Column(db.String(12), default="private")  # private / pending / reviewed
    doctor_diagnosis = db.Column(db.String(40))
    doctor_note = db.Column(db.Text)
    reviewed_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.now)
    patient = db.relationship("User", foreign_keys=[patient_id])
    doctor = db.relationship("User", foreign_keys=[doctor_id])

    @property
    def bars(self):
        return [("Normal", self.p_normal), ("Foot Ulcer (DFU)", self.p_dfu), ("Gangrene", self.p_gangrene)]


with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(uid):
    return db.session.get(User, int(uid))


def role_required(role):
    def deco(f):
        @wraps(f)
        def wrapper(*a, **k):
            if not current_user.is_authenticated:
                return login_manager.unauthorized()
            if current_user.role != role:
                abort(403)
            return f(*a, **k)
        return wrapper
    return deco


def get_case(case_id):
    c = db.session.get(Case, case_id)
    if c is None:
        abort(404)
    owner = current_user.role == "patient" and c.patient_id == current_user.id
    assigned = current_user.role == "doctor" and c.doctor_id == current_user.id
    if not (owner or assigned):
        abort(403)
    return c

# ------------------------------------------------------------------ MODEL + KNOWLEDGE
RECOMMENDATIONS = {
    "Normal": {
        "urgency": "low", "urgency_label": "No Immediate Concern",
        "summary": "No signs of ulcer or gangrene detected. Continue preventive foot care.",
        "steps": [
            "Inspect your feet daily for cuts, blisters, redness, or swelling, especially if you have diabetes.",
            "Wash feet daily with lukewarm water and dry thoroughly, especially between toes.",
            "Moisturize dry skin, but avoid applying lotion between the toes.",
            "Wear well-fitting, protective footwear and avoid walking barefoot.",
            "Keep blood sugar levels well-controlled if you are diabetic.",
            "Schedule regular foot checkups with a doctor or podiatrist, particularly if diabetic.",
            "For a small cut or scrape: clean gently with mild soap and water, pat dry, apply an antiseptic, and cover with a clean bandage.",
            "Change the bandage daily and watch for infection (increasing redness, swelling, warmth, or pus). See a doctor if these appear.",
            "If diabetic, treat even minor cuts seriously and consult a doctor promptly.",
        ]},
    "Foot Ulcer (DFU)": {
        "urgency": "medium", "urgency_label": "Medical Attention Recommended",
        "summary": "Signs consistent with a foot ulcer were detected. Early care significantly reduces the risk of complications.",
        "steps": [
            "See a doctor or podiatrist promptly. Early treatment prevents progression to infection or gangrene.",
            "Keep the wound clean and covered with a sterile, non-adherent dressing.",
            "Avoid putting weight or pressure on the affected foot as much as possible (off-loading).",
            "Do not attempt to cut away dead skin or self-treat the wound at home.",
            "Watch for warning signs: increasing redness, swelling, warmth, odor, or fever. Seek care immediately if these occur.",
            "If diabetic, tightly manage blood sugar levels, as poor control slows healing.",
            "Avoid smoking, which impairs circulation and wound healing.",
        ]},
    "Gangrene": {
        "urgency": "high", "urgency_label": "Urgent Medical Attention Needed",
        "summary": "Signs consistent with gangrene were detected. This is a serious condition that can progress rapidly.",
        "steps": [
            "Seek emergency medical care immediately. Go to the nearest hospital or emergency room.",
            "Do not attempt to treat, clean aggressively, or apply home remedies to the affected area.",
            "Keep the affected limb still and avoid putting weight on it while seeking care.",
            "Do not apply heat, ice, or tight bandages to the area.",
            "Fever, rapid heartbeat, confusion, or spreading discoloration may indicate a spreading infection. Treat as an emergency.",
            "Bring a list of current medications and medical history (especially diabetes) when seeking care.",
        ]},
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
    {"type": "Dry Gangrene", "desc": "Caused by reduced blood flow. Tissue becomes dry, shriveled, and dark brown/black. Usually progresses slowly."},
    {"type": "Wet Gangrene", "desc": "Involves bacterial infection. Tissue is swollen and moist, may blister or smell foul. Spreads rapidly, a medical emergency."},
    {"type": "Gas Gangrene", "desc": "Caused by Clostridium bacteria producing gas within tissue. Rapidly life-threatening; needs immediate emergency treatment."},
]

print("Loading model...")
model = load_model(os.path.join(BASE, "model", "final_model.keras"))
print("Model loaded successfully.")


def prepare_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32")
    return preprocess_input(np.expand_dims(img, 0))


def is_likely_skin_image(path, min_ratio=0.15):
    img = cv2.imread(path)
    if img is None:
        return False
    ycrcb = cv2.cvtColor(cv2.resize(img, (300, 300)), cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([0, 133, 77], np.uint8), np.array([255, 173, 127], np.uint8))
    return np.count_nonzero(mask) / mask.size >= min_ratio


def analyze_severity(path, label):
    """Experimental colour heuristic, NOT a clinical staging tool."""
    if label == "Normal":
        return None
    hsv = cv2.cvtColor(cv2.resize(cv2.imread(path), (300, 300)), cv2.COLOR_BGR2HSV)
    dark = float(np.mean(hsv[:, :, 2] < 60))
    out = {"severity": "Mild" if dark < 0.10 else "Moderate" if dark < 0.25 else "Severe"}
    if label == "Gangrene":
        out["type_hint"] = ("Colour pattern leans closer to Dry Gangrene (darker, less moist appearance)"
                            if float(np.mean(hsv[:, :, 1])) < 60 else
                            "Colour pattern leans closer to Wet Gangrene (more discoloration/moisture indicators)")
    return out

# ------------------------------------------------------------------ PDF REPORT
def make_report(case):
    folder = os.path.join(PDF_DIR, f"patient_{case.patient_id}")
    os.makedirs(folder, exist_ok=True)
    case.report_file = f"case_{case.id}.pdf"
    c = pdfcanvas.Canvas(os.path.join(folder, case.report_file), pagesize=A4)
    w, h = A4
    c.setFillColorRGB(.06, .46, .43); c.rect(0, h - 30 * mm, w, 30 * mm, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, h - 18 * mm, "FootGuard AI - Foot Ulcer & Gangrene Report")
    c.setFont("Helvetica", 10); c.drawString(20 * mm, h - 25 * mm, f"Case #{case.id}  |  {case.created_at:%d %b %Y, %H:%M}")
    y = h - 42 * mm
    c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica", 11)
    c.drawString(20 * mm, y, f"Patient: {case.patient.name}"); y -= 7 * mm
    if case.doctor:
        c.drawString(20 * mm, y, f"Shared with: Dr. {case.doctor.name}"); y -= 7 * mm
    size = 75 * mm
    try:
        c.drawImage(os.path.join(IMG_DIR, f"patient_{case.patient_id}", case.image_file), 20 * mm, y - size,
                    width=size, height=size, preserveAspectRatio=True, anchor="nw")
    except Exception:
        pass
    x = 20 * mm + size + 10 * mm
    c.setFont("Helvetica-Bold", 14); c.drawString(x, y - 8 * mm, case.prediction)
    c.setFont("Helvetica", 11); c.drawString(x, y - 16 * mm, f"Confidence: {case.confidence}%")
    for i, (n, v) in enumerate(case.bars):
        c.drawString(x, y - (26 + i * 7) * mm, f"{n}: {v}%")
    if case.severity:
        c.drawString(x, y - 50 * mm, f"Est. severity (experimental): {case.severity}")
    y -= size + 10 * mm
    if case.reviewed_at:
        c.setFont("Helvetica-Bold", 12); c.drawString(20 * mm, y, f"Doctor review - Dr. {case.doctor.name}"); y -= 7 * mm
        c.setFont("Helvetica", 11); c.drawString(20 * mm, y, f"Diagnosis: {case.doctor_diagnosis}"); y -= 7 * mm
        for line in textwrap.wrap(case.doctor_note or "", 90)[:14]:
            c.drawString(20 * mm, y, line); y -= 5.5 * mm
    c.setFont("Helvetica-Oblique", 8.5); c.setFillColorRGB(.4, .4, .4)
    note = ("Disclaimer: generated by an academic project tool for demonstration only. It is not a certified medical "
            "device. Always consult a qualified healthcare professional for diagnosis and treatment.")
    for i, line in enumerate(textwrap.wrap(note, 110)):
        c.drawString(20 * mm, (22 - i * 4.5) * mm, line)
    c.showPage(); c.save()
    db.session.commit()

# ------------------------------------------------------------------ PUBLIC + AUTH
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        f = request.form
        name, email, pw, role = f.get("name", "").strip(), f.get("email", "").strip().lower(), f.get("password", ""), f.get("role", "patient")
        err = None
        if len(name) < 2: err = "Please enter your name."
        elif "@" not in email: err = "Please enter a valid email."
        elif len(pw) < 6: err = "Password must be at least 6 characters."
        elif role not in ("patient", "doctor"): err = "Invalid role."
        elif User.query.filter_by(email=email).first(): err = "This email is already registered."
        elif role == "doctor" and f.get("code", "") != DOCTOR_CODE: err = "Invalid doctor access code."
        if err:
            flash(err, "error")
            return render_template("register.html", form=f)
        user = User(name=name, email=email, role=role, password_hash=generate_password_hash(pw),
                    specialization=f.get("specialization", "").strip()[:80] or None)
        db.session.add(user); db.session.commit()
        login_user(user)
        flash(f"Welcome, {user.name}!", "success")
        return redirect(url_for("dashboard"))
    return render_template("register.html", form={})


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        user = User.query.filter_by(email=request.form.get("email", "").strip().lower()).first()
        if user and check_password_hash(user.password_hash, request.form.get("password", "")):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Incorrect email or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


@app.errorhandler(403)
@app.errorhandler(404)
def no_access(e):
    flash("You don't have access to that page." if e.code == 403 else "Page not found.", "error")
    return redirect(url_for("dashboard") if current_user.is_authenticated else url_for("home"))

# ------------------------------------------------------------------ DASHBOARD + LISTS
@app.route("/dashboard")
@login_required
def dashboard():
    col = Case.patient_id if current_user.role == "patient" else Case.doctor_id
    q = Case.query.filter(col == current_user.id)
    stats = {"total": q.count(), "pending": q.filter_by(status="pending").count(),
             "reviewed": q.filter_by(status="reviewed").count(),
             "urgent": q.filter_by(prediction="Gangrene").count()}
    recent = q.order_by(Case.created_at.desc()).limit(5).all()
    return render_template("dashboard.html", stats=stats, recent=recent)


@app.route("/cases")
@login_required
def cases():
    col = Case.patient_id if current_user.role == "patient" else Case.doctor_id
    q = Case.query.filter(col == current_user.id)
    status = request.args.get("status")
    if status in ("private", "pending", "reviewed"):
        q = q.filter_by(status=status)
    return render_template("cases.html", cases=q.order_by(Case.created_at.desc()).all(), status=status)

# ------------------------------------------------------------------ PATIENT: SCAN
@app.route("/scan", methods=["GET", "POST"])
@role_required("patient")
def scan():
    doctors = User.query.filter_by(role="doctor").order_by(User.name).all()
    if request.method == "POST":
        f = request.files.get("file")
        ext = f.filename.rsplit(".", 1)[-1].lower() if f and f.filename else ""
        if ext not in ALLOWED:
            flash("Please upload a PNG or JPG image.", "error")
            return redirect(url_for("scan"))
        folder = os.path.join(IMG_DIR, f"patient_{current_user.id}")
        os.makedirs(folder, exist_ok=True)
        name = uuid.uuid4().hex + "." + ext
        path = os.path.join(folder, name)
        f.save(path)
        if not is_likely_skin_image(path):
            os.remove(path)
            flash("This doesn't look like a photo of skin. Please upload a clear photo of the affected foot.", "error")
            return redirect(url_for("scan"))
        probs = model.predict(prepare_image(path), verbose=0)[0]
        i = int(np.argmax(probs))
        label, conf = CLASS_NAMES[i], float(probs[i]) * 100
        if conf < CONFIDENCE_THRESHOLD:
            os.remove(path)
            flash(f"The model is not confident enough ({conf:.1f}%). Please upload a clearer, well-lit photo.", "error")
            return redirect(url_for("scan"))
        sev = analyze_severity(path, label) or {}
        did = request.form.get("doctor_id", type=int)
        doctor = db.session.get(User, did) if did else None
        if doctor and doctor.role != "doctor":
            doctor = None
        case = Case(patient_id=current_user.id, doctor_id=doctor.id if doctor else None, image_file=name,
                    prediction=label, confidence=round(conf, 2), p_normal=round(float(probs[0]) * 100, 2),
                    p_dfu=round(float(probs[1]) * 100, 2), p_gangrene=round(float(probs[2]) * 100, 2),
                    severity=sev.get("severity"), severity_hint=sev.get("type_hint"),
                    patient_note=request.form.get("note", "").strip()[:500],
                    status="pending" if doctor else "private")
        db.session.add(case); db.session.commit()
        make_report(case)
        return redirect(url_for("case_view", case_id=case.id))
    return render_template("scan.html", doctors=doctors)

# ------------------------------------------------------------------ CASE: VIEW / FILES / ACTIONS
@app.route("/case/<int:case_id>")
@login_required
def case_view(case_id):
    c = get_case(case_id)
    ref = None
    if c.prediction == "Foot Ulcer (DFU)":
        ref = {"title": "Wagner Ulcer Classification (Reference)", "rows": [(g["grade"], g["desc"]) for g in WAGNER_STAGES]}
    elif c.prediction == "Gangrene":
        ref = {"title": "Common Gangrene Types (Reference)", "rows": [(g["type"], g["desc"]) for g in GANGRENE_TYPES]}
    doctors = User.query.filter_by(role="doctor").order_by(User.name).all() if current_user.role == "patient" else []
    return render_template("case.html", case=c, rec=RECOMMENDATIONS[c.prediction], ref=ref,
                           doctors=doctors, diagnoses=DIAGNOSES)


@app.route("/case/<int:case_id>/image")
@login_required
def case_image(case_id):
    c = get_case(case_id)
    return send_file(os.path.join(IMG_DIR, f"patient_{c.patient_id}", c.image_file))


@app.route("/case/<int:case_id>/report")
@login_required
def case_report(case_id):
    c = get_case(case_id)
    path = os.path.join(PDF_DIR, f"patient_{c.patient_id}", c.report_file or "")
    if not os.path.isfile(path):
        make_report(c)
        path = os.path.join(PDF_DIR, f"patient_{c.patient_id}", c.report_file)
    return send_file(path, as_attachment=True, download_name=f"MediScan_case_{c.id}.pdf")


@app.route("/case/<int:case_id>/review", methods=["POST"])
@role_required("doctor")
def review(case_id):
    c = get_case(case_id)
    diag = request.form.get("diagnosis")
    if diag not in DIAGNOSES:
        abort(400)
    c.doctor_diagnosis, c.doctor_note = diag, request.form.get("note", "").strip()[:2000]
    c.status, c.reviewed_at = "reviewed", datetime.now()
    db.session.commit(); make_report(c)
    flash("Review saved. The patient can now see your feedback.", "success")
    return redirect(url_for("case_view", case_id=c.id))


@app.route("/case/<int:case_id>/share", methods=["POST"])
@role_required("patient")
def share(case_id):
    c = get_case(case_id)
    did = request.form.get("doctor_id", type=int)
    d = db.session.get(User, did) if did else None
    if d and d.role == "doctor":
        c.doctor_id, c.status = d.id, "pending"
        c.doctor_diagnosis = c.doctor_note = c.reviewed_at = None
        db.session.commit(); make_report(c)
        flash(f"Case shared with Dr. {d.name}.", "success")
    return redirect(url_for("case_view", case_id=c.id))


@app.route("/case/<int:case_id>/delete", methods=["POST"])
@role_required("patient")
def delete_case(case_id):
    c = get_case(case_id)
    for p in (os.path.join(IMG_DIR, f"patient_{c.patient_id}", c.image_file),
              os.path.join(PDF_DIR, f"patient_{c.patient_id}", c.report_file or "")):
        if os.path.isfile(p):
            os.remove(p)
    db.session.delete(c); db.session.commit()
    flash("Record deleted.", "success")
    return redirect(url_for("cases"))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)