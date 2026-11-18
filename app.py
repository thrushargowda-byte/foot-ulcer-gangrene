import os
import uuid
import numpy as np
import cv2
from datetime import datetime
from functools import wraps
from flask import Flask, request, render_template, redirect, url_for, flash, abort, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.csrf import CSRFProtect, CSRFError
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

BASE = os.path.abspath(os.path.dirname(__file__))
IMG_DIR = os.path.join(BASE, "storage", "images")
PDF_DIR = os.path.join(BASE, "storage", "reports")

APP_NAME = "Foot Ulcer and Gangrene Detection"

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "change-this-secret-key"),
    SQLALCHEMY_DATABASE_URI="sqlite:///mediscan.db",
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    TEMPLATES_AUTO_RELOAD=True,
)
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@footguard.local")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "ChangeMe@123")

db = SQLAlchemy(app)
csrf = CSRFProtect(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to continue."
login_manager.login_message_category = "error"

IMG_SIZE = 224
CLASS_NAMES = ["Normal", "Foot Ulcer (DFU)", "Gangrene"]
DIAGNOSES = CLASS_NAMES + ["Inconclusive"]
ALLOWED = {"png", "jpg", "jpeg"}
CONFIDENCE_THRESHOLD = 55.0
NOT_ANALYZED = "Not analyzed"

# Case status flow:
#   submitted -> patient sent it, waiting for the admin
#   pending   -> admin forwarded it, the doctor has to analyze and review it
#   reviewed  -> doctor sent the result and recommendation to the patient
#   returned  -> admin (or doctor via admin) sent it back to the patient
#   private   -> old cases from the previous version of the app
STATUS_LABELS = {"submitted": "Awaiting admin", "pending": "With doctor", "reviewed": "Result ready",
                 "returned": "Returned", "private": "Private"}
STATUS_LABELS_DOCTOR = {"pending": "Needs review", "reviewed": "Sent to patient"}
STATUS_LABELS_ADMIN = {"reviewed": "Completed"}


@app.template_filter("status_label")
def status_label(status, role="patient"):
    if role == "doctor" and status in STATUS_LABELS_DOCTOR:
        return STATUS_LABELS_DOCTOR[status]
    if role == "admin" and status in STATUS_LABELS_ADMIN:
        return STATUS_LABELS_ADMIN[status]
    return STATUS_LABELS.get(status, str(status).capitalize())


@app.context_processor
def inject_globals():
    try:
        css_v = int(os.path.getmtime(os.path.join(BASE, "static", "css", "style.css")))
    except OSError:
        css_v = 0
    return {"APP_NAME": APP_NAME, "CSS_V": css_v}

# ------------------------------------------------------------------ DATABASE
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # "patient", "doctor" or "admin"
    specialization = db.Column(db.String(80))
    active = db.Column(db.Boolean, nullable=False, default=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    diabetic = db.Column(db.String(10))
    contact = db.Column(db.String(30))
    created_at = db.Column(db.DateTime, default=datetime.now)

    @property
    def is_active(self):
        return bool(self.active)


class Case(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    image_file = db.Column(db.String(120), nullable=False)
    report_file = db.Column(db.String(120))
    prediction = db.Column(db.String(30), nullable=False)  # NOT_ANALYZED until the doctor runs the model
    confidence = db.Column(db.Float)
    p_normal = db.Column(db.Float)
    p_dfu = db.Column(db.Float)
    p_gangrene = db.Column(db.Float)
    severity = db.Column(db.String(20))
    severity_hint = db.Column(db.String(200))
    patient_note = db.Column(db.Text)
    status = db.Column(db.String(12), default="submitted")
    doctor_diagnosis = db.Column(db.String(40))
    doctor_note = db.Column(db.Text)
    reviewed_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.now)
    patient = db.relationship("User", foreign_keys=[patient_id])
    doctor = db.relationship("User", foreign_keys=[doctor_id])
    created_by_id = db.Column(db.Integer)
    p_name = db.Column(db.String(80))
    p_age = db.Column(db.Integer)
    p_gender = db.Column(db.String(10))
    p_diabetic = db.Column(db.String(10))
    p_contact = db.Column(db.String(30))
    admin_note = db.Column(db.Text)
    forwarded_at = db.Column(db.DateTime)

    @property
    def display_name(self):
        return self.p_name or self.patient.name

    @property
    def bars(self):
        return [("Normal", self.p_normal), ("Foot Ulcer (DFU)", self.p_dfu), ("Gangrene", self.p_gangrene)]

    @property
    def analyzed(self):
        return self.prediction != NOT_ANALYZED and self.confidence is not None

    @property
    def patient_can_see(self):
        """Patients only see the model result after the doctor has sent it (or for old private cases)."""
        return self.analyzed and self.status in ("reviewed", "private")


with app.app_context():
    db.create_all()
    from sqlalchemy import inspect, text
    insp = inspect(db.engine)
    ucols = {c["name"] for c in insp.get_columns("user")}
    if "active" not in ucols:
        db.session.execute(text('ALTER TABLE "user" ADD COLUMN active BOOLEAN NOT NULL DEFAULT 1'))
    for col, typ in [("age", "INTEGER"), ("gender", "VARCHAR(10)"), ("diabetic", "VARCHAR(10)"), ("contact", "VARCHAR(30)")]:
        if col not in ucols:
            db.session.execute(text(f'ALTER TABLE "user" ADD COLUMN {col} {typ}'))
    have = {c["name"] for c in insp.get_columns("case")}
    for col, typ in [("created_by_id", "INTEGER"), ("p_name", "VARCHAR(80)"), ("p_age", "INTEGER"),
                     ("p_gender", "VARCHAR(10)"), ("p_diabetic", "VARCHAR(10)"), ("p_contact", "VARCHAR(30)"),
                     ("admin_note", "TEXT"), ("forwarded_at", "DATETIME")]:
        if col not in have:
            db.session.execute(text(f'ALTER TABLE "case" ADD COLUMN {col} {typ}'))
    db.session.commit()
    if not User.query.filter_by(role="admin").first():
        db.session.add(User(name="Administrator", email=ADMIN_EMAIL, role="admin", active=True,
                            password_hash=generate_password_hash(ADMIN_PASSWORD)))
        db.session.commit()


@login_manager.user_loader
def load_user(uid):
    u = db.session.get(User, int(uid))
    return u if u and u.active else None


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
    if current_user.role == "admin":
        if c.status == "private":
            abort(403)
        return c
    owner = current_user.role == "patient" and c.patient_id == current_user.id
    assigned = (current_user.role == "doctor" and c.doctor_id == current_user.id
                and c.status in ("pending", "reviewed"))
    if not (owner or assigned):
        abort(403)
    return c


def active_doctors():
    return User.query.filter_by(role="doctor", active=True).order_by(User.name).all()


def doctor_load():
    """{doctor_id: number of cases currently waiting for that doctor}, used to show availability to the admin."""
    rows = (db.session.query(Case.doctor_id, db.func.count(Case.id))
            .filter(Case.status == "pending").group_by(Case.doctor_id).all())
    return {did: n for did, n in rows}

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

print("=== Foot Ulcer and Gangrene Detection | BUILD v3 (patient/doctor chooser, admin assigns doctor) ===")
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
import io as _io
from xml.sax.saxutils import escape as _esc
import reportlab.platypus as rlp
from reportlab.lib import colors as rlc
from reportlab.lib.styles import ParagraphStyle as RLStyle

_R_TEAL = rlc.HexColor("#1E40AF")   # main report colour (blue)
_R_LIGHT = rlc.HexColor("#EAF1FF")
_R_INK = rlc.HexColor("#1F2937")
_R_MUTED = rlc.HexColor("#6B7280")
_R_LINE = rlc.HexColor("#D1D5DB")
_R_AMBER = rlc.HexColor("#B45309")
_R_AMBER_BG = rlc.HexColor("#FEF3C7")
_R_RED = rlc.HexColor("#B91C1C")
_R_GREEN = rlc.HexColor("#15803D")
_BORDERLINE_GAP = 20.0  # top-two probability gap (percentage points) below which a result is flagged


def _class_color(name):
    n = (name or "").lower()
    if "normal" in n:
        return _R_GREEN
    if "gangrene" in n:
        return _R_RED
    return _R_AMBER


def _num(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


class _ProbBars(rlp.Flowable):
    def __init__(self, data, width, highlight):
        rlp.Flowable.__init__(self)
        self.data, self.width, self.highlight = data, width, highlight
        self.height = 21 * max(len(data), 1) + 2

    def draw(self):
        c = self.canv
        lw, pw = 38 * mm, 18 * mm
        bw = self.width - lw - pw
        y = self.height - 18
        for name, pct in self.data:
            pct = max(0.0, min(100.0, pct))
            c.setFillColor(_R_INK)
            c.setFont("Helvetica-Bold" if name == self.highlight else "Helvetica", 9.5)
            c.drawString(0, y + 4, name)
            c.setFillColor(rlc.HexColor("#E5E7EB")); c.roundRect(lw, y, bw, 14, 3, stroke=0, fill=1)
            c.setFillColor(_class_color(name)); c.roundRect(lw, y, max(bw * pct / 100.0, 2), 14, 3, stroke=0, fill=1)
            c.setFillColor(_R_INK); c.setFont("Helvetica-Bold", 9.5)
            c.drawRightString(self.width, y + 4, "%.2f%%" % pct)
            y -= 21


def make_report(case):
    folder = os.path.join(PDF_DIR, f"patient_{case.patient_id}")
    os.makedirs(folder, exist_ok=True)
    case.report_file = f"case_{case.id}.pdf"
    out_path = os.path.join(folder, case.report_file)

    W, H = A4
    LM = RM = 18 * mm
    FW = W - LM - RM
    HEAD = 31 * mm
    BOTTOM = 19.5 * mm
    stamp = f"{case.created_at:%d %b %Y, %H:%M}"

    def on_page(c, doc):
        c.saveState()
        c.setFillColor(_R_TEAL); c.rect(0, H - HEAD, W, HEAD, stroke=0, fill=1)
        c.setFillColor(rlc.white)
        c.setFont("Helvetica-Bold", 16); c.drawString(LM, H - 14.5 * mm, APP_NAME)
        c.setFont("Helvetica", 11); c.drawString(LM, H - 21.5 * mm, "Screening Report")
        c.setFont("Helvetica-Bold", 11); c.drawRightString(W - RM, H - 14.5 * mm, f"Case #{case.id}")
        c.setFont("Helvetica", 10); c.drawRightString(W - RM, H - 21 * mm, stamp)
        c.setStrokeColor(_R_LINE); c.setLineWidth(0.6); c.line(LM, 17 * mm, W - RM, 17 * mm)
        c.setFillColor(_R_MUTED); c.setFont("Helvetica", 8)
        c.drawString(LM, 12 * mm, f"{APP_NAME}  |  Academic project tool  |  Confidential: contains patient information")
        c.drawRightString(W - RM, 12 * mm, f"Page {doc.page}")
        c.restoreState()

    def st(name, **kw):
        return RLStyle(name, fontName=kw.pop("fontName", "Helvetica"), textColor=kw.pop("textColor", _R_INK), **kw)

    h2 = st("h2", fontName="Helvetica-Bold", fontSize=11.5, textColor=_R_TEAL)
    lab = st("lab", fontSize=8.5, textColor=_R_MUTED, leading=11)
    val = st("val", fontName="Helvetica-Bold", fontSize=10.5, leading=14)
    small = st("small", fontSize=8.5, leading=12, textColor=_R_MUTED)
    bul = st("bul", fontSize=9.5, leading=14, leftIndent=12, bulletIndent=0, spaceAfter=2)
    P = rlp.Paragraph

    def section(title):
        t = rlp.Table([[P(title.upper(), h2)]], colWidths=[FW])
        t.setStyle(rlp.TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1, _R_TEAL), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 0)]))
        return t

    # ---------- data ----------
    try:
        bars = [(str(n), _num(v)) for n, v in case.bars]
    except Exception:
        bars = []
    conf = _num(case.confidence)
    ranked = sorted(bars, key=lambda b: b[1], reverse=True)
    gap, second = None, None
    if len(ranked) >= 2:
        gap, second = ranked[0][1] - ranked[1][1], ranked[1][0]
    borderline = gap is not None and gap < _BORDERLINE_GAP
    doc_name = case.doctor.name if getattr(case, "doctor", None) else ""

    story = []

    # ---------- patient & case information ----------
    items = [("PATIENT NAME", case.display_name), ("CASE ID", f"#{case.id}"), ("REPORT DATE", stamp)]
    if case.p_age: items.append(("AGE", str(case.p_age)))
    if case.p_gender: items.append(("GENDER", str(case.p_gender)))
    if case.p_diabetic: items.append(("DIABETIC", str(case.p_diabetic)))
    if case.p_contact: items.append(("CONTACT", str(case.p_contact)))
    if doc_name: items.append(("SCANNED BY" if case.created_by_id else "DOCTOR", f"Dr. {doc_name}"))
    rows = []
    for i in range(0, len(items), 3):
        grp = items[i:i + 3]
        grp += [("", "")] * (3 - len(grp))
        rows.append([P(l, lab) for l, _ in grp])
        rows.append([P(_esc(v), val) for _, v in grp])
    info = rlp.Table(rows, colWidths=[FW * 0.36, FW * 0.30, FW * 0.34])
    sty = [("BACKGROUND", (0, 0), (-1, -1), _R_LIGHT), ("BOX", (0, 0), (-1, -1), 0.6, _R_LINE),
           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 3),
           ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for r in range(2, len(rows), 2):
        sty.append(("TOPPADDING", (0, r), (-1, r), 8))
    info.setStyle(rlp.TableStyle(sty))
    story += [section("Patient & Case Information"), rlp.Spacer(1, 6), info]

    # ---------- image + result ----------
    img_w = 58 * mm
    gutter = 10 * mm
    res_w = FW - img_w - gutter
    try:
        ipath = os.path.join(IMG_DIR, f"patient_{case.patient_id}", case.image_file)
        arr = cv2.imread(ipath)
        h0, w0 = arr.shape[:2]
        s = min(1.0, 900.0 / max(h0, w0))
        if s < 1.0:
            arr = cv2.resize(arr, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
        ih, iw = arr.shape[:2]
        scale = min(img_w / iw, 80 * mm / ih)
        ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        img_cell = [rlp.Image(_io.BytesIO(buf.tobytes()), width=iw * scale, height=ih * scale),
                    rlp.Spacer(1, 3), P("Figure 1. Image submitted for analysis", small)]
    except Exception:
        img_cell = [P("Image not available", small)]

    badge = rlp.Table([[P(f'<font color="white"><b>{_esc(str(case.prediction))}</b></font>', st("bd", fontSize=13, leading=16))]],
                      colWidths=[res_w - 4 * mm])
    badge.setStyle(rlp.TableStyle([("BACKGROUND", (0, 0), (-1, -1), _class_color(case.prediction)),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 7),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    res_rows = [[P("PREDICTED CLASS", lab)], [badge], [rlp.Spacer(1, 8)],
                [P("MODEL CONFIDENCE", lab)],
                [P("%.2f%%" % conf, st("big", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=_R_TEAL))]]
    if case.severity:
        res_rows += [[rlp.Spacer(1, 6)], [P("ESTIMATED SEVERITY", lab)], [P(_esc(f"{case.severity} (experimental)"), val)]]
    if gap is not None:
        res_rows += [[rlp.Spacer(1, 6)], [P("RELIABILITY FLAG", lab)]]
        if borderline:
            txt = (f'<font color="#B45309"><b>Borderline result.</b></font> Top two classes are only '
                   f'{gap:.2f} percentage points apart.')
        else:
            txt = f'Top class leads the next by {gap:.2f} percentage points.'
        res_rows.append([P(txt, st("rf", fontSize=9.5, leading=13))])
    res = rlp.Table(res_rows, colWidths=[res_w])
    res.setStyle(rlp.TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    two = rlp.Table([[img_cell, "", res]], colWidths=[img_w, gutter, res_w])
    two.setStyle(rlp.TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0)]))
    story += [rlp.Spacer(1, 7), section("Image Analysis & Result"), rlp.Spacer(1, 6), two]

    # ---------- probability bars ----------
    if bars:
        story += [rlp.Spacer(1, 7), section("Class Probability Breakdown"), rlp.Spacer(1, 8),
                  _ProbBars(bars, FW, str(case.prediction)), rlp.Spacer(1, 4),
                  P("Probabilities are the model's raw output scores and sum to approximately 100%.", small)]

    # ---------- interpretation notes ----------
    notes = [f"The model's highest-scoring class is <b>{_esc(str(case.prediction))}</b> at <b>{conf:.2f}%</b>."]
    if borderline:
        notes.append(f"The <b>{_esc(second)}</b> score ({ranked[1][1]:.2f}%) is close to the top score. A result this close "
                     f"should be treated as <b>uncertain between {_esc(ranked[0][0])} and {_esc(second)}</b>, "
                     f"not as a confident single diagnosis.")
    if case.severity:
        notes.append("The severity grade is <b>experimental</b> (a colour-based estimate) and has not been clinically "
                     "validated. It should not be used to decide urgency or treatment.")
    notes.append("Image-based analysis cannot assess wound depth, infection, circulation or nerve function. "
                 "<b>In-person clinical examination by a qualified healthcare professional is recommended.</b>")
    story += [rlp.Spacer(1, 7), section("Interpretation Notes"), rlp.Spacer(1, 6)]
    story += [P(n, bul, bulletText="\u2022") for n in notes]

    # ---------- doctor review / clinician review ----------
    if case.reviewed_at:
        story += [rlp.Spacer(1, 7), section("Doctor Review"), rlp.Spacer(1, 6)]
        rv = [("REVIEWED BY", f"Dr. {doc_name}" if doc_name else "Doctor"),
              ("DIAGNOSIS", case.doctor_diagnosis or "Not recorded")]
        try:
            rv.append(("REVIEWED ON", f"{case.reviewed_at:%d %b %Y, %H:%M}"))
        except Exception:
            pass
        rv_t = rlp.Table([[P(l, lab) for l, _ in rv], [P(_esc(v), val) for _, v in rv]],
                         colWidths=[FW / len(rv)] * len(rv))
        rv_t.setStyle(rlp.TableStyle([("BACKGROUND", (0, 0), (-1, -1), _R_LIGHT), ("BOX", (0, 0), (-1, -1), 0.6, _R_LINE),
                                      ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 3),
                                      ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(rv_t)
        if case.doctor_note:
            story += [rlp.Spacer(1, 6), rlp.KeepTogether([P("DOCTOR'S RECOMMENDATION", lab), rlp.Spacer(1, 2),
                      P(_esc(case.doctor_note).replace("\n", "<br/>"), st("dn", fontSize=9.5, leading=14))])]
    else:
        story += [rlp.Spacer(1, 7), section("Clinician Review (to be completed)")]
        rev = rlp.Table([[P("Reviewed by", lab), "", P("Signature", lab), "", P("Date", lab), ""]],
                        colWidths=[22 * mm, 46 * mm, 20 * mm, 40 * mm, 12 * mm, FW - 140 * mm])
        rev.setStyle(rlp.TableStyle([("LINEBELOW", (1, 0), (1, 0), 0.6, _R_INK), ("LINEBELOW", (3, 0), (3, 0), 0.6, _R_INK),
                                     ("LINEBELOW", (5, 0), (5, 0), 0.6, _R_INK), ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 12)]))
        story.append(rev)

    # ---------- disclaimer ----------
    disc = rlp.Table([[P("<b>Disclaimer.</b> This report was generated by an academic project tool for demonstration "
                         "purposes only. It is not a certified medical device and does not replace professional "
                         "diagnosis. Always consult a qualified healthcare professional for diagnosis and treatment.",
                         st("dc", fontSize=8.5, leading=12, textColor=rlc.HexColor("#78350F")))]], colWidths=[FW])
    disc.setStyle(rlp.TableStyle([("BACKGROUND", (0, 0), (-1, -1), _R_AMBER_BG),
                                  ("BOX", (0, 0), (-1, -1), 0.6, rlc.HexColor("#F59E0B")),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                  ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [rlp.Spacer(1, 12), rlp.KeepTogether([disc])]

    doc = rlp.BaseDocTemplate(out_path, pagesize=A4, leftMargin=LM, rightMargin=RM,
                              topMargin=HEAD + 7 * mm, bottomMargin=BOTTOM,
                              title=f"{APP_NAME} - Case #{case.id}", author=APP_NAME)
    frame = rlp.Frame(LM, BOTTOM, FW, H - HEAD - 7 * mm - BOTTOM, leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
    doc.addPageTemplates([rlp.PageTemplate(id="p", frames=[frame], onPage=on_page)])
    doc.build(story)
    db.session.commit()


# ------------------------------------------------------------------ PUBLIC + AUTH
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        logout_user()
    kind = request.args.get("as")
    if kind == "doctor":
        flash("Doctor accounts are created by the admin. Log in with the email and password the admin gave you.", "success")
        return redirect(url_for("login", role="doctor"))
    if request.method == "GET" and kind != "patient":
        return render_template("register_choose.html")
    if request.method == "POST":
        f = request.form
        name, email, pw = f.get("name", "").strip(), f.get("email", "").strip().lower(), f.get("password", "")
        age = f.get("age", type=int)
        age = age if age is not None and 0 <= age <= 120 else None
        gender = f.get("gender") if f.get("gender") in ("Male", "Female", "Other") else None
        diab = f.get("diabetic") if f.get("diabetic") in ("Yes", "No", "Unknown") else None
        contact = f.get("contact", "").strip()[:30] or None
        err = None
        if len(name) < 2: err = "Please enter your name."
        elif "@" not in email: err = "Please enter a valid email."
        elif len(pw) < 6: err = "Password must be at least 6 characters."
        elif User.query.filter_by(email=email).first(): err = "This email is already registered."
        if err:
            flash(err, "error")
            return render_template("register.html", form=f)
        user = User(name=name, email=email, role="patient", active=True, age=age, gender=gender,
                    diabetic=diab, contact=contact, password_hash=generate_password_hash(pw))
        db.session.add(user); db.session.commit()
        login_user(user)
        flash(f"Welcome, {user.name}!", "success")
        return redirect(url_for("dashboard"))
    return render_template("register.html", form={})


@app.route("/login", methods=["GET", "POST"])
def login():
    role = request.values.get("role", "patient")
    if role not in ("patient", "doctor", "admin"):
        role = "patient"
    if current_user.is_authenticated:
        logout_user()  # opening a login page signs the current user out, so you can switch portals
    email = request.form.get("email", "").strip().lower()
    if request.method == "POST":
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, request.form.get("password", "")):
            if user.role != role:
                flash(f"This is a {user.role} account. Please use the {user.role} login.", "error")
                return render_template("login.html", role=role, email=email, admin_email=ADMIN_EMAIL)
            if not user.active:
                flash("Your account is inactive. Please contact the admin.", "error")
                return render_template("login.html", role=role, email=email, admin_email=ADMIN_EMAIL)
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Incorrect email or password.", "error")
    return render_template("login.html", role=role, email=email, admin_email=ADMIN_EMAIL)


@app.route("/logout")
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


@app.errorhandler(CSRFError)
def csrf_error(e):
    flash("Your session expired or the form was invalid. Please try again.", "error")
    return redirect(url_for("dashboard") if current_user.is_authenticated else url_for("login"))


@app.errorhandler(403)
@app.errorhandler(404)
def no_access(e):
    flash("You don't have access to that page." if e.code == 403 else "Page not found.", "error")
    return redirect(url_for("dashboard") if current_user.is_authenticated else url_for("home"))

# ------------------------------------------------------------------ DASHBOARD + LISTS
def my_cases_query():
    """Cases the logged-in patient owns, or the cases an admin has forwarded to the logged-in doctor."""
    if current_user.role == "patient":
        return Case.query.filter(Case.patient_id == current_user.id)
    return Case.query.filter(Case.doctor_id == current_user.id, Case.status.in_(["pending", "reviewed"]))


@app.route("/dashboard")
@login_required
def dashboard():
    if current_user.role == "admin":
        return redirect(url_for("admin_dashboard"))
    q = my_cases_query()
    stats = {"total": q.count(),
             "submitted": q.filter(Case.status == "submitted").count(),
             "pending": q.filter(Case.status == "pending").count(),
             "reviewed": q.filter(Case.status == "reviewed").count(),
             "urgent": q.filter(Case.prediction == "Gangrene").count()}
    recent = q.order_by(Case.created_at.desc()).limit(5).all()
    return render_template("dashboard.html", stats=stats, recent=recent)


@app.route("/cases")
@login_required
def cases():
    if current_user.role == "admin":
        return redirect(url_for("admin_dashboard"))
    q = my_cases_query()
    status = request.args.get("status")
    if status in ("private", "returned", "submitted", "pending", "reviewed"):
        q = q.filter(Case.status == status)
    else:
        status = None
    return render_template("cases.html", cases=q.order_by(Case.created_at.desc()).all(), status=status)

# ------------------------------------------------------------------ PATIENT: SEND A CASE
@app.route("/scan", methods=["GET", "POST"])
@role_required("patient")
def scan():
    doctors = active_doctors()
    if request.method == "POST":
        f = request.files.get("file")
        ext = f.filename.rsplit(".", 1)[-1].lower() if f and f.filename else ""
        if ext not in ALLOWED:
            flash("Please upload a PNG or JPG image.", "error")
            return redirect(url_for("scan"))
        did = request.form.get("doctor_id", type=int)
        doctor = db.session.get(User, did) if did else None
        if not doctor or doctor.role != "doctor" or not doctor.active:
            flash("Please choose a doctor.", "error")
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
        case = Case(patient_id=current_user.id, doctor_id=doctor.id, image_file=name,
                    prediction=NOT_ANALYZED, status="submitted",
                    p_name=current_user.name, p_age=current_user.age, p_gender=current_user.gender,
                    p_diabetic=current_user.diabetic, p_contact=current_user.contact)
        db.session.add(case); db.session.commit()
        flash("Your photo was sent to the admin. You will see the result here once the doctor has reviewed it.", "success")
        return redirect(url_for("case_view", case_id=case.id))
    return render_template("scan.html", doctors=doctors)

# ------------------------------------------------------------------ EMAIL THE REPORT TO THE PATIENT
def email_report(c):
    """Email the case's PDF report to the patient. Returns (ok, message for the doctor)."""
    import smtplib, ssl
    from email.message import EmailMessage
    patient = db.session.get(User, c.patient_id)
    to = (patient.email if patient else "") or ""
    if not to or to.endswith("@walkin.local"):
        return False, "No patient email on file, so the report was not emailed."
    user, pw = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    if not (user and pw):
        return False, "Email is not set up (SMTP_USER / SMTP_PASS missing), so the report was not emailed."
    path = os.path.join(PDF_DIR, f"patient_{c.patient_id}", c.report_file or "")
    if not c.report_file or not os.path.isfile(path):
        return False, "The report PDF could not be found, so it was not emailed."
    msg = EmailMessage()
    msg["Subject"] = "Your foot scan report - Foot Ulcer and Gangrene Detection"
    msg["From"] = user
    msg["To"] = to
    msg.set_content(
        f"Hello {c.p_name or patient.name},\n\n"
        f"Dr. {current_user.name} has reviewed your foot scan. Your report is attached as a PDF.\n\n"
        "This is an academic tool and not a certified medical device. It does not replace "
        "professional diagnosis. Please follow your doctor's advice.\n")
    with open(path, "rb") as fh:
        msg.add_attachment(fh.read(), maintype="application", subtype="pdf",
                           filename=f"Foot_Ulcer_Gangrene_case_{c.id}.pdf")
    try:
        with smtplib.SMTP(os.environ.get("SMTP_HOST", "smtp.gmail.com"),
                          int(os.environ.get("SMTP_PORT", "587")), timeout=20) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(user, pw)
            s.send_message(msg)
    except Exception as e:
        app.logger.exception("Report email failed")
        return False, f"The report could not be emailed ({type(e).__name__}). You can still download it from the case page."
    return True, f"Report emailed to {to}."

def email_login(to, name, temp_pw):
    """Email a new patient their login details. Returns a short message for the doctor."""
    import smtplib, ssl
    from email.message import EmailMessage
    user, pw = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    if not (user and pw):
        return "Email is not set up, so please share these login details with the patient yourself."
    msg = EmailMessage()
    msg["Subject"] = "Your login - Foot Ulcer and Gangrene Detection"
    msg["From"] = user
    msg["To"] = to
    msg.set_content(
        f"Hello {name},\n\n"
        f"Dr. {current_user.name} has created an account for you.\n\n"
        f"Log in: {request.host_url}login?role=patient\n"
        f"Email: {to}\n"
        f"Temporary password: {temp_pw}\n\n"
        "Your result will appear under My Records once your doctor has sent it.\n")
    try:
        with smtplib.SMTP(os.environ.get("SMTP_HOST", "smtp.gmail.com"),
                          int(os.environ.get("SMTP_PORT", "587")), timeout=20) as srv:
            srv.starttls(context=ssl.create_default_context())
            srv.login(user, pw)
            srv.send_message(msg)
    except Exception:
        app.logger.exception("Login email failed")
        return "The login details could not be emailed, so please share them with the patient yourself."
    return f"Login details emailed to {to}."

# ------------------------------------------------------------------ DOCTOR: NEW PATIENT SCAN
@app.route("/doctor/new-scan", methods=["GET", "POST"])
@role_required("doctor")
def doctor_new_scan():
    if request.method == "POST":
        f = request.files.get("file")
        ext = f.filename.rsplit(".", 1)[-1].lower() if f and f.filename else ""
        if ext not in ALLOWED:
            flash("Please upload a PNG or JPG image.", "error")
            return redirect(url_for("doctor_new_scan"))
        p_name = request.form.get("p_name", "").strip()[:80]
        p_age = request.form.get("p_age", type=int)
        p_gender = request.form.get("p_gender", "").strip()[:10]
        p_diabetic = request.form.get("p_diabetic", "").strip()[:10]
        p_contact = request.form.get("p_contact", "").strip()[:30]
        note = request.form.get("patient_note", "").strip()[:1000]
        import re
        p_email = request.form.get("p_email", "").strip().lower()[:120]
        if not (p_name and p_age and p_gender and p_diabetic and p_contact):
            flash("Please fill in all the patient details.", "error")
            return redirect(url_for("doctor_new_scan"))
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", p_email):
            flash("Please enter a valid patient email address.", "error")
            return redirect(url_for("doctor_new_scan"))

        from werkzeug.security import generate_password_hash
        patient = User.query.filter_by(email=p_email).first()
        if patient and patient.role != "patient":
            flash("That email belongs to a doctor or admin account. Please use the patient's own email.", "error")
            return redirect(url_for("doctor_new_scan"))
        new_login = None
        if not patient:
            import secrets
            new_login = secrets.token_hex(4)   # temporary password, 8 characters
            patient = User(name=p_name, email=p_email,
                           password_hash=generate_password_hash(new_login),
                           role="patient", active=True, age=p_age, gender=p_gender,
                           diabetic=p_diabetic, contact=p_contact)
            db.session.add(patient)
            db.session.flush()

        folder = os.path.join(IMG_DIR, f"patient_{patient.id}")
        os.makedirs(folder, exist_ok=True)
        name = uuid.uuid4().hex + "." + ext
        path = os.path.join(folder, name)
        f.save(path)
        if not is_likely_skin_image(path):
            os.remove(path)
            db.session.rollback()
            flash("This doesn't look like a photo of skin. Please upload a clear photo of the affected foot.", "error")
            return redirect(url_for("doctor_new_scan"))

        case = Case(patient_id=patient.id, doctor_id=current_user.id, image_file=name,
                    prediction=NOT_ANALYZED, status="pending", created_by_id=current_user.id,
                    p_name=p_name, p_age=p_age, p_gender=p_gender,
                    p_diabetic=p_diabetic, p_contact=p_contact, patient_note=note or None)
        db.session.add(case)
        db.session.commit()
        flash("Patient case created. Run the analysis below.", "success")
        if new_login:
            info = email_login(p_email, p_name, new_login)
            flash(f"Patient login created. Email: {p_email}   Temporary password: {new_login}. {info}", "success")
        return redirect(url_for("case_view", case_id=case.id))
    return render_template("doctor_scan.html")

# ------------------------------------------------------------------ CASE: VIEW / FILES
@app.route("/case/<int:case_id>")
@login_required
def case_view(case_id):
    if current_user.role == "admin":
        return redirect(url_for("admin_case", case_id=case_id))
    c = get_case(case_id)
    is_doctor = current_user.role == "doctor"
    show_result = c.analyzed and (is_doctor or c.patient_can_see)
    rec = ref = None
    if not is_doctor and show_result:
        key = c.doctor_diagnosis if (c.status == "reviewed" and c.doctor_diagnosis) else c.prediction
        rec = RECOMMENDATIONS.get(key)
        if key == "Foot Ulcer (DFU)":
            ref = {"title": "Wagner Ulcer Classification (Reference)", "rows": [(g["grade"], g["desc"]) for g in WAGNER_STAGES]}
        elif key == "Gangrene":
            ref = {"title": "Common Gangrene Types (Reference)", "rows": [(g["type"], g["desc"]) for g in GANGRENE_TYPES]}
    can_resubmit = current_user.role == "patient" and c.status in ("private", "returned")
    return render_template("case.html", case=c, show_result=show_result, rec=rec, ref=ref,
                           doctors=active_doctors() if can_resubmit else [], diagnoses=DIAGNOSES,
                           can_resubmit=can_resubmit)


@app.route("/case/<int:case_id>/image")
@login_required
def case_image(case_id):
    c = get_case(case_id)
    return send_file(os.path.join(IMG_DIR, f"patient_{c.patient_id}", c.image_file))


@app.route("/case/<int:case_id>/report")
@login_required
def case_report(case_id):
    c = get_case(case_id)
    if not c.analyzed or (current_user.role == "patient" and not c.patient_can_see):
        flash("The report is available after the doctor has sent the result.", "error")
        return redirect(url_for("case_view", case_id=c.id))
    make_report(c)  # always rebuild so the latest layout and doctor review are included
    path = os.path.join(PDF_DIR, f"patient_{c.patient_id}", c.report_file)
    return send_file(path, as_attachment=True, download_name=f"Foot_Ulcer_Gangrene_case_{c.id}.pdf")

# ------------------------------------------------------------------ DOCTOR: ANALYZE + SEND RESULT
@app.route("/case/<int:case_id>/analyze", methods=["POST"])
@role_required("doctor")
def analyze(case_id):
    c = get_case(case_id)
    if c.status != "pending":
        flash("Only cases waiting for your review can be analyzed.", "error")
        return redirect(url_for("case_view", case_id=c.id))
    path = os.path.join(IMG_DIR, f"patient_{c.patient_id}", c.image_file)
    if not os.path.isfile(path):
        flash("The image file could not be found.", "error")
        return redirect(url_for("case_view", case_id=c.id))
    if not is_likely_skin_image(path):
        flash("This image does not look like a clear photo of skin, so it was not analyzed. "
              "Use 'Return to admin' to ask the patient for a better photo.", "error")
        return redirect(url_for("case_view", case_id=c.id))
    probs = model.predict(prepare_image(path), verbose=0)[0]
    i = int(np.argmax(probs))
    label, conf = CLASS_NAMES[i], float(probs[i]) * 100
    sev = analyze_severity(path, label) or {}
    c.prediction, c.confidence = label, round(conf, 2)
    c.p_normal, c.p_dfu, c.p_gangrene = (round(float(probs[k]) * 100, 2) for k in range(3))
    c.severity, c.severity_hint = sev.get("severity"), sev.get("type_hint")
    db.session.commit()
    if conf < CONFIDENCE_THRESHOLD:
        flash(f"Analysis done, but the model confidence is low ({conf:.1f}%). "
              "Consider choosing 'Inconclusive' or asking for a clearer photo.", "error")
    else:
        flash("Analysis complete. Please add your recommendation below.", "success")
    return redirect(url_for("case_view", case_id=c.id))


@app.route("/case/<int:case_id>/review", methods=["POST"])
@role_required("doctor")
def review(case_id):
    c = get_case(case_id)
    if not c.analyzed:
        flash("Run the analysis first.", "error")
        return redirect(url_for("case_view", case_id=c.id))
    diag = request.form.get("diagnosis")
    if diag not in DIAGNOSES:
        abort(400)
    c.doctor_diagnosis, c.doctor_note = diag, request.form.get("note", "").strip()[:2000]
    if request.form.get("action") == "send" or c.status == "reviewed":
        c.status, c.reviewed_at = "reviewed", datetime.now()
        msg = "The result and your recommendation were sent to the patient."
    else:
        msg = "Draft saved. The patient cannot see it until you send it."
    db.session.commit(); make_report(c)
    flash(msg, "success")
    if c.status == "reviewed":
        ok, info = email_report(c)
        flash(info, "success" if ok else "error")
    return redirect(url_for("case_view", case_id=c.id))


@app.route("/case/<int:case_id>/back", methods=["POST"])
@role_required("doctor")
def back_to_admin(case_id):
    c = get_case(case_id)
    if c.status == "pending":
        reason = request.form.get("reason", "").strip()[:300] or "no reason given"
        c.status = "submitted"
        c.admin_note = f"Returned by Dr. {current_user.name}: {reason}"
        db.session.commit()
        flash("The case was returned to the admin.", "success")
        return redirect(url_for("cases"))
    return redirect(url_for("case_view", case_id=c.id))

# ------------------------------------------------------------------ PATIENT: RESUBMIT / DELETE
@app.route("/case/<int:case_id>/resubmit", methods=["POST"])
@role_required("patient")
def resubmit(case_id):
    c = get_case(case_id)
    d = db.session.get(User, request.form.get("doctor_id", type=int) or 0)
    if c.status in ("private", "returned") and d and d.role == "doctor" and d.active:
        c.doctor_id, c.status = d.id, "submitted"
        c.doctor_diagnosis = c.doctor_note = c.reviewed_at = c.admin_note = None
        db.session.commit()
        flash("Case sent to the admin.", "success")
    else:
        flash("Please choose an active doctor.", "error")
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

# ------------------------------------------------------------------ ADMIN
@app.route("/admin")
@role_required("admin")
def admin_dashboard():
    status = request.args.get("status", "submitted")
    flow = ("submitted", "pending", "reviewed")
    stats = {s: Case.query.filter_by(status=s).count() for s in flow}
    q = Case.query.filter(Case.status.in_(flow))
    if status in stats:
        q = q.filter_by(status=status)
    return render_template("admin.html", stats=stats, status=status,
                           cases=q.order_by(Case.created_at.desc()).all())


@app.route("/admin/case/<int:case_id>")
@role_required("admin")
def admin_case(case_id):
    c = get_case(case_id)
    return render_template("admin_case.html", case=c, doctors=active_doctors(), load=doctor_load())


@app.route("/admin/case/<int:case_id>/forward", methods=["POST"])
@role_required("admin")
def admin_forward(case_id):
    c = get_case(case_id)
    d = db.session.get(User, request.form.get("doctor_id", type=int) or 0)
    if c.status not in ("submitted", "pending") or not d or d.role != "doctor" or not d.active:
        flash("Choose an active doctor to forward this case to.", "error")
        return redirect(url_for("admin_case", case_id=c.id))
    c.doctor_id, c.status, c.forwarded_at = d.id, "pending", datetime.now()
    c.admin_note = request.form.get("note", "").strip()[:500]
    db.session.commit()
    flash(f"Case forwarded to Dr. {d.name}.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/case/<int:case_id>/return", methods=["POST"])
@role_required("admin")
def admin_return(case_id):
    c = get_case(case_id)
    if c.status == "submitted":
        c.status, c.doctor_id = "returned", None
        c.admin_note = request.form.get("note", "").strip()[:500]
        db.session.commit()
        flash("Case returned to the patient.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/users")
@role_required("admin")
def admin_users():
    return render_template("admin_users.html",
                           doctors=User.query.filter_by(role="doctor").order_by(User.name).all(),
                           patients=User.query.filter_by(role="patient").order_by(User.name).all(),
                           load=doctor_load())


@app.route("/admin/doctors/new", methods=["POST"])
@role_required("admin")
def admin_new_doctor():
    f = request.form
    name, email, pw = f.get("name", "").strip(), f.get("email", "").strip().lower(), f.get("password", "")
    err = None
    if len(name) < 2: err = "Enter the doctor's name."
    elif "@" not in email: err = "Enter a valid email."
    elif len(pw) < 6: err = "Password must be at least 6 characters."
    elif User.query.filter_by(email=email).first(): err = "This email is already registered."
    if err:
        flash(err, "error")
    else:
        db.session.add(User(name=name, email=email, role="doctor", active=True,
                            password_hash=generate_password_hash(pw),
                            specialization=f.get("specialization", "").strip()[:80] or None))
        db.session.commit()
        flash(f"Doctor account created for Dr. {name}.", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/user/<int:uid>/toggle", methods=["POST"])
@role_required("admin")
def admin_toggle_user(uid):
    u = db.session.get(User, uid)
    if u is None or u.role == "admin":
        abort(403)
    u.active = not u.active
    db.session.commit()
    flash(f"{u.name} is now {'active' if u.active else 'inactive'}.", "success")
    return redirect(url_for("admin_users"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, ssl_context=("cert.pem", "key.pem") if os.path.exists("cert.pem") else "adhoc")
