"""Aplikasi Flask: prediksi risiko gagal bayar kartu kredit (TA-03).

Routes:
    /           – formulir input + hasil prediksi
    /evaluasi   – halaman metrik & grafik evaluasi model
    /health     – health-check JSON
"""
import json, os
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_from_directory

from features import build_features, NUMERIC_COLS

# ── Inisialisasi ───────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE, "model"))

app    = Flask(__name__)
model  = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    META = json.load(f)

THRESHOLD = float(os.environ.get("THRESHOLD", META.get("threshold", 0.5)))

# ── Spesifikasi formulir ───────────────────────────────────────────────────────
# (nama, label, min, max, default)
FIELDS_PROFIL = [
    ("AGE",      "Usia (tahun)",          18,       100,      35),
    ("LIMIT_BAL","Batas Kredit (NT$)",    10_000, 1_000_000, 100_000),
]

SELECTS = {
    "SEX":       ("Jenis Kelamin",     [(1, "Laki-laki"), (2, "Perempuan")]),
    "EDUCATION": ("Pendidikan",        [(1, "Pascasarjana"), (2, "Universitas"),
                                        (3, "SMA"), (4, "Lainnya")]),
    "MARRIAGE":  ("Status Pernikahan", [(1, "Menikah"), (2, "Lajang"), (3, "Lainnya")]),
}

# Riwayat pembayaran 6 bulan (PAY_0 = bulan Sept, PAY_6 = April)
PAY_FIELDS = [
    ("PAY_0",  "Sept (bulan ke-1)", -2, 8, 0),
    ("PAY_2",  "Agst (bulan ke-2)", -2, 8, 0),
    ("PAY_3",  "Juli (bulan ke-3)", -2, 8, 0),
    ("PAY_4",  "Juni (bulan ke-4)", -2, 8, 0),
    ("PAY_5",  "Mei  (bulan ke-5)", -2, 8, 0),
    ("PAY_6",  "Apr  (bulan ke-6)", -2, 8, 0),
]

BILL_FIELDS = [
    ("BILL_AMT1", "Tagihan Sept (NT$)",  -200_000, 1_000_000, 50_000),
    ("BILL_AMT2", "Tagihan Agst (NT$)",  -200_000, 1_000_000, 50_000),
    ("BILL_AMT3", "Tagihan Juli (NT$)",  -200_000, 1_000_000, 50_000),
    ("BILL_AMT4", "Tagihan Juni (NT$)",  -200_000, 1_000_000, 50_000),
    ("BILL_AMT5", "Tagihan Mei  (NT$)",  -200_000, 1_000_000, 50_000),
    ("BILL_AMT6", "Tagihan Apr  (NT$)",  -200_000, 1_000_000, 50_000),
]

PAYAMT_FIELDS = [
    ("PAY_AMT1", "Dibayar Sept (NT$)", 0, 1_000_000, 2_000),
    ("PAY_AMT2", "Dibayar Agst (NT$)", 0, 1_000_000, 2_000),
    ("PAY_AMT3", "Dibayar Juli (NT$)", 0, 1_000_000, 2_000),
    ("PAY_AMT4", "Dibayar Juni (NT$)", 0, 1_000_000, 2_000),
    ("PAY_AMT5", "Dibayar Mei  (NT$)", 0, 1_000_000, 2_000),
    ("PAY_AMT6", "Dibayar Apr  (NT$)", 0, 1_000_000, 2_000),
]

ALL_FIELDS = FIELDS_PROFIL + PAY_FIELDS + BILL_FIELDS + PAYAMT_FIELDS
FIELD_SPEC = {n: (lo, hi) for n, _, lo, hi, _ in ALL_FIELDS}

PAY_HELP = (
    "-2 = tidak ada tagihan  |  -1 = lunas tepat waktu  |  "
    "0 = kredit berputar (minimum payment)  |  1–8 = terlambat sekian bulan"
)


# ── Logika prediksi ────────────────────────────────────────────────────────────

def predict(values: dict):
    df = pd.DataFrame([values])
    X  = build_features(df)
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])
    p  = float(model.predict_proba(X)[0, 1])
    return p, p >= THRESHOLD


def parse_form(form):
    values, errors = {}, []

    for name, label, lo, hi, _ in ALL_FIELDS:
        raw = (form.get(name) or "").strip()
        try:
            v = float(raw)
        except ValueError:
            errors.append(f"{label}: harus diisi dengan angka.")
            continue
        if not lo <= v <= hi:
            errors.append(f"{label}: harus antara {lo:,} dan {hi:,}.")
        values[name] = v

    for name, (label, opts) in SELECTS.items():
        try:
            v = int(form.get(name, ""))
            if v not in [o[0] for o in opts]:
                raise ValueError
            values[name] = v
        except ValueError:
            errors.append(f"{label}: pilihan tidak valid.")

    return values, errors


# ── Route utama ────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    result, errors = None, []
    form = request.form if request.method == "POST" else {}

    if request.method == "POST":
        values, errors = parse_form(request.form)
        if not errors:
            p, risky = predict(values)
            # Level risiko
            if p < 0.30:
                risk_level, risk_color = "Rendah", "low"
            elif p < THRESHOLD:
                risk_level, risk_color = "Sedang", "medium"
            else:
                risk_level, risk_color = "Tinggi", "high"

            result = {
                "prob":       p,
                "pct":        round(p * 100, 1),
                "risky":      risky,
                "label":      "⚠️ Berisiko Gagal Bayar" if risky else "✅ Tidak Berisiko",
                "threshold":  int(THRESHOLD * 100),
                "risk_level": risk_level,
                "risk_color": risk_color,
            }

    return render_template(
        "index.html",
        fields_profil=FIELDS_PROFIL,
        selects=SELECTS,
        pay_fields=PAY_FIELDS,
        bill_fields=BILL_FIELDS,
        payamt_fields=PAYAMT_FIELDS,
        form=form,
        result=result,
        errors=errors,
        pay_help=PAY_HELP,
        threshold=int(THRESHOLD * 100),
        meta=META,
    )


@app.route("/evaluasi")
def evaluasi():
    eval_dir = os.path.join(MODEL_DIR, "eval")
    # Daftar gambar yang tersedia
    charts = {}
    for key, filename in [
        ("class_distribution", "class_distribution.png"),
        ("confusion_matrix",   "confusion_matrix.png"),
        ("roc_curve",          "roc_curve.png"),
        ("feature_importance", "feature_importance.png"),
    ]:
        charts[key] = filename if os.path.exists(os.path.join(eval_dir, filename)) else None

    return render_template("evaluasi.html", meta=META, charts=charts)


@app.route("/model/eval/<path:filename>")
def eval_image(filename):
    eval_dir = os.path.join(MODEL_DIR, "eval")
    return send_from_directory(eval_dir, filename)


@app.route("/health")
def health():
    return {"status": "ok", "auc": META.get("auc"), "threshold": THRESHOLD}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
