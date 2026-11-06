# TA-03 – Prediksi Risiko Gagal Bayar Kartu Kredit

> **Aplikasi web Flask** berbasis **Regresi Logistik + SMOTE** untuk memprediksi kemungkinan nasabah gagal bayar kartu kredit bulan berikutnya.

---

## 📋 Deskripsi

Sistem ini membantu analis kredit memasukkan data nasabah dan menerima prediksi instan:
- **"⚠️ Berisiko Gagal Bayar"** atau **"✅ Tidak Berisiko"**
- Beserta skor probabilitas, level risiko (Rendah/Sedang/Tinggi), dan visualisasi progress bar.

**Dataset:** UCI Credit Card Default (30.000 nasabah, 25 variabel)
**Fitur yang digunakan:** 26 fitur (20 numerik + 6 dummy kategorikal)

---

## 🗂️ Struktur Proyek

```
credit-risk-app/
├── dataset/
│   └── UCI_Credit_Card.csv      ← Dataset UCI (30.000 baris)
├── model/                        ← Hasil training (dibuat otomatis dari notebook)
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── meta.json
│   └── eval/                     ← Grafik evaluasi
│       ├── class_distribution.png
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── feature_importance.png
├── static/
│   └── style.css                 ← Stylesheet (dark mode support)
├── templates/
│   ├── index.html                ← Halaman utama (formulir + hasil)
│   └── evaluasi.html             ← Halaman evaluasi model
├── training_notebook.ipynb       ← Notebook training step-by-step (EDA, SMOTE, LogReg, Evaluasi)
├── features.py                   ← Definisi fitur & preprocessing (shared)
├── app.py                        ← Flask app (2 route utama)
├── requirements.txt
├── Procfile                      ← Untuk Railway / Render
└── README.md
```

---

## 🚀 Langkah Menjalankan

### 1. Training Menggunakan Jupyter Notebook
Buka dan jalankan file notebook:
```bash
jupyter notebook training_notebook.ipynb
# atau buka langsung training_notebook.ipynb di editor (VS Code / JupyterLab)
```
Jalankan sel secara berurutan. Notebook ini akan:
- Menampilkan EDA & visualisasi class imbalance
- Melakukan pembersihan data, One-Hot Encoding, dan penyesuaian skala fitur
- Menyeimbangkan data latih menggunakan SMOTE
- Melatih model Regresi Logistik
- Menghasilkan metrik evaluasi lengkap (Confusion Matrix, Precision, Recall, F1, AUC)
- Menyimpan artefak model ke `model/` (`model.pkl`, `scaler.pkl`, `meta.json`, dan grafik di `model/eval/`)

### 2. Jalankan Aplikasi Web Flask
Setelah model tersimpan, jalankan web app:
```bash
pip install -r requirements.txt
python app.py
```
Buka browser ke: **http://localhost:5000**

---

## 📊 Halaman Aplikasi

| URL | Deskripsi |
|-----|-----------|
| `/` | Formulir input nasabah + hasil prediksi |
| `/evaluasi` | Confusion Matrix, ROC Curve, Feature Importance, Distribusi Kelas |
| `/health` | Health-check JSON |

---

## 🔧 Fitur yang Digunakan Model

**Numerik (di-scale dengan StandardScaler):**
- `LIMIT_BAL` – Batas kredit
- `AGE` – Usia
- `PAY_0` s/d `PAY_6` – Status pembayaran 6 bulan terakhir
- `BILL_AMT1` s/d `BILL_AMT6` – Tagihan bulanan 6 bulan
- `PAY_AMT1` s/d `PAY_AMT6` – Jumlah pembayaran 6 bulan

**Kategorikal (One-hot encoded, drop-first):**
- `SEX` – Jenis kelamin
- `EDUCATION` – Tingkat pendidikan
- `MARRIAGE` – Status pernikahan

---

## 📈 Metodologi

1. **EDA** – Analisis distribusi fitur dan ketidakseimbangan kelas
2. **Preprocessing** – Pembersihan data, encoding kategorikal, StandardScaler
3. **SMOTE** – Mengatasi class imbalance pada data latih
4. **Regresi Logistik** – `LogisticRegression(max_iter=1000)`
5. **Evaluasi** – Confusion Matrix, Precision, Recall, F1-Score, AUC-ROC

> **Recall (Gagal Bayar)** adalah metrik prioritas karena False Negative (nasabah berisiko diprediksi aman) menimbulkan kerugian nyata bagi perusahaan.

---

## ☁️ Deployment

### Render
1. Push folder ini ke GitHub (sertakan folder `model/`)
2. Render → New Web Service → pilih repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`

### Railway
Sudah tersedia `Procfile`:
```
web: gunicorn app:app
```

### Variabel Lingkungan
| Variabel | Default | Keterangan |
|----------|---------|------------|
| `THRESHOLD` | `0.5` | Ambang keputusan (0.4 untuk prioritaskan Recall) |
| `MODEL_DIR` | `./model` | Folder model |
| `PORT` | `5000` | Port server |
