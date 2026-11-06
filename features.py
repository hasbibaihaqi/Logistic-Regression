"""Definisi fitur bersama untuk train.py dan app.py (supaya preprocessing identik).

Dataset: UCI Credit Card Default (30.000 baris, 25 kolom)
Target : default.payment.next.month  (0 = Tidak Gagal Bayar, 1 = Gagal Bayar)
"""
import pandas as pd

# ── Kolom numerik yang akan di-scale dengan StandardScaler ─────────────────────
NUMERIC_COLS = [
    "LIMIT_BAL",
    "AGE",
    # Riwayat pembayaran (6 bulan)
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    # Tagihan bulanan (6 bulan)
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    # Jumlah pembayaran (6 bulan)
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# ── Variabel kategorikal & levelnya ────────────────────────────────────────────
CATEGORICAL_LEVELS = {
    "SEX":       [1, 2],        # 1=Laki-laki, 2=Perempuan
    "EDUCATION": [1, 2, 3, 4],  # 1=Pascasarjana, 2=Universitas, 3=SMA, 4=Lainnya
    "MARRIAGE":  [1, 2, 3],     # 1=Menikah, 2=Lajang, 3=Lainnya
}

# Kolom dummy (drop-first encoding) → urutan kolom final yang dilihat model
DUMMY_COLS = [
    f"{c}_{v}"
    for c, levels in CATEGORICAL_LEVELS.items()
    for v in levels[1:]
]

FEATURE_COLUMNS = NUMERIC_COLS + DUMMY_COLS  # panjang: 20 + 6 = 26


# ── Fungsi preprocessing ───────────────────────────────────────────────────────

def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Perbaiki kode EDUCATION & MARRIAGE yang di luar range standar."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encoding dengan level tetap; kolom numerik tetap mentah
    (scaling dilakukan terpisah oleh StandardScaler).
    """
    df = clean_categories(df)
    out = df[NUMERIC_COLS].astype(float).copy()
    for col, levels in CATEGORICAL_LEVELS.items():
        for v in levels[1:]:
            out[f"{col}_{v}"] = (df[col].astype(int) == v).astype(float)
    return out[FEATURE_COLUMNS]
