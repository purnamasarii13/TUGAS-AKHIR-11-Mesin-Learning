from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os

# ✅ Pastikan Flask tahu folder templates & static saat deploy
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# ===================== 1. Load Model & Preprocessing =====================

MODEL_PATH = os.path.join(BASE_DIR, "model_stroke.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.json")

# Load model & scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def load_train_columns():
    """
    Prioritas:
    1) columns.json (paling aman untuk pd.get_dummies + reindex)
    2) scaler.feature_names_in_ (kalau scaler di-fit dari DataFrame)
    Jika tidak ada keduanya -> None
    """
    if os.path.exists(COLUMNS_PATH):
        with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
            cols = json.load(f)
        return list(cols)

    # fallback: kalau scaler menyimpan nama kolom
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)

    return None

TRAIN_COLUMNS = load_train_columns()

# ===================== 2. Utility: helpers =====================

def _get_required(form_data, key):
    if key not in form_data or form_data[key] is None or str(form_data[key]).strip() == "":
        raise ValueError(f"Field '{key}' wajib diisi.")
    return form_data[key]

def _to_float(val, key):
    try:
        return float(val)
    except Exception:
        raise ValueError(f"Field '{key}' harus angka (float).")

def _to_int(val, key):
    try:
        return int(val)
    except Exception:
        raise ValueError(f"Field '{key}' harus angka (int).")

def prepare_input(form_data):
    """
    Build 1-row DataFrame -> one-hot -> align columns -> scale
    """
    if TRAIN_COLUMNS is None:
        # Tanpa columns.json DAN scaler.feature_names_in_ -> tidak aman
        raise RuntimeError(
            "columns.json tidak ditemukan dan scaler tidak menyimpan feature names. "
            "Tidak bisa memastikan fitur input sama seperti training. "
            "Solusi: generate ulang columns.json dari training."
        )

    # Ambil & validasi data dari form
    data = {
        "gender": _get_required(form_data, "gender"),
        "age": _to_float(_get_required(form_data, "age"), "age"),
        "hypertension": _to_int(_get_required(form_data, "hypertension"), "hypertension"),
        "heart_disease": _to_int(_get_required(form_data, "heart_disease"), "heart_disease"),
        "ever_married": _get_required(form_data, "ever_married"),
        "work_type": _get_required(form_data, "work_type"),
        "Residence_type": _get_required(form_data, "Residence_type"),
        "avg_glucose_level": _to_float(_get_required(form_data, "avg_glucose_level"), "avg_glucose_level"),
        "bmi": _to_float(_get_required(form_data, "bmi"), "bmi"),
        "smoking_status": _get_required(form_data, "smoking_status"),
    }

    df = pd.DataFrame([data])

    # One-hot encoding (sesuai TA-11)
    df_enc = pd.get_dummies(df)

    # Align kolom agar sama persis seperti training
    df_enc = df_enc.reindex(columns=TRAIN_COLUMNS, fill_value=0)

    # Pastikan tidak ada NaN setelah proses (amanin)
    df_enc = df_enc.fillna(0)

    # Scale
    X_scaled = scaler.transform(df_enc).astype("float32")
    return X_scaled

# ===================== 3. Routes Flask =====================

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            X_input = prepare_input(request.form)
            prob = float(model.predict(X_input, verbose=0).ravel()[0])
            probability = round(prob, 4)
            prediction = "RISIKO STROKE (1)" if prob >= 0.5 else "TIDAK RISIKO STROKE (0)"
        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
