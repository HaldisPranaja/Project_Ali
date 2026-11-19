import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="Analisis Avian Influenza (HPAI)",
    page_icon="🦠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "Avian Influenza (HPAI).csv"
REQUIRED = [
    "Year", "Month", "Day", "Time", "Latitude", "Longitude",
    "State", "County", "target_H5_HPAI"
]
NUMERIC = ["Year", "Month", "Day", "Time", "Latitude", "Longitude"]
CATEGORICAL = ["State", "County"]
FEATURES = NUMERIC + CATEGORICAL

st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 3rem;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px;
       background: linear-gradient(120deg,#102a43,#176b87);
       color: white; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin: 0;}
.hero p {color: #e5f3f8; margin: .5rem 0 0 0;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.08);
    border: 1px solid rgba(128,128,128,.22); padding: 14px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>Dashboard Analisis Avian Influenza</h1>
<p>Eksplorasi dataset, data cleaning, pelatihan Random Forest, evaluasi,
dan ekspor hasil prediksi H5/HPAI.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Dataset")
if DEFAULT_DATA.exists():
    st.sidebar.success("Dataset contoh bawaan tersedia.")
else:
    st.sidebar.warning("Dataset bawaan tidak ditemukan. Silakan unggah CSV.")

uploaded = st.sidebar.file_uploader(
    "Unggah CSV lain (opsional, untuk mengganti dataset bawaan)",
    type=["csv"],
)
st.sidebar.divider()
st.sidebar.subheader("Konfigurasi model")
test_size = st.sidebar.slider(
    "Proporsi data testing", 0.10, 0.40, 0.20, 0.05
)
n_estimators = st.sidebar.number_input(
    "Jumlah pohon (n_estimators)", 50, 1000, 200, 50
)
max_depth = st.sidebar.number_input(
    "Kedalaman maksimum (max_depth)", 2, 100, 15, 1
)
random_state = st.sidebar.number_input(
    "Random state", 0, 9999, 42, 1
)
run_analysis = st.sidebar.button(
    "Jalankan analisis", type="primary", use_container_width=True
)

# Gunakan CSV bawaan jika pengguna tidak mengunggah file pengganti.
try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        dataset_source = uploaded.name
    elif DEFAULT_DATA.exists():
        raw = pd.read_csv(DEFAULT_DATA)
        dataset_source = DEFAULT_DATA.name
    else:
        raw = None
        dataset_source = None
except Exception as exc:
    st.error(f"Dataset gagal dibaca: {exc}")
    st.stop()

if raw is None:
    st.warning("Belum ada dataset. Unggah file CSV melalui panel sebelah kiri.")
    st.stop()

missing_cols = [col for col in REQUIRED if col not in raw.columns]
if missing_cols:
    st.error("Kolom wajib tidak ditemukan: " + ", ".join(missing_cols))
    st.write("Kolom yang tersedia:", list(raw.columns))
    st.stop()

st.caption(f"Dataset aktif: **{dataset_source}**")
st.subheader("1. Ringkasan dataset")
m1, m2, m3 = st.columns(3)
m1.metric("Jumlah baris", f"{len(raw):,}")
m2.metric("Jumlah kolom", f"{raw.shape[1]:,}")
m3.metric("Missing values", f"{int(raw.isna().sum().sum()):,}")

with st.expander("Pratinjau dataset", expanded=True):
    st.dataframe(raw.head(10), use_container_width=True)
    st.write("Distribusi target:")
    st.dataframe(
        raw["target_H5_HPAI"].value_counts(dropna=False)
        .rename_axis("Target").to_frame("Jumlah"),
        use_container_width=True,
    )

st.subheader("2. Jalankan analisis")
st.write(
    "Tekan tombol **Jalankan analisis** di panel kiri. "
    "Aplikasi akan membersihkan data, melatih Random Forest, "
    "dan menampilkan evaluasi."
)

if run_analysis:
    try:
        progress = st.progress(0, text="Menyiapkan data...")
        df = raw.copy()
        before_rows = len(df)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        missing_before = df.isna().sum()
        duplicate_count = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()
        df["County"] = df["County"].fillna("Unknown")
        progress.progress(15, text="Membersihkan data...")

        X = df[FEATURES].copy()
        y = df["target_H5_HPAI"].copy()

        for col in NUMERIC:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if y.isna().any():
            raise ValueError("Kolom target_H5_HPAI memiliki nilai kosong.")
        if y.nunique() != 2:
            raise ValueError(
                f"Target harus memiliki 2 kelas. Ditemukan: {y.nunique()}."
            )
        if set(pd.Series(y).unique()) != {0, 1}:
            raise ValueError(
                "Nilai target harus berupa 0 dan 1 agar evaluasi HPAI sesuai."
            )

        numeric_pipe = SimpleImputer(strategy="median")
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL),
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size),
            random_state=int(random_state), stratify=y,
        )
        progress.progress(35, text="Melatih Random Forest...")
        classifier = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            class_weight="balanced",
            random_state=int(random_state),
            n_jobs=-1,
        )
        model = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", classifier),
        ])
        model.fit(X_train, y_train)
        progress.progress(75, text="Menghitung evaluasi...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
        }
        results = X_test.copy()
        results["Actual_HPAI"] = y_test.to_numpy()
        results["Predicted_HPAI"] = y_pred
        results["Probability_HPAI_1"] = y_prob

        st.session_state["hpai_result"] = {
            "df": df,
            "before_rows": before_rows,
            "missing_before": missing_before,
            "duplicate_count": duplicate_count,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "model": model,
            "results": results,
            "metrics": metrics,
        }
        progress.progress(100, text="Analisis selesai.")
        st.success("Analisis berhasil dijalankan.")
    except Exception as exc:
        st.error(f"Analisis gagal: {type(exc).__name__}: {exc}")
        st.exception(exc)

if "hpai_result" not in st.session_state:
    st.info("Dataset sudah dimuat. Klik **Jalankan analisis** pada panel kiri untuk memulai.")
    st.stop()

result = st.session_state["hpai_result"]
st.subheader("3. Data cleaning")
c1, c2, c3 = st.columns(3)
c1.metric("Baris sebelum cleaning", f"{result['before_rows']:,}")
c2.metric("Duplikasi dihapus", f"{result['duplicate_count']:,}")
c3.metric("Baris setelah cleaning", f"{len(result['df']):,}")

with st.expander("Missing values sebelum cleaning"):
    st.dataframe(
        result["missing_before"].rename("Missing Values").to_frame(),
        use_container_width=True,
    )
with st.expander("Data setelah cleaning"):
    st.dataframe(result["df"].head(10), use_container_width=True)

st.subheader("4. Evaluasi model Random Forest")
metric_cols = st.columns(5)
for col, (name, score) in zip(metric_cols, result["metrics"].items()):
    col.metric(name, f"{score:.4f}" if name == "ROC-AUC" else f"{score:.2%}")

tab_cm, tab_fi, tab_compare, tab_roc = st.tabs([
    "Confusion Matrix", "Feature Importance", "Actual vs Predicted", "ROC Curve"
])

with tab_cm:
    cm = confusion_matrix(result["y_test"], result["y_pred"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["0", "1"], yticklabels=["0", "1"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - HPAI")
    st.pyplot(fig)
    plt.close(fig)

with tab_fi:
    names = result["model"].named_steps["preprocessing"].get_feature_names_out()
    values = result["model"].named_steps["classifier"].feature_importances_
    fi = pd.DataFrame({"Feature": names, "Importance": values})
    fi = fi.sort_values("Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=fi.head(10), x="Importance", y="Feature", ax=ax)
    ax.set_title("Top 10 Feature Importance")
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(fi.head(10), use_container_width=True)

with tab_compare:
    actual = pd.Series(result["y_test"]).value_counts()
    predicted = pd.Series(result["y_pred"]).value_counts()
    compare = pd.DataFrame({
        "Actual": [actual.get(0, 0), actual.get(1, 0)],
        "Predicted": [predicted.get(0, 0), predicted.get(1, 0)],
    }, index=["0", "1"])
    st.bar_chart(compare)
    st.dataframe(compare, use_container_width=True)

with tab_roc:
    fpr, tpr, _ = roc_curve(result["y_test"], result["y_prob"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"Random Forest (AUC={result['metrics']['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("5. Classification report")
report = classification_report(
    result["y_test"], result["y_pred"], zero_division=0, output_dict=True
)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.subheader("6. Hasil prediksi dan download")
st.dataframe(result["results"].head(25), use_container_width=True)

model_file = io.BytesIO()
joblib.dump(result["model"], model_file)
model_file.seek(0)
st.download_button(
    "Download model Random Forest (.pkl)",
    data=model_file,
    file_name="HPAI_RandomForest_Model.pkl",
    mime="application/octet-stream",
)
st.download_button(
    "Download hasil prediksi (.csv)",
    data=result["results"].to_csv(index=False).encode("utf-8"),
    file_name="HPAI_Prediction_Results.csv",
    mime="text/csv",
)
metric_df = pd.DataFrame(
    [{"Metric": key, "Score": value} for key, value in result["metrics"].items()]
)
st.download_button(
    "Download evaluasi (.csv)",
    data=metric_df.to_csv(index=False).encode("utf-8"),
    file_name="HPAI_Model_Evaluation.csv",
    mime="text/csv",
)

st.caption(
    "Catatan: hasil ini merupakan keluaran model machine learning dari dataset, "
    "bukan diagnosis atau peringatan resmi kesehatan hewan."
)
