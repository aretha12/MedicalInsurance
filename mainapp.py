import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Sistem Prediksi Risiko Asuransi Kesehatan",
    layout="centered"
)

st.title("🏥 Sistem Prediksi Risiko Kesehatan")
st.markdown(
    "**Aplikasi ini memprediksi apakah seseorang termasuk kategori risiko tinggi (high risk) "
    "berdasarkan data klinis dan gaya hidup.**"
)

@st.cache_resource
def load_models():
    xgb    = joblib.load("XGB_pkl")
    rf     = joblib.load("RF_pkl")
    dt     = joblib.load("DT_pkl")
    lr     = joblib.load("LR_pkl")
    scaler = joblib.load("scaler_pkl")
    return xgb, rf, dt, lr, scaler

try:
    xgb_model, rf_model, dt_model, lr_model, scaler = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.info("Pastikan file XGB_pkl, RF_pkl, DT_pkl, LR_pkl, dan scaler_pkl ada di direktori yang sama.")
    models_loaded = False

F1_SCORE = {
    "XGBoost": 0.86,
    "Random Forest": 0.86,
    "Decision Tree": 0.84,
    "Logistic Regression": 0.81,
}

def hitung_estimasi_biaya(age, bmi, smoker, systolic_bp, diastolic_bp,
                           hba1c, ldl, hospitalizations, medication_count,
                           prob_high):
    """
    Hitung estimasi premi asuransi tahunan (Rupiah)
    berdasarkan risk score dari faktor-faktor klinis.
    """
    TARIF_DASAR = 3_000_000 

    risk_score = 1.0

    if age < 30:
        risk_score *= 1.0
    elif age < 45:
        risk_score *= 1.3
    elif age < 60:
        risk_score *= 1.7
    else:
        risk_score *= 2.2
    if bmi < 18.5:
        risk_score *= 1.1
    elif bmi < 25:
        risk_score *= 1.0
    elif bmi < 30:
        risk_score *= 1.2
    else:
        risk_score *= 1.5
    if smoker == "Yes":
        risk_score *= 1.8
    if systolic_bp >= 140 or diastolic_bp >= 90:
        risk_score *= 1.4
    elif systolic_bp >= 130 or diastolic_bp >= 85:
        risk_score *= 1.2
    if hba1c >= 6.5:
        risk_score *= 1.5
    elif hba1c >= 5.7:
        risk_score *= 1.2
    if ldl >= 190:
        risk_score *= 1.4
    elif ldl >= 160:
        risk_score *= 1.2
    if hospitalizations >= 3:
        risk_score *= 1.5
    elif hospitalizations >= 1:
        risk_score *= 1.2
    if medication_count >= 5:
        risk_score *= 1.3
    elif medication_count >= 3:
        risk_score *= 1.1
    risk_score *= (1 + prob_high / 100)

    estimasi = TARIF_DASAR * risk_score
    batas_bawah = estimasi * 0.85
    batas_atas  = estimasi * 1.15

    return int(estimasi), int(batas_bawah), int(batas_atas)

st.sidebar.markdown("### 🤖 Pilihan Model")
st.sidebar.markdown("Secara default aplikasi menggunakan **XGBoost** (F1-Score tertinggi).")

advanced_mode = st.sidebar.toggle("⚙️ Mode Lanjutan (pilih model sendiri)", value=False)

if advanced_mode:
    model_choice = st.sidebar.selectbox(
        "Pilih Model:",
        ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"]
    )
    st.sidebar.info(f"F1-Score model ini: **{F1_SCORE[model_choice]*100:.0f}%**")
else:
    model_choice = "XGBoost"
    st.sidebar.success("✅ Menggunakan **XGBoost** (Direkomendasikan)")

st.sidebar.markdown("---")
st.sidebar.caption("Developed with 💖 using Streamlit")
st.header("📋 Data Pasien")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Informasi Umum")
    age     = st.number_input("Usia (tahun)", 18, 100, 35)
    sex     = st.radio("Jenis Kelamin", ["Male", "Female"], horizontal=True)
    bmi     = st.number_input("BMI", 10.0, 60.0, 24.0, step=0.1)
    smoker  = st.radio("Perokok?", ["No", "Yes"], horizontal=True)
    alcohol = st.selectbox("Frekuensi Alkohol", ["Never", "Occasionally", "Frequently", "Unknown"])
    employ  = st.selectbox("Status Pekerjaan", ["Employed", "Unemployed", "Self-employed", "Retired"])

with col2:
    st.subheader("🩺 Data Klinis")
    systolic_bp  = st.number_input("Systolic BP (mmHg)", 80, 220, 120)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", 50, 140, 80)
    ldl          = st.number_input("LDL (mg/dL)", 50, 300, 130)
    hba1c        = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5, step=0.1)

st.subheader("🏨 Riwayat Pelayanan Kesehatan")
c1, c2, c3 = st.columns(3)
with c1:
    visits_last_year            = st.number_input("Kunjungan (1 thn terakhir)", 0, 50, 3)
    hospitalizations_last_3yrs  = st.number_input("Rawat Inap (3 thn terakhir)", 0, 20, 0)
    days_hospitalized_last_3yrs = st.number_input("Total Hari Rawat Inap", 0, 365, 0)
    medication_count            = st.number_input("Jumlah Obat Rutin", 0, 30, 2)
with c2:
    proc_imaging_count = st.number_input("Prosedur Imaging", 0, 20, 1)
    proc_surgery_count = st.number_input("Prosedur Operasi", 0, 10, 0)
    proc_physio_count  = st.number_input("Prosedur Fisioterapi", 0, 30, 0)
with c3:
    proc_consult_count = st.number_input("Prosedur Konsultasi", 0, 30, 2)
    proc_lab_count     = st.number_input("Prosedur Lab", 0, 50, 3)
from sklearn.preprocessing import LabelEncoder

def encode_cat(val, classes):
    le = LabelEncoder()
    le.fit(classes)
    return int(le.transform([val])[0])

sex_enc     = encode_cat(sex,     ["Female", "Male"])
smoker_enc  = encode_cat(smoker,  ["No", "Yes"])
alcohol_enc = encode_cat(alcohol, ["Frequently", "Never", "Occasionally", "Unknown"])
employ_enc  = encode_cat(employ,  ["Employed", "Retired", "Self-employed", "Unemployed"])

input_data = np.array([[
    age, sex_enc, bmi, smoker_enc, alcohol_enc,
    systolic_bp, diastolic_bp, ldl, hba1c,
    visits_last_year, hospitalizations_last_3yrs,
    days_hospitalized_last_3yrs, medication_count,
    proc_imaging_count, proc_surgery_count, proc_physio_count,
    proc_consult_count, proc_lab_count
]])

if st.button("🔍 Prediksi Risiko Kesehatan", disabled=not models_loaded):

    input_scaled = scaler.transform(input_data)

    model_map = {
        "XGBoost": xgb_model,
        "Random Forest": rf_model,
        "Decision Tree": dt_model,
        "Logistic Regression": lr_model,
    }
    chosen_model = model_map[model_choice]
    prediction   = chosen_model.predict(input_scaled)[0]
    proba        = chosen_model.predict_proba(input_scaled)[0]
    prob_high    = proba[1] * 100

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    st.metric("Model Digunakan", model_choice)
    st.metric("F1-Score Model", f"{F1_SCORE[model_choice]*100:.0f}%")

    if prediction == 1:
        st.error("🔴 Pasien terindikasi **HIGH RISK** (Risiko Tinggi)")
        st.info(
            "Beberapa parameter klinis atau riwayat kesehatan menunjukkan potensi "
            "risiko tinggi. Disarankan untuk segera berkonsultasi dengan tenaga medis."
        )
    else:
        st.success("🟢 Pasien termasuk **LOW RISK** (Risiko Rendah)")
        st.info(
            "Kondisi kesehatan saat ini relatif stabil berdasarkan data yang diinputkan."
        )

    st.progress(int(prob_high), text=f"Probabilitas High Risk: **{prob_high:.1f}%**")
    st.divider()
    st.subheader("💰 Estimasi Premi Asuransi Tahunan")

    estimasi, bawah, atas = hitung_estimasi_biaya(
        age, bmi, smoker, systolic_bp, diastolic_bp,
        hba1c, ldl, hospitalizations_last_3yrs, medication_count,
        prob_high
    )

    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("Estimasi Premi", f"Rp {estimasi:,.0f}")
    col_e2.metric("Batas Bawah", f"Rp {bawah:,.0f}")
    col_e3.metric("Batas Atas", f"Rp {atas:,.0f}")

    st.caption(
        "⚠️ Estimasi ini dihitung berdasarkan risk score dari faktor klinis dan "
        "probabilitas model. Angka aktual dapat berbeda tergantung kebijakan masing-masing perusahaan asuransi."
    )

    st.markdown("**📌 Faktor yang mempengaruhi estimasi premi kamu:**")
    faktor = []
    if smoker == "Yes":             faktor.append("🚬 Perokok aktif (+80%)")
    if age >= 60:                   faktor.append("👴 Usia di atas 60 tahun (+120%)")
    elif age >= 45:                 faktor.append("🧑 Usia 45-59 tahun (+70%)")
    if bmi >= 30:                   faktor.append("⚖️ BMI obesitas (+50%)")
    if systolic_bp >= 140:          faktor.append("🩺 Hipertensi (+40%)")
    if hba1c >= 6.5:                faktor.append("🩸 HbA1c tinggi/diabetes (+50%)")
    if ldl >= 190:                  faktor.append("🫀 Kolesterol sangat tinggi (+40%)")
    if hospitalizations_last_3yrs >= 3: faktor.append("🏥 Riwayat rawat inap sering (+50%)")
    if prob_high >= 70:             faktor.append(f"🤖 Probabilitas High Risk tinggi ({prob_high:.1f}%)")

    if faktor:
        for f in faktor:
            st.markdown(f"- {f}")
    else:
        st.markdown("✅ Tidak ada faktor risiko signifikan yang terdeteksi.")
    if advanced_mode:
        st.divider()
        st.subheader("🤖 Perbandingan Semua Model")
        cols = st.columns(4)
        model_labels = ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"]
        model_objs   = [xgb_model, rf_model, dt_model, lr_model]
        for col, label, mdl in zip(cols, model_labels, model_objs):
            p      = mdl.predict(input_scaled)[0]
            prob_h = mdl.predict_proba(input_scaled)[0][1] * 100
            with col:
                if p == 1:
                    st.error(f"**{label}**\n\n🔴 High Risk\n\n{prob_h:.1f}%")
                else:
                    st.success(f"**{label}**\n\n🟢 Low Risk\n\n{prob_h:.1f}%")

    st.divider()
    st.subheader("💡 Saran Kesehatan")
    st.markdown("""
    - Lakukan pemeriksaan kesehatan secara rutin setidaknya sekali setahun  
    - Jaga tekanan darah, kadar gula, dan kolesterol dalam batas normal  
    - Hindari merokok dan kurangi konsumsi alkohol  
    - Pertahankan BMI ideal dengan pola makan seimbang dan olahraga teratur  
    - Segera konsultasi ke dokter jika muncul keluhan atau hasil lab tidak normal  
    """)

    st.subheader("📋 Ringkasan Data Input")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Usia", f"{age} thn")
    col_b.metric("BMI", f"{bmi:.1f}")
    col_c.metric("Tekanan Darah", f"{systolic_bp}/{diastolic_bp} mmHg")

st.markdown("---")
st.caption("⚠️ Aplikasi ini hanya alat bantu skrining awal dan **tidak menggantikan diagnosis dokter**.")
