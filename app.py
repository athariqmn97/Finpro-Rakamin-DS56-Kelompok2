
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_models.joblib")

# Home Page
def home():
    st.title("📊 Employee Churn Prediction App")
    st.markdown("""
    Selamat datang di aplikasi prediksi churn karyawan DANN Co.

    Aplikasi ini membantu perusahaan untuk:
    - Memprediksi kemungkinan karyawan akan resign.
    - Menjalankan simulasi prediksi secara individu maupun batch.
    - Menganalisis potensi penghematan biaya jika churn dapat dicegah.
    """)

# Individual Prediction Page
def predict_individual():
    st.title("🔍 Prediksi Churn Karyawan (Individu)")
    st.markdown("Masukkan data karyawan untuk memprediksi kemungkinan churn.")

    # Input fields
    age = st.number_input("Usia", min_value=18, max_value=65, value=30)
    experience_years = st.number_input("Pengalaman Kerja (tahun)", min_value=0, max_value=40, value=5)
    monthly_target = st.number_input("Target Bulanan", value=100)
    target_achievement = st.slider("Pencapaian Target (%)", 0.0, 1.0, 0.7)
    working_hours = st.number_input("Jam Kerja per Minggu", value=40)
    overtime_hours = st.number_input("Jam Lembur per Minggu", value=5)
    salary = st.number_input("Gaji Bulanan (juta)", value=5.0)
    commission_rate = st.slider("Komisi (%)", 0.0, 1.0, 0.05)
    tenure = st.number_input("Lama Kerja di Perusahaan (tahun)", value=2)
    job_satisfaction = st.slider("Kepuasan Kerja (1-4)", 1, 4, 2)
    manager_support = st.slider("Dukungan Manajer (1-4)", 1, 4, 2)
    distance = st.number_input("Jarak ke Kantor (km)", value=10)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    marital_status = st.selectbox("Status Pernikahan", ["Single", "Married"])
    education = st.selectbox("Pendidikan", ["High School", "Diploma", "Bachelor"])
    work_location = st.selectbox("Lokasi Kerja", ["Urban", "Suburban", "Rural"])

    if st.button("Prediksi Churn"):
        input_df = pd.DataFrame([{
            "age": age,
            "experience_years": experience_years,
            "monthly_target": monthly_target,
            "target_achievement": target_achievement,
            "working_hours_per_week": working_hours,
            "overtime_hours_per_week": overtime_hours,
            "salary": salary,
            "commission_rate": commission_rate,
            "company_tenure_years": tenure,
            "job_satisfaction": job_satisfaction,
            "manager_support_score": manager_support,
            "distance_to_office_km": distance,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "work_location": work_location
        }])

        # Encoding & preprocessing should match training pipeline
        # For simplicity, assume model accepts raw input
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.success(f"Prediksi: {'Churn' if prediction == 1 else 'Tidak Churn'}")
        st.info(f"Probabilitas Churn: {prob:.2f}")

# Batch Prediction Page
def predict_batch():
    st.title("📁 Prediksi Churn Batch (Upload CSV)")
    st.markdown("Upload file CSV berisi data karyawan untuk prediksi churn secara massal.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview Data:", df.head())

        predictions = model.predict(df)
        df["Churn Prediction"] = predictions
        churn_count = (predictions == 1).sum()

        st.success(f"Jumlah karyawan yang diprediksi akan churn: {churn_count}")
        st.download_button("Download Hasil Prediksi", df.to_csv(index=False), file_name="churn_predictions.csv")

# Cost Analysis Page
def cost_analysis():
    st.title("💰 Analisis Penghematan Biaya")
    st.markdown("Jika kita berhasil mencegah churn, kita dapat menghemat biaya rekrutmen dan onboarding sebesar **Rp 4.000.000** per karyawan.")

    churn_prevented = st.number_input("Masukkan jumlah churn yang berhasil dicegah", min_value=0, value=10)
    saving = churn_prevented * 4_000_000
    st.metric(label="Total Penghematan Biaya", value=f"Rp {saving:,.0f}")

# Sidebar Navigation
page = st.sidebar.selectbox("Navigasi", ["Home", "Prediksi Individu", "Prediksi Batch", "Analisis Biaya"])
if page == "Home":
    home()
elif page == "Prediksi Individu":
    predict_individual()
elif page == "Prediksi Batch":
    predict_batch()
elif page == "Analisis Biaya":
    cost_analysis()
