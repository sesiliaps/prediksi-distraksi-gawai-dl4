import streamlit as st
import numpy as np
import pickle
from tensorflow import keras
import plotly.graph_objects as go
from datetime import datetime

# Konstanta
EXPECTED_CLASSES = {'Rendah', 'Sedang', 'Tinggi'}
QUESTIONS = [
    {
        "key": "q1",
        "title": "1️⃣ Frekuensi Penggunaan Gadget",
        "question": "Seberapa sering (frekuensi) penggunaan gadget Kamu setiap hari?",
        "description": "1 = Sangat Jarang | 8 = Sangat Sering"
    },
    {
        "key": "q2",
        "title": "2️⃣ Durasi Penggunaan",
        "question": "Berapa durasi penggunaan gadget Kamu, di luar jam perkuliahan?",
        "description": "1 = Sangat Singkat (<1 jam) | 8 = Sangat Lama (>10 jam)"
    },
    {
        "key": "q3",
        "title": "3️⃣ Tujuan Penggunaan",
        "question": "Bagaimana tujuan utama penggunaan gadget Kamu?",
        "description": "1 = Sangat Produktif (Belajar) | 8 = Sangat Tidak Produktif (Hiburan)"
    },
    {
        "key": "q4",
        "title": "4️⃣ Kesulitan Kontrol Waktu",
        "question": "Seberapa sulit Kamu dalam mengontrol waktu penggunaan gadget?",
        "description": "1 = Sangat Mudah Dikontrol | 8 = Sangat Sulit Dikontrol"
    },
    {
        "key": "q5",
        "title": "5️⃣ Persepsi Pengaruh Akademik",
        "question": "Bagaimana persepsi Kamu terhadap pengaruh penggunaan gadget pada kondisi Akademik?",
        "description": "1 = Sangat Negatif | 8 = Sangat Positif"
    },
    {
        "key": "q6",
        "title": "6️⃣ Kemampuan Mengatur Waktu",
        "question": "Bagaimana kemampuan Kamu dalam mengatur waktu antara menggunakan gadget dengan aktivitas akademik lain?",
        "description": "1 = Sangat Buruk | 8 = Sangat Baik"
    },
    {
        "key": "q7",
        "title": "7️⃣ Upaya Mengurangi Intensitas",
        "question": "Bagaimana upaya Kamu mengurangi intensitas penggunaan gadget?",
        "description": "1 = Tidak Ada Upaya | 8 = Upaya Sangat Maksimal"
    }
]


def validate_label_encoder_classes(label_encoder):
    """Validasi bahwa label_encoder memiliki semua kelas yang diharapkan."""
    actual_classes = set(label_encoder.classes_)
    missing = EXPECTED_CLASSES - actual_classes
    if missing:
        raise ValueError(f"Label encoder tidak memiliki kelas: {missing}. Ditemukan: {actual_classes}")
    return True


def build_probabilities(prediction_proba, label_encoder):
    """Build probability dict dengan validasi kelas."""
    validate_label_encoder_classes(label_encoder)
    classes = label_encoder.classes_
    return {cls: float(prediction_proba[0][i]) * 100 for i, cls in enumerate(classes)}


# KONFIGURASI HALAMAN

st.set_page_config(
    page_title="Prediksi Distraksi Gawai",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS UNTUK STYLING

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
        transition: all 0.3s;
    }
    .question-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-rendah {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .result-sedang {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .result-tinggi {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# LOAD MODEL DAN ARTIFACTS

@st.cache_resource
def load_model_artifacts():
    """Load model, scaler, dan label encoder"""
    try:
        model = keras.models.load_model('models/mlp_model.h5')
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, scaler, label_encoder, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, label_encoder, error = load_model_artifacts()


# SIDEBAR

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: #667eea; margin-bottom: 0;'>📱</h1>
            <h2 style='color: #667eea; margin-top: 0;'>Prediksi Tingkat Distraksi Penggunaan Gawai</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        ### 📋 Tentang Aplikasi
        Aplikasi ini menggunakan **Deep Learning** (Multi-Layer Perceptron) untuk 
        memprediksi tingkat distraksi akibat penggunaan gadget pada mahasiswa.
        
        ### 🎯 Tingkat Distraksi
        - **🟢 Rendah**: Penggunaan gadget masih terkontrol dengan baik
        - **🟡 Sedang**: Mulai ada gangguan pada aktivitas sehari-hari
        - **🔴 Tinggi**: Penggunaan gadget sangat mengganggu produktivitas
        
        ### 📖 Cara Menggunakan
        1. Jawab 7 pertanyaan dengan memilih skala 1-8
        2. Klik tombol **Prediksi Tingkat Distraksi**
        3. Lihat hasil prediksi dan rekomendasi
        
        ### 👥 Tim Peneliti
        **Kelompok 4 Deep Learning**
        1. Sesilia Putri Subandi
        2. Tria Yunnani
        3. Sahid Maulana  
        Institut Teknologi Sumatera
        
        ---
        
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            © 2025 | Built with Streamlit & TensorFlow
        </div>
    """, unsafe_allow_html=True)



# MAIN CONTENT

# Header
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #667eea; font-size: 2.5rem;'>
            🎓 Kuesioner Penggunaan Gadget Mahasiswa
        </h1>
        <p style='color: #666; font-size: 1.1rem;'>
            Silakan jawab pertanyaan berikut dengan jujur untuk mendapatkan prediksi tingkat distraksi Anda
        </p>
    </div>
""", unsafe_allow_html=True)

# Check if model loaded successfully
if error:
    st.error(f"""
        ⚠️ **Gagal memuat model!**
        
        Error: {error}
        
        Pastikan folder `models/` berisi file berikut:
        - mlp_model.h5
        - scaler.pkl
        - label_encoder.pkl
    """)
    st.stop()


# PERTANYAAN DAN INPUT

# Initialize session state untuk menyimpan jawaban
if 'answers' not in st.session_state:
    st.session_state.answers = {}

if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False

# Display questions dengan button input
for q in QUESTIONS:
    with st.container():
        st.markdown(f"""
            <div class='question-container'>
                <h3 style='color: #667eea; margin-bottom: 0.5rem;'>{q['title']}</h3>
                <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>{q['question']}</p>
                <p style='color: #666; font-size: 0.9rem; font-style: italic;'>{q['description']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Create 8 columns for buttons 1-8
        cols = st.columns(8)
        for i, col in enumerate(cols, 1):
            with col:
                # Cek apakah button ini yang dipilih
                is_selected = st.session_state.answers.get(q['key']) == i
                button_type = "primary" if is_selected else "secondary"
                
                if col.button(
                    f"**{i}**",
                    key=f"{q['key']}_btn_{i}",
                    type=button_type,
                    use_container_width=True
                ):
                    st.session_state.answers[q['key']] = i
                    st.session_state.prediction_done = False
                    st.rerun()
        
        # Tampilkan nilai yang dipilih
        if q['key'] in st.session_state.answers:
            selected_value = st.session_state.answers[q['key']]
            st.markdown(f"""
                <div style='text-align: center; margin-top: 0.5rem;'>
                    <span style='background-color: #667eea; color: white; padding: 0.3rem 1rem; 
                                 border-radius: 20px; font-weight: 600;'>
                        Jawaban Anda: {selected_value}
                    </span>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)



# TOMBOL SUBMIT DAN PREDIKSI

st.markdown("---")

# Cek apakah semua pertanyaan sudah dijawab
all_answered = len(st.session_state.answers) == 7

if not all_answered:
    st.warning(f"⚠️ Silakan jawab semua pertanyaan. Anda sudah menjawab {len(st.session_state.answers)}/7 pertanyaan.")
else:
    st.success("✅ Semua pertanyaan sudah dijawab! Klik tombol di bawah untuk melihat hasil prediksi.")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(
        "🔮 Prediksi Tingkat Distraksi",
        type="primary",
        disabled=not all_answered,
        use_container_width=True
    )



# PROSES PREDIKSI

if predict_button and all_answered:
    with st.spinner("🔄 Menganalisis data Anda..."):
        # Kumpulkan input dalam urutan yang benar
        input_data = np.array([[
            st.session_state.answers['q1'],
            st.session_state.answers['q2'],
            st.session_state.answers['q3'],
            st.session_state.answers['q4'],
            st.session_state.answers['q5'],
            st.session_state.answers['q6'],
            st.session_state.answers['q7']
        ]])
        
        # Normalisasi
        input_scaled = scaler.transform(input_data)
        
        # Prediksi
        prediction_proba = model.predict(input_scaled, verbose=0)
        predicted_class_idx = np.argmax(prediction_proba[0])
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Build probabilities dengan validasi
        probabilities = build_probabilities(prediction_proba, label_encoder)
        
        # Simpan hasil ke session state
        st.session_state.prediction_result = {
            'class': predicted_class,
            'probabilities': probabilities,
            'confidence': float(prediction_proba[0][predicted_class_idx]) * 100,
            'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        st.session_state.prediction_done = True


# TAMPILKAN HASIL PREDIKSI


if st.session_state.prediction_done and 'prediction_result' in st.session_state:
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
    result = st.session_state.prediction_result
    predicted_class = result['class']
    confidence = result['confidence']
    
    # Tentukan styling berdasarkan kelas
    if predicted_class == "Rendah":
        result_class = "result-rendah"
        emoji = "🟢"
        color = "#667eea"
    elif predicted_class == "Sedang":
        result_class = "result-sedang"
        emoji = "🟡"
        color = "#f5576c"
    else:
        result_class = "result-tinggi"
        emoji = "🔴"
        color = "#fa709a"
    
    # Display hasil utama
    st.markdown(f"""
        <div class='result-box {result_class}'>
            <h1 style='font-size: 3rem; margin: 0;'>{emoji}</h1>
            <h2 style='margin: 1rem 0;'>Tingkat Distraksi: {predicted_class.upper()}</h2>
            <p style='font-size: 1.3rem; margin: 0;'>Confidence: {confidence:.1f}%</p>
            <p style='font-size: 0.9rem; margin-top: 1rem; opacity: 0.9;'>
                Prediksi dilakukan pada: {result['timestamp']}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display probabilitas untuk semua kelas
    st.markdown("### 📊 Distribusi Probabilitas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #667eea; margin-bottom: 0.5rem;'>🟢 Rendah</h3>
                <h2 style='margin: 0;'>{:.1f}%</h2>
            </div>
        """.format(result['probabilities']['Rendah']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #f5576c; margin-bottom: 0.5rem;'>🟡 Sedang</h3>
                <h2 style='margin: 0;'>{:.1f}%</h2>
            </div>
        """.format(result['probabilities']['Sedang']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='metric-card'>
                <h3 style='color: #fa709a; margin-bottom: 0.5rem;'>🔴 Tinggi</h3>
                <h2 style='margin: 0;'>{:.1f}%</h2>
            </div>
        """.format(result['probabilities']['Tinggi']), unsafe_allow_html=True)
    
    # Visualisasi bar chart
    st.markdown("<br>", unsafe_allow_html=True)
    fig = go.Figure(data=[
        go.Bar(
            x=['Rendah', 'Sedang', 'Tinggi'],
            y=[
                result['probabilities']['Rendah'],
                result['probabilities']['Sedang'],
                result['probabilities']['Tinggi']
            ],
            marker_color=['#667eea', '#f5576c', '#fa709a'],
            text=[f"{v:.1f}%" for v in result['probabilities'].values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Probabilitas Prediksi Per Kelas",
        xaxis_title="Tingkat Distraksi",
        yaxis_title="Probabilitas (%)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rekomendasi berdasarkan hasil
    st.markdown("### 💡 Rekomendasi")
    
    if predicted_class == "Rendah":
        st.info("""
            **Selamat! Tingkat distraksi Anda masih rendah.**
            
            Anda sudah menunjukkan kontrol yang baik terhadap penggunaan gadget. Pertahankan kebiasaan positif ini dengan:
            - ✅ Tetap konsisten dengan jadwal yang sudah dibuat
            - ✅ Manfaatkan gadget untuk hal-hal produktif
            - ✅ Jaga keseimbangan antara aktivitas digital dan non-digital
        """)
    elif predicted_class == "Sedang":
        st.warning("""
            **Perhatian! Tingkat distraksi Anda berada di level sedang.**
            
            Mulai ada tanda-tanda penggunaan gadget mengganggu aktivitas. Pertimbangkan untuk:
            - ⚠️ Membatasi waktu penggunaan gadget untuk hiburan
            - ⚠️ Menggunakan aplikasi screen time monitoring
            - ⚠️ Menetapkan zona bebas gadget (saat belajar, makan, tidur)
            - ⚠️ Mencoba digital detox di akhir pekan
        """)
    else:
        st.error("""
            **Peringatan! Tingkat distraksi Anda tinggi.**
            
            Penggunaan gadget Anda berpotensi mengganggu produktivitas secara signifikan. Segera lakukan:
            - 🚨 Konsultasi dengan konselor akademik atau psikolog
            - 🚨 Terapkan time blocking untuk aktivitas belajar tanpa gadget
            - 🚨 Hapus aplikasi yang paling menguras waktu
            - 🚨 Aktifkan mode fokus/do not disturb saat belajar
            - 🚨 Cari aktivitas alternatif pengganti gadget (olahraga, hobi)
        """)
    
    # Ringkasan input user
    with st.expander("📋 Lihat Ringkasan Jawaban Anda"):
        summary_data = {
            "Pertanyaan": [q['question'][:50] + "..." for q in QUESTIONS],
            "Jawaban": [st.session_state.answers[q['key']] for q in QUESTIONS]
        }
        st.table(summary_data)
    
    # Tombol reset
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Ulangi Kuesioner", use_container_width=True):
            st.session_state.answers = {}
            st.session_state.prediction_done = False
            if 'prediction_result' in st.session_state:
                del st.session_state.prediction_result
            st.rerun()


# FOOTER

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p><strong>Disclaimer:</strong> Hasil prediksi ini bersifat estimasi berdasarkan model machine learning 
        dan tidak menggantikan diagnosa profesional. Gunakan sebagai panduan awal untuk memahami pola 
        penggunaan gadget Anda.</p>
        <p style='margin-top: 1rem;'>Jika Anda merasa mengalami kecanduan gadget yang serius, 
        segera konsultasikan dengan profesional kesehatan mental.</p>
    </div>
""", unsafe_allow_html=True)