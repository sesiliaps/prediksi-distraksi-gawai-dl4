📱 Aplikasi Prediksi Tingkat Distraksi Penggunaan Gawai
Aplikasi web interaktif untuk memprediksi tingkat distraksi mahasiswa akibat penggunaan gadget menggunakan Deep Learning (Multi-Layer Perceptron).


📁 Struktur Folder
streamlit_app/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── mlp_model.h5
    ├── scaler.pkl
    └── label_encoder.pkl


🚀 Cara Menjalankan Aplikasi
1. Persiapan File Model
2. Install Dependencies
3. Jalankan Aplikasi


🎯 Fitur Aplikasi Yang Tersedia:
1. Input Interaktif: 7 pertanyaan dengan button skala 1-8
2. Prediksi Real-time: Menggunakan model MLP trained
3. Visualisasi Hasil: Bar chart probabilitas per kelas
4. Rekomendasi: Saran berdasarkan tingkat distraksi
5. UI/UX Modern: Desain responsive dengan gradient colors
6. Sidebar Informatif: Info model dan cara penggunaan


📊 Output Prediksi:
🟢 Rendah: Penggunaan gadget terkontrol
🟡 Sedang: Mulai mengganggu aktivitas
🔴 Tinggi: Sangat mengganggu produktivitas


🔧 Konfigurasi Model
Model yang digunakan:
1. Arsitektur: MLP (32-16-8 neurons)
2. Dropout: 0.2 - 0.1
3. Optimizer: Adam (lr=0.001)
4. Accuracy: 50.0%
5. Training Samples: 87 mahasiswa


📝 Cara Menggunakan Aplikasi
1. Buka aplikasi di browser
2. Baca instruksi di sidebar
3. Jawab 7 pertanyaan dengan klik button 1-8
4. Klik tombol "Prediksi Tingkat Distraksi"
5. Lihat hasil prediksi dan rekomendasi
6. Klik "Ulangi Kuesioner" untuk mengisi ulang


👥 Tim Pengembang
1. Sesilia Putri Subandi (122450012)
2. Tria Yunnani (122450)
3. Sahid Maulana (122450)

📄 Lisensi
Project ini dibuat untuk keperluan tugas mata kuliah Deep Learning.

📞 Kontak
Jika ada pertanyaan atau issue, silakan hubungi tim pengembang.