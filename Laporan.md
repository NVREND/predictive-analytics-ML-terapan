# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Gagal jantung merupakan salah satu penyebab kematian utama di seluruh dunia. Menurut data global, terdapat lebih dari 64 juta kasus gagal jantung dan angka ini diprediksi akan terus meningkat. Hal ini menyebabkan dampak pada individu, masyarakat, dan sistem kesehatan. Gagal jantung merupakan kondisi dimana jantung tidak dapat memompa darah dengan cukup kuat. Kondisi ini menandakan kegagalan fungsi vital jantung, organ yang bertanggung jawab atas kelangsungan hidup manusia. Prediksi gagal jantung adalah alat untuk meningkatkan hasil pasien dan mengurangi beban pada sistem kesehatan. Dengan prediksi yang akurat, kita dapat menyelamatkan nyawa dan membuka peluang untuk masa depan yang lebih cerah bagi pasien dan sistem kesehatan.

## Business Understanding

### Problem Statements

- Kurangnya Alat Sistem prediksi gagal jantung yang ada mungkin tidak cukup akurat/efisien.
- Angka kematian tinggi, lebih dari 64 juta kasus gagal jantung di seluruh dunia, dengan prediksi peningkatan di masa depan.
- Dampak yang signifikan pada individu. Dampak ini menurunkan kualitas hidup secara signifikan, memengaruhi hubungan, pekerjaan, dan kesehatan mental secara keseluruhan.

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Meningkatkan efisiensi dan akurasi alat prediksi gagal jantung dapat membantu tenaga kesehatan mengidentifikasi pasien berisiko tinggi lebih cepat dan tepat, sehingga meningkatkan peluang kesembuhan dan kelangsungan hidup pasien.
- Menurunkan angka kematian global akibat gagal jantung dengan lebih dari 64 juta kasus, sekaligus meningkatkan kualitas hidup pasien di seluruh dunia, agar terhindar dari dampak negatif yang signifikan pada individu, keluarga, dan masyarakat.
- Mengembangkan solusi untuk mengurangi dampak signifikan gagal jantung pada individu, sehingga mereka dapat menjalani kehidupan yang lebih produktif dan bahagia.

## Data Understanding
Project ini menggunakan dataset yang tersedia secara publik di Kaggle: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction. Dataset ini berisi informasi tentang pasien gagal jantung dan dimaksudkan untuk digunakan dalam tugas pemodelan prediktif.

### Variabel-variabel pada heart failur prediction dataset adalah sebagai berikut:
- Age: age of the patient [years]
- Sex: sex of the patient [M: Male, F: Female]
- ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: -Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP: resting blood pressure [mm Hg]
- Cholesterol: serum cholesterol [mm/dl]
- FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
- ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
- Oldpeak: oldpeak = ST [Numeric value measured in depression]
- ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- HeartDisease: output class [1: heart disease, 0: Normal]


## Data Preparation

Teknik data preparation yang digunakan pada project ini adalah standardization menggunakan StandardScaler dari scikit-learn. Standarization dapat meningkatkan performa dan stabilitas numerik pada model. Standarisasi diterapkan pada data training (X_train) dan data testing (X_test).

## Modeling
Dalam project ini, saya menggunakan tiga model machine learning untuk memprediksi risiko penyakit jantung, yaitu K-Nearest Neighbors (KNN) dan Rndom Forest. 
- KNN: menggunakan parameter n_neighbors=3 menetapkan jumlah tetangga terdekat yang akan digunakan untuk klasifikasi. Dalam hal ini, model akan mencari 3 data point terdekat di ruang fitur training data untuk memprediksi kelas dari data baru (titik data testing).
- Random Forest:
  - n_estimators=70 untuk menentukan jumlah pohon keputusan yang akan dibangun dalam model random forest. Semakin banyak pohon, maka akurasi model akan meningkat, tetapi membutuhkan waktu pelatihan yang lebih lama.
  - random_state=42 seeding generator bilangan random untuk memastikan reprodusibilitas hasil.

## Evaluation
Pada project ini saya menggunakan metrik akurasi, precision, recall, dan F1 score.
- Metrik yang digunakan
  - Akurasi: Merupakan metrik untuk menghitung proporsi prediksi yang benar dari semua prediksi yang dilakukan model. Nilai akurasi yang tinggi menunjukkan bahwa model mampu memprediksi sebagian besar kasus dengan benar.
    - Hasil: Random Forest memiliki akurasi testing yang sedikit lebih tinggi dibandingkan KNN. Rndom forest: akurasi training 100% dan akurasi testing 88.3%. KNN: akurasi training 90.4% dan akurasi testing 85.7%.
      
  - Precision:metrik ini berguna untuk mengukur proporsi prediksi positif yang benar-benar positif. Dihitung dengan membagi jumlah positif yang diprediksi dengan benar dengan jumlah total prediksi positif.
     - Hasil: Random Forest memiliki precision yang lebih tinggi untuk kelas positif (penyakit jantung) dibandingkan KNN. KNN memiliki precision yang lebih tinggi untuk kelas negatif (tidak memiliki penyakit jantung) dibandingkan Random Forest.
         - Random Forest: Kelas negatif (0) 0.83, Kelas positif (1) 0.93
         - KNN: Kelas negatif (0) 0.80, Kelas positif (1) 0.90

  - Recall: untuk proporsi kasus positif yang benar-benar diidentifikasi oleh model. Dihitung dengan membagi jumlah positif yang diprediksi dengan benar dengan jumlah total kasus positif yang sebenarnya. Recall yang tinggi menunjukkan bahwa model mampu mengidentifikasi sebagian besar kasus positif.
      - Hasil: Random Forest memiliki recall yang lebih tinggi untuk kelas negatif dibandingkan KNN. KNN memiliki recall yang sedikit lebih tinggi untuk kelas positif dibandingkan Random Forest.
          - Random Forest: Kelas negatif (0) 0.91. Kelas positif (1) 0.86
          - KNN: Kelas negatif (0) 0.88, Kelas positif (1) 0.84
     
  -  F1-Score: Merupakan metrik yang menggabungkan precision dan recall menjadi satu skor. Dihitung dengan rata-rata harmonis precision dan recall.
F1-Score yang tinggi menunjukkan keseimbangan yang baik antara precision dan recall, artinya model dapat memprediksi kedua kelas dengan cukup baik.
      - Hasil: F1-Score untuk kedua model berada di kisaran yang serupa.
          - Random Forest: Kelas negatif (0) 0.87, Kelas positif 0.89
          - KNN: Kelas negatif (0) 0.84, Kelas (positif (1) 0.87

- Kesimpulan: Random Forest memiliki akurasi testing yang sedikit lebih tinggi dan unggul dalam mendeteksi penyakit jantung (precision dan recall untuk kelas positif). Namun, KNN mungkin lebih baik dalam mengidentifikasi pasien yang sehat (precision untuk kelas negatif).

