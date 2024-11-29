# Laporan Proyek Machine Learning - Endritha Pramudya

## Project Domain

Heart disease is also known as cardiovascular disease which is one of the main causes of death globally.  According to WHO, cardiovascular disease claims around 17.9 million lives every year.  Cardiovascular disease is a group of heart and blood vessel disorders that includes coronary heart disease, cerebrovascular disease, rheumatic heart disease, and other conditions.  Identifying heart disease is a challenge because of the various risk factors that contribute, including diabetes, high blood pressure, high cholesterol, abnormal pulse, and several other factors.  Often, there are no symptoms of underlying disease of the blood vessels.  So many people are unaware of the risks until the condition becomes serious.  As computing and data processing capabilities continue to develop, technology can be used to analyze large amounts of data that were previously difficult to process manually.  Using machine learning algorithms, large and complex medical data can be analyzed to identify patterns and relationships that are not easily visible, allowing for more accurate heart disease risk predictions. 
Previous research conducted by (Haganta Depari et al., n.d.) used a data set of heart disease patients known as 'Personal Key Indicators of Heart Disease' and applied the Decision Tree, Naive Bayes, and Random Forest classification algorithms.  This research aims to process and analyze data, as well as apply these methods to the classification of heart disease.  The performance evaluation results show that the accuracy of the Decision Tree method is 71%, Naive Bayes is 72%, and Random Forest is 75%, with Random Forest being the best method for classifying heart disease based on the dataset used.
Machine learning models, such as Random Forest, can be trained using historical patient data to recognize risk factors associated with heart disease, such as blood pressure patterns, cholesterol levels, and medical history.  With classification techniques, this model is able to provide predictions about whether someone is at high risk of experiencing heart disease, as well as identifying risk factors that may not be immediately visible.  The results obtained from this machine learning model are very useful for medical professionals to intervene earlier and provide more appropriate treatment recommendations.  Thus, the application of machine learning can increase accuracy in early detection of heart disease and help reduce deaths related to this disease through more effective prevention.

## Business Understanding

### Problem Statements

- "Bagaimana penerapan algoritma machine learning Random Forest, dapat meningkatkan akurasi dalam mendeteksi dan memprediksi risiko penyakit jantung pada pasien?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Menganalisis performa algoritma machine learning Random Forest, dalam klasifikasi penyakit jantung

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

