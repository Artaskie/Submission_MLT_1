# Laporan Proyek Machine Learning - Achmad Faiz Izzi

## Domain Proyek

Kemajuan teknologi terkini berdampak pada berbagai bidang, seperti bidang kesehatan. Salah satu teknologi yang paling canggih adalah Machine Learning, yang merupakan cabang kecerdasan buatan yang menghasilkan algoritma untuk prediksi, pengenalan pola, dan klasifikasi data. Dalam bidang kesehatan, Machine Learning berpotensi membantu dokter memprediksi atau mendiagnosis penyakit lebih awal, sehingga tindakan medis dapat dilakukan lebih awal dan lebih berhasil.

Salah satu penyakit yang tengah menjadi sorotan adalah kanker payudara, yang merupakan penyakit yang mengancam jiwa dan rentan menyerang kaum wanita saat usianya bertambah. Berdasarkan statistik Global Cancer Observatory dari World Health Organization (WHO), pada tahun 2020 kanker payudara dilaporkan sebagai kanker terbanyak di Indonesia, dengan jumlah penderita mencapai 65.858 kasus atau 30,8% dari seluruh kanker. Tingginya angka ini banyak disebabkan oleh keterlambatan diagnosis dini, sehingga penting untuk menemukan solusi berbasis teknologi untuk mendeteksi penyakit ini sedini mungkin.

Penelitian Majid dan Nawangsih (2024) menunjukkan bahwa penerapan algoritma Machine Learning seperti Decision Tree, Naïve Bayes, dan K-Nearest Neighbor (KNN), khususnya jika dipadukan dengan teknik ensemble learning seperti AdaBoost dan Bagging, dapat membantu meningkatkan akurasi prediksi kanker payudara secara signifikan. Pada penelitian tersebut, teknik ensemble menghasilkan akurasi tertinggi, yaitu Decision Tree + Bagging menghasilkan akurasi sebesar 82,76%, sedangkan KNN + Bagging menghasilkan nilai AUC sebesar 0,950 yang tergolong sangat baik.

Berdasarkan hasil penelitian tersebut, dapat disimpulkan bahwa penerapan metode Machine Learning khususnya dengan pendekatan ensemble merupakan solusi potensial untuk meningkatkan akurasi diagnosis dini kanker payudara. Oleh karena itu, pengembangan sistem prediksi kanker payudara berbasis Machine Learning harus menjadi fokus untuk mencegah kematian akibat keterlambatan diagnosis.
[Perbandingan Metode Ensemble Untuk Meningkatkan Akurasi Algoritma Machine Learning Dalam Memprediksi Penyakit Breast Cancer (Kanker Payudara)]([https://scholar.google.com/](https://ojs.trigunadharma.ac.id/index.php/jis/index))

## Business Understanding

Dalam era perkembangan teknologi yang semakin pesat, pemanfaatan Machine Learning di bidang kesehatan menjadi salah satu inovasi penting untuk meningkatkan kualitas layanan medis. Salah satu tantangan besar dalam bidang ini adalah bagaimana mendeteksi dan memprediksi penyakit secara lebih dini dan akurat, guna meningkatkan tingkat kesembuhan pasien. Kanker, khususnya kanker payudara, merupakan salah satu penyakit dengan tingkat kematian tinggi yang membutuhkan perhatian serius dalam upaya diagnosis dini dan prediksi kelangsungan hidup pasien.

### Problem Statement
Permasalahan yang ingin diselesaikan dalam proyek ini adalah bagaimana memprediksi waktu bertahan hidup pasien kanker berdasarkan fitur klinis seperti stadium kanker, kelenjar getah bening, status hormon, dan ukuran tumor. Melalui penggunaan regresi dalam Machine Learning, diharapkan dapat dikembangkan suatu model prediksi dengan akurasi tinggi yang dapat membantu tenaga medis dalam mengambil keputusan medis yang lebih tepat dan cepat.

### Goals
Untuk mencapai tujuan tersebut, proyek ini menetapkan tiga goals utama, yaitu:
- Memprediksi jumlah bulan kelangsungan hidup (Survival Months) pasien kanker secara akurat.
- Mengidentifikasi fitur-fitur klinis yang paling berpengaruh terhadap lamanya kelangsungan hidup.

### Solution statements
#### Solution 1 : Random Forest Regressor dengan Hyperparameter Tuning (Grid Search)
- Deskripsi:
  Random Forest Regressor dipilih karena kemampuannya menangani data tabular, ketahanan terhadap outlier, dan kemampuannya memodelkan hubungan non-linear.
- Langkah-langkah:
  1. Preprocessing data numerik dan kategorikal.
  2. Split data menjadi training dan testing set.
  3. Membangun model Random Forest.
  4. Melakukan Grid Search untuk menemukan kombinasi hyperparameter terbaik (seperti n_estimators, max_depth, min_samples_split).
- Metrik Evaluasi:
  1. Root Mean Squared Error (RMSE)
  2. Mean Absolute Error (MAE)
  3. R² Score
#### Solution 2: Linear Regression
- Deskripsi:
  Linear Regression digunakan sebagai baseline model untuk membandingkan performansi model lain. Linear Regression sederhana namun efektif dalam memodelkan hubungan linear antar fitur.
- Langkah-langkah:
  1. Preprocessing fitur numerik (scaling jika diperlukan).
  2. Training model Linear Regression.
  3. Evaluasi performa di data training dan testing.
- Metrik Evaluasi:
  1. Root Mean Squared Error (RMSE)
  2. Mean Absolute Error (MAE)
  3. R² Score
#### Solution 3: K-Nearest Neighbors (KNN) Regressor
- Deskripsi:
  KNN digunakan untuk memanfaatkan kemiripan lokal antar data. Karena KNN sensitif terhadap skala, fitur numerik perlu distandarisasi.
- Langkah-langkah:
  1. Standardisasi fitur numerik menggunakan StandardScaler.
  2. Training model KNN dengan pemilihan jumlah tetangga (n_neighbors) yang optimal.
  3. Evaluasi performa di data training dan testing.
- Metrik Evaluasi:
  1. Root Mean Squared Error (RMSE)
  2. Mean Absolute Error (MAE)
  3. R² Score
## Data Understanding
Pada proyek ini, digunakan dataset Breast Cancer yang bersumber dari Kaggle. Dataset ini dapat diunduh melalui tautan berikut: https://www.kaggle.com/datasets/reihanenamdari/breast-cancer . Dataset ini berisi data klinis pasien kanker payudara, yang datanya dapat digunakan untuk memprediksi beberapa status kesehatan pasien, termasuk tingkat kelangsungan hidup. Ada satu pasien per baris dalam kumpulan data ini, dan fitur klinis yang berlaku untuk tujuan prediktif diberikan. Dataset ini berisi 16 fitur (kolom) seperti informasi demografi, status hormon, stadium kanker, dan ukuran tumor.
### Variabel-variabel dalam dataset Breast Cancer adalah sebagai berikut:
Variabel-variabel dalam dataset Breast Cancer adalah sebagai berikut:
- Age : Usia pasien saat diagnosis.
- Race : Ras pasien.
- Marital Status : Status pernikahan pasien.
- T Stage : Stadium tumor berdasarkan ukuran dan penyebaran.
- N Stage : Stadium nodus limfa.
- 6th Stage : Stadium klinis berdasarkan sistem TNM edisi ke-6.
- differentiate : Tingkat diferensiasi sel kanker.
- Grade : Tingkat keganasan tumor.
- A Stage : Stadium akhir kanker pasien.
- Tumor Size : Ukuran tumor dalam milimeter.
- Estrogen Status : Status reseptor estrogen.
- Progesterone Status : Status reseptor progesteron.
- Regional Node Examined : Jumlah kelenjar getah bening regional yang diperiksa.
- Regional Node Positive : Jumlah kelenjar getah bening yang positif kanker.
- Survival Months : Lama kelangsungan hidup pasien dalam bulan.
- Status : Status pasien.
### Exploratory Data Analysis (EDA)
Untuk memahami struktur dan distribusi data lebih baik, beberapa tahapan eksplorasi dilakukan, antara lain:
1. Cek informasi umum dataset
   - Total baris dan kolom.
   - Tipe data masing-masing kolom (object, int, float).
2. Distribusi Target (Survival Months)
   
   Visualisasi histogram ini digunakan untuk melihat apakah distribusinya normal atau skewed. Berikut merupakan visualisasinya.
   
   ![Image](https://github.com/user-attachments/assets/f2369d60-75be-43a2-b0c1-7a1027d07a69)

   Distribusi data Survival Months dalam dataset ini bersifat tiidak simetris, di mana frekuensi bersifat kumulatif dalam nilai Survival Months yang relatif tinggi, yaitu antara 60 dan 100 bulan. Namun relatif lebih rendah dalam nilai yang relatif lebih rendah, yaitu antara 0 hingga 40 bulan. Sesuai pola ini, dapat dinyatakan bahwa distribusi Survival Months termasuk dalam kategori left-skewed (negatively skewed), meskipun jumlah kemiringannya tidak tinggi sama sekali. Distribusi left-skewed ini menandakan bahwa sebagian besar pasien memiliki waktu bertahan hidup yang cukup lama, dan ekor distribusi meluas ke nilai yang lebih kecil.
4. Cek missing values

   Berdasarkan pengerjaannya, diperoleh bahwa dataset Breast Cancer tidak ada missing value.
   
5. Boxplot fitur numerik

Berikut merupakan analisis outlier untuk fitur Age, Tumor Size, Regional Node Examined, Regional Node Positive, dan Survival Months.



Berdasarkan gambar boxplot di atas, dapat dilihat bahwa terdapat sejumlah outlier pada beberapa fitur numerik, yaitu pada Tumor Size, Regional Node Examined, Reginol Node Positive, dan Survival Months. Outlier terlihat jelas sebagai titik-titik yang berada di luar batas bawah dan batas atas (whisker) pada masing-masing boxplot. Khususnya pada fitur Tumor Size dan Reginol Node Positive, jumlah outlier tampak cukup banyak, yang menunjukkan adanya pasien dengan ukuran tumor atau jumlah kelenjar getah bening positif yang jauh lebih besar dibandingkan pasien lainnya. Sementara itu, fitur Age relatif lebih stabil dengan sedikit atau hampir tanpa outlier. Untuk menangani outlier yang teridentifikasi tersebut,sintaks penyelesaiannya sebagai berikut.



Dengan menghilangkan outlier menggunakan metode IQR ini, diharapkan distribusi data menjadi lebih representatif dan model prediksi yang dibangun nantinya menjadi lebih stabil serta tidak bias akibat pengaruh nilai ekstrem.
