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

   ![Image](https://github.com/user-attachments/assets/ffe4cd15-c125-424c-8137-e80b28b41182)

   Berdasarkan gambar boxplot di atas, dapat dilihat bahwa terdapat sejumlah outlier pada beberapa fitur numerik, yaitu pada Tumor Size, Regional Node Examined, Reginol Node Positive, dan Survival Months. Outlier terlihat jelas sebagai titik-titik yang berada di luar batas bawah dan batas atas (whisker) pada masing-masing boxplot. Khususnya pada fitur Tumor Size dan Reginol Node Positive, jumlah outlier tampak cukup banyak, yang menunjukkan adanya pasien dengan ukuran tumor atau jumlah kelenjar getah bening positif yang jauh lebih besar dibandingkan pasien lainnya. Sementara itu, fitur Age relatif lebih stabil dengan sedikit atau hampir tanpa outlier. Untuk menangani outlier yang teridentifikasi tersebut,sintaks penyelesaiannya sebagai berikut.

   <img width="357" alt="Image" src="https://github.com/user-attachments/assets/08e61e0e-ebdf-4da2-9dea-7c4131267e07" />

   Dengan menghilangkan outlier menggunakan metode IQR ini, diharapkan distribusi data menjadi lebih representatif dan model prediksi yang dibangun nantinya menjadi lebih stabil serta tidak bias akibat pengaruh nilai ekstrem.

## Data Preparation
Pada tahap Data Preparation, dilakukan serangkaian proses untuk mempersiapkan data sebelum digunakan dalam pemodelan Machine Learning. Berikut ini tahapan-tahapan data preparation yang dilakukan secara berurutan.
1. Encoding Fitur Kategorikal

   Langkah pertama adalah mengubah fitur kategorikal menjadi format numerik menggunakan teknik One-Hot Encoding. Fitur-fitur seperti Race, Marital Status, T Stage, N Stage, 6th Stage, differentiate, Grade, A Stage, Estrogen Status, Progesterone Status, dan Status dikonversi menjadi beberapa kolom baru dengan nilai biner (0 dan 1). One-Hot Encoding diperlukan karena sebagian besar algoritma Machine Learning hanya dapat memproses data numerik. Teknik ini mencegah model salah dalam menginterpretasikan nilai kategorikal yang berbentuk string.
   Setelah melakukan encoding, fitur-fitur kategorikal asli yang belum diubah dihapus dari dataset menggunakan metode drop(). Hal ini bertujuan untuk menghindari redundansi informasi dan mencegah terjadinya multikolinearitas antar fitur, yang dapat mempengaruhi kualitas dan kestabilan model Machine Learning yang akan dibangun.

2. Splitting Data

   Dataset kemudian dipisahkan menjadi dua bagian utama, yaitu fitur (X) yang berisi seluruh kolom kecuali Survival Months, dan target (y) yang berisi nilai dari Survival Months. Proses ini penting agar model dapat dilatih untuk memprediksi target berdasarkan pola yang ditemukan pada fitur. Selanjutnya, dilakukan pembagian data menggunakan teknik train-test split dengan proporsi 80% untuk training set dan 20% untuk testing set. Penggunaan random_state memastikan pembagian data dilakukan secara acak namun tetap dapat direproduksi. Pembagian ini penting untuk mengevaluasi kemampuan model dalam melakukan prediksi terhadap data yang belum pernah dilihat sebelumnya. 

3. Standardisasi Fitur Numerik

   Fitur numerik seperti Age, Tumor Size, Regional Node Examined, dan Regional Node Positive distandarisasi menggunakan teknik StandardScaler dari pustaka sklearn.preprocessing. Standardisasi ini mengubah distribusi data agar memiliki nilai rata-rata (mean) sebesar 0 dan standar deviasi sebesar 1. Proses ini sangat penting terutama untuk algoritma seperti K-Nearest Neighbor (KNN) dan Linear Regression yang sangat sensitif terhadap skala data, sehingga semua fitur memiliki kontribusi yang seimbang dalam proses pembelajaran.

4. Melihat Statistik Deskriptif Fitur Numerik

   Tahapan terakhir dalam data preparation adalah menampilkan statistik deskriptif untuk fitur numerik menggunakan fungsi describe(). Dengan melihat nilai minimum, maksimum, mean, dan standar deviasi dari fitur yang telah diproses, dapat dipastikan bahwa standardisasi sudah berjalan dengan baik dan tidak terdapat anomali pada distribusi data.

## Modelling
Pada tahap ini, dilakukan pengembangan berbagai model Machine Learning untuk menyelesaikan permasalahan prediksi kelangsungan hidup pasien kanker (Survival Months). Proses model development meliputi pemilihan algoritma, training model, serta evaluasi awal menggunakan metrik Mean Squared Error (MSE). Beberapa model yang digunakan adalah Random Forest Regressor dengan hyperparameter tuning (Grid Search), Linear Regression, dan K-Nearest Neighbors (KNN) Regressor.
1. Menyiapkan Dataframe untuk Analisis Model

   Langkah pertama dalam pengembangan model adalah menyiapkan sebuah DataFrame bernama models dengan index 'train_mse' dan 'test_mse', serta kolom untuk setiap algoritma yang diuji. Tujuannya adalah untuk menyimpan dan membandingkan nilai Mean Squared Error (MSE) dari masing-masing model secara terstruktur. Dengan menggunakan struktur ini, proses analisis performa model menjadi lebih mudah dan sistematis.

2. Random Forest Regressor dengan hyperparameter tuning (Grid Search)

   Model pertama yang dikembangkan adalah Random Forest Regressor. Model ini dibuat menggunakan pustaka sklearn.ensemble dan dituning menggunakan Grid Search dengan 5-fold cross-validation untuk mencari kombinasi hyperparameter terbaik. Parameter yang dicoba meliputi n_estimators (jumlah pohon) sebesar 50, 100, dan 150; max_depth (kedalaman maksimum pohon) sebesar 5, 10, dan 20; serta min_samples_split sebesar 2, 5, dan 10. Teknik Grid Search digunakan untuk meminimalkan risiko overfitting sekaligus menemukan konfigurasi model paling optimal. Random Forest memiliki keunggulan dalam menangani hubungan non-linear antar fitur dan cukup tahan terhadap overfitting. Namun, kekurangannya adalah model ini lebih kompleks dan interpretasinya lebih sulit dibandingkan model sederhana seperti Linear Regression.

3. Linear Regression

   Model kedua yang digunakan adalah Linear Regression, dibangun menggunakan pustaka sklearn.linear_model. Model ini dilatih pada data training tanpa melakukan hyperparameter tuning tambahan. Linear Regression sangat populer karena kesederhanaannya, kecepatan pelatihan yang tinggi, serta kemudahan dalam interpretasi koefisien model. Model ini cocok ketika hubungan antar variabel cenderung linear. Meskipun begitu, Linear Regression memiliki keterbatasan dalam menangkap hubungan non-linear antar fitur dan cukup sensitif terhadap adanya outlier di dalam data.

4. K-Nearest Neighbors (KNN) Regressor

   Model ketiga yang dikembangkan adalah K-Nearest Neighbors (KNN) Regressor dengan nilai n_neighbors=10. Model ini menggunakan pendekatan berbasis kedekatan jarak antar data untuk melakukan prediksi. KNN sangat fleksibel dalam menangkap pola lokal yang kompleks tanpa perlu asumsi bentuk hubungan antar variabel. Namun, kekurangan dari KNN adalah sensitivitas yang tinggi terhadap skala fitur, sehingga standardisasi data menjadi sangat penting. Selain itu, performa prediksi KNN cenderung lambat pada dataset yang besar karena perhitungan jarak yang dilakukan ke seluruh data training untuk setiap prediksi.

Setelah seluruh model dibangun dan dilatih, performa awal masing-masing model diukur menggunakan metrik Mean Squared Error (MSE) pada data training. Model yang dipilih sebagai model terbaik adalah Random Forest Regressor. Pemilihan ini didasarkan pada fleksibilitas Random Forest dalam menangani kompleksitas data, kemampuannya memodelkan hubungan non-linear, serta performa akurasi yang lebih tinggi dibandingkan Linear Regression maupun KNN. Dengan tambahan tuning hyperparameter melalui Grid Search, Random Forest mampu mengurangi risiko overfitting sehingga lebih stabil ketika digunakan untuk memprediksi data baru.

## Evaluasi
Pada proyek ini, karena permasalahan yang dihadapi adalah prediksi nilai kelangsungan hidup pasien kanker payudara dalam bulan, maka digunakan metrik evaluasi untuk regresi. Metrik evaluasi yang digunakan adalah Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), dan R² Score.

### Penjelasan Metrik Evaluasi
1. Root Mean Squared Error (RMSE)
   
   Root Mean Squared Error (RMSE) mengukur seberapa besar rata-rata kesalahan prediksi yang dihasilkan model, dengan penalti lebih besar untuk kesalahan yang besar, karena kesalahan di kuadratkan sebelum dirata-ratakan. RMSE dihitung menggunakan formula berikut :

   $\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }$

   Nilai RMSE yang semakin kecil menunjukkan performa model yang semakin baik.

2. Mean Absolute Error (MAE)
   Mean Absolute Error (MAE) menghitung rata-rata dari seluruh selisih absolut antara nilai aktual dan nilai prediksi. MAE menggunakan rumus berikut.

   $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|$

   MAE dianggap lebih robust terhadap outlier dibandingkan RMSE. Sama seperti RMSE, semakin kecil nilai MAE, semakin baik performa model dalam menghasilkan prediksi yang akurat.

3. R² Score

   R² Score atau koefisien determinasi mengukur seberapa banyak variansi dalam target (Survival Months) yang bisa dijelaskan oleh fitur-fitur prediktor.

   $R^2 = 1 - \frac{ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }{ \sum_{i=1}^{n} (y_i - \bar{y})^2 }$

   Nilai R² berkisar antara -∞ sampai 1, dengan nilai lebih mendekati 1 menunjukkan model yang lebih baik. R² negatif berarti performa model lebih buruk daripada sekadar memprediksi nilai rata-rata.
### Hasil Proyek Berdasarkan Metrik Evaluasi
Evaluasi dilakukan pada tiga model: Random Forest Regressor, Linear Regression, dan K-Nearest Neighbors (KNN) Regressor. Berikut ringkasan hasilnya:

| Model | Train RMSE | Test RMSE | Train MAE | Test MAE | Train R² | Test R² |
| ------ | ---------- | --------- | --------- | -------- | -------- | ------- |
| Random Forest (RF) | 18.90 | 19.98 | 15.80 | 16.29 | 0.228 | 0.129 |
| Linear Regression (LR) | 19.56 | 20.07 | 16.28 | 16.41 | 0.174 | 0.120 |
| K-Nearest Neighbors (KNN) | 19.24 | 21.67 | 15.78 | 17.63 | 0.200 | -0.026 |

   Hasil evaluasi menunjukkan bahwa Random Forest Regressor memiliki performa terbaik dibandingkan dengan Linear Regression dan K-Nearest Neighbors (KNN) Regressor. Random Forest menghasilkan nilai Test RMSE sebesar 19.98, Test MAE sebesar 16.29, dan Test R² sebesar 0.1286. Nilai RMSE dan MAE yang lebih kecil menunjukkan bahwa prediksi Random Forest lebih mendekati nilai aktual dibandingkan model lainnya. Sedangkan nilai R² yang lebih tinggi menunjukkan bahwa model ini sedikit lebih mampu menjelaskan variabilitas data dibandingkan model Linear Regression dan KNN. Sebaliknya, model KNN menunjukkan performa terburuk dengan nilai Test R² negatif, yang berarti KNN bahkan lebih buruk dibandingkan hanya memprediksi rata-rata Survival Months.
   
