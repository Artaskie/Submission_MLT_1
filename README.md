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
