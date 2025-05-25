# Laporan Proyek Machine Learning - Ryanza Aufa Yansa

## Domain Proyek (Bisnis)

Kedai kopi merupakan salah satu sektor bisnis yang berkembang pesat dalam beberapa tahun terakhir. Berdasarkan data Statista, pasar kopi Indonesia menghasilkan pendapatan US$8.84 miliar dari *segmen out-of-home* (kedai kopi/restoran), dengan pertumbuhan CAGR 3.5% [1]. Pertumbuhan ini didorong oleh perubahan preferensi konsumen menuju gaya hidup urban dan konsumsi kopi sebagai bagian dari rutinitas harian [2]. Namun, fluktuasi pendapatan harian menjadi tantangan utama bagi pelaku bisnis, terutama dalam mengelola operasional dan pemasaran. Faktor-faktor seperti persaingan yang ketat, fluktuasi harga bahan baku, dan ketergantungan pada tren pasar dapat memengaruhi stabilitas pendapatan harian kedai kopi [3]. Sebagai contoh, studi kasus pada Kedai Kopi Raga di Bekasi menunjukkan bahwa persaingan dengan 21 kedai kopi dalam radius 1 km menyebabkan penurunan pendapatan yang signifikan [4]. Selain itu, fluktuasi harga bahan baku seperti biji kopi, susu, dan gula yang dipengaruhi oleh berbagai faktor seperti musim panen dan kebijakan impor turut menjadi tantangan dalam menjaga margin keuntungan [3].

Untuk mengatasi tantangan tersebut, dibutuhkan pendekatan berbasis data yang mampu memprediksi pendapatan harian secara akurat dan adaptif. Seiring berkembangnya teknologi, machine learning telah menjadi salah satu solusi unggulan dalam mengolah data dan menghasilkan prediksi yang presisi. Machine learning telah terbukti efektif dalam berbagai konteks bisnis, termasuk dalam memprediksi pendapatan usaha kecil dan menengah berdasarkan data historis seperti jumlah pelanggan, pengeluaran harian, dan tren pembelian. Pendekatan machine learning seperti Random Forest, XGBoost, dan Neural Networks memberikan akurasi yang tinggi dalam peramalan pendapatan harian pada sektor UKM [5]. Oleh sebab itu, *machine learning* akan digunakan untuk memprediksi *daily revenue* pada suatu kedai kopi. Pada proyek ini, terdapat 5 algoritma yang akan digunakan, yaitu *Linear Regression*, *Decision Tree Regressor*, *Random Forest Regressor*, *XGBoost Regressor* dan *Support Vector Regression*. Lalu, dataset yang akan digunakan adalah dataset yang ada pada kaggle dengan nama [**Coffee Shop Daily Revenue Prediction Dataset**](https://www.kaggle.com/datasets/himelsarder/coffee-shop-daily-revenue-prediction-dataset/data).

## Business Understanding

### Problem Statements
Rumusan masalah yang bisa didapatkan:
1. Faktor-faktor apa saja yang paling mempengaruhi *daily revenue* dari suatu kedai kopi.
2. Menemukan performa model terbaik dalam memprediksi *daily revenue* berdasarkan metriks evaluasi.

### Goals:
Tujuan yang bisa didapatkan dari rumusan masalah:
1. Mengidentifikasi faktor-faktor yang paling berpengaruh terhadap *daily revenue*.
2. Mengevaluasi performa model untuk menentukan model terbaik berdasarkan metriks evaluasi seperti *Mean Absolute Error* (MAE), *Root Mean Squared Error* (RMSE) dan *R-squared* (R² Score).

### Solution Statement
1. Mengidentifikasi variabel-variabel yang paling berdampak terhadap pendapatan harian melalui analisis korelasi dan *feature importance* dari model yang digunakan.
2. Menerapkan dan membandingkan beberapa model machine learning dalam melakukan prediksi, seperti Linear Regression, Decision Tree, Random Forest, SVR, dan XGBoost, dengan menggunakan metrik evaluasi seperti MAE, RMSE, dan R².

## Data Understanding
Dataset yang digunakan untuk memprediksi *daily revenue* diambil dari [**Kaggle**](https://www.kaggle.com/datasets/himelsarder/coffee-shop-daily-revenue-prediction-dataset/data). Dataset tersebut dipublikasikan oleh Himel Sarder. Dataset ini berisi data operasional harian dari sebuah kedai kopi fiktif, mencakup informasi seperti jumlah pelanggan, pengeluaran marketing, nilai rata-rata transaksi, hingga jam operasional dan jumlah karyawan. Dataset terdiri dari 1 file CSV yang memuat total **2000 baris data** dan **7 kolom fitur**, dengan target prediksi berupa daily revenue dalam satuan dolar. Dataset ini memiliki tingkat *usability* mencapai 10.00/10.00.

### *Exploratory Data Analysis* (EDA)

#### Memahami Struktur Data
![info_dataset](https://github.com/user-attachments/assets/da924985-2d1d-4ad1-aa24-a6b223c62943)

Dataset memuat total **2000 baris data** dan **7 kolom** diawal. Berikut uraian 7 kolom: 
1. **Number of Customers Per Day**  
    Total jumlah pelanggan yang mengunjungi kedai kopi pada suatu hari tertentu.  
    Rentang: 50 – 500 pelanggan.

2. **Average Order Value ($)**  
    Rata-rata jumlah uang (dalam dolar) yang dihabiskan oleh setiap pelanggan selama kunjungan mereka.  
    Rentang: $2.50 – $10.00.

3. **Operating Hours Per Day**  
    Jumlah total jam kedai kopi buka dan beroperasi setiap harinya.  
    Rentang: 6 – 18 jam.

4. **Number of Employees**  
    Jumlah karyawan yang bekerja pada hari tertentu.  
    Faktor ini dapat memengaruhi kecepatan layanan, kepuasan pelanggan, dan pada akhirnya, penjualan.  
    Rentang: 2 – 15 karyawan.

5. **Marketing Spend Per Day ($)**  
    Jumlah uang yang dikeluarkan untuk kampanye pemasaran atau promosi pada hari tertentu.  
    Rentang: $10 – $500 per hari.

6. **Location Foot Traffic (people/hour)**  
    Jumlah orang yang melintasi kedai kopi setiap jam.  
    Variabel ini mencerminkan lokasi kedai serta potensinya untuk menarik pelanggan. 
    Rentang: 50 – 1000 orang per jam.
   
7. **Daily Revenue ($)**  
    Ini adalah variabel dependen yang merepresentasikan total pendapatan yang dihasilkan oleh kedai kopi setiap hari. Nilai ini dihitung berdasarkan kombinasi jumlah kunjunga
    pelanggan, rata-rata pengeluaran, serta faktor operasional lain seperti pengeluaran untuk promosi dan ketersediaan staf. 
    Rentang: $200 – $10.000 per hari.

#### Identifikasi Missing dan Duplicate Values
![image](https://github.com/user-attachments/assets/c842a8e9-d4a1-4df8-9988-abbbdfd29bdd)

Dataset tidak memiliki **missing values** dan **duplicate values**.

#### Analisis Deskriptif dan Univariate Analysis
|        | Number_of_Customers_Per_Day | Average_Order_Value | Operating_Hours_Per_Day | Number_of_Employees | Marketing_Spend_Per_Day | Location_Foot_Traffic | Daily_Revenue |
|----------------|------------------------------|----------------------|--------------------------|----------------------|--------------------------|------------------------|----------------|
| count          | 2000.000000                  | 2000.000000          | 2000.000000              | 2000.000000          | 2000.000000              | 2000.000000            | 2000.000000    |
| mean           | 274.296000                   | 6.261215             | 11.667000                | 7.947000             | 252.614160               | 534.893500             | 1917.325940    |
| std            | 129.441933                   | 2.175832             | 3.438608                 | 3.742218             | 141.136004               | 271.662295             | 976.202746     |
| min            | 50.000000                    | 2.500000             | 6.000000                 | 2.000000             | 10.120000                | 50.000000              | -58.950000     |
| 25% (Q1)       | 164.000000                   | 4.410000             | 9.000000                 | 5.000000             | 130.125000               | 302.000000             | 1140.085000    |
| 50% (Median)   | 275.000000                   | 6.300000             | 12.000000                | 8.000000             | 250.995000               | 540.000000             | 1770.775000    |
| 75% (Q3)       | 386.000000                   | 8.120000             | 15.000000                | 11.000000            | 375.352500               | 767.000000             | 2530.455000    |
| max            | 499.000000                   | 10.000000            | 17.000000                | 14.000000            | 499.740000               | 999.000000             | 5114.600000    |

Berikut informasi yang didapat dari informasi deskriptif tersebut:
- **Jumlah pelanggan** yang datang ke kafe kurang lebih berkisar dari **50 sampai 500** orang dengan rata-rata per harinya **274** orang. Kafe berada pada kondisi teramai dengan jumlah pelanggan **499 pelanggan** per hari.
- Rata-rata pengeluaran pelanggan adalah **6.26 dollar** per orang dengan pengeluaran tertinggi adalah **10 dollar** dan terendah **2.5 dollar** per orang.
- **Jam opreasional** kafe berkisar dari **6 jam dan 17 jam**. Rata-rata jam kafe beroperasional adalah **11.6 jam**.
- **Jumlah karyawan** yang bekerja berkisar dari **2 sampai 14 karyawan** dengan rata-rata jumlah karyawan kurang lebih sebanyak **8 karyawan**.
- Rata-rata **pengeluaran** yang dilakukan kafe untuk **marketing** dan **promosi** kurang lebih sebesar **252.6 dollar** per hari dengan pengeluaran tertinggi mencapai **499.74 dollar** dan yang terendah adalah **10.12 dollar**.
- **Jumlah orang** yang berlalu lalang di depan kafe pernah mencapai di angka **999 orang/jam**.
- **Total pendapatan** kafe pernah mencapai **5114.60 dollar** per harinya dan mencapai titik terendahnya yaitu **-58.95 dollar** per harinya. Rata- rata **total pendapatan** yang diraih kafe sebesar **1917.32 dollar** per hari.

Berikut visualisasi persebaran data
![dist_data](https://github.com/user-attachments/assets/08256e9e-062e-4227-bece-6a7b196a9439)


Berikut informasi yang didapat dari grafik persebaran data:
1. **Number of Customers Per Day:**

   - Distribusinya **cukup merata**, artinya jumlah pelanggan harian tersebar hampir secara konsisten dari minimum hingga maksimum.
   - Hal ini menunjukkan tidak ada dominasi hari tertentu dengan jumlah pelanggan sangat tinggi/rendah.

2. **Average Order Value:**

   - **Tidak ada nilai yang cukup ekstrem**, dan semua nilai cukup terdistribusi rata, mengindikasikan bahwa rata-rata belanja per pelanggan cenderung seragam antar hari.

3. **Operating Hours Per Day:**

   - Distribusi terlihat **diskrit** dan terbagi dalam beberapa titik. Jumlah data pada setiap jam cukup seimbang, namun ada lebih banyak data di sekitar **12-16 jam**.

4. **Number of Employees:**

   - Fitur ini memiliki distribusi diskrit karena jumlah karyawan biasanya bernilai integer bulat kecil. Jumlah pegawai cenderung berkisar di tengah-tengah .

5. **Marketing Spend Per Day:**

   - Terlihat juga distribusinya **cukup merata**, tanpa dominasi nilai tertentu. Hal ini menunjukkan mungkin strategi marketing dilakukan secara konsisten dalam jangka waktu tertentu.

6. **Location Foot Traffic:**

   - Distribusinya **cukup merata**, tetapi terdapat beberapa fluktuasi (puncak dan lembah). Hal ini bisa dipengaruhi oleh hari dalam minggu atau musim, yang patut dianalisis lebih lanjut bila informasi waktu tersedia.

7. **Daily Revenue:**

   - Distribusi agak skewed **skewed ke kanan** (right-skewed). Banyak nilai revenue harian berada di rentang 1000–2500. Ini menunjukkan bahwa hanya sebagian kecil hari yang menghasilkan pendapatan sangat tinggi, sedangkan mayoritas hari berada di kisaran pendapatan menengah-ke-rendah.

Berikut ini visualisasi boxplot untuk identifikasi outlier
![boxplot_graph](https://github.com/user-attachments/assets/4c4a24d8-47e3-4bf7-86a8-576505d0f985)

Berikut informasi yang didapat dari visualisasi boxplot untuk identifikasi outlier:
- **Daily Revenue:**
    - Terdapat beberapa outlier di nilai yang tinggi.

#### Analisis Korelasi dan Multivariate Analysis
![jumlah_karyawan_daily_revenue](https://github.com/user-attachments/assets/676617b3-287b-4426-afad-3de5912c381d)

![mean_pengeluaran_daily_revenue](https://github.com/user-attachments/assets/1aa476dc-118c-435f-8b0b-6243bcebd9e4)

Berikut hasil yang didapat dari grafik rata-rata *daily revenue* per jumlah karyawan dan rata-rata *daily revenue* per rata-rata pengeluaran pelanggan:
- Rata-rata Daily Revenue **tidak menunjukkan pola** yang sangat jelas terhadap jumlah karyawan. **Terdapat fluktuasi nilai**, namun perbedaannya tidak terlalu signifikan. Artinya, menambah jumlah karyawan **tidak serta-merta meningkatkan** revenue secara **konsisten**.
-  Rata-rata Daily Revenue terhadap jam operasional juga **tidak menunjukkan pola** yang sangat jelas. Beberapa jam operasional yang lebih pendek (seperti 9–11 jam) justru memberikan **mean revenue yang lebih tinggi**. Kemungkinan efisiensi operasional berada pada **jam-jam tertentu** (bukan semakin lama, semakin tinggi).

![pairplot](https://github.com/user-attachments/assets/f9d11dab-889d-428c-b235-13c9c6c45606)

![heatmap](https://github.com/user-attachments/assets/2cef102c-a670-4dbc-a3c2-f76434e12493)

Berikut informasi mengenai korelasi yang didapat dari pairplot dan heatmap:
- Pada pairplot, terlihat jelas bahwa hanya beberapa pasangan fitur yang menunjukkan korelasi visual kuat. **Number of Customers** dan **Average Order Value** adalah dua faktor paling relevan terhadap pendapatan. Fitur lainnya tampak memiliki korelasi lemah atau tidak signifikan secara visual.
- Dan pada heatmap korelasi, nilai korelasi paling kuat terhadap Daily Revenue adalah **Number of Customers** (0.74), **Average Order Value** (0.54) dan **Marketing Spend Per Day** (0.25).

## Data Preparation
Tahap ini mencakup pembersihan dan transformasi data agar siap digunakan dalam pemodelan machine learning. Tahapan penting yang dimaksud seperti menangani missing values, menangani data duplikat, menangani outlier, menangani skewness, melakukan normalisasi atau standarisasi pada fitur numerik dan encoding fitur kategorikal.

### Penanganan Outlier
![del_outlier](https://github.com/user-attachments/assets/e7fb481e-575a-42cd-8c90-c7df504f792b)

Setelah menangani outlier yang ada pada *daily revenue*, struktur data mengalami sedikit perubahan. Jumlah data yang awalnya **2000** menjadi **1991** data.

### Mengecek Skewness Value
![skew_val](https://github.com/user-attachments/assets/cf36a72f-960d-4095-beec-65976335e2aa)

Skewness terbesar dialami oleh **kolom target/Daily Revenue** (0.593703), tetapi sepertinya nilai skewness tersebut tidak terlalu besar dan cenderung moderat, jadi mungkin untuk sekarang akan dibiarkan terlebih dahulu untuk melihat prediksi yang **lebih interpretatif**.

### Feature Engineering dan Selection
![feat_eng](https://github.com/user-attachments/assets/6e4d15c3-5516-45f3-be0a-dd44d86a2c6e)

Berdasarkan korelasi yang didapat, terdapat beberapa fitur baru yang bisa ditambahkan dengan proses **Feature Enggineering**. Fitur-fitur baru ini juga bisa membantu model menangkap hubungan non-linear yang tidak terlihat hanya dari korelasi linier antar fitur asli.
- Berdasarkan korelasi yang tinggi antara jumlah pelanggan dan pendapatan, kamu membentuk fitur baru seperti **Revenue_per_Customer**, yang menunjukkan rata-rata kontribusi pendapatan dari setiap pelanggan. Ini membantu model memahami efisiensi pelanggan secara lebih baik.

- Fitur **Marketing_Efficiency** dibentuk dari rasio antara pendapatan dan pengeluaran marketing. Ini mencerminkan ROI dari strategi pemasaran, yang sangat relevan dalam konteks bisnis.

- Fitur **Customer_Spend** merupakan kombinasi antara jumlah pelanggan dan rata-rata nilai transaksi, yang logis karena perkalian dua fitur penting akan menghasilkan prediksi kasar dari pendapatan harian.

**Feature selection** dilakukan dengan tetap mempertahankan fitur asli yang paling relevan secara statistik dan memperkaya dengan fitur hasil engineering. Proses ini penting untuk menghindari **overfitting**, meningkatkan **efisiensi model**, serta mempertajam kemampuan **generalisasi** terhadap data baru.

### Splitting dan Scaling Data
![train_test_split](https://github.com/user-attachments/assets/ec80d202-c7d6-4974-97ab-a4038e1786e2)

Berdasarkan prinsip evaluasi model yang adil, data dibagi menjadi dua bagian: data latih (80%) dan data uji (20%) menggunakan `train_test_split()`. Hal ini dilakukan untuk memastikan bahwa performa model dapat diukur secara objektif terhadap data yang belum pernah dilihat sebelumnya (test set). Proses scaling menggunakan `StandardScaler` dilakukan agar semua fitur memiliki skala yang setara (mean = 0 dan standard deviation = 1). Ini penting terutama untuk algoritma yang sensitif terhadap skala fitur, seperti **Linear Regression**, **SVM**, **KNN**, dan **algoritma berbasis gradien**.

## Modeling
Pada tahap ini dilakukan proses **pelatihan (training)** model machine learning menggunakan data yang telah diproses sebelumnya. Beberapa algoritma regresi digunakan untuk memprediksi **daily revenue** dari coffee shop, di antaranya **Linear Regression, Decision Tree Regressor, Random Forest Regressor, XGBoost Regressor, dan Support Vector Regressor (SVR)**.

Setelah proses pelatihan, model dievaluasi menggunakan **data uji (testing)** untuk mengukur kinerjanya. Evaluasi dilakukan dengan tiga metrik utama:

- **Mean Absolute Error (MAE)**: Mengukur rata-rata kesalahan absolut antara nilai prediksi dan nilai aktual.

- **Root Mean Squared Error (RMSE)**: Mengukur akar dari rata-rata kuadrat kesalahan; memberikan penalti lebih besar untuk kesalahan yang besar.

- **R-squared (R²)**: Menunjukkan seberapa besar variasi data yang dapat dijelaskan oleh model. Nilai mendekati 1 menandakan model memiliki performa yang baik.

![eval_mode_func](https://github.com/user-attachments/assets/1aeff1ef-4f7a-4f9a-9500-cfcc8b624674)

Fungsi `evaluate_model()` digunakan untuk menghitung dan menampilkan ketiga metrik ini untuk masing-masing model, sehingga dapat dibandingkan secara objektif guna menentukan model terbaik yang akan digunakan untuk prediksi lebih lanjut.

### 1. Linear Regression
Linear Regression adalah metode statistik yang digunakan untuk memprediksi nilai numerik (kontinu) dari sebuah variabel target berdasarkan hubungan linier antara satu atau lebih variabel independen (fitur). Model ini mengasumsikan bahwa terdapat hubungan linier antara input dan output, yang direpresentasikan dengan persamaan garis lurus. Linear regression digunakan secara luas karena kesederhanaannya dan interpretabilitasnya yang tinggi, namun performanya menurun jika hubungan antar variabel tidak linier atau terdapat multikolinearitas.

Model dilatih dengan `X_train_scaled` dan `y_train` sekaligus diuji menggunakan `X_test_scaled` dan `y_test` sebagai data uji. Model ini menggunakan parameter hasil dari cross-validation dan hyperparameter tuning dengan **optuna**. Berikut parameter yang digunakan:
- 'fit_intercept': True
memberitahu model untuk mencari nilai intercept terbaik, dan biasanya pilihan yang aman jika kamu belum melakukan preprocessing khusus pada data fitur

### 2. Decision Tree Regressor
Decision Tree Regressor adalah model prediktif yang menggunakan struktur pohon untuk membagi dataset menjadi subset yang lebih kecil berdasarkan nilai fitur, sehingga menghasilkan prediksi nilai numerik di setiap daun (leaf). Model ini bekerja dengan membuat aturan keputusan (if-else) yang meminimalkan error pada setiap percabangan. Decision tree sangat fleksibel dan dapat menangkap hubungan non-linier antara fitur dan target, namun rentan terhadap overfitting jika tidak dilakukan proses pruning atau pengaturan kedalaman pohon.

Model dilatih dengan `X_train` dan `y_train` sekaligus diuji menggunakan `X_test` dan `y_test` sebagai data uji. Model ini menggunakan parameter hasil dari cross-validation dan hyperparameter tuning dengan **optuna**. Berikut parameter yang digunakan:
- max_depth = 12
Menandakan kedalaman maksimal pohon keputusan cukup dalam untuk menangkap pola data secara detail, tapi belum terlalu dalam sehingga tidak overfitting.

- min_samples_split = 6
Minimal 6 data sampel harus ada pada suatu node agar bisa dilakukan split. Ini membantu mencegah pohon membelah terlalu kecil, sehingga model tidak terlalu kompleks.

- min_samples_leaf = 4
Setiap leaf node minimal harus memiliki 4 sampel, ini juga bentuk regularisasi agar tidak ada leaf terlalu kecil yang membuat model overfit.

### 3. Random Forest Regressor
Random Forest Regressor adalah model ensemble yang terdiri dari banyak decision tree yang dibangun menggunakan data subset acak dan subset fitur acak. Prediksi akhir diperoleh dengan merata-ratakan hasil dari seluruh pohon. Model ini memperbaiki kelemahan decision tree tunggal dengan cara mengurangi overfitting dan meningkatkan generalisasi. Random forest sangat efektif dalam menangani dataset dengan banyak fitur dan hubungan kompleks, serta cukup robust terhadap noise dan outlier.

Model dilatih dengan `X_train` dan `y_train` sekaligus diuji menggunakan `X_test` dan `y_test` sebagai data uji. Model ini menggunakan parameter hasil dari cross-validation dan hyperparameter tuning dengan **optuna**. Berikut parameter yang digunakan:
- n_estimators = 233
Jumlah pohon dalam ensemble sebanyak 233, memberikan keseimbangan antara performa dan waktu komputasi.

- max_depth = 19
Maksimal kedalaman pohon 19, cukup dalam untuk menangkap pola yang kompleks tanpa overfitting berlebihan.

- min_samples_split = 2
Setiap node minimal memiliki 2 sampel untuk bisa di-split, ini mengizinkan pohon untuk membelah hingga cukup rinci.

- min_samples_leaf = 1
Leaf node minimal memiliki 1 sampel, memungkinkan pembentukan leaf yang sangat spesifik.

### 4. XGBoost Regressor
XGBoost Regressor adalah algoritma berbasis gradient boosting yang dikembangkan untuk efisiensi dan performa tinggi. Model ini membangun pohon-pohon keputusan secara bertahap, di mana setiap pohon baru berusaha memperbaiki kesalahan dari pohon sebelumnya. XGBoost menggunakan teknik regularisasi, penanganan missing value, dan optimasi berbasis histogram untuk meningkatkan akurasi dan kecepatan pelatihan. XGBoost sangat populer dalam kompetisi machine learning karena kemampuannya dalam menangani dataset besar dan kompleks dengan performa tinggi.

Model dilatih dengan `X_train` dan `y_train` sekaligus diuji menggunakan `X_test` dan `y_test` sebagai data uji. Model ini menggunakan parameter hasil dari cross-validation dan hyperparameter tuning dengan **optuna**. Berikut parameter yang digunakan:
- n_estimators (300): Jumlah pohon keputusan yang digunakan cukup banyak, memungkinkan model menangkap pola data dengan baik.

- max_depth (4): Kedalaman pohon relatif dangkal, ini membantu menghindari overfitting dengan membuat model lebih sederhana.

- learning_rate (0.1538): Learning rate moderat, artinya setiap pohon berkontribusi secara bertahap dalam pembelajaran, menjaga kestabilan model.

- subsample (0.6901): Model menggunakan sekitar 69% data training secara acak setiap iterasi, teknik ini untuk menambah keanekaragaman model (bagging) dan mengurangi overfitting.

- colsample_bytree (0.8753): Sekitar 87% fitur dipilih secara acak untuk setiap pohon, membantu meningkatkan generalisasi model.


### 5. Support Vector Reggresor (SVR)
Support Vector Regressor (SVR) adalah varian dari Support Vector Machine (SVM) yang digunakan untuk tugas regresi. SVR mencoba mencari garis (atau hiperplane) terbaik yang memiliki deviasi maksimal ε dari semua titik data dalam batas toleransi tertentu. Tujuannya bukan meminimalkan error secara langsung, tetapi menjaga agar sebagian besar prediksi berada dalam margin ε dari nilai sebenarnya. SVR sangat efektif untuk data berdimensi tinggi dan dapat menggunakan kernel untuk menangani hubungan non-linier, namun sensitif terhadap pemilihan parameter dan skala fitur.

Model dilatih dengan `X_train_scaled` dan `y_train` sekaligus diuji menggunakan `X_test_scaled` dan `y_test` sebagai data uji. Model ini menggunakan parameter hasil dari cross-validation dan hyperparameter tuning dengan **optuna**. Berikut parameter yang digunakan:
- C=32.91127633637073
Parameter regularisasi yang mengontrol trade-off antara kesalahan pada data pelatihan dan kompleksitas model. Nilai besar (32.9) menunjukkan model memberi toleransi lebih rendah terhadap kesalahan pelatihan, sehingga model berusaha fit lebih dekat ke data, tapi tetap menjaga generalisasi.

- epsilon=0.07958824918507265
Parameter ini menentukan margin toleransi kesalahan di dalam fungsi loss SVR — artinya, prediksi yang berbeda dari target dalam jarak epsilon tidak dianggap sebagai kesalahan. Nilai kecil 0.0795 menunjukkan model sangat sensitif terhadap kesalahan kecil, sehingga berusaha fit data dengan presisi tinggi.

- gamma='auto'
Gamma menentukan jangkauan pengaruh tiap titik data terhadap model.
Dengan nilai 'auto', gamma disetel ke 1 / n_features, artinya pengaruh tiap titik data menyesuaikan dengan jumlah fitur input secara otomatis. Ini membantu menyeimbangkan kompleksitas model pada data dengan fitur berjumlah sedang.

- kernel='linear'
Fungsi kernel linear digunakan, yang berarti model memetakan data ke ruang fitur asli tanpa transformasi nonlinear.

Setelah parameter model telah disesuaikan dengan hyperparameter tuning, selanjutnya model akan dijalankan. Hasil yang didapatkan akan dievaluasi dengan metriks evaluasi. Berikut hasil yang didapat:
| Model             | MAE        | RMSE       | R2       |
|-------------------|------------|------------|----------|
| Linear Regression | 123.648025 | 157.140958 | 0.974647 |
| Decision Tree     | 133.900078 | 175.833154 | 0.968256 |
| Random Forest     | 94.366597  | 126.860647 | 0.983476 |
| XGBoost           | 43.534817  | 57.049491  | 0.996658 |
| SVR               | 123.934737 | 158.110272 | 0.974333 |

Berdasarkan hasil yang didapat, **XGBoost Regressor** dipilih sebagai model terbaik karena model tersebut memiliki nilai MAE dan RMSE terendah serta R² tertinggi  di antara semua model yang dijalankan. XGBoost memberikan keseimbangan terbaik antara kesalahan kecil **(MAE dan RMSE rendah)** dan kemampuan menjelaskan varians data **(R² tinggi)**.

## Evaluation
Pada proyek ini, penilaian model dilakukan menggunakan metrik evaluasi Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan R-squared (R²) untuk masing-masing model regresi. Sebelum membahas hasil evaluasi, akan dijelaskan terlebih dahulu cara kerja serta interpretasi dari ketiga metrik ini: 
- **Mean Absolute Error (MAE)** mengukur rata-rata dari selisih absolut antara nilai prediksi dan nilai aktual. Metrik ini menunjukkan seberapa besar kesalahan rata-rata model dalam satuan asli data, tanpa memperhitungkan arah kesalahan (positif atau negatif).

- **Root Mean Squared Error (RMSE)** adalah akar dari rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual. RMSE lebih sensitif terhadap kesalahan besar dibanding MAE karena kesalahan dikuadratkan terlebih dahulu sebelum dirata-ratakan, sehingga memberikan penalti lebih besar terhadap prediksi yang jauh meleset.

- **R-squared (R²)** atau koefisien determinasi, mengukur proporsi varians dalam data target yang dapat dijelaskan oleh model. Nilai R² berkisar antara 0 hingga 1; semakin mendekati 1, semakin baik model menjelaskan variasi dalam data.

Ketiga metrik ini memberikan gambaran komprehensif mengenai performa model dari sisi ketepatan prediksi dan seberapa baik model menjelaskan data. Pemilihan model terbaik didasarkan pada kombinasi nilai MAE dan RMSE yang paling rendah serta nilai R² yang paling tinggi.

### Penerapannya dalam Pemilihan Model
#### 1. Linear Regression
- **MAE**: 123.65:
rata-rata kesalahan prediksi model ini sekitar 123.65 satuan. Nilai ini tergolong cukup besar jika dibandingkan dengan model lain.
- **RMSE**: 157.14:
Menunjukkan adanya kesalahan prediksi besar yang cukup signifikan, karena **RMSE** memberikan penalti lebih besar pada error ekstrem.
- **R²**: 0.9746:
Model ini dapat menjelaskan sekitar 97.46% variabilitas dalam data target. Meski cukup baik, masih kalah dibanding model dengan **R²** lebih tinggi.

#### 2. Decision Tree Regressor
- **MAE**: 133.900078
Nilai **MAE** tertinggi di antara semua model, menunjukkan kesalahan rata-rata prediksi yang besar.
- **RMSE**: 175.833154:
Juga merupakan nilai RMSE tertinggi, menunjukkan bahwa model sering melakukan prediksi dengan kesalahan besar.
- **R²**: 0.968256:
Merupakan R² terendah, yang berarti model ini memiliki kemampuan penjelasan variabilitas target yang paling lemah di antara semua model yang diuji.

#### 3. Random Forest Regressor
- **MAE**: 94.366597
Lebih baik dibanding Linear Regression dan Decision Tree. Artinya, rata-rata kesalahan prediksi lebih kecil.
- **RMSE**: 126.860647
Nilai ini juga lebih rendah, menunjukkan Random Forest mampu mengurangi kesalahan besar lebih baik dibanding dua model sebelumnya.
- **R²**: 0.983476
Model ini dapat menjelaskan sekitar 98.35% variabilitas target. Hasil ini menunjukkan performa yang sangat baik.

#### 4. XGBoost Regressor
- Nilai **MAE (Mean Absolute Error)** Terendah: 43.534817
**MAE XGBoost** adalah 43.534817, yang berarti rata-rata kesalahan prediksi model ini paling kecil dibanding model lain. Semakin kecil MAE, semakin akurat model dalam memprediksi nilai sebenarnya.
- Nilai **RMSE (Root Mean Squared Error)** Terendah: 57.049491
**RMSE XGBoost** sebesar 57.049491, yang juga merupakan nilai terkecil di antara semua model. RMSE lebih sensitif terhadap error besar, jadi nilai RMSE yang kecil menunjukkan model ini minim kesalahan prediksi besar.
- Nilai **R² (Koefisien Determinasi)** Tertinggi: 0.996658
R² XGBoost adalah 0.996658, mendekati 1. Ini berarti model dapat menjelaskan sekitar 99.66% variabilitas data target, menunjukkan fit yang sangat baik.

#### 5. Support Vector Reggresor (SVR)
- **MAE**: 123.934737
Hampir setara dengan Linear Regression, menunjukkan kesalahan rata-rata yang cukup besar.
- **RMSE**: 158.110272
Hampir sama dengan Linear Regression, menandakan adanya kesalahan besar dalam beberapa prediksi.
- **R²**: 0.974333
Cukup tinggi, namun masih di bawah Random Forest dan XGBoost.

Berikut visualisasi perbandingan metriks evaluasi setiap model yang digunakan:
![perbandingan_mae](https://github.com/user-attachments/assets/fa50db58-cd61-42a2-9eab-5355016228f0)
![perbandingan_rmse](https://github.com/user-attachments/assets/cafdad2c-9c44-4de2-9104-d7481961d649)
![perbandingan_rsquared](https://github.com/user-attachments/assets/e9d75c2a-7c8f-4dff-b74f-6b2f12752960)

Berdasarkan hasil yang didapat, **XGBoost Regressor** dipilih sebagai model terbaik karena model tersebut memiliki nilai MAE dan RMSE terendah serta R² tertinggi  di antara semua model yang dijalankan. XGBoost memberikan keseimbangan terbaik antara kesalahan kecil **(MAE dan RMSE rendah)** dan kemampuan menjelaskan varians data **(R² tinggi)**. Hal ini membuat model ini sangat andal untuk **prediksi yang akurat**.

### Visualisasi Nilai Prediksi VS Aktual
Berikut visualisasi nilai prediksi dan aktual
![prediksi_vs_aktual](https://github.com/user-attachments/assets/ba538dfb-0a6d-4f2b-a2ea-87ae44aad228)

Pola sebaran titik-titik sangat dekat dengan garis merah putus-putus (garis y = x), ini menunjukkan bahwa prediksi model sangat akurat, karena semakin dekat titik ke garis tersebut, semakin kecil selisih antara nilai aktual dan prediksi. Distribusi prediksi merata di sepanjang rentang nilai target, artinya model tidak hanya bagus di rentang rendah atau tinggi, tapi juga stabil di seluruh distribusi nilai target. Tidak terlihat pola lengkung atau deviasi besar yang bisa menunjukkan underfitting atau overfitting. Artinya model terkalibrasi dengan baik.

### Faktor yang Paling Mempengaruhi *Daily Revenue*
Berikut visualisasi *feature importances*
![feat_importances](https://github.com/user-attachments/assets/0a632788-da2d-49bd-b4fa-a29d9264d81d)

Pada visualisasi tersebut, terlihat bahwasanya 3 faktor yang sangat berpengaruh pada nilai daily revenue pada suatu kedai kopi ialah pengeluaran pelanggan selama berada di kedai kopi, jumlah pelanggan yang datang ke kedai kopi per hari dan efisiensi marketing. **Kesimpulannya**, jika ingin mendapatkan hasil yang maksimal kedai harus meningkatkan **rata-rata pembelanjaan per pelanggan** melalui strategi seperti upselling atau promosi menu bernilai tinggi, menarik **lebih banyak pelanggan** ke kedai melalui peningkatan kualitas layanan, lokasi strategis, atau kerja sama komunitas dan mengoptimalkan **efektivitas kampanye marketing**, agar biaya promosi menghasilkan dampak maksimal terhadap jumlah kunjungan dan pembelian.

## Referensi
[1] Statista. (2025). Coffee - Indonesia: Market Forecast. Diakses pada 22 Mei 2025 dari: https://www.statista.com/outlook/cmo/hot-drinks/coffee/indonesia

[2] Sumara, R. (2024). Integrating SWOT Analysis and Business Model Canvas: A Strategic Approach for Indonesian Coffee Shops. eCo-Buss, 7(1), 588-589.

[3] Kompas.id. (2025). Mengintip Tren dan Tantangan Bisnis Kopi 2025. Diakses pada 22 Mei 2025 dari: https://www.kompas.id/artikel/mengintip-tren-dan-tantangan-bisnis-kopi-2025

[4] Wibowo, A. N. (2022). Strategi Pemasaran Kedai Kopi Raga, Kota Bekasi. Universitas Islam Negeri Syarif Hidayatullah Jakarta. Diakses pada 22 Mei 2025 dari: https://repository.uinjkt.ac.id/dspace/bitstream/123456789/65171/1/ANGGUN%20NOVANDRIE%20WIBOWO-FST.pdf

[5] Ramli, R., et al. (2024). Revenue Forecasting for Small and Medium Enterprises Using Machine Learning Approaches: A Systematic Literature Review. Discover Artificial Intelligence.

[6] Dicoding. Diakses pada 22 Mei 2025 dari: https://www.dicoding.com/academies/319-machine-learning-terapan
