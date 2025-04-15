Laporan Proyek Machine Learning - Wildan Abdurrasyid
=============================================

* * * * *

Domain Proyek
-------------

**Latar Belakang Masalah:**
Keterlambatan pembayaran faktur merupakan permasalahan yang krusial bagi perusahaan karena dapat mengganggu *cash flow* dan operasional. Dengan memprediksi status pembayaran faktur (lunas atau belum), perusahaan dapat mengantisipasi potensi keterlambatan dan mengoptimalkan strategi penagihan. Masalah ini harus diselesaikan untuk meningkatkan efisiensi pengelolaan keuangan dan mengurangi risiko piutang tak tertagih.

**Mengapa Masalah Ini Penting:**

-   **Manajemen Keuangan yang Lebih Baik:** Memprediksi status faktur membantu perusahaan menjaga arus kas yang lebih stabil dengan mengidentifikasi faktur yang berpotensi terlambat.
-   **Strategi Penagihan yang Tepat:** Dengan mengetahui faktur mana yang masih terbuka (*isOpen* = 1), perusahaan bisa memprioritaskan upaya penagihan pada akun-akun tersebut.

**Referensi dan Riset Terkait:**

-   [Optimizing Cash Flow Management](https://scholar.google.com/) -- Referensi yang mendalam mengenai pengaruh prediksi pembayaran terhadap manajemen keuangan.
-   [Predictive Analytics in Finance](https://scholar.google.com/) -- Studi kasus dan metode machine learning dalam dunia keuangan.

* * * * *

Business Understanding
----------------------

**Problem Statements:**

-   **Pernyataan Masalah 1:** Banyak perusahaan mengalami kesulitan dalam memprediksi status pembayaran faktur (apakah akan lunas atau masih terbuka melewati tenggat waktu), yang berdampak pada perencanaan *cash flow*.
-   **Pernyataan Masalah 2:** Kurangnya identifikasi dini terhadap faktur yang berpotensi masih terbuka membuat strategi penagihan menjadi kurang proaktif dan efektif.

**Goals:**

-   **Tujuan 1:** Memprediksi status faktur (*isOpen*) berdasarkan data historis sehingga perusahaan dapat mengidentifikasi faktur yang berisiko belum lunas dan merencanakan arus kas dengan lebih baik.
-   **Tujuan 2:** Mengembangkan model klasifikasi yang akurat untuk membedakan antara faktur yang sudah lunas (*isOpen* = 0) dan yang masih terbuka (*isOpen* = 1), guna mendukung strategi penagihan yang lebih efektif.

### Solution Statements

-   **Solusi Pertama:**
    Membangun *baseline model* menggunakan **Logistic Regression** untuk memprediksi status faktur (*isOpen*). Model ini akan memberikan acuan performa awal dan mudah diinterpretasikan karena sifat linearnya.

-   **Solusi Kedua:**
    Mengimplementasikan **Random Forest** sebagai model alternatif yang diharapkan mampu menangani kompleksitas dan hubungan non-linear dalam data dengan lebih baik. Dilakukan juga *hyperparameter tuning* menggunakan **GridSearchCV** untuk mencari kombinasi parameter terbaik dan meningkatkan performa model Random Forest.

    *Evaluasi kedua model (Logistic Regression dan Random Forest sebelum/sesudah tuning) dilakukan dengan metrik seperti classification report (precision, recall, f1-score), confusion matrix, dan ROC-AUC untuk mengukur dan membandingkan performa secara kuantitatif.*

* * * * *

Data Understanding
------------------

Dataset yang digunakan dalam proyek ini berasal dari Kaggle dan berisi informasi historis pembayaran faktur.

**Sumber Dataset:**
Dataset dapat diunduh melalui tautan: [Kaggle Payment Date Dataset](https://www.kaggle.com/datasets/rajattomar132/payment-date-dataset)

**Informasi Dasar Dataset:**

-   **Jumlah Data:** Dataset terdiri dari 50.000 baris (sampel) dan 19 kolom (fitur).
-   **Kondisi Data:**
    -   *Missing Values:* Terdapat *missing value* pada kolom `clear_date` (10.000 nilai) dan `invoice_id` (6 nilai). Kolom `area_business` memiliki *missing value* di seluruh baris (50.000 nilai).
    -   *Duplikat:* Tidak dilakukan pengecekan duplikat secara eksplisit dalam notebook, namun ini bisa menjadi pertimbangan untuk analisis lebih lanjut.
    -   *Tipe Data:* Terdapat campuran tipe data (object, int64, float64). Beberapa kolom tanggal awalnya bertipe object atau int64 dan perlu dikonversi.

**Uraian Fitur pada Data (Total 19 Fitur Awal):**

1.  **business_code:** Kode unik untuk unit bisnis (Kategorikal).
2.  **cust_number:** Nomor identifikasi unik pelanggan (Kategorikal/Identifier).
3.  **name_customer:** Nama pelanggan (Kategorikal/Identifier).
4.  **clear_date:** Tanggal pembayaran faktur diselesaikan (Tanggal/Object, memiliki missing values).
5.  **buisness_year:** Tahun transaksi bisnis (Numerik/Tanggal).
6.  **doc_id:** ID unik dokumen/faktur (Numerik/Identifier).
7.  **posting_date:** Tanggal transaksi dicatat (Tanggal/Object).
8.  **document_create_date:** Tanggal dokumen dibuat (Numerik/Tanggal).
9.  **document_create_date.1:** Tanggal dokumen dibuat (duplikat/redundant, Numerik/Tanggal).
10. **due_in_date:** Tanggal jatuh tempo pembayaran (Numerik/Tanggal).
11. **invoice_currency:** Mata uang yang digunakan dalam faktur (Kategorikal).
12. **document type:** Tipe dokumen transaksi (Kategorikal).
13. **posting_id:** ID posting (Numerik, tampaknya konstan).
14. **area_business:** Area bisnis (Numerik/Float, semua nilainya null).
15. **total_open_amount:** Jumlah total yang masih terbuka/belum dibayar (Numerik).
16. **baseline_create_date:** Tanggal dasar pembuatan transaksi (Numerik/Tanggal).
17. **cust_payment_terms:** Kode syarat pembayaran pelanggan (Kategorikal).
18. **invoice_id:** ID unik faktur (Numerik/Float, memiliki sedikit missing values).
19. **isOpen:** Status faktur (Target: 1 jika terbuka, 0 jika sudah lunas) (Numerik/Kategorikal).

**Exploratory Data Analysis (EDA) & Visualisasi:**

-   **Statistik Deskriptif:** Menggunakan `df.describe()` menunjukkan ringkasan statistik untuk fitur numerik seperti `total_open_amount`, `buisness_year`, dll. Terlihat bahwa `total_open_amount` memiliki rentang nilai yang cukup lebar. `posting_id` memiliki nilai konstan 1.
-   **Distribusi Fitur:**
    -   Histogram `total_open_amount` menunjukkan distribusi yang *right-skewed*, artinya sebagian besar faktur memiliki jumlah terbuka yang relatif kecil, namun ada beberapa faktur dengan jumlah sangat besar.
    -   Countplot `invoice_currency` menunjukkan bahwa mayoritas transaksi menggunakan mata uang USD dibandingkan CAD.
    -   Countplot `cust_payment_terms` menunjukkan variasi syarat pembayaran, dengan beberapa kode (seperti NAH4, NAA8) jauh lebih dominan daripada yang lain.
-   **Analisis Tren Waktu:** Plot jumlah transaksi harian (`posting_date`) menunjukkan adanya fluktuasi dalam volume transaksi dari waktu ke waktu, mungkin dipengaruhi oleh siklus bisnis atau musiman. Plot total *open amount* harian juga menunjukkan variasi nilai transaksi.
-   **Analisis Korelasi:** Heatmap korelasi antar variabel numerik menunjukkan korelasi yang sangat tinggi (mendekati 1) antara beberapa kolom tanggal (`document_create_date`, `document_create_date.1`, `baseline_create_date`, `due_in_date`) dan juga antara `doc_id` dan `invoice_id`. Kolom `posting_id` dan `buisness_year` memiliki korelasi rendah dengan fitur lain. Korelasi tinggi antar fitur tanggal mengindikasikan potensi redundansi.

* * * * *

Data Preparation
----------------

Tahapan persiapan data dilakukan untuk memastikan data siap digunakan dalam pemodelan machine learning:

1.  **Konversi Tipe Data Tanggal:**
    -   *Proses:* Kolom-kolom yang merepresentasikan tanggal (`due_in_date`, `posting_date`, `baseline_create_date`, `document_create_date.1`, `document_create_date`) dikonversi dari tipe data numerik (int64) atau object menjadi tipe data datetime menggunakan `pd.to_datetime()`.
    -   *Alasan:* Memungkinkan ekstraksi fitur berbasis waktu (jika diperlukan nanti) dan memastikan interpretasi data tanggal yang benar.

2.  **Penanganan Missing Value (Penghapusan Kolom):**
    -   *Proses:* Kolom `area_business` dihapus dari dataset karena semua nilainya adalah null (`df.drop(columns=['area_business'])`).
    -   *Alasan:* Kolom tanpa variasi atau informasi tidak akan memberikan nilai tambah pada model prediksi. Missing value pada `clear_date` dan `invoice_id` tidak ditangani secara eksplisit karena kolom ini tidak termasuk dalam fitur yang dipilih untuk modeling.

3.  **Seleksi Fitur (Feature Selection):**
    -   *Proses:* Dipilih subset fitur yang dianggap relevan untuk memprediksi `isOpen`. Fitur yang dipilih adalah: `business_code`, `buisness_year`, `doc_id`, `total_open_amount`, `invoice_currency`, `document type`, `cust_payment_terms`. Targetnya adalah `isOpen`.
    -   *Alasan:* Mengurangi dimensi data, fokus pada informasi yang paling berpengaruh, dan menghindari kompleksitas berlebih atau noise dari fitur yang kurang relevan (seperti `name_customer`, `cust_number`, `invoice_id`, dan beberapa kolom tanggal yang berkorelasi tinggi).

4.  **Pemisahan Data (Train-Test Split):**
    -   *Proses:* Dataset dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split` dari scikit-learn dengan `random_state=42` untuk reproduktifitas.
    -   *Alasan:* Memungkinkan pelatihan model pada sebagian besar data dan evaluasi performa model pada data yang belum pernah dilihat sebelumnya, sehingga memberikan estimasi kinerja yang lebih objektif dan menghindari overfitting.

5.  **Definisi Pipeline Preprocessing:**
    -   *Proses:* Dibuat pipeline terpisah untuk fitur numerik dan kategorikal menggunakan `ColumnTransformer`.
        -   Fitur Numerik (`buisness_year`, `doc_id`, `total_open_amount`): Diterapkan `StandardScaler` untuk menstandarisasi skala nilai.
        -   Fitur Kategorikal (`business_code`, `invoice_currency`, `document type`, `cust_payment_terms`): Diterapkan `OneHotEncoder` (dengan `handle_unknown='ignore'`) untuk mengubah fitur kategorikal menjadi representasi numerik biner.
    -   *Alasan:* Standardisasi diperlukan agar fitur numerik dengan skala berbeda tidak mendominasi model. One-hot encoding diperlukan agar model machine learning dapat memproses fitur kategorikal. `handle_unknown='ignore'` mencegah error jika ada kategori baru di data uji. Pipeline memastikan proses preprocessing diterapkan secara konsisten pada data latih dan uji.

* * * * *

Modeling
--------

Dua model klasifikasi dibangun dan dievaluasi untuk memprediksi status faktur (`isOpen`):

1.  **Model Baseline: Logistic Regression**
    -   *Cara Kerja:* Logistic Regression adalah model linear yang memprediksi probabilitas suatu kelas (dalam kasus ini, `isOpen`=1 atau `isOpen`=0) menggunakan fungsi logistik (sigmoid) pada kombinasi linear dari fitur-fitur input. Model ini baik sebagai baseline karena interpretasinya yang mudah.
    -   *Implementasi:* Model diimplementasikan dalam `Pipeline` scikit-learn yang menggabungkan `preprocessor` (StandardScaler + OneHotEncoder) dengan classifier `LogisticRegression`.
    -   *Parameter:* Parameter `max_iter` diatur ke 1000 untuk memastikan konvergensi. Parameter lain menggunakan nilai *default* dari scikit-learn (misalnya, regularization='l2', solver='lbfgs').

2.  **Model Alternatif: Random Forest**
    -   *Cara Kerja:* Random Forest adalah model *ensemble* yang terdiri dari banyak *decision tree*. Prediksi dilakukan berdasarkan voting mayoritas (untuk klasifikasi) dari hasil prediksi masing-masing tree. Model ini kuat dalam menangani hubungan non-linear dan interaksi antar fitur, serta cenderung tahan terhadap overfitting dibandingkan decision tree tunggal.
    -   *Implementasi Awal (Baseline RF):* Model diimplementasikan dalam `Pipeline` serupa, menggunakan `RandomForestClassifier` dengan `random_state=42` untuk reproduktifitas. Parameter lain menggunakan nilai *default* (misalnya, `n_estimators=100`, `max_depth=None`, `min_samples_split=2`).
    -   *Improvement (Hyperparameter Tuning):*
        -   *Metode:* Menggunakan `GridSearchCV` dengan 5-fold cross-validation (`cv=5`) dan metrik `scoring='f1'` untuk mencari kombinasi parameter terbaik.
        -   *Parameter yang Di-tuning:*
            -   `n_estimators`: [50, 100, 200] (Jumlah pohon dalam forest)
            -   `max_depth`: [None, 10, 20] (Kedalaman maksimum pohon)
            -   `min_samples_split`: [2, 5, 10] (Jumlah minimum sampel untuk membagi node)
        -   *Hasil Parameter Terbaik:* Hasil `GridSearchCV` menunjukkan kombinasi terbaik adalah: `n_estimators=50`, `max_depth=10`, `min_samples_split=2`.
    -   *Implementasi Akhir (Best RF):* Model Random Forest terbaik (hasil tuning) digunakan untuk prediksi akhir pada data uji.

* * * * *

Evaluation
----------

Performa ketiga model (Logistic Regression, Random Forest baseline, Random Forest terbaik) dievaluasi pada data uji menggunakan metrik berikut:

-   **Classification Report:** Memberikan rincian precision, recall, dan f1-score untuk kedua kelas (0 dan 1).
-   **Confusion Matrix:** Menunjukkan jumlah prediksi True Positive, True Negative, False Positive, dan False Negative.
-   **ROC-AUC:** Area di bawah kurva ROC, mengukur kemampuan diskriminasi model secara keseluruhan.

**Hasil Evaluasi:**

1.  **Logistic Regression (Baseline):**
    -   Accuracy: 0.90
    -   Precision (Kelas 1): 0.66
    -   Recall (Kelas 1): 0.96
    -   F1-Score (Kelas 1): 0.78
    -   AUC: 0.95
    -   *Analisis:* Model ini memiliki recall yang sangat tinggi untuk kelas 1 (faktur terbuka), artinya mampu mengidentifikasi sebagian besar faktur terbuka, namun precision-nya lebih rendah, menunjukkan adanya sejumlah faktur lunas yang salah diklasifikasikan sebagai terbuka (False Positives).

2.  **Random Forest (Baseline - Default Parameters):**
    -   Accuracy: 1.00 (mendekati)
    -   Precision (Kelas 1): 0.99
    -   Recall (Kelas 1): 0.99
    -   F1-Score (Kelas 1): 0.99
    -   AUC: 1.00 (mendekati)
    -   *Analisis:* Performa sangat tinggi di semua metrik, menunjukkan kemampuan yang jauh lebih baik daripada Logistic Regression dalam membedakan kedua kelas. Hanya sedikit kesalahan prediksi.

3.  **Random Forest (Terbaik - Hasil Tuning GridSearchCV):**
    -   Accuracy: 1.00 (mendekati)
    -   Precision (Kelas 1): 0.99
    -   Recall (Kelas 1): 0.99
    -   F1-Score (Kelas 1): 0.99
    -   AUC: 1.00 (mendekati)
    -   *Analisis:* Hasil tuning dengan `max_depth=10` mempertahankan performa yang sangat tinggi, mirip dengan Random Forest baseline. Meskipun tidak ada peningkatan signifikan dari baseline RF (karena baseline sudah sangat baik), model hasil tuning ini mungkin sedikit lebih sederhana (kedalaman pohon dibatasi) dan berpotensi lebih generalizable. Jumlah kesalahan pada confusion matrix sedikit lebih rendah dibandingkan baseline RF.

**Kesimpulan Evaluasi:**
Kedua model Random Forest (baseline dan hasil tuning) menunjukkan performa yang sangat superior dibandingkan Logistic Regression, dengan akurasi dan AUC mendekati sempurna. Model Random Forest terbaik hasil tuning (`max_depth=10`, `min_samples_split=2`, `n_estimators=50`) dipilih sebagai model final karena mempertahankan performa tinggi dengan potensi generalisasi yang baik (karena pembatasan `max_depth`).

* * * * *

Hubungan dengan Business Understanding
---------------------------------------

Evaluasi model yang telah dilakukan memberikan jawaban dan dampak terhadap *Business Understanding* yang telah dirumuskan:

-   **Menjawab Problem Statement 1 (Kesulitan Prediksi Status Faktur):**
    -   Model Random Forest terbaik berhasil memprediksi status `isOpen` dengan akurasi dan AUC mendekati 1.00. Ini berarti model sangat efektif dalam membedakan faktur yang sudah lunas dan yang masih terbuka. Dengan demikian, model ini secara signifikan mengurangi kesulitan dalam memprediksi status pembayaran, memungkinkan perusahaan merencanakan *cash flow* dengan lebih baik berdasarkan prediksi faktur yang berisiko belum lunas.

-   **Menjawab Problem Statement 2 (Strategi Penagihan Kurang Efektif):**
    -   Dengan kemampuan prediksi yang tinggi (Precision dan Recall kelas 1 = 0.99), model dapat secara akurat mengidentifikasi faktur-faktur yang masih terbuka (*isOpen*=1). Informasi ini memungkinkan tim penagihan untuk fokus pada faktur yang benar-benar memerlukan tindak lanjut, membuat strategi penagihan menjadi lebih proaktif, efisien, dan tepat sasaran.

-   **Mencapai Goal 1 (Prediksi Status Faktur untuk Perencanaan Arus Kas):**
    -   Tujuan ini tercapai dengan sangat baik. Model Random Forest memberikan prediksi status `isOpen` yang sangat akurat, yang merupakan input penting untuk perencanaan arus kas. Perusahaan dapat mengestimasi penerimaan kas dengan lebih realistis.

-   **Mencapai Goal 2 (Klasifikasi Akurat untuk Strategi Penagihan):**
    -   Tujuan ini juga tercapai dengan sangat baik. Tingginya nilai F1-Score (0.99) untuk kelas 1 menunjukkan bahwa model memiliki keseimbangan yang baik antara precision dan recall dalam mengidentifikasi faktur terbuka, mendukung efektivitas strategi penagihan.

-   **Dampak Solusi Statement:**
    -   *Solusi 1 (Logistic Regression):* Berhasil memberikan *baseline* performa (AUC 0.95, F1-Score kelas 1 0.78). Meskipun kinerjanya di bawah Random Forest, model ini menunjukkan bahwa data historis memang memiliki pola yang dapat dipelajari untuk prediksi status faktur dan memberikan titik awal perbandingan yang berguna.
    -   *Solusi 2 (Random Forest & Tuning):* Terbukti sangat efektif. Implementasi Random Forest secara signifikan meningkatkan performa dibandingkan Logistic Regression (AUC mendekati 1.00, F1-Score kelas 1 0.99). Proses *hyperparameter tuning* dengan GridSearchCV memastikan bahwa parameter model optimal telah dipilih, meskipun dalam kasus ini peningkatannya tidak drastis karena performa baseline RF sudah sangat tinggi. Solusi ini memberikan dampak positif yang besar terhadap pencapaian *goals* proyek.

Secara keseluruhan, model yang dikembangkan, khususnya Random Forest hasil tuning, berhasil menjawab permasalahan bisnis dan mencapai tujuan yang ditetapkan dengan memberikan kemampuan prediksi status faktur yang sangat akurat.

* * * * *

Kesimpulan dan Rekomendasi
--------------------------

**Kesimpulan:**
Proyek ini berhasil mengembangkan model *machine learning* untuk memprediksi status pembayaran faktur (*isOpen*). Proses dimulai dengan pemahaman data, diikuti persiapan data yang mencakup konversi tipe data, penanganan *missing value*, seleksi fitur, dan pemisahan data. Dua model, Logistic Regression (sebagai *baseline*) dan Random Forest (dengan *hyperparameter tuning* menggunakan GridSearchCV), dibangun dan dievaluasi. Hasil evaluasi menunjukkan bahwa model Random Forest terbaik (dengan parameter `max_depth=10`, `min_samples_split=2`, `n_estimators=50`) memiliki performa superior dengan akurasi dan AUC mendekati 1.00, serta F1-Score 0.99 untuk kelas faktur terbuka. Model ini secara efektif menjawab kebutuhan bisnis untuk memprediksi status faktur guna mendukung perencanaan arus kas dan strategi penagihan yang lebih baik.

**Rekomendasi:**

-   **Implementasi Model:** Model Random Forest terbaik direkomendasikan untuk diintegrasikan ke dalam sistem operasional perusahaan (misalnya, sistem CRM atau ERP) untuk memberikan prediksi status faktur secara *real-time* atau periodik, mendukung tim keuangan dan penagihan.
-   **Pengembangan Lebih Lanjut (Fitur & Model):**
    -   *Feature Engineering:* Pertimbangkan untuk membuat fitur baru dari data tanggal (misalnya, selisih hari antara tanggal posting dan jatuh tempo, hari dalam seminggu, bulan) untuk mungkin menangkap pola temporal yang lebih kompleks.
    -   *Model Alternatif:* Eksplorasi model *ensemble* lain seperti Gradient Boosting (XGBoost, LightGBM) atau model *deep learning* jika diperlukan peningkatan akurasi lebih lanjut, meskipun performa Random Forest saat ini sudah sangat tinggi.
-   **Ekspansi Prediksi:** Jika data `clear_date` dapat diisi atau dimodelkan, proyek selanjutnya dapat fokus pada prediksi *tanggal pembayaran* yang sebenarnya atau *kategori umur faktur* (misalnya, 0-15 hari terlambat, 16-30 hari, dst.) untuk analisis yang lebih mendalam.
-   **Monitoring dan Retraining:** Setelah implementasi, lakukan monitoring performa model secara berkala dan lakukan *retraining* dengan data baru untuk memastikan model tetap akurat seiring perubahan pola pembayaran pelanggan.
