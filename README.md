Laporan Proyek Machine Learning - Wildan Abdurrasyid
=============================================

* * * * *

Domain Proyek
-------------

**Latar Belakang Masalah:**\
Keterlambatan pembayaran faktur merupakan permasalahan yang krusial bagi perusahaan karena dapat mengganggu cash flow dan operasional. Dengan memprediksi tanggal pembayaran serta mengkategorikan umur faktur, perusahaan dapat mengantisipasi keterlambatan dan mengoptimalkan strategi penagihan. Masalah ini harus diselesaikan untuk meningkatkan efisiensi pengelolaan keuangan dan mengurangi risiko operasional.

**Mengapa Masalah Ini Penting:**

-   **Manajemen Keuangan yang Lebih Baik:** Meminimalkan keterlambatan pembayaran membantu perusahaan menjaga arus kas yang stabil.

-   **Strategi Penagihan yang Tepat:** Dengan mengetahui kategori umur faktur, perusahaan bisa menerapkan pendekatan penagihan yang lebih tepat sasaran.

**Referensi dan Riset Terkait:**

-   [Optimizing Cash Flow Management](https://scholar.google.com/) -- Referensi yang mendalam mengenai pengaruh prediksi pembayaran terhadap manajemen keuangan.

-   [Predictive Analytics in Finance](https://scholar.google.com/) -- Studi kasus dan metode machine learning dalam dunia keuangan.

* * * * *

Business Understanding
----------------------

**Problem Statements:**

-   **Pernyataan Masalah 1:** Banyak perusahaan mengalami kesulitan dalam memprediksi tanggal pembayaran faktur yang berdampak pada cash flow.

-   **Pernyataan Masalah 2:** Kurangnya klasifikasi yang tepat terhadap umur faktur membuat strategi penagihan menjadi kurang efektif.

**Goals:**

-   **Tujuan 1:** Memprediksi tanggal pembayaran faktur berdasarkan data historis sehingga perusahaan dapat merencanakan arus kas dengan lebih akurat.

-   **Tujuan 2:** Mengklasifikasikan umur faktur (misalnya: tepat waktu, terlambat, sangat terlambat) untuk mendukung strategi penagihan yang lebih efektif.

### Solution Statements

-   **Solusi Pertama:**\
    Membangun baseline model menggunakan **Logistic Regression** untuk memprediksi status faktur. Model ini akan memberikan acuan awal dan kemudahan interpretasi karena sifat linear-nya.

-   **Solusi Kedua:**\
    Mengimplementasikan **Random Forest** sebagai model alternatif yang mampu menangani kompleksitas data secara non-linear. Dilakukan juga hyperparameter tuning (menggunakan GridSearchCV) untuk meningkatkan performa model.\
    *Evaluasi kedua model dilakukan dengan metrik seperti classification report, confusion matrix, dan ROC-AUC sehingga solusi yang diberikan dapat terukur secara kuantitatif.*

* * * * *

Data Understanding
------------------

Dataset yang digunakan dalam proyek ini terdiri dari 50.000 sampel data dengan 19 kolom. Dataset memuat informasi mengenai transaksi faktur seperti:

-   **business_code:** Kode bisnis dari perusahaan.

-   **cust_number:** Nomor pelanggan.

-   **name_customer:** Nama pelanggan.

-   **clear_date:** Tanggal penyelesaian pembayaran (jika ada).

-   **buisness_year:** Tahun bisnis terkait transaksi.

-   **doc_id:** ID dokumen faktur.

-   **posting_date:** Tanggal posting transaksi.

-   **document_create_date:** Tanggal pembuatan dokumen.

-   **due_in_date:** Tanggal jatuh tempo pembayaran.

-   **invoice_currency:** Mata uang faktur.

-   **document type:** Jenis dokumen.

-   **cust_payment_terms:** Syarat pembayaran yang diterapkan.

-   **isOpen:** Status faktur (misalnya: masih terbuka atau sudah lunas).

**Sumber Dataset:**\
Dataset dapat diunduh melalui tautan: [Kaggle Payment Date Dataset](https://www.kaggle.com/datasets/rajattomar132/payment-date-dataset)

**Exploratory Data Analysis (EDA):**

-   **Statistik Deskriptif:** Informasi mengenai sebaran data, mean, dan standar deviasi tiap kolom.

-   **Visualisasi Data:** Grafik distribusi untuk variabel numerik seperti *total_open_amount*, countplot untuk variabel kategorikal, serta tren waktu berdasarkan *posting_date*.

-   **Analisis Korelasi:** Heatmap untuk mengetahui hubungan antar variabel numerik guna menentukan fitur yang paling relevan untuk pemodelan.

* * * * *

Data Preparation
----------------

Pada tahap ini, dilakukan serangkaian proses untuk membersihkan dan mempersiapkan data sehingga layak digunakan dalam pemodelan:

-   **Konversi Tipe Data:**\
    Mengkonversi kolom tanggal (misalnya: *due_in_date*, *posting_date*, *document_create_date*, dll.) ke format datetime agar mendukung analisis tren dan perhitungan interval waktu.

-   **Penanganan Missing Value:**\
    Menghapus kolom yang memiliki missing value ekstrem (misalnya kolom 'area_business' yang seluruh nilainya null) untuk menghindari distorsi analisis.

-   **Pemisahan Fitur dan Target:**\
    Memilih fitur-fitur yang relevan untuk prediksi status faktur (*isOpen*), kemudian membagi data menjadi set training dan testing agar model dapat dievaluasi secara objektif.

**Alasan Proses Data Preparation:**

-   **Konsistensi Data:** Konversi tipe data memastikan bahwa analisis yang berbasis waktu berjalan dengan benar.

-   **Kebersihan Data:** Menghapus kolom yang tidak relevan meningkatkan kualitas data dan meminimalkan noise pada model.

-   **Validasi Model:** Pemisahan data membantu memastikan bahwa model tidak mengalami overfitting dan mampu menggeneralisasi pada data baru.

* * * * *

Modeling
--------

Pada tahap modeling, dua pendekatan digunakan untuk membangun model prediksi:

-   **Model Baseline (Logistic Regression):**

    -   *Implementasi:* Model Logistic Regression diterapkan menggunakan pipeline yang mencakup preprocessing (StandardScaler untuk fitur numerik dan OneHotEncoder untuk fitur kategorikal).

    -   *Kelebihan:* Mudah diinterpretasikan dan menjadi acuan awal dalam pengukuran performa.

    -   *Kekurangan:* Kemampuan terbatas dalam menangani hubungan non-linear yang kompleks.

-   **Model Alternatif (Random Forest):**

    -   *Implementasi:* Model Random Forest digunakan dengan pipeline serupa, dilengkapi dengan proses hyperparameter tuning menggunakan GridSearchCV untuk menentukan kombinasi parameter optimal seperti `n_estimators`, `max_depth`, dan `min_samples_split`.

    -   *Kelebihan:* Lebih baik dalam menangani variabilitas dan non-linearitas pada data.

    -   *Kekurangan:* Memerlukan waktu komputasi lebih tinggi dan lebih kompleks untuk interpretasi.

**Proses Improvement:**

-   **Hyperparameter Tuning:** GridSearchCV dilakukan pada Random Forest untuk meningkatkan performa model dengan mencari parameter terbaik secara sistematis.

-   **Perbandingan Model:** Kedua model dievaluasi menggunakan metrik evaluasi yang telah ditentukan. Model dengan performa terbaik (dalam hal ROC-AUC dan classification report) akan dipilih sebagai solusi final.

* * * * *

Evaluation
----------

Pada tahap evaluasi, metrik-metrik berikut digunakan untuk menilai performa model:

-   **Classification Report:** Menampilkan precision, recall, dan f1-score untuk setiap kelas sehingga dapat mengetahui keseimbangan performa model.

-   **Confusion Matrix:** Menyajikan distribusi kesalahan prediksi antara kelas yang sebenarnya dan kelas yang diprediksi.

-   **ROC-AUC:** Mengukur kemampuan model dalam membedakan antara kelas positif dan negatif, dengan nilai AUC yang mendekati 1 menunjukkan performa yang baik.

**Penjelasan Metrik Evaluasi:**

-   **Precision:** Proporsi prediksi positif yang benar.

-   **Recall:** Kemampuan model untuk menemukan seluruh data positif.

-   **F1-Score:** Harmonik rata-rata dari precision dan recall, memberikan keseimbangan antara kedua metrik tersebut.

-   **ROC-AUC:** Area di bawah kurva ROC yang menunjukkan trade-off antara true positive rate dan false positive rate.

**Hasil Evaluasi:**

-   Model Logistic Regression memberikan baseline yang cukup baik dengan interpretasi yang mudah.

-   Model Random Forest, setelah dilakukan hyperparameter tuning, menunjukkan peningkatan performa dengan nilai ROC-AUC yang lebih tinggi serta peningkatan di metrics precision, recall, dan f1-score.

-   Model terbaik dipilih berdasarkan hasil evaluasi kuantitatif yang menunjukkan kinerja optimal dalam prediksi status faktur.

* * * * *

Kesimpulan dan Rekomendasi
--------------------------

**Kesimpulan:**\
Proses analisis dimulai dengan eksplorasi dan pembersihan data, dilanjutkan dengan penerapan dua model machine learning untuk prediksi status faktur. Evaluasi mendalam dengan berbagai metrik menunjukkan bahwa model Random Forest, terutama setelah dilakukan tuning, memberikan performa yang lebih unggul dibandingkan model baseline. Hal ini mendukung pengambilan keputusan terkait strategi penagihan dan perbaikan arus kas.

**Rekomendasi:**

-   **Implementasi Model:** Model terbaik (Random Forest) dapat diintegrasikan ke dalam sistem manajemen penagihan untuk mendukung keputusan bisnis secara real-time.

-   **Pengembangan Lebih Lanjut:** Pertimbangkan pengembangan model lebih lanjut dengan teknik ensemble atau deep learning untuk meningkatkan akurasi prediksi.

-   **Penambahan Fitur:** Eksplorasi fitur tambahan atau teknik feature engineering untuk menangkap lebih banyak variabilitas data yang kompleks.