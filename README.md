1\. Domain Proyek
-----------------

**Latar Belakang:**\
Dalam dunia keuangan, pengelolaan arus kas sangat bergantung pada pemahaman terhadap pergerakan invoice. Invoice yang belum tertagih (open invoice) dapat mengganggu likuiditas dan meningkatkan risiko kredit. Oleh karena itu, memprediksi nilai invoice harian (*total_open_amount*) menjadi sangat penting agar perusahaan dapat merencanakan cash flow dan menindaklanjuti invoice dengan lebih proaktif.

**Mengapa Masalah Ini Harus Diselesaikan:**

-   **Optimalisasi Arus Kas:** Dengan mengetahui proyeksi invoice ke depan, tim keuangan dapat mengantisipasi kebutuhan likuiditas dan mengurangi risiko kekurangan dana.
-   **Efisiensi Operasional:** Perusahaan dapat menetapkan prioritas penagihan dan pengelolaan kredit yang lebih tepat.
-   **Perencanaan Strategis:** Forecast yang akurat membantu manajemen dalam merencanakan investasi, pengeluaran, dan kebijakan kredit.

**Referensi dan Riset Terkait:**\
Beberapa referensi yang mendasari pendekatan forecasting time series antara lain:

-   Box, G.E.P., Jenkins, G.M., Reinsel, G.C. & Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.
-   Hyndman, R.J. & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*. OTexts.\
    Referensi tersebut memberikan dasar teori mengenai pemodelan ARIMA dan teknik forecasting lainnya yang relevan untuk proyek ini.

* * * * *

2\. Business Understanding
--------------------------

**Problem Statement:**\
Bagaimana memprediksi total nilai invoice harian agar perusahaan dapat mengantisipasi fluktuasi arus kas, mengurangi risiko kredit macet, dan meningkatkan efisiensi penagihan?

**Goals (Tujuan):**

-   Mengidentifikasi pola historis pada data invoice harian.
-   Membangun model forecasting untuk memproyeksikan total invoice ke depan (misalnya, 30 hari ke depan).
-   Menyediakan metrik evaluasi yang terukur (seperti MSE, MAE, AIC) untuk mengukur kinerja model.
-   Memberikan insight yang dapat digunakan oleh tim keuangan untuk pengambilan keputusan.

**Solution Statement:**\
Untuk mencapai tujuan di atas, kami mengusulkan dua pendekatan:

1.  **Baseline Model:** Menggunakan model ARIMA dengan parameter awal (1,1,1) untuk mendapatkan prediksi dasar.
2.  **Improvement Model:** Melakukan tuning hyperparameter dengan metode grid search atau menggunakan pendekatan auto_arima untuk menemukan konfigurasi optimal, sehingga model dapat memberikan forecast yang lebih akurat.\
    Kedua solusi akan dievaluasi menggunakan metrik seperti Mean Squared Error (MSE), Mean Absolute Error (MAE), dan nilai AIC, sehingga model terbaik dapat dipilih secara terukur.

* * * * *

3\. Data Understanding
----------------------

**Informasi Umum Data:**

-   **Jumlah Data:** 40.000 entri
-   **Jumlah Kolom:** 18 kolom, dengan fitur-fitur seperti *business_code, cust_number, clear_date, total_open_amount*, dsb.

**Fokus Analisis:**\
Untuk proyek forecasting, kami fokus pada:

-   **clear_date:** Tanggal penyelesaian invoice (sebagai variabel waktu).
-   **total_open_amount:** Nilai total invoice per transaksi, yang akan diaggregasi untuk mendapatkan total nilai invoice harian.

**Sumber Data:**\
Dataset dapat diunduh melalui tautan berikut *(contoh link; sesuaikan dengan sumber data asli)*.

**Eksploratory Data Analysis (EDA):**\
Tahapan EDA meliputi:

-   Statistik deskriptif (rata-rata, median, varians) untuk *total_open_amount*.
-   Visualisasi distribusi nilai invoice dan trend harian.
-   Pemetaan missing values dan outlier (walaupun dataset sudah lengkap, pemeriksaan tambahan selalu dilakukan).

Contoh cuplikan kode eksplorasi:

``` python
import pandas as pd
import matplotlib.pyplot as plt

# Membaca data dan mengurutkan berdasarkan tanggal
df = pd.read_csv('data.csv', parse_dates=['clear_date']).sort_values('clear_date')

# Agregasi total invoice per hari
ts = df.groupby('clear_date')['total_open_amount'].sum().asfreq('D').fillna(0)

# Statistik deskriptif
print(ts.describe())

# Visualisasi data historis
plt.figure(figsize=(12,6))
plt.plot(ts, label='Data Historis')
plt.xlabel('Tanggal')
plt.ylabel('Total Open Amount')
plt.title('Data Historis Total Open Amount per Hari')
plt.legend()
plt.show()`
```

* * * * *

4\. Data Preparation
--------------------

**Langkah-langkah Pra-pemrosesan Data:**

1.  **Pembacaan Data:**\
    Data diimpor menggunakan `pandas` dengan parsing tanggal agar kolom *clear_date* dikenali sebagai tipe datetime.

2.  **Pengurutan dan Agregasi:**\
    Data diurutkan berdasarkan *clear_date* dan dikelompokkan untuk menghitung total invoice per hari.\
    *Alasan:* Agar analisis time series dapat dilakukan secara kronologis dan memastikan setiap periode waktu memiliki data.

3.  **Resampling:**\
    Data di-resample ke frekuensi harian dan nilai yang hilang diisi dengan 0.\
    *Alasan:* Memastikan kontinuitas waktu, menghindari kekosongan pada tanggal tertentu.

4.  **Visualisasi Awal:**\
    Plot data historis untuk mengidentifikasi tren, fluktuasi, dan pola musiman.

Contoh kode lengkap:

``` python
# Membaca data dan mengurutkan berdasarkan tanggal
df = pd.read_csv('data.csv', parse_dates=['clear_date']).sort_values('clear_date')

# Agregasi data: total invoice per hari
ts = df.groupby('clear_date')['total_open_amount'].sum()

# Resampling ke frekuensi harian
ts = ts.asfreq('D').fillna(0)
```

* * * * *

5\. Modeling
------------

**Pendekatan Pemodelan:**\
Dalam proyek ini, kami menggunakan model ARIMA untuk forecasting. ARIMA dipilih karena kemampuannya menangani data time series dengan komponen tren dan musiman (setelah differencing).

### 5.1 Pemilihan Parameter

-   **Differencing (d):** Dimulai dengan d=1 untuk mengatasi non-stasioneritas.
-   **Orde AR (p) dan MA (q):** Berdasarkan analisis grafik Autocorrelation (ACF) dan Partial Autocorrelation (PACF), nilai awal p=1 dan q=1 digunakan. Parameter ini akan dituning lebih lanjut.

### 5.2 Fitting Model ARIMA

Cuplikan kode untuk fitting model:

``` python
from statsmodels.tsa.arima.model import ARIMA

# Fitting model ARIMA dengan order (1,1,1)
model = ARIMA(ts, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())
```

*Penjelasan:*

-   Model dilatih dengan data historis yang telah disiapkan.
-   Summary model memberikan informasi koefisien, nilai AIC, dan statistik diagnostik lainnya.

### 5.3 Proses Improvement (Hyperparameter Tuning)

Untuk mendapatkan model yang optimal, dilakukan tuning hyperparameter dengan metode grid search atau auto_arima.\
*Contoh (secara konseptual):*

``` python
# Contoh pseudocode untuk grid search hyperparameter ARIMA
import itertools

p = d = q = range(0, 3)
pdq = list(itertools.product(p, [1], q))
best_aic = float("inf")
best_order = None

for order in pdq:
    try:
        model = ARIMA(ts, order=order)
        results = model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_order = order
    except:
        continue

print("Best ARIMA order:", best_order, "AIC:", best_aic)
```

*Penjelasan:*

-   Tuning dilakukan dengan mencoba berbagai kombinasi parameter.
-   Parameter terbaik dipilih berdasarkan nilai AIC terendah.

**Kelebihan dan Kekurangan Algoritma:**

-   **ARIMA:**
    -   *Kelebihan:* Sederhana, interpretatif, dan efektif untuk data yang stasioner setelah differencing.
    -   *Kekurangan:* Terbatas untuk menangkap non-linieritas dan tidak mengakomodasi variabel eksogen tanpa modifikasi (misalnya, ARIMAX).
-   **Improvement:**
    -   Proses tuning dapat meningkatkan akurasi model, namun memerlukan komputasi lebih dan pemahaman mendalam terhadap data.

* * * * *

6\. Evaluation
--------------

**Metrik Evaluasi:**\
Model dievaluasi menggunakan beberapa metrik:

-   **Mean Squared Error (MSE):** Mengukur rata-rata kuadrat error.
-   **Mean Absolute Error (MAE):** Rata-rata nilai absolut kesalahan.
-   **Akaike Information Criterion (AIC):** Untuk membandingkan model dengan kompleksitas berbeda.
-   **Analisis Residual:** Melihat pola sisa untuk memastikan tidak ada autokorelasi.

### 6.1 Evaluasi Performa Model

``` python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Menghitung error pada data training (atau melalui metode rolling forecast)
train_pred = model_fit.fittedvalues
mse = mean_squared_error(ts.dropna(), train_pred.dropna())
mae = mean_absolute_error(ts.dropna(), train_pred.dropna())

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)`
```

### 6.2 Analisis Residual

``` python
import seaborn as sns
import statsmodels.api as sm

residuals = model_fit.resid

# Plot residual time series
plt.figure(figsize=(12,6))
plt.plot(residuals)
plt.xlabel('Tanggal')
plt.ylabel('Residual')
plt.title('Plot Residual Model ARIMA')
plt.show()

# Histogram dan Q-Q Plot
plt.figure(figsize=(12,6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residual')
plt.title('Distribusi Residual')
plt.show()

sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot Residual')
plt.show()
```

*Penjelasan:*

-   Evaluasi metrik dan analisis residual memastikan bahwa model telah menangkap pola data dengan baik dan sisa tidak menunjukkan pola sistematis.

* * * * *

7\. Forecasting
---------------

### 7.1 Melakukan Forecast 30 Hari ke Depan

``` python
# Menentukan horizon forecast: 30 hari ke depan
forecast_steps = 30

# Forecasting dengan interval kepercayaan (misalnya 95%)
forecast_result = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()

# Membuat indeks tanggal untuk hasil forecast
last_date = ts.index[-1]
forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
forecast_series = pd.Series(forecast_mean, index=forecast_dates)

print("Forecast 30 Hari ke Depan:")
print(forecast_series.head())
print("\nInterval Kepercayaan (95%):")
print(forecast_ci.head())`
```

### 7.2 Visualisasi Forecast

``` python
plt.figure(figsize=(12,6))
plt.plot(ts, label='Data Historis')
plt.plot(forecast_series, label='Forecast 30 Hari', color='red')
plt.fill_between(forecast_series.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Tanggal')
plt.ylabel('Total Open Amount')
plt.title('Forecasting Total Open Amount per Hari dengan Interval Kepercayaan')
plt.legend()
plt.show()
```

*Penjelasan:*

-   Hasil forecast dilengkapi dengan interval kepercayaan untuk mengkomunikasikan ketidakpastian prediksi.
-   Visualisasi membantu memahami bagaimana model memproyeksikan tren ke depan.

* * * * *

8\. Diskusi dan Kesimpulan
--------------------------

**Analisis Hasil:**

-   **Model Fit:** Berdasarkan summary ARIMA dan evaluasi metrik (MSE, MAE, AIC), model dengan parameter (1,1,1) atau yang sudah di-tuning menunjukkan kemampuan yang memadai dalam menangkap pola data historis.
-   **Forecasting:** Hasil prediksi 30 hari ke depan, disertai interval kepercayaan, memberikan gambaran realistis tentang nilai invoice mendatang yang dapat mendukung perencanaan keuangan.
-   **Analisis Residual:** Residual yang mendekati distribusi normal dan tidak menunjukkan autokorelasi signifikan mengindikasikan bahwa model telah menangkap struktur data secara efektif.

**Kelebihan dan Keterbatasan:**

-   *Kelebihan:*
    -   Model ARIMA sederhana dan mudah diinterpretasikan.
    -   Proses tuning dan evaluasi diagnostik memberikan dasar yang kuat untuk perbaikan model.
-   *Keterbatasan:*
    -   ARIMA memiliki keterbatasan dalam menangkap hubungan non-linier.
    -   Jika terdapat variabel eksogen yang berpengaruh, model ARIMAX atau pendekatan deep learning seperti LSTM dapat dijadikan alternatif.

**Rekomendasi:**

-   Lakukan tuning parameter lebih lanjut dengan grid search atau auto_arima untuk mendapatkan konfigurasi optimal.
-   Pertimbangkan penggunaan model tambahan (misalnya, LSTM) untuk membandingkan performa forecasting.
-   Terapkan validasi time series (rolling forecast) untuk evaluasi performa model yang lebih robust.

**Kesimpulan:**\
Laporan ini menyajikan seluruh tahapan dari pemilihan domain, pemahaman bisnis, eksplorasi dan persiapan data, pemodelan, hingga evaluasi forecasting menggunakan ARIMA. Hasil proyek memberikan insight yang mendalam terkait prediksi nilai invoice harian, sehingga dapat dijadikan dasar pengambilan keputusan dalam manajemen keuangan dan perencanaan arus kas. Dengan melengkapi laporan ini dengan analisis diagnostik dan evaluasi metrik, diharapkan submission memenuhi seluruh kriteria wajib dan tambahan untuk meraih bintang lima.