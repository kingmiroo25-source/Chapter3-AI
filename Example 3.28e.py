import matplotlib.pyplot as plt
import numpy as np
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. Load dataset
X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# 2. Ambil 1.000 data pertama saja agar proses CANGGIH & KILAT (hitungan detik)
X_train, X_test, y_train, y_test = train_test_split(
    X.iloc[:1000], y.iloc[:1000], test_size=0.1, random_state=1
)

# 3. Tambahkan verbose=1 agar kamu bisa melihat proses tiap model yang selesai
print("Sedang memproses model, mohon tunggu sebentar...\n")
reg = LazyRegressor(verbose=1, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# 4. Tampilkan tabel hasil
print("\n--- HASIL EVALUASI REGRESI ---")
print(models)

# 5. Visualisasi Grafik
plt.figure(figsize=(10, 5))
plt.plot(models.index, models['R-Squared'], '-s')
plt.xticks(rotation=90)
plt.title("Perbandingan R-Squared Model Regresi")
plt.xlabel("Model")
plt.ylabel("R-Squared")
plt.tight_layout()

# 6. Wajib ada agar jendela Figure 1 terbuka
plt.show()