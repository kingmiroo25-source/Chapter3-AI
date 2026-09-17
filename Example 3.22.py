import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. Load dataset regresi pengganti yang kompatibel
X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# 2. Ambil 1.000 sampel data agar eksekusi cepat (hitungan detik)
X_train, X_test, y_train, y_test = train_test_split(
    X.iloc[:1000], y.iloc[:1000], test_size=0.2, random_state=42
)

# 3. Jalankan pemodelan regresi otomatis (AutoML)
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# 4. Tampilkan tabel perbandingan semua model
print(models)