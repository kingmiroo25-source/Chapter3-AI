import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# 1. Load dataset regresi
X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# 2. Bagi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Inisialisasi & latih predictor/regressor
ml_predictor = GradientBoostingRegressor(random_state=42)
ml_predictor.fit(X_train, y_train)

# 4. Hitung skor R^2 pada data uji
score = ml_predictor.score(X_test, y_test)
print(f"Skor R^2 Regresi: {score:.4f}")