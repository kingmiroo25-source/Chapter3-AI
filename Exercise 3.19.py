# Exercise 3.19: Classification using breast cancer data (LazyPredict alternative)
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. Load dataset
cancer = load_breast_cancer(as_frame=True)
data = cancer.data.copy()
data["Target"] = cancer.target

# 2. Split data menjadi Features (X) dan Target (y)
X = data.drop(columns=["Target"])
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Inisialisasi dan jalankan evaluasi model
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# 4. Tampilkan hasil perbandingan model
print(models)