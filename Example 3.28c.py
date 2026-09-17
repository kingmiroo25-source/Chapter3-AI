import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Load dataset Iris
X, y = load_iris(return_X_y=True)

# 2. Bagi dataset menjadi data latih dan data uji (25% data uji)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

# 3. Inisialisasi dan jalankan LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# 4. Tampilkan tabel hasil evaluasi model
print(models)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(models.index, models['Accuracy'])
plt.show()