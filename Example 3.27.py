# Example 3.27: PyCaret Alternative using LazyPredict
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 1. Load dataset Iris
iris = datasets.load_iris(as_frame=True)
X = iris.data
y = iris.target

# 2. Bagi dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Bandingkan semua model klasifikasi otomatis (AutoML)
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# 4. Tampilkan tabel perbandingan model
print(models)