# Example 3.3 Python SVM Iris CSV Classifications
import pandas as pd
from sklearn import svm

url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
df = pd.read_csv(url)
X = df.values[:, :2]
s = df["species"]
d = {label: index for index, label in enumerate(sorted(set(s)))}
y = [d[label] for label in s]

clf = svm.SVC()
clf.fit(X, y)

# Predict the flower for a given sepal length and width
p = clf.predict([[5.4, 3.2]])
print(p)
