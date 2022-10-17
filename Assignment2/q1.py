import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ATTRIBUTES = ['sepal_length', 'sepal_width', 'petal_length','petal_width']
TARGET = 'class'

df = pd.read_csv('iris.data', sep=',', names=ATTRIBUTES + [TARGET])

x = df[ATTRIBUTES]
y = df[TARGET].tolist()

pca = PCA(0.95)
x_pca = pca.fit_transform(x)

print(x_pca.shape)


