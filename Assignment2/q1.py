import numpy as np
import pandas as pd

ATTRIBUTES = ['sepal_length', 'sepal_width', 'petal_length','petal_width','class']

df = pd.read_csv('iris.data', sep=',', names=ATTRIBUTES)
print(df.head())