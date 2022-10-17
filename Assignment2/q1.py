import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ATTRIBUTES = ['sepal_length', 'sepal_width', 'petal_length','petal_width']
TARGET = 'class'

df = pd.read_csv('iris.data', sep=',', names=ATTRIBUTES + [TARGET])

x = df[ATTRIBUTES]
y = df[TARGET].tolist()


x_scaled = StandardScaler().fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2)

