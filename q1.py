import pandas as pd
import numpy as np
from sklearn import model_selection


dataset = pd.read_csv("Train_B_Tree.csv")

train, test = model_selection.train_test_split(dataset,test_size=0.3)
print(train)