import pandas as pd
import numpy as np

# for random test/train split
def test_train_split(db,train_size=0.3) -> tuple:
    random_suffled = db.iloc[np.random.permutation(len(db))]
    split_point = int(len(db)*0.3)
    return random_suffled[:split_point].reset_index(drop=True),random_suffled[split_point:].reset_index(drop=True)

dataset = pd.read_csv("Train_B_Tree.csv")

train, test = test_train_split(dataset,test_size=0.3)