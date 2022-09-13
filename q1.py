import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# load csv file
dataset = pd.read_csv("Train_B_Tree.csv")


# column names
ATTRIBUTES = ['cement', 'slag', 'flyash', 'water',
              'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age']
TARGET_ATTRIBUTE = 'csMPa'


# accuracy function : acc% = 100 - (100 *(|y_true-y_pred|/y_true))
def get_score(y_true: np.ndarray, y_pred: np.ndarray):
    err = abs(y_true-y_pred)
    err = (err / y_true)
    err = err * 100
    return 100 - (err.mean())


# Regression Tree
class RegressionTree:
    def __init__(self, db: pd.DataFrame, depth=0, isRoot=True, LIMIT_SIZE=20, MAX_DEPTH=10):
        '''
            db -> Training Dataset
            LIMIT_SIZE -> number of dataset in a leaf node
            MAX_DEPTH -> Maximum depth of tree 


        '''
        self.isRoot = isRoot
        self.isLeaf = True
        self.return_ans = db[TARGET_ATTRIBUTE].mean()
        if len(db) <= LIMIT_SIZE or depth >= MAX_DEPTH:
            return
        else:
            self.isLeaf = False
            min_sse = -1
            self.attrb = ATTRIBUTES[0]
            self.left_value = db[ATTRIBUTES[0]].mean()

            for atrribute in ATTRIBUTES:
                new_db = db[[atrribute, TARGET_ATTRIBUTE]]
                new_db.sort_values(by=[atrribute])
                for i in range(len(new_db)):
                    left_db = new_db[TARGET_ATTRIBUTE].iloc[:i]
                    right_db = new_db[TARGET_ATTRIBUTE].iloc[i:]
                    mean1 = left_db.mean()
                    mean2 = right_db.mean()
                    cur_sse = 0
                    cur_sse += sum([(left_db.iloc[j]-mean1) **
                                   2 for j in range(len(left_db))])
                    cur_sse += sum([(right_db.iloc[j]-mean2) **
                                   2 for j in range(len(right_db))])

                    if min_sse == -1 or min_sse > cur_sse:
                        self.attrb = atrribute
                        min_sse = cur_sse
                        self.left_value = new_db[atrribute].iloc[i]

            self.L = RegressionTree(
                db.loc[db[self.attrb] < self.left_value], depth+1, False, LIMIT_SIZE, MAX_DEPTH)
            self.R = RegressionTree(
                db.loc[db[self.attrb] >= self.left_value], depth+1, False, LIMIT_SIZE, MAX_DEPTH)

    def fit(self, one_row):
        '''

        '''
        if (self.isLeaf):
            return self.return_ans
        if (one_row[self.attrb] < self.left_value):
            return self.L.fit(one_row)

        return self.R.fit(one_row)

    def get_output(self, test_input: pd.DataFrame):
        '''

        '''
        return np.array([self.fit(test_input.iloc[i]) for i in range(len(test_input))])

    def get_accuracy(self, test_input: pd.DataFrame, y_true: np.ndarray):
        '''

        '''
        y_pred = self.get_output(test_input)
        return get_score(y_true, y_pred)

    def prune(self, test_input: pd.DataFrame, root, y_true: np.ndarray):
        '''

        '''
        if (self.isLeaf):
            return
        if (not self.isRoot):
            prev_acc = root.get_accuracy(test_input, y_true)
            self.isLeaf = True
            cur_acc = root.get_accuracy(test_input, y_true)
            if (cur_acc >= prev_acc):
                return

            self.isLeaf = False

        self.L.prune(test_input, root, y_true)
        self.R.prune(test_input, root, y_true)


# function to split dataset into train set and test set
def test_train_split(db: pd.DataFrame, train_size=0.3) -> tuple:
    random_suffled = db.iloc[np.random.permutation(len(db))]
    split_point = int(len(db)*train_size)
    return random_suffled[:split_point].reset_index(drop=True), random_suffled[split_point:].reset_index(drop=True)


# perform ten random splits and plot accuracy
def ten_random_splits():

    print("Accuracy of 10 Random Splits")
    accuracy = []
    accuracy_prune = []
    x_axis = []

    for i in range(10):
        print(f"Split ({i+1}):-")
        x_axis.append(i+1)
        test, train = test_train_split(dataset, 0.3)

        reg = RegressionTree(train, LIMIT_SIZE=1, MAX_DEPTH=10)
        y_true = np.array(test[TARGET_ATTRIBUTE])

        accuracy.append(reg.get_accuracy(test, test[TARGET_ATTRIBUTE]))
        reg.prune(test, reg, y_true)
        accuracy_prune.append(reg.get_accuracy(test, test[TARGET_ATTRIBUTE]))

    plt.plot(x_axis, accuracy, label="accuracy")
    plt.plot(x_axis, accuracy_prune, label="prune accuracy")
    plt.xlabel('Split number')
    plt.ylabel('Accuracy Percentage')
    plt.title('Accuracy of 10 random splits')
    plt.legend()
    plt.savefig('Accuracy_of_10_random_splits.png')
    plt.clf()


# perform prediction with different limit sizes and plot accuracy vs limit size graph
def different_limit_size():

    print("Accuracy of Different Limit Size")
    accuracy = []
    accuracy_prune = []
    x_axis = []

    test, train = test_train_split(dataset, 0.3)
    for min_size in range(1, 62, 6):
        print(f"Min_size ({min_size}):-")
        x_axis.append(min_size)
        reg = RegressionTree(train, LIMIT_SIZE=min_size, MAX_DEPTH=9)
        y_true = np.array(test[TARGET_ATTRIBUTE])
        accuracy.append(reg.get_accuracy(test, test[TARGET_ATTRIBUTE]))
        reg.prune(test, reg, y_true)
        accuracy_prune.append(reg.get_accuracy(test, test[TARGET_ATTRIBUTE]))

    plt.plot(x_axis, accuracy, label="accuracy")
    plt.plot(x_axis, accuracy_prune, label="prune accuracy")
    plt.xlabel('Limit Size')
    plt.ylabel('Accuracy Percentage')
    plt.title('Accuracy v/s Limit Size')
    plt.legend()
    plt.savefig('Accuracy_vs_limit_size.png')
    plt.clf()


# perform prediction with different max depth and plot accuracy vs max depth graph
def different_max_depths():

    print("Accuracy of Different Max Depth")
    accuracy = []
    accuracy_prune = []
    x_axis = []

    test, train = test_train_split(dataset, 0.3)
    for depth in range(1, 10):
        print(f"Max Depth ({depth}):-")
        x_axis.append(depth)
        reg = RegressionTree(train, LIMIT_SIZE=1, MAX_DEPTH=depth)
        y_true = np.array(test[TARGET_ATTRIBUTE])
        accuracy.append(reg.get_accuracy(test, test[TARGET_ATTRIBUTE]))
        reg.prune(test, reg, y_true)
        accuracy_prune.append(reg.get_accuracy(test, test[TARGET_ATTRIBUTE]))

    plt.plot(x_axis, accuracy, label="accuracy")
    plt.plot(x_axis, accuracy_prune, label="prune accuracy")
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy Percentage')
    plt.title('Accuracy v/s Max Depth')
    plt.legend()
    plt.savefig('Accuracy_vs_max_depth.png')
    plt.clf()


if __name__ == '__main__':
    ten_random_splits()
    different_limit_size()
    different_max_depths()
