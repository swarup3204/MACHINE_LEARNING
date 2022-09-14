import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


# load csv file
dataset = pd.read_csv("Train_B_Tree.csv")


# column names
ATTRIBUTES = ['cement', 'slag', 'flyash', 'water',
              'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age']
TARGET_ATTRIBUTE = 'csMPa'


def get_score(y_true: np.ndarray, y_pred: np.ndarray):
    '''
        Input : Expected output and Predicted output in numpy array type
        Output : Accuracy in percentage
        Accuracy function : acc% = 100 - (100 *(|y_true-y_pred|/y_true))
    '''
    err = abs(y_true-y_pred)
    err = (err / y_true)
    err = err * 100
    return 100 - (err.mean())


# Regression Tree
class RegressionTree:
    def __init__(self, db: pd.DataFrame, depth=1, isRoot=True, LIMIT_SIZE=20, MAX_DEPTH=15):
        '''
            db -> Training Dataset
            LIMIT_SIZE -> number of dataset in a leaf node
            MAX_DEPTH -> Maximum depth of tree 
        '''
        self.min_sse = -1
        self.depth = depth
        self.dataset_count = len(db)
        self.isRoot = isRoot
        self.isLeaf = True
        self.return_ans = db[TARGET_ATTRIBUTE].mean()

        # Leaf node Condition
        if len(db) <= LIMIT_SIZE or depth >= MAX_DEPTH:
            return
        else:
            # non-leaf node
            self.isLeaf = False
            min_sse = -1
            self.attrb = None
            self.left_value = None

            # calculating minimun sum square error for all attributes
            for atrribute in ATTRIBUTES:

                # creating new database with only current attribute and target attribute
                new_db = db[[atrribute, TARGET_ATTRIBUTE]]

                #sorting it with respect to attribute
                new_db = new_db.sort_values(by=[atrribute])

                # iterating over all indexes for finding best split
                for i in range(len(new_db)):

                    # dividing dataset into left and right dataset
                    left_db = new_db[TARGET_ATTRIBUTE].iloc[:i]
                    right_db = new_db[TARGET_ATTRIBUTE].iloc[i:]

                    # calcualting mean of both halves
                    mean1 = np.mean(left_db)
                    mean2 = np.mean(right_db)

                    # calculating current sum square error
                    cur_sse = np.sum((left_db-mean1)**2) + np.sum((right_db-mean2)**2)

                    # updating best split
                    if min_sse == -1 or min_sse >= cur_sse:
                        self.attrb = atrribute
                        min_sse = cur_sse
                        self.left_value = new_db[atrribute].iloc[i]

            # storing best split's sum square error            
            self.min_sse = min_sse

            # Recursively creating Left and Right Tree using best split
            self.L = RegressionTree(
                db.loc[db[self.attrb] < self.left_value], depth+1, False, LIMIT_SIZE, MAX_DEPTH)
            self.R = RegressionTree(
                db.loc[db[self.attrb] >= self.left_value], depth+1, False, LIMIT_SIZE, MAX_DEPTH)

    def fit(self, one_row):
        '''
            Input : One Dataset
            Output : Predicted Target Attribute value
        '''
        if (self.isLeaf):
            return self.return_ans
        if (one_row[self.attrb] < self.left_value):
            return self.L.fit(one_row)

        return self.R.fit(one_row)

    def get_output(self, test_input: pd.DataFrame):
        '''
            Input : List of Dataset
            Output : List of Predicted values
        '''
        return np.array([self.fit(test_input.iloc[i]) for i in range(len(test_input))])

    def get_accuracy(self, test_input: pd.DataFrame, y_true: np.ndarray):
        '''
            Input : List of Testing dataset, Expected Output
            Output : accuracy%
        '''
        y_pred = self.get_output(test_input)
        return get_score(y_true, y_pred)

    def prune(self, test_input: pd.DataFrame, root, y_true: np.ndarray):
        '''
            Logic : if making current node a leaf node ... increases accuracy then let current node a leaf node permanently.
            Otherwise revert back changes and Recursively call function for its left and right child
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

    def print_tree(self):
        '''
            Pre-Order Traversal of Tree for printing information
        '''
        if (self.isLeaf):
            print("---"*self.depth, end='>')
            print(
                f" Dataset count : {self.dataset_count}, isLeaf : True, Predicted Value : {self.return_ans}", end="\n")
        else:
            print("---"*self.depth, end='>')
            print(
                f" Dataset count : {self.dataset_count}, isLeaf : False, Split Rule : {self.attrb} < {self.left_value}, min_sse : {self.min_sse}", end='\n')
            self.L.print_tree()
            self.R.print_tree()

    def get_depth(self) -> int:
        '''
            DFS to get depth of tree
        '''
        if(self.isLeaf):
            return 1
        return max(self.L.get_depth(),self.R.get_depth()) + 1

def test_train_split(db: pd.DataFrame, train_size=0.3) -> tuple:
    '''
        Input : Panda Dataframe, train dataset size
        Output : Tuple (Testing Dataset, Training Dataset)
        Function to split dataset into train set and test set
    '''
    random_suffled = db.iloc[np.random.permutation(len(db))]
    split_point = int(len(db)*train_size)
    return random_suffled[:split_point].reset_index(drop=True), random_suffled[split_point:].reset_index(drop=True)


def save_model_tree(model, filename: str, test: pd.DataFrame):
    '''
        Input : Trained model, filename, test dataset to get accuracy
        Tree Diagram stored in given filename directory of given trained tree model
    '''

    # storing original stdout
    original_stdout = sys.stdout

    with open(filename, "w") as f:

        # assigning stdout to given file
        sys.stdout = f

        # printing Accuracy of given model 
        print(f"Accuracy : {model.get_accuracy(test,test[TARGET_ATTRIBUTE])}")

        # printing tree
        model.print_tree()

        # assigning back original stdout
        sys.stdout = original_stdout



def ten_random_splits():
    '''
        Perform ten random splits and plot accuracy
        Plot PNG File : Accuracy_of_10_random_splits.png
    '''

    print("Accuracy of 10 Random Splits")

    # list for plotting graph
    accuracy = []
    accuracy_prune = []
    x_axis = []

    max_acc = 0
    depth_of_best_acc = -1

    for i in range(10):
        print(f"Split ({i+1}):-")
        x_axis.append(i+1)

        # creating a random split
        test, train = test_train_split(dataset, 0.3)

        # traing model
        reg = RegressionTree(train, LIMIT_SIZE=1, MAX_DEPTH=15)
        y_true = np.array(test[TARGET_ATTRIBUTE])

        # accuracy without pruning
        ori_acc = reg.get_accuracy(test, test[TARGET_ATTRIBUTE])
        accuracy.append(ori_acc)
        
        # updating max accuracy
        if max_acc < ori_acc:
            max_acc = ori_acc
            depth_of_best_acc = reg.get_depth()

        #pruning
        reg.prune(test, reg, y_true)

        #accuracy with prune
        accuracy_prune.append(reg.get_accuracy(test, test[TARGET_ATTRIBUTE]))

    # Plotting graph
    plt.plot(x_axis, accuracy, label="accuracy")
    plt.plot(x_axis, accuracy_prune, label="prune accuracy")
    plt.xlabel('Split number')
    plt.ylabel('Accuracy Percentage')
    plt.title('Accuracy of 10 random splits')
    plt.legend()
    plt.savefig('plot/Accuracy_of_10_random_splits.png')
    plt.clf()
    
    print("Plot save in plot/Accuracy_of_10_random_splits.png")
    print(f"Best Accuracy : {max_acc} with depth : {depth_of_best_acc}")

    # storing original stdout
    original_stdout = sys.stdout
    filename = "accuracy_results/ten_random_splits.txt"
    with open(filename, "w") as f:

        # assigning stdout to given file
        sys.stdout = f  

        print(f"Accuracy Without Pruning : {accuracy}")
        print(f"Accuracy With Pruning : {accuracy_prune}")

        # assigning back original stdout
        sys.stdout = original_stdout


def different_limit_size():
    '''
        Perform prediction with different limit sizes and plot accuracy vs limit size graph
        Limit Size : [1,7,13,...,61]
        Plot PNG File : Accuracy_vs_limit_size.png
    '''
    print("Accuracy of Different Limit Size")

    # list for plotting graph
    accuracy = []
    accuracy_prune = []
    x_axis = []

    # splitting dataset
    test, train = test_train_split(dataset, 0.3)

    # min size : [1,7,13,...61]
    for min_size in range(1, 62, 6):
        print(f"Min_size ({min_size}):-")
        x_axis.append(min_size)

        # traing model with current limit size
        reg = RegressionTree(train, LIMIT_SIZE=min_size, MAX_DEPTH=9)
        y_true = np.array(test[TARGET_ATTRIBUTE])

        # accuracy without pruning
        accuracy.append(reg.get_accuracy(test, test[TARGET_ATTRIBUTE]))

        # pruning
        reg.prune(test, reg, y_true)

        # accuracy with pruning
        accuracy_prune.append(reg.get_accuracy(test, test[TARGET_ATTRIBUTE]))

    # Plotting Graph
    plt.plot(x_axis, accuracy, label="accuracy")
    plt.plot(x_axis, accuracy_prune, label="prune accuracy")
    plt.xlabel('Limit Size')
    plt.ylabel('Accuracy Percentage')
    plt.title('Accuracy v/s Limit Size')
    plt.legend()
    plt.savefig('plot/Accuracy_vs_limit_size.png')
    plt.clf()

    print("Plot save in plot/Accuracy_vs_limit_size.png")

    # storing original stdout
    original_stdout = sys.stdout
    filename = "accuracy_results/different_limit_size.txt"
    with open(filename, "w") as f:

        # assigning stdout to given file
        sys.stdout = f  

        print(f"Limit Size : {x_axis}")
        print(f"Accuracy Without Pruning : {accuracy}")
        print(f"Accuracy With Pruning : {accuracy_prune}")

        # assigning back original stdout
        sys.stdout = original_stdout


def different_max_depths():
    '''
        Perform prediction with different max depth and plot accuracy vs max depth graph
        For depth -> [1,14]
        Plot PNG file : Accuracy_vs_max_depth.png 
    '''

    print("Accuracy of Different Max Depth")

    # list for plotting graphs
    accuracy = []
    accuracy_prune = []
    x_axis = []

    #splitting dataset
    test, train = test_train_split(dataset, 0.3)
    y_true = np.array(test[TARGET_ATTRIBUTE])
    for depth in range(1, 15):
        print(f"Max Depth ({depth}):-")
        x_axis.append(depth)

        # creating tree with custom max_depth as depth
        reg = RegressionTree(train, LIMIT_SIZE=1, MAX_DEPTH=depth)

        # appending accuracy without pruning
        accuracy.append(reg.get_accuracy(test, test[TARGET_ATTRIBUTE]))

        # pruning tree
        reg.prune(test, reg, y_true)

        #appending accuracy with pruning
        accuracy_prune.append(reg.get_accuracy(test, test[TARGET_ATTRIBUTE]))

    # plotting accuracy vs max depth
    plt.plot(x_axis, accuracy, label="accuracy")
    plt.plot(x_axis, accuracy_prune, label="prune accuracy")
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy Percentage')
    plt.title('Accuracy v/s Max Depth')
    plt.legend()
    plt.savefig('plot/Accuracy_vs_max_depth.png')
    plt.clf()
    print("Plot save in plot/Accuracy_vs_max_depth.png")

        # storing original stdout
    original_stdout = sys.stdout
    filename = "accuracy_results/different_max_depths.txt"
    with open(filename, "w") as f:

        # assigning stdout to given file
        sys.stdout = f  

        print(f"Max Depth : {x_axis}")
        print(f"Accuracy Without Pruning : {accuracy}")
        print(f"Accuracy With Pruning : {accuracy_prune}")

        # assigning back original stdout
        sys.stdout = original_stdout


def prune_testing():
    '''
        Perform prune testing and prints before and after pruning version of tree in corresponding txt files
        tree_before_pruning.txt : Tree diagram without pruning
        tree_after_pruning.txt : Tree diagram with pruning
    '''
    print("Prune Testing")
    test, train = test_train_split(dataset, 0.3)

    # creating tree with max_depth=5
    reg = RegressionTree(train, MAX_DEPTH=5)
    save_model_tree(reg, "printed_tree/tree_before_pruning.txt", test)
    
    # pruning
    reg.prune(test, reg, np.array(test[TARGET_ATTRIBUTE]))
    save_model_tree(reg, "printed_tree/tree_after_pruning.txt", test)

    print("Saved tree printed_tree/")


if __name__ == '__main__':
    q = 1
    while(1):
        print("\n---------------------------------------------------------------------------------------------------------------------")
        print("0) Exit")
        print("1) Ten Random Split : Perform ten random splits and plot accuracy")
        print("2) Accuracy vs Limit Size : Perform prediction with different limit sizes and plot accuracy vs limit size graph")
        print("3) Accuracy vs Max Depth : Perform prediction with different max depth and plot accuracy vs max depth graph")
        print("4) Prune Test : Perform prune testing and prints before and after pruning version of tree in corresponding txt files")
        print("---------------------------------------------------------------------------------------------------------------------\n")

        q = int(input("Enter Choice : "))

        if q == 0:
            break
        elif q == 1:
            ten_random_splits()
        elif q == 2:
            different_limit_size()
        elif q == 3:
            different_max_depths()
        elif q == 4:
            prune_testing()
        else:
            print("Wrong Choice")
