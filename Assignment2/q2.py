import pandas as pd
import math
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# in_built function not allowed


def standard_scalar_normalize(df):
    # do standard scalar normalisation on all columns except class column
    for col in df.columns:
        if col == "class":
            break
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df

# in_built function not allowed
# class column categorically encoded as follows
# 0 --> Iris-setosa
# 1 --> Iris-versicolor
# 2 --> Iris-virginica


def categorical_encoding(df):
    # do categorical encoding on class column
    for i in range(len(df)):
        # print(flower_class.iloc[i,0])
        if (df.iloc[i, 4] == 'Iris-setosa'):
            df.iloc[i, 4] = 0
        if (df.iloc[i, 4] == 'Iris-versicolor'):
            df.iloc[i, 4] = 1
        if (df.iloc[i, 4] == 'Iris-virginica'):
            df.iloc[i, 4] = 2
    return df

# in_built function not allowed


def sample_dataframe(df):
    random_suffled = df.iloc[np.random.permutation(len(df))]
    return random_suffled

# split dataframe into 80:20 ratio for training and testing


def train_test_split(df) -> tuple:
    split_point = int(len(df)*0.8)
    return df[:split_point].reset_index(drop=True), df[split_point:].reset_index(drop=True)


def find_linearly_separable(df):
    # find linearly separable classes
    # return class which is linearly separable
    # if no class is linearly separable return -1
    train, test = train_test_split(df)
    train_x = train.iloc[:, 0:4].to_numpy().astype('float')
    train_y = train.iloc[:, 4].to_numpy().astype('float')
    test_x = test.iloc[:, 0:4].to_numpy().astype('float')
    test_y = test.iloc[:, 4].to_numpy().astype('float')
    # print(test_x)
    # print(test_y)
    acc=[]
    for k in range(3):
        acc.append(find_accuracy_SVM(train_x,train_y,test_x, test_y, k,basis = 'linear',C = 4000000))
        # print(acc)

    max_acc = max(acc)
    return acc.index(max_acc)+1


# calculate accuracy for a particular kernel and print it

def find_accuracy_SVM(train_x,train_y,test_x, test_y, k, basis,C=1.0,degree=3):
    # if poly kernel degree considered else ignored
    clf = SVC(C=C,kernel=basis,degree=degree)
    clf.fit(train_x,train_y)
    y_pred = clf.predict(test_x)
    return accuracy_score(test_y, y_pred)

# in built function allowed

def print_accuracy_SVM(df, k):
    # print accuracy of of SVM on given dataframe
    # print accuracy using linear, quadratic and rbf kernel
    # iterate test_y and set label of class k(one assumed as linearly separable) as 1
    # and other classes as -1 and fit SVM classifier on it
    # split dataframe into 80:20 ratio for training and testing
    train,test = train_test_split(df)
    train_x = train.iloc[:, 0:4].to_numpy().astype('float')
    train_y = train.iloc[:, 4].to_numpy().astype('float')
    test_x = test.iloc[:, 0:4].to_numpy().astype('float')
    test_y = test.iloc[:, 4].to_numpy().astype('float')
    acc1,acc2,acc3 = 0,0,0
    for i in range(len(test_y)):
        if (test_y[i] == k):
            test_y[i] = 1
        else:
            test_y[i] = -1
    for i in range(len(train_y)):
        if (train_y[i] == k):
            train_y[i] = 1
        else:
            train_y[i] = -1
    acc1=find_accuracy_SVM(train_x,train_y,test_x, test_y, k, 'linear')
    acc2=find_accuracy_SVM(train_x,train_y,test_x, test_y, k, 'poly',degree=2)
    acc3=find_accuracy_SVM(train_x,train_y,test_x, test_y, k, 'rbf')

    print(f"Accuracy of SVM using linear kernel is {acc1}")
    print(f"Accuracy of SVM using quadratic kernel is {acc2}")
    print(f"Accuracy of SVM using radial basis function kernel is {acc3}")

def find_accuracy_MLP(train,test,hidden_layers,learning_rate):
    # hidden layers is a list of number of neurons in each hidden layer
    # for example [10,20,30] means 3 hidden layers with 10,20,30 neurons respectively
    # print accuracy of MLP classifier on given dataframe
    train_x = train.iloc[:, 0:4].to_numpy().astype('float')
    train_y = train.iloc[:, 4].to_numpy().astype('float')
    test_x = test.iloc[:, 0:4].to_numpy().astype('float')
    test_y = test.iloc[:, 4].to_numpy().astype('float')
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers,solver='sgd',batch_size=32,learning_rate_init=learning_rate)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    return accuracy_score(test_y, y_pred)


# Builds 2 MLP classifiers as given in question
# compares and prints accuracy
# returns most accurate model
# 1 denotes model with 1 hidden layer of 16 nodes
# 2 denotes model with 2 hidden layers of 256 and 16 nodes
# optimiser is stochastic gradient descent
def get_accurate_MLP(df):
    train,test = train_test_split(df)

    acc1 = find_accuracy_MLP(train,test,(16),0.001)
    # print accuracy with one hidden layer of 16 nodes
    print(f"The accuracy of MLP classifier with one hidden layer of 16 nodes is {acc1}")
    
    acc2 = find_accuracy_MLP(train,test,(256,16),0.001)

    # print accuracy with 2 hidden layers of size 256 and 16 nodes

    print(f"The accuracy of MLP classifier with two hidden layers of 256 and 16 nodes is {acc2}")

    if acc1 > acc2:
        return 1
    else:
        return 2

# in built function not allowed
# performs backward elmination of features and prints the best set of features
# find the feature whose removal causes least error and remove if error less than original error
# repeat until no feature can be removed,i.e, original error is less than error after removing a feature
# error = number of samples*(1 - accuracy)
# since number of samples is constant (80:20 split),error is proportional to (1-accuracy) 
# therefore we can compare accuracies instead of errors
def perform_backward_elimination(df,hidden_layers,learning_rate):
    train,test = train_test_split(df)
    org_acc=find_accuracy_MLP(train,test,hidden_layers,learning_rate)
    print("BACKWARD ELIMINATION STARTED")
    # removing features one by one
    # find best accuracy after removing each feature and remove it if it gives better accuracy than original
    while True:
        print("Current accuracy is ",org_acc)
        acc_list = []
        k = len(train.columns)-1
        if k == 1:
            break
        for i in range(k):
            train1 = train.drop(train.columns[i],axis=1)
            test1 = test.drop(test.columns[i],axis=1)
            train_x = train1.iloc[:, 0:k-1].to_numpy().astype('float')
            train_y = train1.iloc[:, k-1].to_numpy().astype('float')
            test_x = test1.iloc[:, 0:k-1].to_numpy().astype('float')
            test_y = test1.iloc[:, k-1].to_numpy().astype('float')
            clf = MLPClassifier(hidden_layer_sizes=hidden_layers,solver='sgd',batch_size=32,learning_rate_init=learning_rate)
            clf.fit(train_x, train_y)
            y_pred = clf.predict(test_x)
            acc = accuracy_score(test_y, y_pred)
            acc_list.append(acc)
        max_acc = max(acc_list)
        # print accuracy on removing each feature
        for i in range(k):
            print(f"Accuracy after removing feature {i} is {acc_list[i]}")
        if max_acc > org_acc:
            org_acc = max_acc
            train = train.drop(train.columns[acc_list.index(max_acc)],axis=1)
            test = test.drop(test.columns[acc_list.index(max_acc)],axis=1)
            print(f"Removing feature {acc_list.index(max_acc)}")
        else:
            print(f"Stopped removing features")
            break
        
    print(f"The best set of features is {train.columns[:-1].astype('str').to_list()}")
        
    

# performs max voting technique with 3 models and prints accuracy
# SVM with quadratic
# SVM with rbf
# best MLP model found in part 3
def ensemble_max_voting_technique(df,hidden_layers):
    train,test = train_test_split(df)
    train_x = train.iloc[:, 0:4].to_numpy().astype('float')
    train_y = train.iloc[:, 4].to_numpy().astype('float')
    test_x = test.iloc[:, 0:4].to_numpy().astype('float')
    test_y = test.iloc[:, 4].to_numpy().astype('float')
    clf1 = SVC(kernel='poly',degree=2)
    clf1.fit(train_x,train_y)
    clf2 = SVC(kernel='rbf')
    clf2.fit(train_x,train_y)
    clf3 = MLPClassifier(hidden_layer_sizes=hidden_layers,solver='sgd',batch_size=32,learning_rate_init=0.001)
    clf3.fit(train_x,train_y)
    y_pred_list = []
    y_pred_list.append(clf1.predict(test_x))
    y_pred_list.append(clf2.predict(test_x))
    y_pred_list.append(clf3.predict(test_x))
    # print(clf1.predict_proba(test_x))
    # y_prob1 = clf1.predict_proba(test_x)
    # y_prob2 = clf2.predict_proba(test_x)
    # y_prob3 = clf3.predict_proba(test_x)
    acc_list=[]
    acc_list.append(accuracy_score(test_y, y_pred_list[0]))
    acc_list.append(accuracy_score(test_y, y_pred_list[1]))
    acc_list.append(accuracy_score(test_y, y_pred_list[2]))
    y_pred = []
    for i in range(len(y_pred_list[0])):
        if y_pred_list[0][i] == y_pred_list[1][i] or y_pred_list[0][i] == y_pred_list[2][i] or y_pred_list[1][i] == y_pred_list[2][i]:
            y_pred.append(y_pred_list[0][i])
        else:   # all 3 models predict different classes for a particular test case, assign class with highest accuracy
            y_pred.append(y_pred_list[acc_list.index(max(acc_list))])
            # if y_prob1[i][0] > y_prob1[i][1] and y_prob1[i][0] > y_prob1[i][2]:
            #     y_pred.append(0)
            # elif y_prob1[i][1] > y_prob1[i][0] and y_prob1[i][1] > y_prob1[i][2]:
            #     y_pred.append(1)
            # else:
            #     y_pred.append(2)
            
    print(f"Accuracy of ensemble model is {accuracy_score(test_y, y_pred)}")


if __name__ == '__main__':
    # Read the data
    df = pd.read_csv('iris_data.csv')
    # Remove rows with missing values
    df.dropna(inplace=True)
    # reset the indexing of rows after removing rows with missing values
    #print(df)
    df.reset_index(drop=True, inplace=True)
    # standard scalar normalisation
    df = standard_scalar_normalize(df)
    # categorical encoding of "class" column
    df = categorical_encoding(df)
    # shuffle the dataframe
    #print(df)
    df = sample_dataframe(df)
    df.reset_index(drop=True, inplace=True)
    #print(df)
    '''
       SVMs with linear kernel find the longest margin that separates train data.
       If we set the C hyperparameter to a very high number (e.g. 2^32), we will force the optimizer to make 0 error in 
       classification in order to minimize the loss function. 
       Thus, we will overfit the data.
       If we can overfit it with a linear model, that means the data is linearly separable.
    '''
    k = find_linearly_separable(df)
    if k == 0:
        print("Class Iris-setosa is linearly separable than other two classes, Iris-versicolor and Iris-virginica")
    elif k == 1:
        print("Class Iris-versicolor is linearly separable than other two classes, Iris-setosa and Iris-virginica")
    elif k == 2:
        print("Class Iris-virginica is linearly separable than other two classes, Iris-setosa and Iris-versicolor")
    else:
        print("No class is linearly separable than other two classes")

    # print accuracy of SVM on given dataframe with the linearly
    # separable class as 1 and other classes as -1
    print_accuracy_SVM(df, k)

    # get 1 if first option given in question gives more accuracy ,else 2
    opt = get_accurate_MLP(df)
    
    if opt == 1:
        print("First option given in question gives more accuracy,hidden layers = (16)")
        layers_hidden = (16)
    else:
        print("Second option given in question gives more accuracy,hidden layers = (256,16)")
        layers_hidden = (256,16)
    
    # vary learning rate of most optimal MLP classifier and print its accuracy
    # also plot on graph 
    learning_rates = [0.0001,0.001,0.01,0.1,1]
    accuracies = []
    classifiers = []
    # split data into test and train
    train,test = train_test_split(df)
    for i in learning_rates:
        accuracies.append(find_accuracy_MLP(train,test,layers_hidden,i))
    print("Learning rates are ",learning_rates)
    print("Accuracies are ",accuracies)
    print(f"Maximum accuracy learning rate is {learning_rates[accuracies.index(max(accuracies))]}")
    plt.plot(learning_rates,accuracies)
    plt.xlabel("Learning rates")
    plt.ylabel("Accuracy")
    # save it as a png file
    plt.savefig("MLP accuracy vs learning_rate.png")

    # use backward elimination on best accuracy MLP classifier above and print the best set of features
    perform_backward_elimination(df,layers_hidden,0.01)

    # perform ensemble learning (max voting technique) with 3 models and print accuracy
    ensemble_max_voting_technique(df,layers_hidden)