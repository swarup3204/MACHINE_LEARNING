# Assignment:1
# Grp 25
# Name: Swarup Padhi
# Name: Grace Sharma

import pandas as pd
import numpy as np
import math


# splitting the dataset into train and test 70:30 ratio
def test_train_split(db) -> tuple:
    random_suffled = db.iloc[np.random.permutation(len(db))]
    split_point = int(len(db)*0.3)
    return random_suffled[:split_point].reset_index(drop=True), random_suffled[split_point:].reset_index(drop=True)


def remove_outlier(df_in, col_name, THRESHOLD):
    # remove outlier based on threshold
    # find maximum outlier feature number of a row
    # max
    max_outlier = 0
    for i in range(len(df_in)):
        outlier = 0
        for col in col_name:
            if df_in[col][i] > THRESHOLD[col]:
                outlier += 1
        max_outlier = max(outlier, max_outlier)

    for i in range(len(df_in)):
        outlier = 0
        for col in col_name:
            if df_in[col][i] > THRESHOLD[col]:
                outlier += 1
        if outlier == max_outlier:
            df_in = df_in.drop(i)

    return df_in

# function to normalize dataset


def normalize(dataset):
    normalized_dataset = (dataset-dataset.mean())/dataset.std()
    normalized_dataset['is_patient'] = dataset['is_patient']
    return normalized_dataset

# function to calculate value of probablitiy by normal distribution at a point


def normal_pdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def divide_data(dataset, k):
  # five fold cross validation
    dataset.sample(frac=1, random_state=200)
    # obtaining list of column names
    column_list = dataset.columns
    # convert pandas dataframe to numpy array
    dataset_arr = dataset.values[:, 0:11]
    # obtain 5 almost equal sized arrays
    sets = np.array_split(dataset_arr, k)

    df_set = []   # to store pandas dataframe 5 equal sized sets => the numpy arrays after 5 equal sized splits

    for ind, ele in enumerate(sets):
        # convert numpy arrays to pandas dataframe
        df_set.append(pd.DataFrame(ele, columns=column_list))
        df_set[ind].reset_index(drop=True)              # re -indexing

    return df_set


def calculate_parameters(train, laplace_flag):

    # split train into train_positive and train_negative
    # 1 indicates person is liver patient and 2 not
    train_positive = train[train['is_patient'] == 1]
    train_negative = train[train['is_patient'] == 2]
    # calcuating mean and standard of columns of train_positive and train_negative
    MEAN_1 = train_positive.mean()
    STD_1 = train_positive.std()

    MEAN_2 = train_negative.mean()
    STD_2 = train_negative.std()

    # since gender is a discrete variable,probability of a particular class given an outcome computed from dataset
    train_positive_gender_prob = []
    train_negative_gender_prob = []
    x1 = len(train_positive[train_positive['gender'] == 0])
    y1 = len(train_positive[train_positive['gender'] == 1])
    x2 = len(train_negative[train_negative['gender'] == 0])
    y2 = len(train_negative[train_negative['gender'] == 1])

    if laplace_flag == 1:
        x1 = max(x1, 1)
        y1 = max(y1, 1)
        x2 = max(x2, 1)
        y2 = max(y2, 1)

    # storing probability of positve outcome given female and male in a list
    train_positive_gender_prob.append(x1/(x1+y1))
    train_positive_gender_prob.append(y2/(x2+y2))

    # storing probability of negative outcome given female and male in a list
    train_negative_gender_prob.append(x1/(x1+y1))
    train_negative_gender_prob.append(y2/(x2+y2))

    # just computing number of patients is enough as denominator is same for both
    prior_is_patient = len(train_positive)
    prior_is_not_patient = len(train_negative)

    return MEAN_1, MEAN_2, STD_1, STD_2, train_positive_gender_prob, train_negative_gender_prob, prior_is_patient, prior_is_not_patient


def predict(test_data, column_headers, MEAN_1, STD_1, MEAN_2, STD_2, train_positive_gender_prob, train_negative_gender_prob, prior_is_patient, prior_is_not_patient):
    prob_1 = prior_is_patient
    prob_2 = prior_is_not_patient

    # now multiplying prior with the likelihood given is patient and not patient
    for feature in column_headers:
        if feature == "gender":  # if feature is "gender" use probability computed from data
            prob_1 = prob_1 * \
                train_positive_gender_prob[int(test_data["gender"])]
            prob_2 = prob_2 * \
                train_negative_gender_prob[int(test_data["gender"])]
        if feature == "is_patient":  # last column of dataset which is the result column so ignored
            break
        prob_1 = prob_1 * \
            normal_pdf(test_data[feature], MEAN_1[feature], STD_1[feature])
        # calculating probability of a particular feature given is patient
        prob_2 = prob_2 * \
            normal_pdf(test_data[feature], MEAN_2[feature], STD_2[feature])
        # calculating probability of a particular feature given is not patient

    # now comparing the probability of is patient and is not patient and returning the class with higher probability
    prediction = 1 if prob_1 > prob_2 else 2
    # returning true if class of higher probablity matches the class of the test data
    if test_data["is_patient"] == prediction:
        return True
    else:
        return False


def find_accuracy(test, MEAN_1, MEAN_2, STD_1, STD_2, train_positive_gender_prob, train_negative_gender_prob, prior_is_patient, prior_is_not_patient):
    column_headers = list(test.columns)
    # getting list of column names of test data

    test_positive = test[test['is_patient'] == 1]
    test_negative = test[test['is_patient'] == 2]
    # splitting test data into test_positive and test_negative

    tp = 0      # true positive
    tn = 0      # true negative
    fp = 0      # false positive
    fn = 0      # false negative

    test_positive = test_positive.dropna().reset_index(drop=True)
    test_negative = test_negative.dropna().reset_index(drop=True)

    for row in test_positive.iterrows():
        # checking prediction of each row of test set
        ret = predict(row[1], column_headers, MEAN_1, STD_1, MEAN_2, STD_2, train_positive_gender_prob,
                      train_negative_gender_prob, prior_is_patient, prior_is_not_patient)
        if ret == True:
            tp = tp+1
        else:
            fn = fn+1

    for row in test_negative.iterrows():
      # checking prediction of each row of test set
        ret = predict(row[1], column_headers, MEAN_1, STD_1, MEAN_2, STD_2, train_positive_gender_prob,
                      train_negative_gender_prob, prior_is_patient, prior_is_not_patient)
        if ret == True:
            tn = tn+1
        else:
            fp = fp+1

    accuracy = (tp+tn)/(tp+tn+fp+fn)

    # we can calculate other accuracy measures too

    return accuracy*100


def main():

    f = open("output_2.txt", "w")
    # read the dataset
    dataset = pd.read_csv("Train_B_Bayesian.csv")
    dataset = dataset.dropna().reset_index(drop=True)
    # dropping rows which have missing values in any feature
    print("Length of dataset", len(dataset), file=f)
    # getting the gender column as a series
    static = pd.get_dummies(dataset['gender'], drop_first=True)

    # 0 indicates Female and 1 indicates Male

    # removing earlier gender column and replacing it with new column of gender after categorical encoding of the gender column
    dataset = pd.concat([dataset, static], axis=1)
    dataset = dataset.drop('gender', axis=1)
    dataset = dataset.rename(columns={'Male': 'gender'})
    col = dataset.pop('gender')
    dataset.insert(0, 'gender', col)
    # inserting at 0th column

    MEAN = dataset.mean()
    STD = dataset.std()
    THRESHOLD = 2*MEAN+5*STD
    # calculating mean ,standard deviation and threshold for all columns to remove outliers based on the condition, x > 2*mean + 5*std

    dataset_filtered = remove_outlier(dataset, dataset.columns, THRESHOLD)
    dataset_filtered = dataset_filtered.dropna().reset_index(drop=True)

    print('Length of dataset after removing outliers: ',
          len(dataset_filtered), file=f)
    # filtering dataset by removing outliers and reset the indexing of the rows

    # dataset_filtered=normalize(dataset_filtered)
    # print(dataset_filtered)

    test, train = test_train_split(dataset_filtered)

    # training
    print('Final set of features of dataset', file=f)
    for feature in dataset_filtered.columns:
        print(feature, file=f)

    MEAN_1, MEAN_2, STD_1, STD_2, train_positive_gender_prob, train_negative_gender_prob, prior_is_patient, prior_is_not_patient = calculate_parameters(
        train, False)

    # printing accuracy of training given test data
    print('Accuracy of training by 70:30 random split', file=f)
    print(find_accuracy(test, MEAN_1, MEAN_2, STD_1, STD_2, train_positive_gender_prob,
                        train_negative_gender_prob, prior_is_patient, prior_is_not_patient), file=f)

    # Laplace correction

    MEAN_1_laplace, MEAN_2_laplace, STD_1_laplace, STD_2_laplace, train_positive_gender_prob_laplace, train_negative_gender_prob_laplace, prior_is_patient_laplace, prior_is_not_patient_laplace = calculate_parameters(
        train, True)

    print('Accuracy of training after Laplace correction', file=f)
    print(find_accuracy(test, MEAN_1_laplace, MEAN_2_laplace, STD_1_laplace, STD_2_laplace, train_positive_gender_prob_laplace,
                        train_negative_gender_prob_laplace, prior_is_patient_laplace, prior_is_not_patient_laplace), file=f)

    dataset_filtered = dataset_filtered.sample(frac=1)
    dataset_filtered = dataset_filtered.reset_index(drop=True)
    df_set = divide_data(dataset_filtered, 5)
    # getting 5 equal sized sets of dataset

    # considering test sets each of 5 sets one by one
    acc = []
    # to store accuracy of each instance of 5 fold cross validation

    print('Accuracy result of each iteration of 5 fold cross validation', file=f)

    for ind, ele in enumerate(df_set):
        # series contains all sets except the current set which is test set
        series = [(ser) for index, ser in enumerate(df_set) if index != ind]

        test_new = df_set[ind]

        # train_new is the union of all sets except the current set
        train_new = pd.concat(series, axis=0)

        MEAN_1_new, MEAN_2_new, STD_1_new, STD_2_new, train_positive_gender_prob_new, train_negative_gender_prob_new, cnt_new, cnt_not_new = calculate_parameters(
            train_new, False)
        acc.append(find_accuracy(test_new, MEAN_1_new, MEAN_2_new, STD_1_new, STD_2_new,
                                 train_positive_gender_prob_new, train_negative_gender_prob_new, cnt_new, cnt_not_new))
        print(acc[ind], file=f)

    print('Average accuracy of 5 fold cross validation is', np.mean(acc), file=f)

    # 5 fold cross validation with laplace correction
    print('Accuracy result of each iteration of 5 fold cross validation with Laplace correction', file=f)

    for ind, ele in enumerate(df_set):
        # series contains all sets except the current set which is test set
        series = [(ser) for index, ser in enumerate(df_set) if index != ind]

        test_new = df_set[ind]

        # train_new is the union of all sets except the current set
        train_new = pd.concat(series, axis=0)

        MEAN_1_new, MEAN_2_new, STD_1_new, STD_2_new, train_positive_gender_prob_new, train_negative_gender_prob_new, cnt_new, cnt_not_new = calculate_parameters(
            train_new, True)
        acc.append(find_accuracy(test_new, MEAN_1_new, MEAN_2_new, STD_1_new, STD_2_new,
                                 train_positive_gender_prob_new, train_negative_gender_prob_new, cnt_new, cnt_not_new))
        print(acc[ind], file=f)

    print('Average accuracy of 5 fold cross validation with Laplace correction',
          np.mean(acc), file=f)

    f.close()


if __name__ == "__main__":
    main()
