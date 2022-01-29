import pandas as pd
import warnings
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from math import sqrt
import argparse


parser = argparse.ArgumentParser(description="Simple way to run k-nearest-neighbors algorithm with 10 fold cross "
                                             "validation. I wrote this long time ago for personal use so the "
                                             "algorithm only supports up to two labels, I will add "
                                             "support for more labels and input file types soon. Please contact me if "
                                             "you have any questions from kcan@marun.edu.tr "
                                 )
parser.add_argument('-i', '--input', type=str, metavar='', required=True, help="Path to the xlsx file. Please refer "
                                                                               "to the demo input file when preparing"
                                                                               " the input.")
parser.add_argument('-k', '--k_number', type=int, metavar='', required=False, default=3, help="K value for the "
                                                                                              "algorithm. Default is 3")

args = parser.parse_args()


def get_avg_list(some_list):
    return sum(some_list) / len(some_list)


def eucdist(A, B):
    if len(A) != len(B):
        print('length between vectors differ.')
        quit()
    length = len(A)
    mysum = 0
    for i in range(length):
        mysum = (A[i] - B[i]) ** 2 + mysum

    return sqrt(mysum)


def k_nearest_neighboors(data, predict, k=15):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting group')
    distances = []
    for group in data:
        for features in data[group]:
            euc_distance = eucdist(features, predict)
            distances.append([euc_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


rawdata = pd.read_excel(args.input)

cols_no_label = rawdata.columns[1:]
y_raw = rawdata['label'].values
x_raw = rawdata.drop(labels='label', axis=1).values
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(x_raw, y_raw)

accuracy_h_list = []
accuracy_d_list = []
sensitivity_h_list = []
sensitivity_d_list = []
specifity_h_list = []
specifity_d_list = []
precision_h_list = []
precision_d_list = []
f1_score_h_list = []
f1_score_d_list = []

for train_index, test_index in skf.split(x_raw, y_raw):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x_raw[train_index], x_raw[test_index]
    y_train, y_test = y_raw[train_index], y_raw[test_index]

    X_train = pd.DataFrame(X_train, columns=cols_no_label)
    X_test = pd.DataFrame(X_test, columns=cols_no_label)
    train_df = pd.DataFrame(y_train, columns=['label'])
    test_df = pd.DataFrame(y_test, columns=['label'])

    for col_ in cols_no_label:
        train_df[col_] = X_train[col_]
        test_df[col_] = X_test[col_]

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    train_data_list = train_df.astype(float).values.tolist()
    test_data_list = test_df.astype(float).values.tolist()
    train_set = {0: [], 1: []}  # 0 is disease, 1 is healthy sample
    test_set = {0: [], 1: []}

    for i in train_data_list:
        train_set[i[0]].append(i[1:])
    for i in test_data_list:
        test_set[i[0]].append(i[1:])

    TP_h = 0
    FP_h = 0
    FN_h = 0
    TN_h = 0

    TP_d = 0
    FP_d = 0
    FN_d = 0
    TN_d = 0

    for group in test_set:
        for testfeatures in test_set[group]:
            vote = k_nearest_neighboors(train_set, testfeatures, k=args.k_number)
            if group == 1 and vote == 1:
                TP_h += 1
                TN_d += 1
            elif group == 1 and vote == 0:
                FN_h += 1
                FP_d += 1
            elif group == 0 and vote == 1:
                FP_h += 1
                FN_d += 1
            elif group == 0 and vote == 0:
                TN_h += 1
                TP_d += 1

    sensitivity_h = TP_h / (TP_h + FN_h)
    specifity_h = TN_h / (TN_h + FP_h)
    precision_h = TP_h / (TP_h + FP_h)
    f1_score_h = (2 * precision_h * sensitivity_h) / (precision_h + sensitivity_h)
    accuracy_h = (TP_h + TN_h) / (TP_h + TN_h + FP_h + FN_h)

    sensitivity_d = TP_d / (TP_d + FN_d)
    specifity_d = TN_d / (TN_d + FP_d)
    precision_d = TP_d / (TP_d + FP_d)
    f1_score_d = (2 * precision_d * sensitivity_d) / (precision_d + sensitivity_d)
    accuracy_d = (TP_d + TN_d) / (TP_d + TN_d + FP_d + FN_d)

    accuracy_h_list.append(accuracy_h);
    accuracy_d_list.append(accuracy_d)
    sensitivity_h_list.append(sensitivity_h);
    sensitivity_d_list.append(sensitivity_d)
    specifity_h_list.append(specifity_h);
    specifity_d_list.append(specifity_d)
    precision_h_list.append(precision_h);
    precision_d_list.append(precision_d)
    f1_score_h_list.append(f1_score_h);
    f1_score_d_list.append(f1_score_d)

print('\nAccuracy_h: ' + str(get_avg_list(accuracy_h_list)))
print('Sensitivity_h: ' + str(get_avg_list(sensitivity_h_list)))
print('specificity_h: ' + str(get_avg_list(specifity_h_list)))
print('precision_h: ' + str(get_avg_list(precision_h_list)))
print('f1 score_h: ' + str(get_avg_list(f1_score_h_list)))
print('\n')
print('Accuracy_d: ' + str(get_avg_list(accuracy_d_list)))
print('Sensitivity_d: ' + str(get_avg_list(sensitivity_d_list)))
print('specificity_d: ' + str(get_avg_list(specifity_d_list)))
print('precision_d: ' + str(get_avg_list(precision_d_list)))
print('f1 score_d: ' + str(get_avg_list(f1_score_d_list)))

print('\nf1_avg: ' + str((get_avg_list(f1_score_h_list) + get_avg_list(f1_score_d_list)) / 2))
