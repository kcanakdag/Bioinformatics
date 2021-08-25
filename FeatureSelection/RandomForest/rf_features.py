import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="This is a simple tool to extract important features from a dataset via "
                                             "entropy. "
                                 )
parser.add_argument('-i', '--input', type=str, metavar='', required=True, help=".csv dataset file")
parser.add_argument('-min', '--min_samples', type=int, metavar='', required=False, help="Min samples required to "
                                                                                        "classify dataset in decision "
                                                                                        "tree, lower values may lead "
                                                                                        "to overfitting", default=10)
parser.add_argument('-nf', '--num_select_features', type=int, metavar='', required=False, help="Number of features "
                                                                                               "randomly selected "
                                                                                               "with each decision "
                                                                                               "tree training. Higher "
                                                                                               "values may yield "
                                                                                               "better accuracy but "
                                                                                               "it also requires more "
                                                                                               "time.",
                    default=100)
parser.add_argument('-tt', '--tt_percent', type=int, metavar='', required=False, help="train-test split size"
                                                                                        " %%", default=20)
parser.add_argument('-rs', '--rs_percent', type=int, metavar='', required=False, help="randomly selected samples"
                                                                                      " %%", default=20)
parser.add_argument('-n', '--num_trees', type=int, metavar='', required=False, help="number of trees generated for "
                                                                                    "random forest", default=20)
parser.add_argument('-acc', '--acc_cutoff', type=int, metavar='', required=False, help="accuracy cutoff %%", default=90)
parser.add_argument('-o', '--output_name', type=str, metavar='', required=False, help="name of the output file",
                    default='./rf_features_result.csv')

args = parser.parse_args()


# Train-Test split pandas df function
def train_test_split(data, testsize=args.tt_percent/100):
    train_data = data.sample(frac=(1 - testsize), random_state=np.random.randint(1000))
    test_data = data.drop(train_data.index)
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    train_data.drop(columns=['index'], inplace=True)
    test_data.drop(columns=['index'], inplace=True)
    return train_data, test_data


# Data purity checker function # Checks if the labels are the same in a data group
def purity_checker(data):
    labels = data[:, -1]
    if len(np.unique(labels)) == 1:  # Only one label, data is pure
        return True
    else:
        return False


# Classifier function
def classify(data):
    labels = data[:, -1]
    unique_labels, counts_unique_classes = np.unique(labels, return_counts=True)
    index = counts_unique_classes.argmax()  # returns the index of highest count
    return unique_labels[index]


# Potential Splits
def get_potential_splits(data):
    columncount = data.shape[1]
    potential_splits = {}

    for column_index in range(columncount - 1):  # -1 for labels
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)
        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                prev_value = unique_values[index - 1]
                potential_split = (current_value + prev_value) / 2
                potential_splits[column_index].append(potential_split)

    return potential_splits


# Split Data Function
def split_data(data, attribute_column, split_value):
    condition = data[:, attribute_column] <= split_value
    slicedunder = data[condition]
    slicedover = data[(np.invert(condition))]

    return slicedover, slicedunder


# Lowest Overall Entropy Function to determine best split
def get_entropy(data, label_column=-1):
    labels = data[:, label_column]
    counts = np.unique(labels, return_counts=True)[1]
    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy


# Using overall entropy formula to compare data under-over the split
def get_overall_entropy(data_over, data_under):
    n_data_points = len(data_over) + len(data_under)
    p_data_under = len(data_under) / n_data_points
    p_data_over = len(data_over) / n_data_points
    overall_entropy = (p_data_under * get_entropy(data_under) + p_data_over * get_entropy(data_over))
    return overall_entropy


def determine_best_split(data, potential_splits):
    overall_entropy = 999
    best_split_column = 0
    best_split_value = 0
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_over, data_under = split_data(data, attribute_column=column_index, split_value=value)
            current_overall_entropy = get_overall_entropy(data_over, data_under)
            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


# Main Algorithm

def get_decision_tree(data_df, column_names, counter=0, min_samples=args.min_samples):
    if counter == 0:
        data = data_df.values
    else:
        data = data_df

    # base case
    if purity_checker(data) or len(data) < min_samples:
        classification = classify(data)
        return classification
    # recursive part
    else:
        #  This part repeats until purity is reached
        counter += 1
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_under, data_over = split_data(data, split_column, split_value)

        # Sub tree creation
        feature_name = column_names[split_column]
        question = "%s ≤ %s" % (feature_name, split_value)
        sub_tree = {question: []}

        # find answers (recursion)
        yes_answer = get_decision_tree(data_under, column_names, counter)
        no_answer = get_decision_tree(data_over, column_names, counter)

        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)

        return sub_tree


def classify_sample(example, d_tree):
    if not isinstance(d_tree, dict):
        answer = d_tree

        return answer
    question = list(d_tree.keys())[0]
    feature_name, comparison_operator, value = question.split()
    if example[feature_name] <= float(value):
        answer = d_tree[question][1]
    else:
        answer = d_tree[question][0]
    if isinstance(answer, dict):
        return classify_sample(example, answer)
    if not isinstance(answer, dict):
        return answer


def determine_accuracy(test_datadf, d_tree):
    test_data = test_datadf
    labels = np.unique(test_data['label'].values).tolist()
    accuracies = []
    for label in labels:
        test_classify_data = test_data[test_data['label'] == label].copy()
        test_classify_data['classification'] = test_classify_data.apply(classify_sample, axis=1, args=(d_tree,))
        test_classify_data['classification_correct'] = test_classify_data.classification == test_classify_data.label
        accuracies.append(test_classify_data.classification_correct.mean())
    avg_acc = sum(accuracies) / len(accuracies)

    return avg_acc


def get_random_forest(mtrain_df, mtest_df, num_trees=args.num_trees, num_select_features=args.num_select_features,
                      accuracy_cutoff=args.acc_cutoff/100,
                      sample_percentage=args.rs_percent/100):
    num_features = len(mtrain_df.columns) - 1
    if num_features < num_select_features:
        num_select_features = num_features
    tree_list = []
    tree_counter = 0
    while len(tree_list) < num_trees:
        label_column = mtrain_df['label'].copy()
        train_data_random = mtrain_df.drop(['label'], axis=1).copy()
        train_data_random = train_data_random.sample(num_select_features, axis=1)
        train_data_random['label'] = label_column
        train_data_random = train_data_random.sample(frac=sample_percentage)
        tree_counter += 1
        print(f"Creating tree {tree_counter}...")
        dec_tree = get_decision_tree(train_data_random, train_data_random.columns)
        avg_acc = determine_accuracy(mtest_df, dec_tree)
        print(f"Average accuracy: {avg_acc} \n")
        if avg_acc >= accuracy_cutoff:
            tree_list.append(dec_tree)
    return tree_list


def get_feature_counts(tree_list):
    global features
    features = []

    def get_features(decision_tree):
        if not isinstance(decision_tree, dict):
            return None
        else:
            question = list(decision_tree.keys())[0]
            feature_name, comparison_operator, value = question.split()
            features.append(feature_name)
            path_one = decision_tree[question][0]
            path_two = decision_tree[question][1]

            for path in [path_one, path_two]:
                if isinstance(path, dict):
                    return get_features(path)
                else:
                    pass

    for tree in tree_list:
        get_features(tree)
    feature_dict = {}
    for feature in features:
        if feature in feature_dict.keys():
            new_count = feature_dict[feature] + 1
            feature_dict[feature] = new_count
        else:
            feature_dict[feature] = 1

    feature_df = pd.DataFrame.from_dict(feature_dict, orient='index', columns=['Feature Counts'])
    return feature_df


# input args;
INPUT_FILE_PATH = args.input

df = pd.read_csv(INPUT_FILE_PATH, header=0, index_col=None)
df = df.rename(columns={str(df.columns[-1]): 'label'})
df.iloc[:, -1] = df.iloc[:, -1].apply(str).values  # convert last column (labels) into str
num_labels = len(np.unique(df['label'].values).tolist())
train_df, test_df = train_test_split(df)
test_num_labels = len(np.unique(test_df['label'].values).tolist())
train_num_labels = len(np.unique(train_df['label'].values).tolist())
while test_num_labels != num_labels and train_num_labels != num_labels:
    test_num_labels = len(np.unique(test_df['label'].values).tolist())
    train_num_labels = len(np.unique(train_df['label'].values).tolist())
    train_df, test_df = train_test_split(df)  # Split train data for testing created trees
tree_l = get_random_forest(mtrain_df=train_df, mtest_df=test_df)
feature_counts = get_feature_counts(tree_l)
feature_counts.to_csv(path_or_buf=args.output_name)
