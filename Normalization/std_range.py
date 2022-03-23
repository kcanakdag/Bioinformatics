import pandas as pd
import numpy as np


def s_normalization(input_array, input_columns, sigma_coefficient=1):
    my_array = input_array.astype(float)
    # print('my_array')
    # print(my_array)
    std_list = np.std(my_array, axis=1).tolist()
    # print('std_list')
    # print(std_list)
    mean_list = np.mean(my_array, axis=1).tolist()
    # print('mean_list')
    print(mean_list)
    for row_i in range(my_array.shape[0]):
        my_dist = std_list[row_i] * sigma_coefficient
        my_mean = mean_list[row_i]
        upper_dist = my_mean + my_dist
        lower_dist = my_mean - my_dist
        for col_i in range(my_array.shape[1]):
            if my_array[row_i, col_i] >= upper_dist:
                my_array[row_i, col_i] = 1
            elif my_array[row_i, col_i] <= lower_dist:
                my_array[row_i, col_i] = -1
            else:
                my_array[row_i, col_i] = 0
    return pd.DataFrame(my_array, columns=input_columns)


def n_normalization(input_array, input_columns, sigma_coefficient=2):
    pass


if __name__ == "main":
    input_path = r"~/PycharmProjects/Bioinformatics/test_datasets/normalize_test.csv"
    df = pd.read_csv(input_path, header=0, index_col=None)
    # normalizer_obj = normalizer(input_df=df)
    normalized_df = s_normalization(input_array=df.values, input_columns=df.columns, sigma_coefficient=1)
    print(normalized_df)