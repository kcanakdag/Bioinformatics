import pandas as pd
import numpy as np


# class normalizer:
#     def __init__(self, input_df, sigma_coefficient=1, simple_normalization=True):
#
#         self.input_array = input_df.values
#         self.input_columns = input_df.columns
#         self.simple_normalization = simple_normalization
#         self.sigma_coefficient = sigma_coefficient
#
#     def s_normalization(self):
#         my_array = self.input_array
#         std_list = np.std(my_array, axis=0).tolist()
#         mean_list = np.mean(my_array, axis=0).tolist()
#         for col_i in range(my_array.shape[1]):
#             my_dist = std_list[col_i] * self.sigma_coefficient
#             my_mean = mean_list[col_i]
#             upper_dist = my_mean + my_dist
#             lower_dist = my_mean - my_dist
#             for row_i in range(my_array.shape[0]):
#                 if my_array[row_i, col_i] >= upper_dist:
#                     my_array[row_i, col_i] = 1
#                 elif my_array[row_i, col_i] <= lower_dist:
#                     my_array[row_i, col_i] = -1
#                 else:
#                     my_array[row_i, col_i] = 0
#         return pd.DataFrame(my_array, columns=self.input_columns)
#
#     def normalization(self):
#         pass

def s_normalization(input_array, input_columns, sigma_coefficient=1):
    my_array = input_array
    std_list = np.std(my_array, axis=0).tolist()
    mean_list = np.mean(my_array, axis=0).tolist()
    for col_i in range(my_array.shape[1]):
        my_dist = std_list[col_i] * sigma_coefficient
        my_mean = mean_list[col_i]
        upper_dist = my_mean + my_dist
        lower_dist = my_mean - my_dist
        for row_i in range(my_array.shape[0]):
            if my_array[row_i, col_i] >= upper_dist:
                my_array[row_i, col_i] = 1
            elif my_array[row_i, col_i] <= lower_dist:
                my_array[row_i, col_i] = -1
            else:
                my_array[row_i, col_i] = 0
    return pd.DataFrame(my_array, columns=input_columns)


if __name__ == "__main__":
    input_path = r"~/PycharmProjects/Bioinformatics/test_datasets/normalize_test.csv"
    df = pd.read_csv(input_path, header=0, index_col=None)
    # normalizer_obj = normalizer(input_df=df)
    normalized_df = s_normalization(input_array=df.values, input_columns=df.columns, sigma_coefficient=1)
    print(normalized_df)