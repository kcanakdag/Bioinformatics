import pandas as pd
import numpy as np
from Normalization import std_range

PATH_TRANSCRIPTOME = r"~/PycharmProjects/Bioinformatics/test_datasets/ppi_test.csv"
PATH_PPI = r"~/PycharmProjects/Bioinformatics/test_datasets/ppi_test.csv"
PATH_Q_TABLE_SAVE = r"~/PycharmProjects/Bioinformatics/q_table.csv"


def count_ppi_states(input_df, ppi_arr):
    result_dict = {}
    for i_ppi in range(len(ppi_arr)):
        ppi_int = ppi_arr[i_ppi]
        if ppi_int[0] in input_df.columns and ppi_int[1] in input_df.columns:
            interaction_name = f"{ppi_int[0]}-{ppi_int[1]}"
            for i_df in range(len(input_df)):
                f_protein_num = input_df.iloc[i_df].loc[ppi_int[0]]
                s_protein_num = input_df.iloc[i_df].loc[ppi_int[1]]
                my_state = f"{f_protein_num}x{s_protein_num}"
                if interaction_name in result_dict.keys():
                    if my_state in result_dict[interaction_name].keys():
                        prev_count = result_dict[interaction_name][my_state]
                        result_dict[interaction_name][my_state] = prev_count + 1
                    else:
                        result_dict[interaction_name][my_state] = 1
                else:
                    result_dict[interaction_name] = {my_state: 1}

    return result_dict  # result_dict = {interaction_name:{state:count}}


def get_q_table(c_states, t_states, c_size, t_size):
    def q_func(n_tumor_state, n_control_state, n_tumor_samples, n_control_samples):
        q = (n_tumor_state / n_tumor_samples) / (
                (n_control_state / n_control_samples) + (n_tumor_state / n_tumor_samples))
        return q

    result_dict = {'PPI': [], 'state': [], 'q-value': []}
    for ppi in c_states.keys():
        for state in c_states[ppi].keys():
            n_c_state = c_states[ppi][state]  # n_control_state
            if state in t_states[ppi].keys():
                n_t_state = t_states[ppi][state]  # n_tumor_state
            else:
                n_t_state = 0
            q_value = q_func(n_tumor_state=n_t_state, n_control_state=n_c_state, n_tumor_samples=t_size,
                             n_control_samples=c_size)
            print(q_value)
            if q_value < 0.10 or q_value > 0.90:
                if n_t_state / t_size >= 0.20 and n_c_state / c_size >= 0.20:
                    result_dict['PPI'].append(ppi)
                    result_dict['state'].append(state)
                    result_dict['q-value'].append(q_value)

    for ppi in t_states.keys():
        for state in t_states[ppi].keys():
            n_t_state = t_states[ppi][state]  # n_tumor_state
            if state in c_states[ppi].keys():
                n_c_state = c_states[ppi][state]  # n_control_state
            else:
                n_c_state = 0
            q_value = q_func(n_tumor_state=n_t_state, n_control_state=n_c_state, n_tumor_samples=t_size,
                             n_control_samples=c_size)
            print(q_value)
            if q_value < 0.10 or q_value > 0.90:
                if n_t_state / t_size >= 0.20 and n_c_state / c_size >= 0.20:
                    result_dict['PPI'].append(ppi)
                    result_dict['state'].append(state)
                    result_dict['q-value'].append(q_value)

    return result_dict


input_path_transcriptome = PATH_PPI
df = pd.read_csv(input_path_transcriptome, header=0, index_col=None)
df = df.rename(columns={str(df.columns[-1]): 'label'})
df.iloc[:, -1] = df.iloc[:, -1].apply(str).values  # convert last column (labels) into str
labels = np.unique(df['label'].values).tolist()
if len(labels) > 2:
    print('Number of labels exceed 2')
    quit()

# Normalize Data

c_labeled = df.loc[df.label == labels[0]]
c_labels = c_labeled.label
c_labeled = c_labeled.drop(['label'], axis=1)

t_labeled = df.loc[df.label == labels[1]]
t_labels = t_labeled.label
t_labeled = t_labeled.drop(['label'], axis=1)

std_range.s_normalization(input_array=c_labeled.values, input_columns=c_labeled.columns, sigma_coefficient=1)
std_range.s_normalization(input_array=t_labeled.values, input_columns=t_labeled.columns, sigma_coefficient=1)

# PPI data
input_path_ppi = PATH_TRANSCRIPTOME
df_ppi = pd.read_csv(input_path_ppi, header=None, index_col=None)
ppi_array = df_ppi.values

c_ppi_states = count_ppi_states(c_labeled, ppi_array)
t_ppi_states = count_ppi_states(t_labeled, ppi_array)

q_table = get_q_table(c_states=c_ppi_states, t_states=t_ppi_states, c_size=len(c_labeled), t_size=len(t_labeled))
q_table_df = pd.DataFrame.from_dict(q_table)

q_table_df.to_csv(path_or_buf=PATH_Q_TABLE_SAVE)
