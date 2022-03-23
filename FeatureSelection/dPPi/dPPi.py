import pandas as pd
import numpy as np
from Normalization import std_range
import requests
import re
import zipfile
import io


PATH_TRANSCRIPTOME = r"/home/kcan/Desktop/Bioinformatics_stuff/dppi_karsılastırmak_için/coadexp_prepped.csv"
# PATH_PPI = r"/home/kcan/Desktop/Bioinformatics_stuff/dppi_karsılastırmak_için/ppi_biogrid.csv"
PATH_Q_TABLE_SAVE = r"/home/kcan/PycharmProjects/Bioinformatics/q_table_new.csv"
# #
# PATH_TRANSCRIPTOME = r"~/PycharmProjects/Bioinformatics/test_datasets/dppi_test.csv"
# PATH_PPI = r"~/PycharmProjects/Bioinformatics/test_datasets/ppi_test.csv"
# PATH_Q_TABLE_SAVE = r"~/PycharmProjects/Bioinformatics/q_table_s2.csv"

transpose_q = input("Do you want to transpose transcriptome data? (y/n)\n")
transpose_transcriptome = False

ACCESS_KEY = "06baf59435fdab65f7382fc0ef582b21"

update_ppi = input("Do you want to update ppi libraries? (y/n)\n")

if update_ppi == 'y':
    print("Updating ppi libraries...")
    r = requests.get("https://downloads.thebiogrid.org/BioGRID/Release-Archive/")
    my_content = r.content.decode("utf-8")
    z = re.findall("BIOGRID-*\d.\d.+", my_content)  # Finding all for now, may add option to use specific release
    latest_release = z[0].split('/')[0].split('-')[1]

    download_link = f"https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.207/BIOGRID-ORGANISM-{latest_release}.tab3.zip"

    r = requests.get(download_link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("./PPI_Library/")
    print("updated")

# set_sep = input("Do you want to specify separator for csv files?")


if transpose_q == 'y':
    print('Transposing input...')
    transpose_transcriptome = True
elif transpose_q == 'n':
    print('Will not transpose...')
    transpose_transcriptome = False
else:
    print('please use only y or n')
    quit()


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

    result_dict = {'PPI': [], 'state': [], 'q-value': [], 'Nt': [], 'Ni': [], 'Nc': [], 'N0': []}
    for ppi in c_states.keys():
        for state in c_states[ppi].keys():
            n_c_state = c_states[ppi][state]  # n_control_state
            if state in t_states[ppi].keys():
                n_t_state = t_states[ppi][state]  # n_tumor_state
            else:
                n_t_state = 0
            q_value = q_func(n_tumor_state=n_t_state, n_control_state=n_c_state, n_tumor_samples=t_size,
                             n_control_samples=c_size)
            # print(q_value)
            if q_value < 0.10 or q_value > 0.90:
                # if n_t_state / t_size >= 0.20 and n_c_state / c_size >= 0.20:
                result_dict['PPI'].append(ppi)
                result_dict['state'].append(state)
                result_dict['q-value'].append(q_value)
                result_dict['Nt'].append(n_t_state)
                result_dict['Ni'].append(t_size)
                result_dict['Nc'].append(n_c_state)
                result_dict['N0'].append(c_size)

    for ppi in t_states.keys():
        for state in t_states[ppi].keys():
            n_t_state = t_states[ppi][state]  # n_tumor_state
            if state in c_states[ppi].keys():
                n_c_state = c_states[ppi][state]  # n_control_state
            else:
                n_c_state = 0
            q_value = q_func(n_tumor_state=n_t_state, n_control_state=n_c_state, n_tumor_samples=t_size,
                             n_control_samples=c_size)
            # print(q_value)
            if q_value < 0.10 or q_value > 0.90:
                # if n_t_state / t_size >= 0.20 and n_c_state / c_size >= 0.20:
                result_dict['PPI'].append(ppi)
                result_dict['state'].append(state)
                result_dict['q-value'].append(q_value)
                result_dict['Nt'].append(n_t_state)
                result_dict['Ni'].append(t_size)
                result_dict['Nc'].append(n_c_state)
                result_dict['N0'].append(c_size)

    return result_dict


input_path_transcriptome = PATH_TRANSCRIPTOME
if transpose_transcriptome:
    df = pd.read_csv(input_path_transcriptome, header=None, index_col=0, low_memory=False)
    df = df.T
else:
    df = pd.read_csv(input_path_transcriptome, header=0, index_col=None, low_memory=False)

df = df.rename(columns={str(df.columns[-1]): 'label'})

# print(df.head())
df.iloc[:, -1] = df.iloc[:, -1].apply(str).values  # convert last column (labels) into str
labels = np.unique(df['label'].values).tolist()
if len(labels) > 2:
    print('Number of labels exceed 2')
    quit()

# Normalize Data
df.fillna(0, inplace=True)  # replace NA with 0
c_labeled = df.loc[df.label == labels[0]]
c_labels = c_labeled.label
c_labeled = c_labeled.drop(['label'], axis=1)

t_labeled = df.loc[df.label == labels[1]]
t_labels = t_labeled.label
t_labeled = t_labeled.drop(['label'], axis=1)

c_labeled = pd.DataFrame(
    std_range.s_normalization(input_array=c_labeled.values, input_columns=c_labeled.columns, sigma_coefficient=1))
t_labeled = pd.DataFrame(
    std_range.s_normalization(input_array=t_labeled.values, input_columns=t_labeled.columns, sigma_coefficient=1))

# PPI data

input_path_ppi = r"./PPI_Library/BIOGRID-ORGANISM-Homo_sapiens-4.4.207.tab3.txt"
df_ppi = pd.read_csv(input_path_ppi, index_col=None, sep=None, engine='python')
ppi_array = df_ppi[df_ppi.columns[7:9]]
ppi_array.columns = range(ppi_array.columns.size)
ppi_array = ppi_array.values


c_ppi_states = count_ppi_states(c_labeled, ppi_array)
t_ppi_states = count_ppi_states(t_labeled, ppi_array)

q_table = get_q_table(c_states=c_ppi_states, t_states=t_ppi_states, c_size=len(c_labeled), t_size=len(t_labeled))
q_table_df = pd.DataFrame.from_dict(q_table)

q_table_df.to_csv(path_or_buf=PATH_Q_TABLE_SAVE)
