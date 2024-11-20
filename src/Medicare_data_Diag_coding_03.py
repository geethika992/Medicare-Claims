import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
import util as util

# Load configuration
config_data = util.config_load()

############################### Read Medical Code Files ######################

def read_med_code_data(config: dict,filetype) -> pd.DataFrame:
    # Create variable to store raw dataset
    ICD_code_data = pd.DataFrame()

    # Raw Dataset Dir
    raw_dataset_dir = config["raw_dataset_dir_med_codes"]

     # List files in the directory and filter those with filetype in the filename
    files = [f for f in os.listdir(raw_dataset_dir) if filetype in f and f.endswith('.xlsx')]

    # Process and concatenate each filtered file
    for i in tqdm(files):
        file_path = os.path.join(raw_dataset_dir, i)
        ICD_code_data = pd.concat([ICD_code_data, pd.read_excel(file_path)], ignore_index=True,axis=0)


    # Return the concatenated DataFrame
    return ICD_code_data


########################## Read Medical Code Data #############################

# Load various medical code datasets
ICD_10_data = read_med_code_data(config_data, 'ICD10')
ICD_9_old = read_med_code_data(config_data, 'ICD9_old')
ICD_9_new = read_med_code_data(config_data, 'ICD9_new')


# Process ICD-10 data
ICD_10_data.drop(columns=['SHORT DESCRIPTION (VALID ICD-10 FY2024)', 'NF EXCL'], inplace=True,axis=1)
ICD_10_data.rename(columns={'LONG DESCRIPTION (VALID ICD-10 FY2024)': 'Description'}, inplace=True)
ICD_10_data['File_name'] = 'ICD_10_data'
ICD_10_data['Priority'] = 1

# Process ICD-9 old data
ICD_9_old.drop(columns=['SHORT DESCRIPTION'], inplace=True,axis=1)
ICD_9_old.rename(columns={'DIAGNOSIS CODE': 'CODE', 'LONG DESCRIPTION': 'Description'}, inplace=True)
ICD_9_old['File_name'] = 'ICD_9_old_data'
ICD_9_old['Priority'] = 3

# Process ICD-9 new data

ICD_9_new.drop(columns=["NF EXCL"], axis=1, inplace=True)
ICD_9_new.rename(columns={'LONG DESCRIPTION (VALID ICD-9 FY2024)': 'Description'}, inplace=True)
ICD_9_new['File_name'] = 'ICD_9_new_data'
ICD_9_new['Priority'] = 2

# Merge datasets
merged_df = (
    ICD_9_old
    .merge(ICD_9_new, on=['CODE', 'Description'], how='outer', suffixes=('_old', '_new'))
    .merge(ICD_10_data, on=['CODE', 'Description'], how='outer', suffixes=('', '_10'))
)

# Combine file names and priorities
merged_df['file_name_combined'] = merged_df[['File_name', 'File_name_new', 'File_name_old']].bfill(axis=1).iloc[:, 0]
merged_df['Priority_combined'] = merged_df[['Priority', 'Priority_new', 'Priority_old']].bfill(axis=1).iloc[:, 0]

# Select relevant columns and deduplicate
result = merged_df[['CODE', 'Description', 'Priority_combined', 'file_name_combined']]
result.sort_values(by='Priority_combined', inplace=True)
data_ICD_deduplicated = result.drop_duplicates(subset='CODE', keep='first').dropna()

# Generate embeddings for ICD codes

util.pickle_dump(data_ICD_deduplicated, config_data["raw_dataset_path_ICD_codes"])