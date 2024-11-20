import pandas as pd
import numpy as np
import util as util
import yaml
import copy
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

config_data=util.config_load()
df_simil_in=util.pickle_load(config_data['raw_dataset_path_data_simil'])

##########################Create Jaccard Index #############################

def jaccard_similarity_index(df):
    df = df[['Provider','BeneID',
        'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
        'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
        'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
        'ClmDiagnosisCode_10']]

    # Step 1: Create a mapping from ICD codes to sets of providers
    icd_to_providers = defaultdict(set)

    # Combine diagnosis code columns and fill the mapping
    diagnosis_columns = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]

    for idx, row in df.iterrows():
        Provider=row['Provider']
        for col in diagnosis_columns:
            icd_code = row[col]
            if pd.notna(icd_code):  # Check if the ICD code is not NaN
                icd_to_providers[icd_code].add(Provider)

    # Step 2: Get unique ICD codes
    icd_codes = list(icd_to_providers.keys())
    m = len(icd_codes)

    # Step 3: Initialize the Jaccard similarity matrix
    jaccard_matrix = np.zeros((m, m))

    # Function to calculate Jaccard similarity
    def jaccard_similarity(set_a, set_b):
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union != 0 else 0

    # Step 4: Populate the Jaccard similarity matrix
    for i in range(m):
        for j in range(m):
            if i == j:
                jaccard_matrix[i, j] = 1  # Jaccard similarity with itself is 1
            elif j > i:  # Calculate only for upper triangle to avoid redundancy
                similarity = jaccard_similarity(icd_to_providers[icd_codes[i]], icd_to_providers[icd_codes[j]])
                jaccard_matrix[i, j] = similarity
                jaccard_matrix[j, i] = similarity  # Symmetric matrix

    # Step 5: Create a DataFrame for the Jaccard similarity matrix
    jaccard_df = pd.DataFrame(jaccard_matrix, index=icd_codes, columns=icd_codes)
    return jaccard_df

#########################Create Count matrix for ICD code tables############
def ICD_count_matrix(df):
    df_group_data_diag_simil=pd.DataFrame(df['Provider'].unique(),columns=['Provider'])
    for i in [
        'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
        'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
        'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
        'ClmDiagnosisCode_10']:
        pivot_data = df[['Provider',i]].pivot_table(
        index='Provider',                      # Set 'Provider' as the index
        columns=[i],      # Set diagnosis codes as columns
        values=[i],       # Values to aggregate
        aggfunc={i:'count'},                       # Count occurrences
        fill_value=0                           # Fill missing values with 0
    ).reset_index() 
        pivot_data.columns = ['_'.join(map(str, col)) for col in pivot_data.columns.values]
        pivot_data.rename(columns={'Provider_':'Provider'},inplace=True)
        df_group_data_diag_simil=df_group_data_diag_simil.merge(pivot_data,on='Provider',how='left')
    df_group_data_diag_simil.columns = df_group_data_diag_simil.columns.str.split('_').str[-1]
    df_group_data_diag_simil=df_group_data_diag_simil.T.groupby(df_group_data_diag_simil.columns).sum()
    df_group_data_diag_simil=df_group_data_diag_simil.T
    data_reordered=df_group_data_diag_simil.drop(columns='Provider').copy()
    return data_reordered,df_group_data_diag_simil


#4. Create df with P(provider)XM(every ICD code in each columns) count values of each ICD code used by each provider 
data_reordered,df_group_data_diag_simil=ICD_count_matrix(df_simil_in)
data_reordered=data_reordered.astype(int)

#Calculate Jaccard index

jaccard_df=jaccard_similarity_index(df_simil_in)
jaccard_df = jaccard_df[data_reordered.columns]
jaccard_df=round(jaccard_df,4)
jaccard_df_rearranged = jaccard_df.reindex(data_reordered.columns)



#5. Matrix multiplication of these metrices to create PXN matrix
data= data_reordered.dot(jaccard_df_rearranged.values)

data.columns=jaccard_df_rearranged.columns
data['Provider']=df_group_data_diag_simil['Provider']
data.fillna(0,inplace=True)
util.pickle_dump(data, config_data["raw_dataset_path_data_matrix"])
util.pickle_dump(df_group_data_diag_simil['Provider'],config_data['raw_dataset_provider_data'])
util.pickle_dump(data_reordered,config_data['raw_dataset_path_usage'])
util.pickle_dump(jaccard_df_rearranged,config_data['raw_dataset_path_simil_sub'])