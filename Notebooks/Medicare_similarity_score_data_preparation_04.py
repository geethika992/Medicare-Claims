import pandas as pd
import numpy as np
import src.util as util
import yaml
import copy
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

config_data=util.config_load()
df_simil_in=util.pickle_load(config_data['raw_dataset_path_data_simil'])
df_ICD_embeddings=util.pickle_load(config_data['raw_dataset_path_ICD_embeddings_combined'])
df_DRG_embeddings=util.pickle_load(config_data['raw_dataset_path_DRG_embeddings'])
#df_PRC_embeddings=util.pickle_load(config_data['raw_dataset_path_PROC_embeddings'])

##########################ICD embedding lookup########################
def convert_ICD_to_embeddings(df,Lookup_df,Coding_ind):
    dict_icd=Lookup_df.set_index('CODE')['embeddings'].to_dict() 
    def ICD_check_embeddings(code):
        if isinstance(code,str) and code.startswith('V'):
            key=code[:3]
            return dict_icd.get(key,[0.0]*768)
        else:
            return dict_icd.get(code,[0.0]*768)
    if Coding_ind.startswith("ICD"):    
        for i in ['Value']:
            df[i+'_embeddings'] = df[i].map(ICD_check_embeddings)
    elif Coding_ind.startswith("DRG"):
        for i in ['DiagnosisGroupCode']:
            df[i+'_embeddings'] = df[i].map(ICD_check_embeddings)
    else:
        if  Coding_ind.startswith("PRC"):
            for i in ['ClmProcedureCode_1', 'ClmProcedureCode_2',
            'ClmProcedureCode_3']:
                df[i+'_embeddings'] = df[i].map(ICD_check_embeddings)

########################## Calculate Similarity Matrix#####################


def compute_cosine_similarity(df):  # Rename the function to avoid conflict
    embeddings = np.vstack(df['Value_embeddings'].to_numpy())
    similarity_matrix = cosine_similarity(embeddings)  
    similarity_matrix_full = pd.DataFrame(similarity_matrix, index=df['Value'], columns=df['Value'])
    return similarity_matrix_full

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

    for _, row in df.iterrows():
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
    data_reordered=df_group_data_diag_simil[df_similarity.index]
    return data_reordered,df_group_data_diag_simil

#1. Create a list of all ICD codes from all columns
columns=[ 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
       'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
       'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
       'ClmDiagnosisCode_10']
non_zero_rows = []
for col in columns:
    non_zero_values = df_simil_in[col]
    for value in non_zero_values:
        if value != 0 and pd.notna(value):
            non_zero_rows.append({'Column': col, 'Value': value})
non_zero_df = pd.DataFrame(non_zero_rows)
unique_non_zero_df = non_zero_df.drop_duplicates()

#2. Convert all codes to embeddings by looking up the embeddings created previously.
convert_ICD_to_embeddings(unique_non_zero_df,df_ICD_embeddings,'ICD')
unique_non_zero_df['column_value']=unique_non_zero_df['Column']+'_'+unique_non_zero_df['Value']
unique_non_zero_df.reset_index(drop=True,inplace=True)
unique_non_zero_df.drop_duplicates(subset=['Value'],inplace=True)

#3 Calculate the cosine similarity in chunks and create M(every ICD code in diag column) xN(unique ICD code) matrix for each icd code
df_similarity = compute_cosine_similarity(unique_non_zero_df)
df_similarity.index=unique_non_zero_df['Value']
df_similarity.fillna(0,inplace=True)
jaccard_df=jaccard_similarity_index(df_simil_in)
jaccard_df = jaccard_df[df_similarity.index]
jaccard_df_rearranged = jaccard_df.reindex(df_similarity.index)


#5. Create the ICD substitutability matrix by multiplying effects of jaccard and cosine similarity matrices.
#df_avg_simil=jaccard_df_rearranged*df_similarity



#4. Create df with P(provider)XM(every ICD code in each columns) count values of each ICD code used by each provider 
data_reordered,df_group_data_diag_simil=ICD_count_matrix(df_simil_in)
data_reordered=data_reordered.astype(int)
#5. Matrix multiplication of these metrices to create PXN matrix
data= data_reordered.dot(jaccard_df_rearranged.values)

data.columns=jaccard_df_rearranged.columns
data['Provider']=df_group_data_diag_simil['Provider']
data.fillna(0,inplace=True)
util.pickle_dump(data, config_data["raw_dataset_path_data_matrix"])
util.pickle_dump(df_group_data_diag_simil['Provider'],config_data['raw_dataset_provider_data'])
util.pickle_dump(data_reordered,config_data['raw_dataset_path_usage'])
util.pickle_dump(jaccard_df_rearranged,config_data['raw_dataset_path_simil_sub'])