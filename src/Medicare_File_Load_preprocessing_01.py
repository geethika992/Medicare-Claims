import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
import  util as util
import yaml
import copy
from tqdm import tqdm
import os
from datetime import datetime
config_data=util.config_load()

#############Read Data######################################
def read_raw_data(config: dict,filetype) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw Dataset Dir
    raw_dataset_dir = config["raw_dataset_dir"]

     # List files in the directory and filter those with filetype in the filename
    files = [f for f in os.listdir(raw_dataset_dir) if filetype in f and f.endswith('.csv')]

    # Process and concatenate each filtered file
    for i in tqdm(files):
        file_path = os.path.join(raw_dataset_dir, i)
        raw_dataset = pd.concat([raw_dataset, pd.read_csv(file_path)], ignore_index=True)


    # Return the concatenated DataFrame
    return raw_dataset

#############################Type conversion#####################
def type_conv(set_data, config_data,datetime_columns,obj_columns,int_columns):
  
    
    # --- Convert columns to datetime format --- #
    for col in config_data[datetime_columns]:
        if set_data[col].dtype != 'datetime64[ns]':
            set_data[col] = pd.to_datetime(set_data[col])

    # --- Convert specified columns to object type --- #
    for col in config_data[obj_columns]:
        if set_data[col].dtype != 'object':
                set_data[col] = set_data[col].astype(str)

    # --- Convert specified float columns to integer format --- #
    for col in config_data[int_columns]:
        set_data[col]=set_data[col].fillna(0)
        if set_data[col].dtype != 'int64':
            set_data[col] = set_data[col].astype(int)

    # --- Return the modified DataFrame --- #
    return set_data


######################Feature Addition-Beneficiary################################
def feature_addition_ben(dataset_conv):
    #Age from max value from set
    max_bene_DOD = max(dataset_conv['DOD'].dropna().unique()[1:])
    dataset_conv['DOD_imputed']=dataset_conv['DOD']
    dataset_conv['DOD_imputed']=dataset_conv['DOD_imputed'].apply(lambda i: i if pd.notna(i)  else max_bene_DOD )
    dataset_conv['AGE'] = np.round(((dataset_conv['DOD_imputed'] - dataset_conv['DOB']).dt.days)/365.0,1)
    dataset_conv['AGE']= dataset_conv['AGE'].astype('int64')
    dataset_conv['DOD_Flag']=dataset_conv['DOD'].apply(lambda i: 1 if pd.notna(i)  else 0 )
    dataset_conv['DOD_Flag']= dataset_conv['DOD_Flag'].astype('object')
    dataset_conv['TotalIPAnnualAmt']=dataset_conv['IPAnnualReimbursementAmt']+dataset_conv['IPAnnualReimbursementAmt']
    dataset_conv['TotalOPAnnualAmt']= dataset_conv['OPAnnualReimbursementAmt']+dataset_conv['OPAnnualDeductibleAmt']
    dataset_conv.drop(columns=['DOB','DOD','DOD_imputed'],axis=1,inplace=True)
    return dataset_conv
######################Feature addition Inpatient##################################
def Feature_addition_inp_outp(df,type_of_data):
    df['Claim_period']=np.round(((df['ClaimEndDt'] - df['ClaimStartDt']).dt.days),1)
    df['Beneficiary_cost']=(df['InscClaimAmtReimbursed'] - df['DeductibleAmtPaid'])
    diagnosis_code_columns = ['ClmDiagnosisCode_1','ClmDiagnosisCode_2','ClmDiagnosisCode_3','ClmDiagnosisCode_4',
                                'ClmDiagnosisCode_5','ClmDiagnosisCode_6','ClmDiagnosisCode_7','ClmDiagnosisCode_8','ClmDiagnosisCode_9','ClmDiagnosisCode_10']
    diagnosis_proc_columns = ['ClmProcedureCode_1','ClmProcedureCode_2','ClmProcedureCode_3']
    df['Count_diag_code']=df[diagnosis_code_columns].notna().sum(axis=1)
    for i in diagnosis_proc_columns:
            df[i]=df[i].replace('nan', np.nan)
    df['Count_proc_code']=df[diagnosis_proc_columns].notna().sum(axis=1)

    if type_of_data.lower()=='inpatient':
        df['Admit_Period']=np.round(((df['DischargeDt'] - df['AdmissionDt']).dt.days),1)
        df=df.drop(columns=['ClaimEndDt','ClaimStartDt','DischargeDt','AdmissionDt','ClmProcedureCode_6','ClmProcedureCode_4',
                                'ClmProcedureCode_5'],axis=1)
        df['Is_admit']=1
    elif type_of_data.lower().strip()=='outpatient':
        df=df.drop(columns=['ClaimEndDt','ClaimStartDt','ClmProcedureCode_6','ClmProcedureCode_4','ClmProcedureCode_5'],axis=1)
        df['Is_admit']=0
    return df
 
   
   

######### Diagnosis and proccode standardization #######################################################
def pad_code(code):
    if pd.isna(code):
        return code  # Keep NaN as is
    return str(code).zfill(4)  # Pad with zeros to make it 4 characters


def diag_proccode_stnd(df):
    for i in [ 'ClmAdmitDiagnosisCode', 'DiagnosisGroupCode',
        'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
        'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
        'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
        'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
        'ClmProcedureCode_3']:
        df[i].astype(str)
        df[i]=df[i].replace(r'\.0$', '', regex=True)
        df[i]=df[i].replace(r'nan', np.nan, regex=True)
        df[i]=df[i].apply(pad_code)
        return df
################Chronic condition standardization##################################################
    
def standardize_conditions(df):
    for i in ['RenalDiseaseIndicator','ChronicCond_Alzheimer',
        'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
        'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
        'ChronicCond_Depression', 'ChronicCond_Diabetes',
        'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
        'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke','Gender']:
        if i=='RenalDiseaseIndicator':
            df[i] = df[i].apply(lambda val: 1 if val =='Y' else 0)
        else:
            df[i] = df[i].apply(lambda val: 0 if val ==2 else 1)
    return df
    
#1. Read Data
df_ben=read_raw_data(config_data,'Beneficiary')
df_inp=read_raw_data(config_data,'Inpatient')
df_outp=read_raw_data(config_data,'Outpatient')

#2. Convert datatype to standardize
df_ben_conv=type_conv(df_ben,config_data,"datetime_columns_ben","obj_columns_ben","int_columns_ben")
df_inp_conv=type_conv(df_inp,config_data,"datetime_columns_inp","obj_columns_inp","int_columns_inp")
df_outp_conv=type_conv(df_outp,config_data,"datetime_columns_outp","obj_columns_outp","int_columns_outp")

#3. Initial Feature addition
df_ben_conv=feature_addition_ben(df_ben_conv)
df_inp_conv=Feature_addition_inp_outp(df_inp_conv,'inpatient')
df_outp_conv=Feature_addition_inp_outp(df_outp_conv,'outpatient')

#4. Diagnosis code and proccode standardization for NA and padding
df_inp_conv=diag_proccode_stnd(df_inp_conv)
df_outp_conv=diag_proccode_stnd(df_outp_conv)

#5. Chronic condition standardization for beneficiary
df_ben_conv=standardize_conditions(df_ben_conv)

#6. Pickled preprocessed dataset
util.pickle_dump(df_ben_conv, config_data["raw_dataset_path_train_ben"])
util.pickle_dump(df_inp_conv, config_data["raw_dataset_path_train_inp"])
util.pickle_dump(df_outp_conv, config_data["raw_dataset_path_train_outp"])