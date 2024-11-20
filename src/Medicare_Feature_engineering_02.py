import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
import util as util
import yaml
import copy
from tqdm import tqdm
import os
from datetime import datetime
config_data=util.config_load()
df_inp=util.pickle_load(config_data['raw_dataset_path_train_inp'])
df_outp=util.pickle_load(config_data['raw_dataset_path_train_outp'])
df_ben=util.pickle_load(config_data['raw_dataset_path_train_ben'])

###### Merge Data ###############################
def merge(df_inp,df_outp,df_ben):
    common_cols=[i for i in df_inp.columns if i in df_outp.columns]
    df_inp_outp=pd.merge(left=df_inp, right=df_outp, left_on=common_cols,right_on=common_cols,how='outer')
    df_inp_outp_ben=pd.merge(left=df_inp_outp,right=df_ben,left_on='BeneID',right_on='BeneID',how='left')
    return df_inp_outp_ben


###################Provider amount###############
def df_group_amt_prv(df_inp_outp_ben):
    # Initialize the DataFrame with unique 'Provider' values
    df_group_data_prv_amt = pd.DataFrame(df_inp_outp_ben['Provider'].unique(), columns=['Provider'])

    # Group by 'Provider' and calculate the sum of the columns
    df_sum = df_inp_outp_ben.groupby('Provider')[['DeductibleAmtPaid', 'InscClaimAmtReimbursed', 'Admit_Period', 'Claim_period']].sum()

    # Merge the summed values with the 'df_group_data_prv_amt' DataFrame
    df_group_data_prv_amt = df_group_data_prv_amt.merge(df_sum, on='Provider', how='left')

    # Rename columns to match the desired naming convention
    df_group_data_prv_amt.rename(columns={
        'DeductibleAmtPaid': 'PRV_TotalDeductibleAmtPaid',
        'InscClaimAmtReimbursed': 'PRV_TotalInscClaimAmtReimbursed',
        'Admit_Period': 'PRV_TotalAdmit_Period',
        'Claim_period': 'PRV_TotalClaimPeriod'
    }, inplace=True)

    return df_group_data_prv_amt
        

##################Feature engineering level 1 ########################################
def Feature_engineering_level1(df_inp_outp_ben):
     df=pd.DataFrame()
     df_group_data_prv=pd.DataFrame(df_inp_outp_ben['Provider'].unique(),columns=['Provider'])
     for i in ['Provider']:
          for j in ['BeneID',
     'AttendingPhysician',
     'OtherPhysician',
     'OperatingPhysician',
     'ClmAdmitDiagnosisCode',
     'ClmProcedureCode_1',
     'ClmProcedureCode_2',
     'ClmProcedureCode_3',
     'ClmDiagnosisCode_1',
     'ClmDiagnosisCode_2',
     'ClmDiagnosisCode_3',
     'ClmDiagnosisCode_4',
     'ClmDiagnosisCode_5',
     'ClmDiagnosisCode_6',
     'ClmDiagnosisCode_7',
     'ClmDiagnosisCode_8',
     'ClmDiagnosisCode_9',
     'ClmDiagnosisCode_10',
     'DiagnosisGroupCode'
     ]:
               prefix=i+"_"+j
               df=df_inp_outp_ben.groupby([i,j])['ClaimID'].count().reset_index()
               df=df.rename(columns={'ClaimID':'Clm_cnt_'+prefix})
               df=df.groupby(['Provider'])['Clm_cnt_'+prefix].sum().reset_index()
               df_group_data_prv=df_group_data_prv.merge(df,on='Provider',how='left')
     return df_group_data_prv

##################Feature engineering level 3 - 1###########################################
def Feature_engineering_level3_phy(df_inp_outp_ben):
     df=pd.DataFrame()
     df_group_data_prv_ben_phy=pd.DataFrame(df_inp_outp_ben['Provider'].unique(),columns=['Provider'])
     for i in ['Provider']:
          for j in ['BeneID']:
               for k in [
          'AttendingPhysician',
          'OtherPhysician',
          'OperatingPhysician']:
                    for l in ['ClmAdmitDiagnosisCode',
          'ClmProcedureCode_1',
          'ClmProcedureCode_2',
          'ClmProcedureCode_3',
          'ClmDiagnosisCode_1',
          'ClmDiagnosisCode_2',
          'ClmDiagnosisCode_3',
          'ClmDiagnosisCode_4',
          'ClmDiagnosisCode_5',
          'ClmDiagnosisCode_6',
          'ClmDiagnosisCode_7',
          'ClmDiagnosisCode_8',
          'ClmDiagnosisCode_9',
          'ClmDiagnosisCode_10',
          'DiagnosisGroupCode']:
                         prefix=i+"_"+j+"_"+k+"_"+l
                         df=df_inp_outp_ben.groupby([i,j,k,l])['ClaimID'].count().reset_index()
                         df=df.rename(columns={'ClaimID':'Clm_cnt_'+prefix})
                         df=df.groupby(['Provider'])['Clm_cnt_'+prefix].sum().reset_index()
                         df_group_data_prv_ben_phy=df_group_data_prv_ben_phy.merge(df,on='Provider',how='left')
     return df_group_data_prv_ben_phy
##################Feature engineering level 3 - 2###########################################
def Feature_engineering_level3_diag(df_inp_outp_ben):
     df=pd.DataFrame()
     df_group_data_prv_ben_diag=pd.DataFrame(df_inp_outp_ben['Provider'].unique(),columns=['Provider'])
     for i in ['Provider']:
          for j in ['BeneID']:
               for k in [
          'ClmDiagnosisCode_1',
          'ClmDiagnosisCode_2',
          'ClmDiagnosisCode_3',
          'ClmDiagnosisCode_4',
          'ClmDiagnosisCode_5',
          'ClmDiagnosisCode_6',
          'ClmDiagnosisCode_7',
          'ClmDiagnosisCode_8',
          'ClmDiagnosisCode_9',
          'ClmDiagnosisCode_10',]:
                    for l in [
          'ClmProcedureCode_1',
          'ClmProcedureCode_2',
          'ClmProcedureCode_3',
          ]:
                         prefix=i+"_"+j+"_"+k+"_"+l
                         df=df_inp_outp_ben.groupby([i,j,k,l])['ClaimID'].count().reset_index()
                         df=df.rename(columns={'ClaimID':'Clm_cnt_'+prefix})
                         df=df.groupby(['Provider'])['Clm_cnt_'+prefix].sum().reset_index()
                         df_group_data_prv_ben_diag=df_group_data_prv_ben_diag.merge(df,on='Provider',how='left')
     return df_group_data_prv_ben_diag

##################Unique count of codes#################
def Feature_engineering_unique_data(df_inp_outp_ben):
     df=pd.DataFrame()
     df_group_data_prv_unq=pd.DataFrame(df_inp_outp_ben['Provider'].unique(),columns=['Provider'])
     for i in ['DiagnosisGroupCode','ClmAdmitDiagnosisCode']:
            
                         df=df_inp_outp_ben.groupby(['Provider'])[i].nunique().reset_index()
                         df=df.rename(columns={i:'Prv_unq_cnt_'+i})
                         df=df.groupby(['Provider'])['Prv_unq_cnt_'+i].sum().reset_index()
                         df_group_data_prv_unq=df_group_data_prv_unq.merge(df,on='Provider',how='left')
     return df_group_data_prv_unq



df_inp_outp_ben=merge(df_inp,df_outp,df_ben)
df_provider_stats = df_inp_outp_ben.groupby('Provider').agg(
    claims_count=('ClaimID', 'count'),
    beneficiaries_count=('BeneID', 'nunique'),
    total_claim_amount=('InscClaimAmtReimbursed', 'sum')
).reset_index()
df_group_data_prv_amt=df_group_amt_prv(df_inp_outp_ben)
df_grp_data_prv=Feature_engineering_level1(df_inp_outp_ben)
df_grp_data_prv_ben_phy=Feature_engineering_level3_phy(df_inp_outp_ben)
df_grp_data_prv_ben_diag=Feature_engineering_level3_diag(df_inp_outp_ben)
df_grp_data_prv_unique=Feature_engineering_unique_data(df_inp_outp_ben)
df_inp_outp_ben=df_inp_outp_ben.drop(columns=['ClaimID','InscClaimAmtReimbursed',
       'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician', 'DeductibleAmtPaid','Claim_period', 'Beneficiary_cost',
       'Count_diag_code', 'Count_proc_code', 'Admit_Period','Race','State', 'County','IPAnnualReimbursementAmt',
       'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
       'OPAnnualDeductibleAmt', 'AGE','TotalIPAnnualAmt',
       'TotalOPAnnualAmt'])
df_column_grouping=df_inp_outp_ben[['Provider','Is_admit', 'Gender', 'RenalDiseaseIndicator', 'ChronicCond_Alzheimer',
       'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
       'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
       'ChronicCond_Depression', 'ChronicCond_Diabetes',
       'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
       'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke', 'DOD_Flag']]
df_inp_outp_ben_grp=df_column_grouping.groupby('Provider').sum()
data_combined=df_grp_data_prv.merge(df_grp_data_prv_ben_phy,on='Provider').merge(df_grp_data_prv_ben_diag,on='Provider').merge(df_group_data_prv_amt,on='Provider').merge(df_grp_data_prv_unique,on='Provider').merge(df_inp_outp_ben_grp,on='Provider')
data_combined.fillna(0,inplace=True)
df_inp_outp_ben_simil_data=df_inp_outp_ben[['Provider','BeneID','Is_admit','ClmAdmitDiagnosisCode','DiagnosisGroupCode', 'ClmDiagnosisCode_1',
       'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
       'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
       'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
       'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3']]


util.pickle_dump(df_inp_outp_ben_simil_data, config_data["raw_dataset_path_data_simil"])
util.pickle_dump(data_combined, config_data["raw_dataset_path_data_combined"])
util.pickle_dump(df_provider_stats, config_data["raw_dataset_path_provider_stats"])