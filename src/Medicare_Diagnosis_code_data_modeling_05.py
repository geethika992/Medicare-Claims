import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
import util as util
import yaml
import copy
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import rrcf
from pyod.models.feature_bagging import FeatureBagging

config_data=util.config_load()
data_matrix=util.pickle_load(config_data['raw_dataset_path_data_matrix'])



#########################Isolation Forest##########################
def Isolation_forest(df,df_scaled,contamination,estimators,m_features):

# Fit Isolation Forest
    model = IsolationForest(contamination=contamination,n_estimators=estimators,max_features=m_features)  # Adjust contamination rate as needed
    model.fit(df_scaled)

    anomaly_scores = model.decision_function(df_scaled)
    # Predict anomalies
    anomalies = model.predict(df_scaled)

    # -1 for anomaly, 1 for normal
    df_scaled['Anomaly_ind']=anomalies
    df['Anomaly_ind']=anomalies
    df['Anomaly_score']=anomaly_scores
    df_scaled['Anomaly_score']=anomaly_scores
    return df,df_scaled,model

############################Local Outlier Factor #######################

def Local_outlier_factor(df,df_scaled):
# Fit LOF
    lof = FeatureBagging(base_estimator=None, n_estimators=10, contamination=0.1)
    anomalies = lof.fit_predict(df_scaled)
    lof_scores = lof.decision_scores_  
    lof_scores = lof_scores.reshape(-1, 1)  # Reshape for compatibility

    # Identify anomalies (Anomalies labeled as -1)
    df_scaled['Anomaly_lof']=anomalies
    df['Anomaly_lof']=anomalies
    df_scaled['Anomaly_score_lof']=lof_scores
    df['Anomaly_score_lof']=lof_scores
    return df,df_scaled

##########################RRCF #####################################

def RRCF(df,df_scaled):
    n=len(df)
    # Set forest parameters
    num_trees = 100
    tree_size = 256
    sample_size_range = (n // tree_size, tree_size)

    # Construct forest
    forest = []
    while len(forest) < num_trees:
        # Select random subsets of points uniformly
        ixs = np.random.choice(n, size=sample_size_range,
                            replace=False)
        # Add sampled trees to forest
        trees = [rrcf.RCTree(scaled_data_level1.iloc[ix].values, index_labels=ix)
                for ix in ixs]
        forest.extend(trees)

    # Compute average CoDisp
    avg_codisp = pd.Series(0.0, index=np.arange(n))
    index = np.zeros(n)
    for tree in forest:
        codisp = pd.Series({leaf : tree.codisp(leaf)
                        for leaf in tree.leaves})
        avg_codisp[codisp.index] += codisp
        np.add.at(index, codisp.index.values, 1)
    avg_codisp /= index
    df_scaled['Anomaly_score_rrcf'] =avg_codisp
    df['Anomaly_score_rrcf']=avg_codisp
    avg_codisp_quantile=avg_codisp.quantile(0.95)
    return df,df_scaled,avg_codisp_quantile

########Dataframe Creation###########################
def dataframe_creation(df,df_scaled,anomaly_ind_column,score_column,ranking_order_ascending):
    df_anom=df[df[anomaly_ind_column]==1]
    df_normal=df[df[anomaly_ind_column]==0]
    df_anom_scale=df_scaled[df_scaled[anomaly_ind_column]==1]
    df_normal_scale=df_scaled[df_scaled[anomaly_ind_column]==0]
    dataset_anom_rank=df_anom.sort_values(by=score_column,ascending=ranking_order_ascending)
    dataset_anom_rank['Rank']=range(1,len(dataset_anom_rank)+1)
    return df_anom,df_normal,df_anom_scale,df_normal_scale,dataset_anom_rank


######Scaling Data############
scaler=RobustScaler()
scaled_data_level1=scaler.fit_transform(data_matrix.drop(columns=['Provider']))
scaled_data_level1 = pd.DataFrame(scaled_data_level1, columns=data_matrix.drop(columns=['Provider']).columns)

#1. Isolation forest, ranking
data_if,data_scaled_if,if_model=Isolation_forest(data_matrix,scaled_data_level1,0.1,1000,100)
data_if['Anomaly_ind'] = data_if['Anomaly_score'].apply(lambda x: 1 if x <= -0.1 else 0)
data_scaled_if['Anomaly_ind'] = data_scaled_if['Anomaly_score'].apply(lambda x: 1 if x <=-0.1 else 0)
df_anom_if,df_normal_if,df_anom_scale_if,df_normal_scale_if,dataset_anom_rank_if=dataframe_creation(data_if,data_scaled_if,'Anomaly_ind','Anomaly_score',True)
data_if_rank=util.pickle_dump(dataset_anom_rank_if,config_data['raw_dataset_path_if_rank'])
util.pickle_dump(if_model,config_data['Isolation_forest_model'])

#2. Local Outlier factor
data_lof,data_scaled_lof=Local_outlier_factor(data_matrix,scaled_data_level1)
# data_scaled_lof['Anomaly_lof'] = data_scaled_lof['Anomaly_lof'].apply(lambda x: 1 if x == -1 else 0)
# data_lof['Anomaly_lof'] = data_lof['Anomaly_lof'].apply(lambda x: 1 if x ==-1 else 0)
df_anom_lof,df_normal_lof,df_anom_scale_lof,df_normal_scale_lof,dataset_anom_rank_lof=dataframe_creation(data_lof,data_scaled_lof,'Anomaly_lof','Anomaly_score_lof',False)
data_lof_rank=util.pickle_dump(dataset_anom_rank_lof,config_data['raw_dataset_path_lof_rank'])


#3. RRCF
data_rrcf,data_scaled_rrcf,avg_codisp_quantile=RRCF(data_matrix,scaled_data_level1)
data_scaled_rrcf['Anomaly_rrcf'] = data_scaled_rrcf['Anomaly_score_rrcf'].apply(lambda x: 1 if x >= avg_codisp_quantile else 0)
data_rrcf['Anomaly_rrcf'] = data_rrcf['Anomaly_score_rrcf'].apply(lambda x: 1 if x >= avg_codisp_quantile else 0)
df_anom_rrcf,df_normal_rrcf,df_anom_scale_rrcf,df_normal_scale_rrcf,dataset_anom_rank_rrcf=dataframe_creation(data_rrcf,data_scaled_rrcf,'Anomaly_rrcf','Anomaly_score_rrcf',False)
data_rrcf_rank=util.pickle_dump(dataset_anom_rank_rrcf,config_data['raw_dataset_path_rrcf_rank'])



#4. Isolation Forest- Provider amount dimension

data_combined_Fe=util.pickle_load(config_data["raw_dataset_path_data_combined"])
columns=['Provider','Clm_cnt_Provider_BeneID',
'Clm_cnt_Provider_OperatingPhysician',
'Clm_cnt_Provider_ClmDiagnosisCode_9',
'Clm_cnt_Provider_ClmDiagnosisCode_10',
'Clm_cnt_Provider_DiagnosisGroupCode',
'Clm_cnt_Provider_BeneID_OtherPhysician_ClmProcedureCode_1',
'PRV_TotalInscClaimAmtReimbursed']
dataset_prv=data_combined_Fe[columns]
dataset_prv_scaled=scaler.fit_transform(dataset_prv.drop(['Provider'],axis=1))
df_prv_scaled = pd.DataFrame(dataset_prv_scaled, columns=dataset_prv.drop(['Provider'],axis=1).columns)
df_prv_scaled.fillna(0,inplace=True)
data_prv_if,data_scaled_prv_if,if_prv_model=Isolation_forest(dataset_prv,df_prv_scaled,0.1,500,5)
data_scaled_prv_if['Anomaly_ind'] = data_scaled_prv_if['Anomaly_score'].apply(lambda x: 1 if x <= -0.02 else 0)
data_prv_if['Anomaly_ind'] = data_prv_if['Anomaly_score'].apply(lambda x: 1 if x <=-0.02 else 0)
df_anom_prv_if,df_normal_prv_if,df_anom_scale_prv_if,df_normal_scale_prv_if,dataset_anom_rank_prv_if=dataframe_creation(data_prv_if,data_scaled_prv_if,'Anomaly_ind','Anomaly_score',True)
dataset_anom_rank_prv_if=dataset_anom_rank_prv_if.set_index('Rank',drop=True)
util.pickle_dump(dataset_anom_rank_prv_if,config_data['raw_dataset_path_if_amt_rank'])
util.pickle_dump(if_prv_model,config_data['if_prv_model'])

