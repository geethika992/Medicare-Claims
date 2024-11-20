import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import util as util
import yaml
import copy
from tqdm import tqdm
import os
from datetime import datetime
import openpyxl
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import shap
from sklearn.ensemble import RandomForestRegressor

config_data=util.config_load()

dataset_anom_if=util.pickle_load(config_data['raw_dataset_path_if_rank'])
dataset_anom_lof=util.pickle_load(config_data['raw_dataset_path_lof_rank'])
dataset_anom_rrcf=util.pickle_load(config_data['raw_dataset_path_rrcf_rank'])
dataset_anom_prv_amt=util.pickle_load(config_data['raw_dataset_path_if_amt_rank'])
model_if=util.pickle_load(config_data['Isolation_forest_model'])
model_prv_amt=util.pickle_load(config_data['if_prv_model'])


#################Isolation Forest####################
explainer=shap.TreeExplainer(model_if)
shap_values_if=explainer(dataset_anom_if.drop(columns=['Provider','Anomaly_ind','Anomaly_score','Rank'],axis=1))
util.pickle_dump(shap_values_if,config_data['raw_dataset_shap_if'])
print('Done')

# ######################Local Outlier Factor###########
lof_scores = dataset_anom_lof['Anomaly_score_lof'].values # Invert for easier interpretation

# Fit a Random Forest model on the LOF scores
rf_model = RandomForestRegressor()
rf_model.fit(dataset_anom_lof.drop(columns=['Provider','Anomaly_ind','Anomaly_score','Anomaly_lof','Anomaly_score_lof','Rank']), lof_scores)

# Use SHAP to explain the Random Forest model
explainer = shap.TreeExplainer(rf_model, dataset_anom_lof.drop(columns=['Provider','Anomaly_ind','Anomaly_score','Anomaly_lof','Anomaly_score_lof','Rank']))
shap_values_lof = explainer(dataset_anom_lof.drop(columns=['Provider','Anomaly_ind','Anomaly_score','Anomaly_lof','Anomaly_score_lof','Rank']),check_additivity=False)
util.pickle_dump(shap_values_lof,config_data['raw_dataset_shap_lof'])

#######################RRCF###################################
rrcf_scores = dataset_anom_rrcf['Anomaly_score_rrcf'].values # Invert for easier interpretation
# Fit a Random Forest model on the LOF scores
rf_model = RandomForestRegressor()
rf_model.fit(dataset_anom_rrcf.drop(columns=['Provider','Anomaly_ind','Anomaly_score','Anomaly_lof','Anomaly_score_lof','Anomaly_rrcf','Anomaly_score_rrcf']), rrcf_scores)

# Use SHAP to explain the Random Forest model
explainer = shap.Explainer(rf_model, dataset_anom_rrcf.drop(columns=['Provider','Anomaly_ind','Anomaly_score','Anomaly_lof','Anomaly_score_lof','Anomaly_rrcf','Anomaly_score_rrcf']))
shap_values_rrcf = explainer(dataset_anom_rrcf.drop(columns=['Provider','Anomaly_ind','Anomaly_score','Anomaly_lof','Anomaly_score_lof','Anomaly_rrcf','Anomaly_score_rrcf']))
util.pickle_dump(shap_values_rrcf,config_data['raw_dataset_shap_rrcf'])

###################Isolation Forest provider claim count########
print(model_prv_amt)
explainer=shap.TreeExplainer(model_prv_amt)
shap_values_prv=explainer(dataset_anom_prv_amt.drop(columns=['Provider','Anomaly_ind','Anomaly_score'],axis=1),check_additivity=False)
util.pickle_dump(shap_values_prv,config_data['raw_dataset_shap_if_prv'])
