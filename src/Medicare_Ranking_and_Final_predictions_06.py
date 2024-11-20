import pandas as pd
import numpy as np
import util as util
import yaml
import copy
import os
from datetime import datetime

config_data=util.config_load()

##############Load Providers ranked by different algorithms################
data_if_rank=util.pickle_load(config_data['raw_dataset_path_if_rank'])
data_lof_rank=util.pickle_load(config_data['raw_dataset_path_lof_rank'])
data_rrcf_rank=util.pickle_load(config_data['raw_dataset_path_rrcf_rank'])
data_if_prv_amt_rank=util.pickle_load(config_data['raw_dataset_path_if_amt_rank'])

################Borda Counting#####################################

def borda_rankings(rankings):
    # Dictionary to store points for each provider
    points = {}
    
    # List to store the sum of points for the first 3 rankings and last ranking
    first_three_points = {}
    last_points = {}
    
    # Iterate through each ranking (model)
    for i, ranking in enumerate(rankings):
        num_candidates = len(ranking)
        
        # Iterate through each provider in the ranking
        for index, candidate in ranking.items():
            if candidate not in points:
                points[candidate] = 0
            
            # Points are calculated based on the rank in the current ranking
            points[candidate] += (num_candidates - index) / (num_candidates - 1)
            
            # For the first three rankings, add points to the sum
            if i < 3:
                if candidate not in first_three_points:
                    first_three_points[candidate] = 0
                first_three_points[candidate] += (num_candidates - index) / (num_candidates - 1)
            
            # For the last ranking, add points to the sum
            if i == len(rankings) - 1:
                if candidate not in last_points:
                    last_points[candidate] = 0
                last_points[candidate] += (num_candidates - index) / (num_candidates - 1)
    
    # Prepare the data for the DataFrame
    df_results = pd.DataFrame(list(points.items()), columns=['Provider', 'Borda_points'])
    
    # Add the sum of points from the first three rankings and the last ranking
    df_results['Sum_first_three_points'] = df_results['Provider'].apply(lambda x: first_three_points.get(x, 0))
    df_results['Sum_last_points'] = (df_results['Provider'].apply(lambda x: last_points.get(x, 0)))
    df_results['avg_ICD'] = (df_results['Sum_first_three_points']/3)
    df_results['Total_avg'] = ((df_results['Sum_first_three_points']/3)+( df_results['Sum_last_points']))
    df_results['ICD_norm']=round((df_results['avg_ICD']/ df_results['Total_avg']),2)
    df_results['Clm_cnt_norm']=round((df_results['Sum_last_points']/ df_results['Total_avg']),2)
    

    # Sort the dataframe based on Borda points in descending order
    df_results_sorted = df_results.sort_values('Total_avg', ascending=False)
    
    return df_results_sorted

rankings_rrcf=data_if_rank.set_index('Rank')['Provider']
rankings_lof=data_lof_rank.set_index('Rank')['Provider']
rankings_if=data_rrcf_rank.set_index('Rank')['Provider']
rankings_if_prv_amt=data_if_prv_amt_rank['Provider']
rankings=[
    rankings_rrcf, 
    rankings_lof,   
    rankings_if,
    rankings_if_prv_amt 
]

df_borda_rankings=borda_rankings(rankings)
df_borda_rankings=df_borda_rankings[df_borda_rankings.Total_avg>=1]

df_borda_rankings['Rank']=range(1,len(df_borda_rankings)+1)
print(df_borda_rankings)
util.pickle_dump(df_borda_rankings, config_data["Dataset_borda_prv_list"])