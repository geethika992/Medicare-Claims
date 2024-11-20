import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import time
import src.util as util
import yaml
import datetime
import json
import os
import subprocess
import shap
import streamlit.components.v1 as components
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
config_data=util.config_load()
metadata_filename = "model_metadata.json"
st.set_page_config(layout="wide")
# Cache data for faster Data Retrieval
# Streamlit app title
st.title("Anomolous Providers")

data_borda=util.pickle_load(config_data["Dataset_borda_prv_list"])
df_provider_stats=util.pickle_load(config_data["raw_dataset_path_provider_stats"])
shap_values_if=util.pickle_load(config_data['raw_dataset_shap_if'])
shap_values_lof=util.pickle_load(config_data['raw_dataset_shap_lof'])
shap_values_rrcf=util.pickle_load(config_data['raw_dataset_shap_rrcf'])
shap_values_if_prv=util.pickle_load(config_data['raw_dataset_shap_if_prv'])
dataset_anom_if=util.pickle_load(config_data['raw_dataset_path_if_rank'])
dataset_anom_lof=util.pickle_load(config_data['raw_dataset_path_lof_rank'])
dataset_anom_rrcf=util.pickle_load(config_data['raw_dataset_path_rrcf_rank'])
dataset_anom_prv_amt=util.pickle_load(config_data['raw_dataset_path_if_amt_rank'])
data_usage=util.pickle_load(config_data['raw_dataset_path_usage'])
data_simil_sub=util.pickle_load(config_data['raw_dataset_path_simil_sub'])
data_provider=util.pickle_load(config_data['raw_dataset_provider_data'])



dataset_anom_if=dataset_anom_if.reset_index(drop=True)
dataset_anom_lof=dataset_anom_lof.reset_index(drop=True)
dataset_anom_rrcf=dataset_anom_rrcf.reset_index(drop=True)
dataset_anom_prv_amt=dataset_anom_prv_amt.reset_index(drop=True)
data_borda_ind = data_borda.set_index('Rank',drop=True)
ICD_Data=util.pickle_load(config_data["raw_dataset_path_ICD_codes"])
ICD_Data=ICD_Data.set_index("CODE",drop=True)


st.write(f"**Total number of Anomolous Provider: {len(data_borda)}** ")

# Display the result

tab1, tab2 = st.tabs(["**Anomolous Providers**", "**Provider Interpretation**"])

with tab1:
    n = st.slider("Select number of top anomalous providers", min_value=1, max_value=len(data_borda),value=10, step=1)
    filtered_data=df_provider_stats[df_provider_stats.Provider.isin(data_borda['Provider'].iloc[:n].tolist())]
    # Display the anomalous providers and plot side by side using columns
    col1, col2 = st.columns(2)

    # Display the list of anomalous providers in the first column
    with col1:
        st.subheader("Anomalous Providers List")
        st.write(data_borda_ind[['Provider']].head(n))

    # Create and display the Altair bar chart in the second column
    with col2:
        select_axis=st.selectbox("Choose the parameter for visualization",['Claims','Beneficiaries','Total claim amount reimbursement'])
        if select_axis == 'Claims':
            # Create a bar chart using Altair for claims count
            bar_chart = alt.Chart(filtered_data).mark_bar().encode(
                y=alt.Y('Provider:O', title='Provider ID'),  # Categorical x-axis (provider IDs)
                x=alt.X('claims_count:Q', title='Total Claim Count'),  # Quantitative y-axis (claim count)
                tooltip=['Provider',
                          'claims_count']  # Add tooltips for provider_id and claims_count
            ).properties(
                title='Total Claim Count by Provider'
            )
            st.altair_chart(bar_chart, use_container_width=True)

        elif select_axis == 'Beneficiaries':
            # Create a bar chart using Altair for beneficiaries count
            bar_chart = alt.Chart(filtered_data).mark_bar().encode(
                y=alt.Y('Provider:O', title='Provider ID'),
                x=alt.X('beneficiaries_count:Q', title='Total Beneficiaries Count'),
                tooltip=['Provider', 'beneficiaries_count']
            ).properties(
                title='Total Beneficiaries Count by Provider'
            )
            st.altair_chart(bar_chart, use_container_width=True)

        else:
            # Create a bar chart using Altair for total claim amount
            bar_chart = alt.Chart(filtered_data).mark_bar().encode(
                y=alt.Y('Provider:O', title='Provider ID'),
                x=alt.X('total_claim_amount:Q', title='Total Claim Amount'),
                tooltip=['Provider', 'total_claim_amount']
            ).properties(
                title='Total Claim Amount by Provider'
            )
            st.altair_chart(bar_chart, use_container_width=True)

with tab2:
    ICD_code=st.sidebar.text_input("Enter the ICD code for the description")
    if not ICD_Data[ICD_Data.index==ICD_code]['Description'].empty:
        Description=ICD_Data[ICD_Data.index==ICD_code]['Description']
    elif len(ICD_code)==0:
        Description=''
    else : Description="The description for the code is not available."
    st.sidebar.write(Description)
    input_method = st.radio("Choose input method", ('Select from list', 'Enter provider number'),horizontal=True)

# Based on the selected method, display the appropriate input widget
    if input_method == 'Select from list':
        prv1 = st.selectbox("Choose Provider from the list", data_borda['Provider'].to_list())
    # Hide the manual input box for entering provider number
        prv2 = None
        prv=prv1
    else:
        prv2 = st.text_input("Enter the Provider number")
        # Hide the selectbox if the user is manually entering the provider number
        prv1 = None
        prv=prv2

    if len(prv)!=0:
        provider_rank = data_borda.loc[data_borda['Provider'] == prv]['Rank'].values
        if provider_rank!= None:
            st.write(f"Provider **{prv}** is falgged as anomolous with a rank of **{provider_rank[0]}**")
            Total_points = round(data_borda[data_borda.Provider == prv]['Total_avg'].iloc[0],2)
            st.write(f"Total points : **{Total_points}**")
            data_borda_prv=data_borda[data_borda.Provider==prv][['ICD_norm','Clm_cnt_norm']]
            data_borda_prv_rename=data_borda_prv.rename(columns={'ICD_norm':'ICD code Anomaly','Clm_cnt_norm':'Claim count Anomaly'})
            df_melted = data_borda_prv_rename.melt(var_name='Category', value_name='Value')
            
            pie = alt.Chart(df_melted).mark_arc(innerRadius=85).encode(
                theta=alt.Theta(field="Value", type="quantitative", stack=True, scale=alt.Scale(type="linear",rangeMax=1.5708, rangeMin=-1.5708 )),
                color=alt.Color(field="Category", type="nominal")
            ).properties(title="Contribution of Category to the Anomaly Score",
                    width=500,  # Increase the chart width
                    height=500)  # Increase the chart height
            pie + pie.mark_text(radius=180, fontSize=18).encode(text='Value')
            st.header("**Interpretation of Anomaly**")
            category=st.radio("Choose a category",['ICD code','Claims'],horizontal=True)
            if category=='ICD code':
                index_if=dataset_anom_if[dataset_anom_if.Provider==prv].index.to_numpy()
                index_lof=dataset_anom_lof[dataset_anom_lof.Provider==prv].index.to_numpy()
                index_rrcf=dataset_anom_rrcf[dataset_anom_rrcf.Provider==prv].index.to_numpy()
                if len(index_if)!=0:
                    shap.waterfall_plot(shap_values_if[index_if[0]])
                    fig = plt.gcf()
                    st.pyplot(fig)   
                elif len(index_lof)!=0:
                    shap.waterfall_plot()
                    fig = plt.gcf()
                    st.pyplot(fig)     
                elif len(index_rrcf)!=0:
                    shap.waterfall_plot(shap_values_rrcf[index_rrcf[0]])
                    fig = plt.gcf()
                    st.pyplot(fig)
                else: 
                    print("Incorrect provider selected")
                data_usage_prv=data_usage.join(data_provider)
                prv_usage_data=data_usage_prv[data_usage_prv.Provider==prv]
                icd_code=st.selectbox("Select the ICD code to find the exact usage by the provider",data_usage_prv.columns.to_list())
                icd_count=prv_usage_data[icd_code]
                st.write("Total Count: "+ str(icd_count.values[0]))
                data_list_simil=data_simil_sub[data_simil_sub[icd_code]!=0][icd_code].index.to_list()
                filtered_df_prv=data_usage_prv[data_usage_prv.Provider==prv]
                matching_columns=filtered_df_prv.columns[filtered_df_prv.columns.isin(data_list_simil)]
                result=filtered_df_prv[matching_columns]
                melted_data=pd.melt(result,var_name='ICD Codes',value_name='Count')
                df_simil_data=pd.DataFrame(data_simil_sub[data_simil_sub[icd_code]!=0][icd_code]).reset_index()
                #df_simil_data = df_simil_data.sort_values(by='Value')
                melted_data = melted_data.sort_values(by='ICD Codes')
                data_joined=melted_data.join(df_simil_data)
                data_joined['Final_score']=data_joined['Count']*data_joined[icd_code]
                sorted_df=data_joined.sort_values(by='Final_score',ascending=False)
                n1=st.slider("Select number of ICD code", min_value=1, max_value=20,value=5, step=1)
                bar_chart = alt.Chart(sorted_df[0:n1]).mark_bar().encode(
                y=alt.Y('ICD Codes:O', title='ICD Code', sort='-x'),
                x=alt.X('Count:Q', title='Count'),
                tooltip=[alt.Tooltip(icd_code, title='Similarity Score'),alt.Tooltip('Final_score', title='Total Score'),alt.Tooltip('Count', title='Count')]
                    ).properties(title='Similar ICD codes used by the Provider',
                    height=400,
                    width=400
                     )
                st.altair_chart(bar_chart, use_container_width=True)


            else:
               
                index_prv_if=dataset_anom_prv_amt[dataset_anom_prv_amt.Provider==prv].index.to_numpy()
                if len(index_prv_if)!=0:
                        shap.plots.waterfall(shap_values_if_prv[index_prv_if[0]])
                        fig = plt.gcf()
                        st.pyplot(fig)
            


