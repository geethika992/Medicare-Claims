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

config_data=util.config_load()

tab1, tab2= st.tabs(["**Introduction**", "**About Data**"])
with tab1: 
        st.header('Unsupervised Method to Identify Anomalous Healthcare Providers')
        st.markdown("""A powerful tool built to identify unusual patterns in healthcare provider data using unsupervised machine learning techniques. Traditional methods for detecting fraud or inefficiencies often rely on labeled data, which can be scarce or unavailable. \nBelow are some of the important features provided :\n 1. Leverages unsupervised machine learning algorithms to detect anomalies without the need for labeled data. \n 2. Automatically identifies unusual patterns in healthcare provider's past data such as overuse of ICD codes and anomalies in claims.\n3. Provides a rank for each provider based on the extend of anomaly detected.  \n 4. Offers interpretation and nature of detected anomalies suggesting possible areas for further investigation.""")
        st.subheader("How to Use the App")
        st.markdown("""
        1. **Upload Your Data**: Fork the github repository. Upload a CSV or Excel file containing healthcare provider data into the folder /data/dataset. Ensure that only required files are in the folder.
        2. **Run the Analysis**: Once the data is uploaded,Goto the page Model Training and execute the scripts. 
        3. **Review the Results**: The app will display a summary of flagged anomalies, along with visualizations highlighting potential issues.
        """)
with tab2: 
                st.markdown("**The dataset consists of three distinct categories: Beneficiary Information, Inpatient Claims, and Outpatient Claims**")
                st.write("If a new dataset is available, place it in the path data/dataset and Load the file")
                if st.button("Click here to Load the data"):
            # Run the scripts using subprocess
                    # Run each Python script
                    st.write("Process 1: Loading the files and Preprocessing...")
                    subprocess.run(["python", "src/Medicare_File_Load_preprocessing_01.py"], check=True)
                df_ben=util.pickle_load(config_data["raw_dataset_path_train_ben"])
                df_inp=util.pickle_load(config_data["raw_dataset_path_train_inp"])
                df_outp=util.pickle_load(config_data["raw_dataset_path_train_outp"])
                if os.path.exists(config_data["raw_dataset_path_train_ben"]):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: 
                        Total_beneficiaries=df_ben['BeneID'].nunique()
                        st.metric(label="Total Beneficiaries", value=Total_beneficiaries)
                    with col2: 
                        Total_Inpatient_claims=df_inp['ClaimID'].nunique()
                        st.metric(label="Total Inpatient Claims", value=Total_Inpatient_claims)
                    with col3:
                        Total_outpatient_claims=df_outp['ClaimID'].nunique()
                        st.metric(label="Total Outpatient Claims", value=Total_outpatient_claims)
                    with col4:
                        Total_providers=pd.concat([df_inp['Provider'],df_outp['Provider']]).nunique()
                        st.metric(label="Total Providers", value=Total_providers)
                    Total_Amount=pd.concat([df_inp['InscClaimAmtReimbursed'],df_outp['InscClaimAmtReimbursed']]).sum()
                    Total_Amount=Total_Amount/1000000
                    st.metric(label="Total Reimbursement Amount", value=f"${Total_Amount}M")