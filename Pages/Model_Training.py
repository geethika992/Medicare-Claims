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

config_data=util.config_load()
metadata_filename = "model_metadata.json"
st.set_page_config(layout="wide")
# Cache data for faster Data Retrieval
import subprocess
# Streamlit app title
st.header("Re-Training the model with New Data")

st.write("**Previous Training details**")
with open(metadata_filename, 'r') as f:
            metadata = json.load(f)
st.write(f"Model last trained on: {metadata['last_trained_time']}")
st.write("Dataset files used for training:")
st.write(metadata['dataset_files'])



st.write("Steps to retrain the model \n 1. Ensure only the required source files are in the folder ""data/dataset"" \n 2. Move any existing data to the archive folder.  \n3. If you have a new set of ICD code files, place them in data/dataset/med_codes folder and execute the ICD data load. \n ")
# Button to run all scripts

if st.button("Click here for ICD Data Load"):
        # Run each Python script
        st.write("ICD Data creation....")
        subprocess.run(["python", "src/Medicare_data_Diag_coding_03.py"], check=True)
        st.success("ICD data loaded successfully!")
if st.button("Click here to Execute the Training"):
    # Run the scripts using subprocess
    try:
        progress_bar = st.progress(0)
        # Run each Python script
        st.write("Process 1: Loading the files and Preprocessing...")
        subprocess.run(["python", "src/Medicare_File_Load_preprocessing_01.py"], check=True)
        
        progress_bar.progress(20) 
        st.write("Process 2: Feature Engineering...")
        subprocess.run(["python", "src/Medicare_Feature_engineering_02.py"], check=True)
        
        progress_bar.progress(40) 
        st.write("Process 3: Data Preparation...")
        subprocess.run(["python", "src/Medicare_ICD_Matrix_data_preparation_04.py"], check=True)
        
        progress_bar.progress(60) 
        st.write("Process 4: Fitting the Models...")
        subprocess.run(["python", "src/Medicare_Diagnosis_code_data_modeling_05.py"], check=True)

        progress_bar.progress(80) 
        st.write("Process 5: Data Ranking...")
        subprocess.run(["python", "src/Medicare_Ranking_and_Final_predictions_06.py"], check=True)
        
        progress_bar.progress(100) 
        st.success("All scripts executed successfully and Models are saved!")
        last_retrained = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")        
        st.write(f"Training Completed: {last_retrained}")
        
        files=os.listdir(config_data['raw_dataset_dir'])
        files = [f for f in files if os.path.isfile(os.path.join(config_data['raw_dataset_dir'], f))and f != ".DS_Store"]
    # Prepare metadata to save
        metadata = {
                    "last_trained_time": last_retrained,
                    "dataset_files": files
                    }

                # Save the metadata to a JSON file
        with open(metadata_filename, "w") as f:
                    json.dump(metadata, f, indent=4)
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)
        st.write("Dataset files used for training:")
        st.write(metadata['dataset_files'])
        
    except subprocess.CalledProcessError as e:
            st.error(f"Error occurred while executing scripts: {e}")


