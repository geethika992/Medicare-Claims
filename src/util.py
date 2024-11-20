import yaml
import joblib
from datetime import datetime

config_dir="config/config.yaml"

def config_load()->dict:
    try:
        with open(config_dir,"r") as file:
            config=yaml.safe_load(file)
    except:
        raise RuntimeError("parameter file not found in path")

    # Return params in dictionary format
    return config
def pickle_load(file_path: str):
    # Load and return pickle file
    return joblib.load(file_path)
def pickle_dump(data, file_path: str) -> None:
    # Dump into file
    joblib.dump(data, file_path)
           


