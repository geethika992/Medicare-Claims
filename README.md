# Unsupervised Method to Identify Anomalous Healthcare Providers

A powerful tool designed to identify unusual patterns in healthcare provider data using unsupervised machine learning techniques. This approach eliminates the reliance on labeled data, offering a novel solution to detect fraud, inefficiencies, and other anomalies in healthcare provider behavior.

## Features

- **Unsupervised Machine Learning**: Anomaly detection without the need for labeled data, making it suitable for real-world scenarios where such data is often scarce or unavailable.
  
- **Anomaly Detection**: It automatically detects unusual patterns in healthcare provider data, such as:
  - Overuse or underuse of specific ICD codes.
  - Irregularities in healthcare claims.

- **Ranking**: Each provider is assigned a rank based on the extent of anomaly detected, helping prioritize further investigation.

- **Interpretability**: The tool provides insights into the detected anomalies, offering suggestions on what areas need further investigation or refinement.

## How to Use the App

Follow these steps to run the analysis on your healthcare provider data:

### 1. **Upload Your Data**
- Fork this GitHub repository.
- Upload a CSV or Excel file containing healthcare provider data into the `/data/dataset` folder. Ensure that only the relevant files are in this folder to avoid processing errors.
- Make sure the dataset follows the format expected by the model for optimal performance.

### 2. **Run the Analysis**
- Navigate to the **Model Training** page in the repository.
- Execute the necessary scripts to start the model training and anomaly detection process. The scripts will process the dataset and detect anomalies based on the unsupervised learning model.

### 3. **Review the Results**
- Once the analysis is complete, the app will display a summary of flagged anomalies.
- Visualizations will be presented to highlight areas of concern, including potential issues with overuse of medical codes, fraud in claims, or inefficiencies in provider behavior.
  
### 4. **Further Investigation**
- The app provides a ranked list of healthcare providers based on the severity of detected anomalies.
- Use this ranking to prioritize providers for further review and investigation.
