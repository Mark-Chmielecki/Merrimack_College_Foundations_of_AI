"""
Created by Mark Chmielecki for CS6313 - Foundations of Artificial Intelligence
Project 5 due Monday, October 5, 2026.   

Generate synthetic data for patient health records.  The data will be used to train a machine learning model to predict patient health risk
based on various features, age, bmi, and blood sugar level for 500 patients.  The data will be generated using random number generation and 
will be saved to a CSV file patient_health_data.csvfor later use in the project.
"""

import numpy as np
import pandas as pd     # save the data to a CSV file
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# access CSV file 
health_data_frame = pd.read_csv('patient_health_data.csv')

# set the imputer to fill in missing values with the median of each column
imputer = SimpleImputer(strategy="median")

#input the health data frame into the imputer and transform it to fill in missing values
health_data_imputed = imputer.fit_transform(health_data_frame)

scaler = StandardScaler()

health_data_scaled = scaler.fit_transform(health_data_imputed)

# save updates back to the CSV
health_data_scaled_df = pd.DataFrame(health_data_scaled, columns=health_data_frame.columns)
health_data_scaled_df.to_csv('patient_health_data_scaled.csv', index=False)   

#need to add splitting