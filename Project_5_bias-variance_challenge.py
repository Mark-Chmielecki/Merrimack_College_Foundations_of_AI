"""
Created by Mark Chmielecki for CS6313 - Foundations of Artificial Intelligence
Project 5 due Monday, October 5, 2026.   

Generate synthetic data for patient health records.  The data will be used to train a machine learning model to predict patient health risk
based on various features, age, bmi, and blood sugar level for 500 patients.  The data will be generated using random number generation and 
will be saved to a CSV file patient_health_data.csvfor later use in the project.
"""

from sklearn.linear_model import LinearRegression
import pandas as pd     # save the data to a CSV file

# access CSV file 
health_data_frame = pd.read_csv('patient_health_data.csv')

#x = health_data_frame[['Age', 'BMI', 'Blood Sugar']]
x = health_data_frame[['Age']]
y = health_data_frame['Risk Score']

model = LinearRegression()

model.fit(x, y)  

print(model.coef_)
print(model.intercept_)