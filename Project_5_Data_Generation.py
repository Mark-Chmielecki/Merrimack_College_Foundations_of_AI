"""
Created by Mark Chmielecki for CS6313 - Foundations of Artificial Intelligence
Project 5 due Monday, October 5, 2026.   

Generate synthetic data for patient health records.  The data will be used to train a machine learning model to predict patient health risk
based on various features, age, bmi, and blood sugar level for 500 patients.  The data will be generated using random number generation and 
will be saved to a CSV file patient_health_data.csvfor later use in the project.
"""

import numpy as np      # generate random numbers for age, bmi, and blood sugar level
import pandas as pd     # save the data to a CSV file

# Make results reproducible
np.random.seed(42)

# 500 patients
age = np.random.randint(18, 80, 500)
bmi = np.random.normal(25, 5, 500)
blood_sugar = np.random.normal(120, 30, 500)

# create a risk score based on age, bmi, and blood sugar level
age_risk = np.where(age < 40, 0, np.where((age >= 40) & (age < 50), 1, np.where((age >= 50) & (age < 60), 2, 3)))

bmi_risk = np.where(bmi < 25, 0, np.where((bmi >= 25) & (bmi < 30), 1, np.where((bmi >= 30) & (bmi < 40), 2, 3)))

blood_sugar_risk = np.where(blood_sugar < 100, 0, np.where((blood_sugar >= 100) & (blood_sugar < 125), 4, 5))

health_risk_score = age_risk + bmi_risk + blood_sugar_risk

# create a risk category based on the risk score
# risk_category = np.where(health_risk_score < 1.5, 'Low', np.where(health_risk_score < 2.5, 'Medium', 'High'))

# save the data to a CSV file
data = pd.DataFrame({'Age': age, 'BMI': bmi, 'Blood Sugar': blood_sugar, 'Risk Score': health_risk_score})
data.to_csv('patient_health_data.csv', index=False)