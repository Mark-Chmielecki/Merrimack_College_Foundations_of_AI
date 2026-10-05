"""
Created by Mark Chmielecki for CS6313 - Foundations of Artificial Intelligence
Project #5 due Monday, October 5, 2026.   

Project requirements:
1) Generate synthetic health data with age, BMI, and blood sugar level, and save to a CSV file called patient_health_data.csv.
2) Preprocess the data by filling in missing values and scaling the features, implement Median Imputation and z-score scaling.
3) Train and evaluate three different models (underfit, overfit, and optimal) using the training and testing sets, and print the MSE and R^2 metrics for each model in the terminal.
4) Allow the user to input their own age, BMI, and blood sugar level to predict their health risk score, probability of risk, and final diagnosis using the optimal model.

This program meets all requirements and is designed to be user-friendly, with clear prompts and messages in the terminal. The program also includes error handling for user input and provides informative messages about the progress of the program. 
"""
import numpy as np                                                      # need for synthetic data generation and calculations
import pandas as pd                                                     # needed for data manipulation and saving to CSV
import time                                                             # add time to processing time to make it realistic

from sklearn.impute import SimpleImputer                                # needed for handling missing values in the data
from sklearn.preprocessing import StandardScaler, PolynomialFeatures    # needed for scaling the features and creating polynomial features for the overfit model
from sklearn.linear_model import LinearRegression, LogisticRegression   # needed for training the models
from sklearn.pipeline import make_pipeline                              # needed for creating a pipeline for the overfit model
from sklearn.metrics import mean_squared_error, r2_score                # needed for evaluating the models
from sklearn.model_selection import train_test_split                    # needed for splitting the data into training and testing sets                                           

def create_synthetic_data(num_rows):
    """Function to generate synthetic health data with age, BMI, and blood sugar level, part of requirement #1."""
    
    # Make results reproducible
    np.random.seed(42)                                                  # uses a fixed seed for the random number generator to ensure that the same synthetic data is generated each time the program is run, 42 is generally accepted as the answer to life, the universe, and everything, so it is a good choice for a seed value (Hitchhiker's Guide to the Galaxy).

    # Generate synthetic data
    age = np.random.randint(18, 80, num_rows)
    bmi = np.random.normal(25, 5, num_rows)
    blood_sugar = np.random.normal(120, 30, num_rows)

    return age, bmi, blood_sugar

def calculate_risk_score(age, bmi, blood_sugar):
    """Function to calculate a health risk score based on age, BMI, and blood sugar level, part of requirement #1."""
    
    # create a risk score based on age, bmi, and blood sugar level
    age_risk = np.where(age < 40, 0, np.where((age >= 40) & (age < 50), 10, np.where((age >= 50) & (age < 60), 25, 33.33)))

    bmi_risk = np.where(bmi < 25, 0, np.where((bmi >= 25) & (bmi < 30), 10, np.where((bmi >= 30) & (bmi < 40), 25, 33.33)))

    blood_sugar_risk = np.where(blood_sugar < 100, 0, np.where((blood_sugar >= 100) & (blood_sugar < 125), 25, 33.34))

    health_risk_score = age_risk + bmi_risk + blood_sugar_risk

    # save the data to a CSV file
    data = pd.DataFrame({'Age': age, 'BMI': bmi, 'Blood Sugar': blood_sugar, 'Risk Score': health_risk_score})
    return data 

def save_data_to_csv(data, filename):
    """Generic function to save the data to a CSV file."""
    data.to_csv(filename, index=False)

def load_data_from_csv(filename):
    """Generic function to load the data from a CSV file."""
    return pd.read_csv(filename)

def preprocess_data(data):
    """Function to preprocess the data by filling in missing values and scaling the features, part of requirement #2."""
    
    scaler = None   # Initialize scaler and imputer to None before preprocessing
    imputer = None  # Initialize scaler and imputer to None before preprocessing

    # Separate features from target
    X = data[['Age', 'BMI', 'Blood Sugar']]
    y = data['Risk Score']

    # Fill missing values in the features
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Scale only the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Put the scaled features back into a DataFrame
    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=['Age', 'BMI', 'Blood Sugar']
    )

    # Add the original Risk Score back
    X_scaled_df['Risk Score'] = y.values

    # Save the processed data
    X_scaled_df.to_csv('patient_health_data.csv', index=False)

    return scaler, imputer

def split_data(data, test_size=0.2):
    """Function to split the data into training and testing sets, required before you move orequirement #3."""
    from sklearn.model_selection import train_test_split

    # Split the data into features and target variable
    X = data[['Age', 'BMI', 'Blood Sugar']]
    y = data['Risk Score']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model_underfit(X_train, y_train):
    """Function to train a linear regression model on the training data, part of requirement #3."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_model_overfit(X_train, y_train):
    """Function to train an overfit model on the training data, part of requirement #3."""
    # Create a pipeline that first transforms the features to polynomial features and then fits a linear regression model
    model = make_pipeline(PolynomialFeatures(degree=10), LinearRegression())
    model.fit(X_train, y_train)
    return model

def train_model_optimal(X_train, y_train):
    """Function to train the optimal model on the training data, part of requirement #3."""
    # Create a pipeline that first transforms the features to polynomial features and then fits a linear regression model
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X_train, y_train)
    return model

def optimal_model_predict(new_patient):
    """Function to make a prediction using the optimal model."""
    return optimal_model.predict(new_patient)

def train_logistic_model(X_train, y_train):
    """Train a logistic regression model to predict probability of risk."""

    # Convert the risk score into two categories:
    # 0 = HEALTHY
    # 1 = AT RISK
    y_class = (y_train >= 60).astype(int)

    model = LogisticRegression()
    model.fit(X_train, y_class)

    return model

def main():
    """
    Main execution function, drives the user experience.  
    I inserted time.sleep() calls to simulate processing time and make the program feel more realistic, kind of like the Dominoes Pizza Tracker.
    """

    # Prompt the user for the number of rows of synthetic data to generate, ensuring that it is at least 500 rows per the project requirements.
    while True:
        rows = input("How many rows of synthetic data would you like to generate? Must be at least 500 rows: ").strip()

        if rows.isdigit():
            rows = int(rows)

            if rows >= 500:
                break
            else:
                print("Please enter at least 500 rows.")
        else:
            print("Please enter a valid number.")

    print(f"Generating {rows} rows of synthetic data...")
    time.sleep(5)  # Simulate a delay for data generation
    age, bmi, blood_sugar = create_synthetic_data(rows)
    print("Saving data to CSV file...")
    time.sleep(5)  # Simulate a delay for saving data
    save_data_to_csv(calculate_risk_score(age, bmi, blood_sugar), 'patient_health_data.csv')

    print("Preprocessing the data...")
    health_data_frame = load_data_from_csv('patient_health_data.csv')
    scaler, imputer = preprocess_data(health_data_frame)
    time.sleep(5)  # simulate a delay for preprocessing
    print("Data preprocessing complete. Scaled data saved to 'patient_health_data.csv'.")

    print("Loading the scaled data...")
    health_data_frame = load_data_from_csv('patient_health_data.csv')
    time.sleep(5)  # simulate a delay for loading data    

    print("Splitting the data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(health_data_frame)
    time.sleep(5)  # simulate a delay for splitting data

    print("Training the underfit model...")
    underfit_model = train_model_underfit(X_train, y_train) 
    mse_underfit = mean_squared_error(y_test, underfit_model.predict(X_test))

    print(f"Underfit model MSE: {mse_underfit}")    
    r2_underfit = r2_score(y_test, underfit_model.predict(X_test))

    print(f"Underfit model R^2: {r2_underfit}")
    time.sleep(5)  # simulate a delay for training the underfit model

    print("Training the overfit model...")
    overfit_model = train_model_overfit(X_train, y_train)
    mse_overfit = mean_squared_error(y_test, overfit_model.predict(X_test))
    print(f"Overfit model MSE: {mse_overfit}")
    r2_overfit = r2_score(y_test, overfit_model.predict(X_test))
    print(f"Overfit model R^2: {r2_overfit}")
    time.sleep(5)  # simulate a delay for training the overfit model   

    print("Training the optimal model...")
    optimal_model = train_model_optimal(X_train, y_train)
    mse_optimal = mean_squared_error(y_test, optimal_model.predict(X_test))

    print(f"Optimal model MSE: {mse_optimal}")      
    r2_optimal = r2_score(y_test, optimal_model.predict(X_test))

    print(f"Optimal model R^2: {r2_optimal}")
    time.sleep(5)  # simulate a delay for training the optimal model

    # Determine which model performs best.
    # Lower MSE is better, while higher R^2 is better.

    if (mse_underfit <= mse_overfit and mse_underfit <= mse_optimal and
            r2_underfit >= r2_overfit and r2_underfit >= r2_optimal):

        print("The underfit model is performing the best on the test data.")

    elif (mse_overfit <= mse_underfit and mse_overfit <= mse_optimal and
          r2_overfit >= r2_underfit and r2_overfit >= r2_optimal):

        print("The overfit model is performing the best on the test data.")

    elif (mse_optimal <= mse_underfit and mse_optimal <= mse_overfit and
          r2_optimal >= r2_underfit and r2_optimal >= r2_overfit):

        print("The optimal model is performing the best on the test data.")

    else:
        print("The model results are mixed. No single model is best on both MSE and R^2.")

    print("Training the logistic regression model...")
    logistic_model = train_logistic_model(X_train, y_train)

    time.sleep(5)
    
    print("Model evaluation complete.")

    print("Give me your patient's age, BMI, and blood sugar level to predict your health risk score.")

    age = float(input("Enter your patient's age: "))
    bmi = float(input("Enter your patient's BMI: "))
    blood_sugar = float(input("Enter your patient's blood sugar level: "))

    new_patient = pd.DataFrame(
    [[age, bmi, blood_sugar]],
    columns=['Age', 'BMI', 'Blood Sugar']
    )

    new_patient_imputed = imputer.transform(new_patient)

    new_patient_scaled = scaler.transform(new_patient_imputed)

    new_patient_scaled_df = pd.DataFrame(
        new_patient_scaled,
        columns=['Age', 'BMI', 'Blood Sugar']
    )

    prediction = optimal_model.predict(new_patient_scaled_df)
    prediction = round(prediction[0], 2)

    time.sleep(5)

    print(f"Your predicted health risk score is: {prediction:.2f}")

    probability = logistic_model.predict_proba(new_patient_scaled_df)[0][1]

    print(f"Your patient's health risk probability is: {probability * 100:.2f}%")

    if prediction < 60:
        print("Your patient is HEALTHY.")
    else:
        print("Your patient is AT RISK.")

if __name__ == "__main__":
    main()