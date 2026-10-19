"""
Created by Mark Chmielecki for CS6313 - Foundations of Artificial Intelligence
Project #6 due Monday, October 12, 2026.   

Project requirements:
1) Data Generation & Persistence - script must generate 500 synthetic patient records and save them as classifier_patient_data.csv in Current Working Directory. 
2) Scaling and Preprocessing - used essentially same function as project 5, and used on all three models (only required on k-NN)
3) Training - train 3 classification models on training data using functions from tree, ensemple, and neighbors in sklearn
4) Terminal Interference and Accuracy - report accuracy scores in terminal
5) Visualization - I added a 2*3 grid, including feature importance and tree structure for each model.  Since 
6) Interactive Terminal Interference - prompt user, run through each model and vote on "Healthy" or "At Risk" 

UPDATED 10/15: moved splitting to before preprossing in main(), updated preprocessing function to take into account training and test date.
I got this wrong in Project 5, go the okay to resubmit per conversation with Maggie M on 10/15.  This was the only change in the code.
"""
import numpy as np                                                      # need for synthetic data generation and calculations
import pandas as pd                                                     # needed for data manipulation and saving to CSV
import time                                                             # add time to processing time to make it realistic

from sklearn.impute import SimpleImputer                                # needed for handling missing values in the data
from sklearn.preprocessing import StandardScaler, PolynomialFeatures    # needed for scaling the features, PolynomialFeatures used in Project 5 no longer needed in Project 6
#from sklearn.linear_model import LinearRegression, LogisticRegression  # from Project 5, not needed for Project 6
from sklearn.pipeline import make_pipeline                              # needed for creating a pipeline for the overfit model
from sklearn.metrics import confusion_matrix                            # needed for viaual on the k-NN model
from sklearn.model_selection import train_test_split                    # needed for splitting the data into training and testing sets
from sklearn.tree import DecisionTreeClassifier, plot_tree              # needed for training the decision tree model and for visualization of the decision tree
from sklearn.ensemble import RandomForestClassifier                     # needed for training the random forest model
from sklearn.neighbors import KNeighborsClassifier                      # needed for training the k-NN model   
from sklearn.metrics import accuracy_score                              # needed for evaluating the models    
from sklearn.inspection import permutation_importance                   # needed for calculating feature importance for the k-NN model
import matplotlib.pyplot as plt                                         # needed for visualizing the data and model performance
import seaborn as sns                                                   # needed for visualizing the data and model performance        

def create_synthetic_data(num_rows):
    """Function to generate synthetic health data with age, BMI, and blood sugar level, part of requirement #1."""
    # Make results reproducible
    np.random.seed(42)         # uses a fixed seed for the random number generator to ensure that the same synthetic data is generated each time the program is run, 42 is generally accepted as the answer to life, the universe, and everything, so it is a good choice for a seed value (Hitchhiker's Guide to the Galaxy).

    # Generate synthetic data
    age = np.random.randint(18, 90, num_rows)
    bmi = np.random.normal(26, 6, num_rows)
    blood_sugar = np.random.normal(105, 25, num_rows)

    return age, bmi, blood_sugar

def calculate_diagnosis(age, bmi, blood_sugar):
    """Calculate diagnosis labels and return patient data as a DataFrame.  Part of requirement #1
    0 = HEALTHY, 1 = AT RISK.
    """
    risk_logic = ((age * 0.3) + (bmi * 1.2) + (blood_sugar * 0.2))

    diagnosis = (risk_logic > 75).astype(int)

    # Return the complete DataFrame for saving to CSV
    data = pd.DataFrame({'Age': age,'BMI': bmi,'Blood Sugar': blood_sugar,'Diagnosis': diagnosis})

    return data

def save_data_to_csv(data, filename):
    """Generic function to save the data to a CSV file."""
    data.to_csv(filename, index=False)

def load_data_from_csv(filename):
    """Generic function to load the data from a CSV file."""
    return pd.read_csv(filename)

def preprocess_data(X_train, X_test):
    """Function to preprocess the data by filling in missing values and scaling the features, part of requirement #2.
       Updated on 10/15 to input and scale after splitting, preventing data leakage. 
    """
    # STEP 1: Create the imputer
    imputer = SimpleImputer(strategy="median")

    # STEP 2: Learn the medians from training data only
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_imputed_df = pd.DataFrame(X_train_imputed,columns=X_train.columns,index=X_train.index)

    # STEP 3: Use those same medians on testing data
    X_test_imputed = imputer.transform(X_test)
    X_test_imputed_df = pd.DataFrame(X_test_imputed,columns=X_test.columns,index=X_test.index)

    # STEP 4: Create the scaler
    scaler = StandardScaler()

    # STEP 5: Learn the scaling parameters from training data only
    X_train_scaled = scaler.fit_transform(X_train_imputed_df)

    # STEP 6: Apply the same scaling parameters to testing data
    X_test_scaled = scaler.transform(X_test_imputed_df)

    # STEP 7: Convert results back to DataFrames with feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled,columns=X_train.columns,index=X_train.index)

    X_test_scaled_df = pd.DataFrame(X_test_scaled,columns=X_test.columns,index=X_test.index)

    # Return the transformed data and fitted preprocessing objects
    return X_train_scaled_df, X_test_scaled_df, scaler, imputer

def split_data(data, test_size=0.2):
    """Function to split the data into training and testing sets, required before you move to requirement #3, training."""
    from sklearn.model_selection import train_test_split

    # Split the data into features and target variable
    X = data[['Age', 'BMI', 'Blood Sugar']]
    y = data['Diagnosis']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

def train_decision_tree_classifier_model(X_train, y_train):
    """Function to train a decision tree classifier on the training data, part of requirement #3."""
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest_classifier_model(X_train, y_train):
    """Function to train a random forest classifier on the training data, part of requirement #3."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_knn_classifier_model(X_train, y_train):
    """Function to train a k-NN classifier on the training data, part of requirement #3."""
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

def report_accuracy_score(model, X_test, y_test):
    """Function to report the accuracy score of a model on the test data, part of requirement #4."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def decision_tree_classifier_model_predict(model, new_patient):
    """Function to make a prediction using the decision tree model, requirement #6."""
    return model.predict(new_patient)

def random_forest_classifier_model_predict(model, new_patient):
    """Function to make a prediction using the random forest model, requirement #6."""
    return model.predict(new_patient)

def knn_classifier_model_predict(model, new_patient):
    """Function to make a prediction using the k-NN model, requirement #6."""
    return model.predict(new_patient)    

"""UI functions called from main() are listed below"""
def get_how_many_synthetic_records():
    """Gathers from user how may synthetic data rows are needed, passes back to main function"""
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
    return rows

def main():
    """
    Main execution function, drives the user experience.  I significanlty reduced the terminal prints from Project 5, to stay on course
    for Project 6 requirements.   
    """
    # Prompt the user for the number of rows of synthetic data to generate - Requirement 1
    rows = get_how_many_synthetic_records()                                                         
    
    # Create raw synthetic data - Requirement 1
    age, bmi, blood_sugar = create_synthetic_data(rows)
   
    # Calculate diagnosis on synthetic data, and save to .csv - Requirement 1
    save_data_to_csv(calculate_diagnosis(age, bmi, blood_sugar), 'classifier_patient_data.csv')
    # load unscaled data frame with raw data - Requirement 1
    health_data_frame = load_data_from_csv('classifier_patient_data.csv')
 
    # Split the data BEFORE preprocessing (change on 10/15 per conversation with Maggie M)
    X_train, X_test, y_train, y_test = split_data(health_data_frame)

    # Preprocess after splitting (change on 10/15 per conversation with Maggie M)
    X_train, X_test, scaler, imputer = preprocess_data(X_train, X_test)
    
    # Train 3 models, report accururacy of each - Requirements 3 and 4
    decision_tree_model = train_decision_tree_classifier_model(X_train, y_train)
    accuracy_decision_tree_model = report_accuracy_score(decision_tree_model, X_test, y_test)     

    random_forest_model = train_random_forest_classifier_model(X_train, y_train)
    accuracy_random_forest_model = report_accuracy_score(random_forest_model, X_test, y_test)
    
    knn_model = train_knn_classifier_model(X_train, y_train)
    accuracy_knn_model = report_accuracy_score(knn_model, X_test, y_test)

    # Report accuracy in table format per the requirements
    accuracy_table = pd.DataFrame({"Classifier Model": [
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "k-Nearest Neighbors (k-NN)"],
        "Accuracy": [
            accuracy_decision_tree_model,
            accuracy_random_forest_model,
            accuracy_knn_model]
    })
    print("CLASSIFIER MODEL ACCURACY COMPARISON")
    print(accuracy_table.to_string(index=False))

    # Next section of code is building the brain reveal for the Decision Tree Classifier model, requirement #5.  
    plt.figure(figsize=(12, 12)) # building a 6 plot dashboard window for the three models, part of requirement #4.
      
    """top left box of the dashboard window"""
    plt.subplot(3, 2, 1)
    plt.title("Feature Importance - Decision Tree Classifier Model")
    importance = pd.DataFrame({"Feature": X_train.columns,"Importance": decision_tree_model.feature_importances_})
    sns.barplot(data=importance,x= "Importance", y="Feature")      
    
    """top right box of the dashboard window"""
    plt.subplot(3, 2, 2)
    plt.title("Decision Tree Rules - Decision Tree  Classifier Model", fontsize=10)  
    plot_tree(decision_tree_model,feature_names=X_train.columns,class_names=["Healthy", "At Risk"],filled=True,rounded=True)

    """middle left box of the dashboard window"""
    plt.subplot(3, 2, 3)
    plt.title("Feature Importance - Random Forest Classifier Model")
    importance = pd.DataFrame({"Feature": X_train.columns,"Importance": random_forest_model.feature_importances_})
    sns.barplot(data=importance,x= "Importance", y="Feature") 

    """middle right box of the dashboard window"""
    plt.subplot(3, 2, 4)
    plt.title("Decision Tree Rules - Random Forest Classifier Model", fontsize=10)
    # I only give you the first random tree due to space concerns, at least on my monitor.
    plot_tree(random_forest_model.estimators_[0],feature_names=X_train.columns,class_names=["Healthy", "At Risk"],filled=True,rounded=True)

    """bottom left box of the dashboard window"""
    plt.subplot(3, 2, 5)
    plt.title("Feature Importance - k-NN Classifier Model")
    # Note - I had to research this method, its different because feature_importances since k-NN does not have feature importances built in.
    perm_results = permutation_importance(knn_model,X_test,y_test,n_repeats=10,random_state=42,scoring="accuracy") # calculate permutation
    importance = pd.DataFrame({"Feature": X_test.columns,"Importance": perm_results.importances_mean})
    importance = importance.sort_values(by="Importance",ascending=False) # Sort features from most important to least important
    sns.barplot(data=importance,x="Importance",y="Feature")

    """bottom right box of the dashboard window"""
    plt.subplot(3, 2, 6)
    plt.title("Confusion Matrix - k-NN")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    knn_predictions = knn_model.predict(X_test)
    # Same note as before, I had to research what a confusion matrix was because k-NN does not use a decision tree.
    # My confusion matrix shows correctly predicted on the downward diagnal, and errors on the upward angle.
    # I thought it was a good alternative to a visual tree structure for the other models,
    cm = confusion_matrix(y_test,knn_predictions,labels=[0, 1])
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Healthy", "At Risk"],yticklabels=["Healthy", "At Risk"])

    # Printing the plot
    plt.subplots_adjust(hspace=0.5,wspace=0.3) #horizontal and vertical spacing between subplots
    plt.show()
    #End of Brain Reveal - Requirement #5
        
    #Patient Evaluation - Requrirement #6
    print("Provide me your patient's age, BMI, and blood sugar level to predict your health risk score.")

    age = float(input("Enter your patient's age: "))
    bmi = float(input("Enter your patient's BMI: "))
    blood_sugar = float(input("Enter your patient's blood sugar level: "))

    feature_columns = ['Age', 'BMI', 'Blood Sugar']

    new_patient = pd.DataFrame(
        [[age, bmi, blood_sugar]],columns=feature_columns)   

    # Use the same imputer and scaler fitted on the training data.
    new_patient_imputed = imputer.transform(new_patient)
    new_patient_imputed_df = pd.DataFrame(new_patient_imputed,columns=feature_columns)
    new_patient_scaled = scaler.transform(new_patient_imputed_df)
    new_patient_scaled_df = pd.DataFrame(new_patient_scaled,columns=feature_columns)

    # Predict for all three models (a bit more elegant than Project #5).
    models = [("Decision Tree Classifier", decision_tree_model),("Random Forest Classifier", random_forest_model),("k-Nearest Neighbors (k-NN)", knn_model)]

    for name, model in models:
        prediction = model.predict(new_patient_scaled_df)[0]
        if prediction == 0:
            print(f"Your patient is HEALTHY according to: {name}")
        else:
            print(f"Your patient is AT RISK according to: {name}")   
    
if __name__ == "__main__":
    main()