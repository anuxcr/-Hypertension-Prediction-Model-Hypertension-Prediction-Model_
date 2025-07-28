# Hypertension-Prediction-Model :- Predictive analysis using Machine Learning 

#Hypertension Prediction Model : Predictive analysis using Machine Learning 
This repository contains a machine learning model designed to predict the likelihood of an individual having hypertension based on various health and lifestyle factors. The project utilizes a classification algorithm to analyze a provided dataset, offering insights into the key indicators of hypertension.

#Project Overview
Hypertension (high blood pressure) is a common condition that can lead to serious health problems. Early prediction and identification of risk factors are crucial for prevention and management. This project aims to build a robust machine learning model that can assist in predicting hypertension using a structured dataset. The model is trained, evaluated, and its performance is visualized to provide clear insights.

#Features
Data Preprocessing: Handles numerical scaling and one-hot encoding for categorical features.

Machine Learning Pipeline: Uses scikit-learn pipelines for streamlined data transformation and model training.

Random Forest Classifier: Implements a powerful ensemble learning method for classification.

Model Evaluation: Provides standard classification metrics including Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

Visualizations: Generates insightful plots for:

Confusion Matrix Heatmap

Receiver Operating Characteristic (ROC) Curve with AUC score

Feature Importances

Example Prediction: Demonstrates how to use the trained model to predict outcomes for new data.


Install the required libraries:
Create a requirements.txt file in your project root with the following content:

pandas
scikit-learn
matplotlib
seaborn
numpy

Then install them:

pip install -r requirements.txt

Usage
Place the Dataset:
Ensure the hypertension_dataset.csv file is in the same directory as the hypertension_predictor.py script.

Run the Script:
Execute the Python script from your terminal:

python hypertension_predictor.py

The script will:

Load and preprocess the data.

Train the machine learning model.

Print evaluation metrics to the console.

Display three plots (Confusion Matrix, ROC Curve, Feature Importances).

Show an example prediction for a sample data point.

#Dataset
The hypertension_dataset.csv file contains various features related to an individual's health and lifestyle, along with a target variable indicating whether they have hypertension.

Columns:

Age: Age of the individual.

Salt_Intake: Daily salt intake.

Stress_Score: A score indicating stress levels.

BP_History: History of blood pressure (e.g., Normal, Prehypertension, Hypertension).

Sleep_Duration: Average hours of sleep per night.

BMI: Body Mass Index.

Medication: Current medication (e.g., None, ACE Inhibitor, Diuretic, Beta Blocker, Other).

Family_History: Indicates if there's a family history of hypertension (Yes/No).

Exercise_Level: Level of physical exercise (e.g., Low, Moderate, High).

Smoking_Status: Smoking habits (e.g., Smoker, Non-Smoker).

Has_Hypertension: Target variable, indicating if the individual has hypertension (Yes/No).

#Model
The model used is a RandomForestClassifier from scikit-learn. It's chosen for its ability to handle various data types, its robustness to overfitting, and its capacity to provide feature importances. The model is integrated into a Pipeline to ensure consistent preprocessing.

#Results and Visualizations
Upon running the script, you will see console output detailing the model's accuracy, a classification report, and a confusion matrix. Additionally, the following plots will be generated:

#Confusion Matrix Heatmap: Helps visualize the performance of a classification model, showing true positives, true negatives, false positives, and false negatives.

#ROC Curve: Illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. The Area Under the Curve (AUC) provides a single measure of overall performance.

#Feature Importances Bar Chart: Shows which features contributed most significantly to the model's predictions, helping to understand the underlying factors influencing hypertension.
