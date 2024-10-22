**Loan Default Prediction Model Evaluation**

This project evaluates different machine learning models for predicting loan defaults using a dataset with several features related to loan applicants. The models compared include Random Forest, XGBoost, Support Vector Machine (SVM), Neural Network, Naive Bayes, and Logistic Regression.

**Project Overview**

This project aims to:

1. Preprocess the dataset: Standardizing the features and applying PCA for dimensionality reduction.

2. Train multiple models: Train and compare six different models:
   
   -Random Forest
   
   -XGBoost
   
   -Support Vector Machine (SVM)
   
   -Neural Network (MLP)
   
   -Naive Bayes
   
   -Logistic Regression

4. Evaluate model performance: Metrics like Accuracy, Precision, Recall, F1-Score, and Confusion Matrix are used to evaluate model performance.
Display results: Results are visualized using confusion matrix heatmaps and bar charts for performance comparison.

**Files used**

Loan_default.csv: Dataset containing loan applicant information.
model_evaluation.py: The Python script for model training, evaluation, and visualization.

**Requirements**

To run this project, you need to have the following Python libraries installed:

pandas

numpy

scikit-learn

xgboost

matplotlib

seaborn

**How to run:**

Clone the repository:

git clone https://github.com/your-username/loan-default-prediction

cd loan-default-prediction

Add the dataset (Loan_default.csv) to the working directory.

Run the Python script to evaluate the models:

python model_evaluation.py

View the results of the model evaluation, including confusion matrices and bar plots comparing metrics like Accuracy, Precision, Recall, and F1-Score.

**Model Performance**

After running the evaluation, the best model based on the F1-Score will be displayed. All models' performance will be shown using bar plots and confusion matrix heatmaps in a 2x3 grid.

**Contributions**

Shreya Soundharya, Dhanush, Harshathan, Rhithik Rhagav

