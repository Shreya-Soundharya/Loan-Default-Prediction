#Loan Default Prediction Model Evaluation
This project evaluates different machine learning models for predicting loan defaults using a dataset with several features related to loan applicants. The models compared include Random Forest, XGBoost, Support Vector Machine (SVM), Neural Network, Naive Bayes, and Logistic Regression.

Project Overview
This project aims to:

Preprocess the dataset: Standardizing the features and applying PCA for dimensionality reduction.
Train multiple models: Train and compare six different models:
Random Forest
XGBoost
Support Vector Machine (SVM)
Neural Network (MLP)
Naive Bayes
Logistic Regression
Evaluate model performance: Metrics like Accuracy, Precision, Recall, F1-Score, and Confusion Matrix are used to evaluate model performance.
Display results: Results are visualized using confusion matrix heatmaps and bar charts for performance comparison.
Files
Loan_default.csv: Dataset containing loan applicant information.
model_evaluation.py: The Python script for model training, evaluation, and visualization.
README.md: Explanation of the project.
Requirements
To run this project, you need to have the following Python libraries installed:

pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
You can install these packages using the following command:

bash
Copy code
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
Instructions
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/loan-default-prediction
cd loan-default-prediction
Add the dataset (Loan_default.csv) to the working directory.

Run the Python script to evaluate the models:

bash
Copy code
python model_evaluation.py
View the results of the model evaluation, including confusion matrices and bar plots comparing metrics like Accuracy, Precision, Recall, and F1-Score.

Model Performance
After running the evaluation, the best model based on the F1-Score will be displayed. All models' performance will be shown using bar plots and confusion matrix heatmaps in a 2x3 grid.

Contribution
Feel free to contribute by submitting a pull request or reporting any issues.

License
This project is licensed under the MIT License - see the LICENSE file for details.
