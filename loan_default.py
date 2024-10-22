import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# 1. Load and Preprocess the Dataset
# Here we load a sample of 70% of the dataset to speed up processing time
data = pd.read_csv('Loan_default.csv')
data = data.sample(frac=0.7, random_state=42)

# Selecting the relevant features for modeling and the target variable (Default status)
features = data[['Age', 'Income', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']]
target = data['Default']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Standardizing the features to improve model performance (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality reduction using PCA to reduce computational complexity
# We reduce the features to 3 principal components for simplicity
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 2. Define the machine learning models to be evaluated

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=50, n_jobs=-1)

# XGBoost Classifier
xgb_model = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')

# Support Vector Machine (SVM) with balanced class weights (due to possible imbalance in data)
svm_model = SVC(kernel='linear', class_weight='balanced', max_iter=1000)

# Neural Network (MLP Classifier) with one hidden layer of 50 neurons
nn_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)

# Naive Bayes classifier
nb_model = GaussianNB()

# Logistic Regression with balanced class weights
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')

# 3. Function to train and evaluate each model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Train the model and evaluate its performance on test data.
    Returns accuracy, precision, recall, F1-score, and confusion matrix.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return accuracy, precision, recall, f1, confusion_matrix(y_test, y_pred)

# Initialize lists to store model evaluation results
models = [rf_model, xgb_model, svm_model, nn_model, nb_model, lr_model]
model_names = ['Random Forest', 'XGBoost', 'SVM', 'Neural Network', 'Naive Bayes', 'Logistic Regression']
results = []

# Evaluate each model and collect the results
for model, name in zip(models, model_names):
    accuracy, precision, recall, f1, cm = evaluate_model(model, X_train_pca, X_test_pca, y_train, y_test)
    results.append((name, accuracy, precision, recall, f1, cm))

# Convert the results into a DataFrame for easier visualization and comparison
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Confusion Matrix'])

# 4. Plot the Confusion Matrices in a 2x3 grid for easier comparison

fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Create a 2x3 grid for plots
axes = axes.ravel()  # Flatten the axes array for easy iteration

# Iterate over the models and plot confusion matrices
for i, (name, cm) in enumerate(zip(model_names, results_df['Confusion Matrix'])):
    cm_new = np.array([[cm[1, 1], cm[1, 0]],  # True Positives, False Negatives
                       [cm[0, 1], cm[0, 0]]])  # False Positives, True Negatives

    sns.heatmap(cm_new, annot=True, fmt='g', cmap='Blues', cbar=False, ax=axes[i],
                xticklabels=['Pred 1', 'Pred 0'], yticklabels=['Actual 1', 'Actual 0'])
    axes[i].set_title(f'Confusion Matrix for {name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 5. Plot the Comparison of Evaluation Metrics (Accuracy, Precision, Recall, F1-Score)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Create subplots for each metric comparison
plt.figure(figsize=(16, 12))
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    sns.barplot(x='Model', y=metric, data=results_df, palette='coolwarm', dodge=False)
    plt.title(f'Comparison of {metric} Across Models')
    plt.ylim(0, 1)  # Ensure all values are on the same scale for easy comparison
    plt.xlabel('Model')
    plt.ylabel(metric)

plt.tight_layout()
plt.show()

# 6. Print the performance metrics for easy reference
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']])

# 7. Identify the best model based on F1-Score
# F1-Score is chosen here as it balances precision and recall, which is crucial in imbalanced datasets
best_model = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
print(f"\nThe best model based on F1-Score is: {best_model}")
