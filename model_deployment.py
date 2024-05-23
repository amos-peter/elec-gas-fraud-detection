import os
import subprocess
import sys

# Function to install a package using pip
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure imbalanced-learn is installed
try:
    import imblearn
except ImportError:
    install("imbalanced-learn")
finally:
    from imblearn.over_sampling import SMOTE

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# Function to load data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to train XGBoost model
def train_model(X_train, y_train, scale_pos_weight):
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

# Streamlit interface
st.title("XGBoost Model Trainer and Fraud Predictor")

# Automatically load training data
train_file_path = 'model_train.csv'
train_data = load_data(train_file_path)
st.write("Training Data Loaded Successfully.")
st.write("Training Data Preview:", train_data.head())

# Automatically detect target and feature columns
target_column = 'target'
features = train_data.columns.difference([target_column, 'client_id']).tolist()

# Prepare data for training
X = train_data[features]
y = train_data[target_column]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes in the training set
if 'SMOTE' in locals():
    sm = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = y_train_resampled.value_counts()[0] / y_train_resampled.value_counts()[1]

    # Train the model
    model = train_model(X_train_resampled, y_train_resampled, scale_pos_weight)
    st.success("Model trained successfully.")
    st.write("You can now use this trained model for predictions.")

    # Evaluate the model using weighted average for metrics
    y_train_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_train_pred)
    precision = precision_score(y_test, y_train_pred, average='weighted')
    recall = recall_score(y_test, y_train_pred, average='weighted')
    f1 = f1_score(y_test, y_train_pred, average='weighted')

    # Display evaluation metrics
    st.subheader("Model Evaluation Metrics")
    evaluation_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [accuracy, precision, recall, f1]
    })
    st.write(evaluation_metrics)
else:
    st.error("SMOTE not available. Model training cannot proceed.")

# File uploader for testing data
uploaded_test_file = st.file_uploader("Choose a CSV file to predict", type="csv")
if uploaded_test_file is not None:
    test_data = load_data(uploaded_test_file)
    st.write("Test Data Preview:", test_data.head())

    # Assuming the test data contains the same features as the training data
    X_test = test_data[features]
    client_ids = test_data['client_id']

    # Make predictions
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.7).astype(int)

    # Create a DataFrame with client_id, prediction probabilities, and predicted values
    results = pd.DataFrame({
        'client_id': client_ids,
        'fraud_probability': y_pred_prob,
        'predicted_value': y_pred
    })

    st.write("Prediction Results:", results)
    st.download_button(
        label="Download Predictions",
        data=results.to_csv(index=False),
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

    # Visualization of prediction results
    st.subheader("Prediction Results Visualization")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    results['predicted_value'].value_counts().plot(kind='bar', ax=ax[0])
    ax[0].set_title("Count of Predicted Fraudulent and Non-Fraudulent Transactions")
    ax[0].set_xlabel("Predicted Value (1=Fraud, 0=Non-Fraud)")
    ax[0].set_ylabel("Count")

    sns.histplot(results['fraud_probability'], bins=20, kde=True, ax=ax[1])
    ax[1].set_title("Distribution of Fraud Probability Scores")
    ax[1].set_xlabel("Fraud Probability")
    ax[1].set_ylabel("Frequency")

    st.pyplot(fig)

    # Feature importance
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': features, 'importance': importance})
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr_matrix = test_data[features].corr()
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

    # Additional demographic visualizations
    st.subheader("Demographic Information Visualization")
    demography_column = st.selectbox("Select a demographic column to visualize", test_data.columns.difference(['client_id', target_column]))

    fig, ax = plt.subplots(figsize=(10, 6))
    test_data[demography_column].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f"Distribution of {demography_column}")
    ax.set_xlabel(demography_column)
    ax.set_ylabel("Count")
    st.pyplot(fig)
