import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import lightgbm as lgb
import streamlit as st

# Function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    return df

# Function to load data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to train LightGBM model
def train_model(X_train, y_train, params):
    model = lgb.LGBMClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    return model

# Streamlit interface
st.title("Fraud Detection in Electricity and Gas Consumption")

# Sidebar with description and download links
st.sidebar.header("About")
st.sidebar.write("""
This app trains a LightGBM model to predict fraud based on the provided dataset. `model_train` is used to train the model, and `model_test` is used to evaluate the model and make predictions.  
You can download the example datasets using the links below. 
""")
st.sidebar.download_button(
    label="Download model_train.csv",
    data=open("model_train.csv", "rb").read(),
    file_name="model_train.csv",
    mime="text/csv"
)
st.sidebar.download_button(
    label="Download model_test.csv",
    data=open("model_test.csv", "rb").read(),
    file_name="model_test.csv",
    mime="text/csv"
)

# Automatically load training data
train_file_path = 'model_train.csv'
model_train = load_data(train_file_path)
st.subheader("Training Data Loaded Successfully.")
st.write("Training Data Preview:")
st.write(model_train.head())

# List of categorical columns that need encoding
categorical_columns = [
    'disrict', 'client_catg', 'region', 'tarif_type_elec_mode', 'tarif_type_gaz_mode'
]

# Ensure the categorical columns exist in the DataFrame
existing_categorical_columns = [col for col in categorical_columns if col in model_train.columns]

# Apply one-hot encoding to the categorical columns
model_train_encoded = pd.get_dummies(model_train, columns=existing_categorical_columns, drop_first=True)

# Clean column names
model_train_encoded = clean_column_names(model_train_encoded)

# Convert boolean columns to integers
for col in model_train_encoded.columns:
    if model_train_encoded[col].dtype == 'bool':
        model_train_encoded[col] = model_train_encoded[col].astype(int)

# List of numerical features to be scaled
numerical_features = [
    'counter_statue_elec_mode', 'counter_statue_gaz_mode', 'reading_remarque_elec_mode', 'reading_remarque_gaz_mode',
    'counter_coefficient_elec_mode', 'counter_coefficient_gaz_mode', 'invoice_date_elec_count', 'invoice_date_gaz_count',
    'mth_avg_index_diff_elec_mean', 'mth_avg_index_diff_gaz_mean'
]

# Scale numerical features
scaler = StandardScaler()
model_train_encoded[numerical_features] = scaler.fit_transform(model_train_encoded[numerical_features])

# Display the processed data
st.subheader("Data after Encoding and Scaling")
st.write(model_train_encoded.head())

# Define features and target
X = model_train_encoded.drop(['target', 'client_id'], axis=1)
y = model_train_encoded['target']

# Apply Lasso for feature selection
lasso = LassoCV()
lasso.fit(X, y)

# Select features with non-zero coefficients
selected_features_lasso_0 = X.columns[(lasso.coef_ != 0)]
st.subheader("Selected Features after Lasso")
st.write(selected_features_lasso_0)

# Prepare the training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X[selected_features_lasso_0], y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution
st.subheader("Class distribution after SMOTE:")
class_distribution = pd.Series(y_train_resampled).value_counts().reset_index()
class_distribution.columns = ['target', 'count']
st.write(class_distribution)

# Calculate scale_pos_weight
scale_pos_weight = y_train_resampled.value_counts()[0] / y_train_resampled.value_counts()[1]

# Best parameters found from the RandomizedSearchCV
best_params = {
    'subsample': 1.0,
    'scale_pos_weight': scale_pos_weight,
    'num_leaves': 15,
    'n_estimators': 200,
    'min_child_samples': 50,
    'max_depth': 7,
    'learning_rate': 0.2,
    'force_row_wise': True,
    'colsample_bytree': 0.6
}

# Train the model
best_lgb_model = train_model(X_train_resampled, y_train_resampled, best_params)
st.success("Model trained successfully.")
st.write("You can now use this trained model for predictions.")

# Evaluate the model using weighted average for metrics
lgb_y_pred = best_lgb_model.predict(X_test)
accuracy = accuracy_score(y_test, lgb_y_pred)
precision = precision_score(y_test, lgb_y_pred, average='weighted')
recall = recall_score(y_test, lgb_y_pred, average='weighted')
f1 = f1_score(y_test, lgb_y_pred, average='weighted')

# Display evaluation metrics
st.subheader("Model Evaluation Metrics")
evaluation_metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Score': [accuracy, precision, recall, f1]
})
st.write(evaluation_metrics)

# Feature importance
st.subheader("Feature Importance")
importance = best_lgb_model.feature_importances_
feature_importance = pd.DataFrame({'feature': selected_features_lasso_0, 'importance': importance})
feature_importance = feature_importance.sort_values(by='importance', ascending=False).head(10)  # Show top 10 features

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
ax.set_title("Top 10 Feature Importance")
st.pyplot(fig)

# File uploader for testing data
uploaded_test_file = st.sidebar.file_uploader("Choose a CSV file for prediction", type="csv")
if uploaded_test_file is not None:
    test_data = load_data(uploaded_test_file)
    st.write("Test Data Preview:", test_data.head())

    # Ensure the categorical columns exist in the test DataFrame
    existing_test_categorical_columns = [col for col in categorical_columns if col in test_data.columns]

    # Apply the same preprocessing steps to the test data
    test_data_encoded = pd.get_dummies(test_data, columns=existing_test_categorical_columns, drop_first=True)
    test_data_encoded = clean_column_names(test_data_encoded)
    for col in test_data_encoded.columns:
        if test_data_encoded[col].dtype == 'bool':
            test_data_encoded[col] = test_data_encoded[col].astype(int)
    test_data_encoded[numerical_features] = scaler.transform(test_data_encoded[numerical_features])

    # Select the same features as used in training
    X_test_final = test_data_encoded[selected_features_lasso_0]
    client_ids = test_data['client_id']

    # Make predictions
    y_pred_prob = best_lgb_model.predict_proba(X_test_final)[:, 1]
    y_pred_prob = np.round(y_pred_prob, 1)  # Format to 1 decimal place
    y_pred = (y_pred_prob >= 0.6).astype(int)  # Threshold set to 0.60 for fraud

    # Create a DataFrame with client_id, prediction probabilities, and predicted values
    results = pd.DataFrame({
        'client_id': client_ids,
        'fraud_probability': y_pred_prob,
        'predicted_value': y_pred
    })

    st.write("Prediction Results:")
    st.write(results)
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

    # Feature importance for test prediction
    st.subheader("Feature Importance for Test Prediction")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
    ax.set_title("Top 10 Feature Importance for Test Prediction")
    st.pyplot(fig)

    # Correlation matrix for top 10 features
    st.subheader("Correlation Matrix for Top 10 Features")
    top_10_features = feature_importance['feature']
    corr_matrix = test_data_encoded[top_10_features].corr()
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix for Top 10 Features")
    st.pyplot(fig)
