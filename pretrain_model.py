import subprocess
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to install required packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure imbalanced-learn is installed
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    install("imbalanced-learn==0.8.0")
    from imblearn.over_sampling import SMOTE

# Function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    return df

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

train_file_path = 'model_train.csv'
model_train = load_data(train_file_path)
model_train = clean_column_names(model_train)

# Apply one-hot encoding and scale numerical features
categorical_columns = ['disrict', 'client_catg', 'region', 'tarif_type_elec_mode', 'tarif_type_gaz_mode']
model_train_encoded = pd.get_dummies(model_train, columns=categorical_columns, drop_first=True)
numerical_features = [col for col in model_train_encoded.columns if model_train_encoded[col].dtype in ['int64', 'float64'] and col != 'target']
scaler = StandardScaler()
model_train_encoded[numerical_features] = scaler.fit_transform(model_train_encoded[numerical_features])

# Define features and target
X = model_train_encoded.drop(['target', 'client_id'], axis=1)
y = model_train_encoded['target'].astype(int)

# Feature selection with Lasso
lasso = LassoCV()
lasso.fit(X, y)
selected_features = X.columns[(lasso.coef_ != 0)]

# Prepare the training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Calculate scale_pos_weight
num_neg = (y_train_resampled == 0).sum()
num_pos = (y_train_resampled == 1).sum()
scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1  # Ensure no division by zero

# Best parameters from the RandomizedSearchCV
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

# Train the LightGBM model and save training metrics
model = lgb.LGBMClassifier(random_state=42, **best_params)
model.fit(X_train_resampled, y_train_resampled, eval_set=[(X_test, y_test)], eval_metric='binary_logloss')

# Evaluate the model
y_pred = model.predict(X_test)
evaluation_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, average='weighted'),
    'Recall': recall_score(y_test, y_pred, average='weighted'),
    'F1-Score': f1_score(y_test, y_pred, average='weighted')
}

# Save the trained model
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Save the selected features
with open('selected_features.pkl', 'wb') as file:
    pickle.dump(selected_features.tolist(), file)

# Save evaluation metrics
with open('evaluation_metrics.json', 'w') as file:
    json.dump(evaluation_metrics, file)

print("Model training complete and saved.")
