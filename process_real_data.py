import pandas as pd
import pdfplumber
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import os

pdf_path = 'Hypertension.csv.pdf'
csv_path = 'Hypertension_Cleaned.csv'

print("Extracting data from PDF using pdfplumber...")
all_data = []
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        table = page.extract_table()
        if table:
            all_data.extend(table)

if not all_data:
    print("Could not extract any tables from the PDF.")
    exit(1)

# The first row is the header
headers = all_data[0]
data_rows = all_data[1:]

df = pd.DataFrame(data_rows, columns=headers)

# Clean Duplicates as requested
initial_rows = len(df)
df = df.drop_duplicates()
final_rows = len(df)
print(f"Removed {initial_rows - final_rows} duplicate rows. Total unique rows: {final_rows}")

# Clean column names (strip whitespace)
df.columns = df.columns.str.strip()

# Save cleaned data to CSV
df.to_csv(csv_path, index=False)
print(f"Saved cleaned data to {csv_path}")

# ==========================================
# MODEL TRAINING PREPARATION
# ==========================================
print("\nPreparing data for training...")

# Define expected features based on app.py
expected_features = [
    'gender', 'age', 'family_history', 'medication', 'symptom_severity',
    'shortness_of_breath', 'visual_changes', 'nosebleeds', 'sys_bp', 'dia_bp', 'diet_control'
]

# We need to map the PDF columns to our app's features
# PDF Columns: Gender, Age, History, Patient, TakeMedication, Severity, BreathShortness, VisualChanges, NoseBleeding, Whendiagnoused, Systolic, Diastolic, ControlledDiet, Stages
rename_map = {
    'Gender': 'gender',
    'C': 'gender', # PDF parsing sometimes misreads 'Gender' as 'C'
    'Age': 'age',
    'History': 'family_history',
    'TakeMedication': 'medication',
    'Severity': 'symptom_severity',
    'BreathShortness': 'shortness_of_breath',
    'VisualChanges': 'visual_changes',
    'NoseBleeding': 'nosebleeds',
    'Systolic': 'sys_bp',
    'd Systolic': 'sys_bp', # Fix for mangled systolic
    'Diastolic': 'dia_bp',
    'ControlledDiet': 'diet_control',
    'Stages': 'target'
}

# Rename columns if they match the map
df_renamed = df.rename(columns=lambda x: rename_map.get(x.strip(), x))

# Filter to keep only the columns we need plus target
columns_to_keep = [col for col in expected_features if col in df_renamed.columns]
if 'target' in df_renamed.columns:
    columns_to_keep.append('target')

df_filtered = df_renamed[columns_to_keep].copy()

# Print out which columns we found
print("Features found in dataset for training:", df_filtered.columns.tolist())

# Now we must ONE-HOT ENCODE categorical variables to match what app.py does (or label encode)
# In standard ML, we convert strings like "Male"/"Female" to numbers.
# We will use simple Label Encoding for demonstration since we don't know the exact preprocessing pipeline the user wants.
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder
import pandas as pd

encoders = {}
for col in df_filtered.columns:
    if not pd.api.types.is_numeric_dtype(df_filtered[col]) or df_filtered[col].dtype == 'object':
        le = LabelEncoder()
        # Convert all to string first to handle any NaNs or mixed types gracefully
        df_filtered[col] = df_filtered[col].astype(str)
        df_filtered[col] = le.fit_transform(df_filtered[col])
        encoders[col] = le

X = df_filtered.drop(columns=['target'])
y = df_filtered['target']

print(f"Training on shape X: {X.shape}, y: {y.shape}")

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Save
with open('logreg_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Also save the encoders so the flask app can use them if needed!
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("SUCCESS! Cleaned duplicates, extracted data, and saved new 'logreg_model.pkl' and 'scaler.pkl'.")
