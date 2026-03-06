import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# Create dummy training data
# 11 features: gender, age, family_history, medication, symptom_severity, shortness_of_breath, visual_changes, nosebleeds, sys_bp, dia_bp, diet_control
X = np.random.randint(0, 4, size=(100, 11))
# Target: 0 (Normal), 1 (Stage-1), 2 (Stage-2), 3 (Crisis)
y = np.random.randint(0, 4, size=(100,))

# Train a scaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train a dummy logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Save the model and scaler
with open('logreg_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Dummy 'logreg_model.pkl' and 'scaler.pkl' created successfully!")
