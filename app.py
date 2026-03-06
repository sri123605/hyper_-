from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'logreg_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

def load_models():
    # Load the model and scaler if they exist
    model = None
    scaler = None
    encoders = None
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")

    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"Error loading scaler from {SCALER_PATH}: {e}")
        
    try:
        with open(os.path.join(BASE_DIR, 'encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
    except Exception as e:
        print(f"Error loading encoders from encoders.pkl: {e}")
        
    return model, scaler, encoders

def preprocess_input(form_data, scaler, encoders):
    """
    Preprocess the incoming form data to the format expected by the model.
    """
    try:
        feature_keys = [
            'gender', 'age', 'family_history', 'medication', 'symptom_severity',
            'shortness_of_breath', 'visual_changes', 'nosebleeds', 'sys_bp', 'dia_bp', 'diet_control'
        ]
        
        raw_features = []
        for key in feature_keys:
            val = form_data.get(key, '')
            val_str = str(val).strip()
            
            if encoders and key in encoders:
                try:
                    # Transform standard value
                    encoded_val = encoders[key].transform([val_str])[0]
                except ValueError:
                    # If the value wasn't in training data, default to 0
                    encoded_val = 0
                raw_features.append(encoded_val)
            else:
                try:
                    numeric_val = float(val) if val else 0.0
                except (ValueError, TypeError):
                    numeric_val = 0.0
                raw_features.append(numeric_val)
                
        features = np.array([raw_features])
        
        if scaler:
            features = scaler.transform(features)
            
        return features
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def get_recommendation(stage):
    recommendations = {
        0: "Your blood pressure is within the normal range. Continue maintaining a healthy lifestyle with a balanced diet and regular exercise.",
        1: "You are exhibiting signs of Stage-1 Hypertension. It's advisable to adopt lifestyle changes such as a low-sodium diet and increased physical activity. Consider scheduling a routine check-up.",
        2: "You are exhibiting signs of Stage-2 Hypertension. Seeking medical advice is strongly recommended. A healthcare professional may suggest medication alongside lifestyle modifications.",
        3: "CRITICAL: You are exhibiting signs of a Hypertensive Crisis. Please seek emergency medical assistance immediately."
    }
    return recommendations.get(stage, "Unable to provide a recommendation. Please consult a doctor.")

def get_stage_info(stage):
    info = {
        0: {'label': 'Normal', 'color': 'green', 'risk': 'Low'},
        1: {'label': 'Stage-1', 'color': 'yellow', 'risk': 'Moderate'},
        2: {'label': 'Stage-2', 'color': 'orange', 'risk': 'High'},
        3: {'label': 'Crisis', 'color': 'red', 'risk': 'Severe'}
    }
    return info.get(stage, {'label': 'Unknown', 'color': 'gray', 'risk': 'Unknown'})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model, scaler, encoders = load_models()
        
        if model is None:
            return render_template('index.html', error="Model not loaded. Please ensure 'logreg_model.pkl' is in the project directory.", form_data=request.form)
        
        features = preprocess_input(request.form, scaler, encoders)
        if features is None:
            return render_template('index.html', error="Error processing input. Please check your data.", form_data=request.form)
            
        try:
            # Predict
            prediction = model.predict(features)[0]
            stage = int(prediction)
            
            # Predict probabilities for confidence
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                confidence = round(max(probabilities) * 100, 2)
            else:
                confidence = "N/A"
            
            stage_info = get_stage_info(stage)
            recommendation = get_recommendation(stage)
            
            result = {
                'stage_label': stage_info['label'],
                'color': stage_info['color'],
                'confidence': confidence,
                'recommendation': recommendation
            }
            
            return render_template('index.html', result=result, form_data=request.form)
            
        except Exception as e:
            return render_template('index.html', error=f"Error making prediction: {e}", form_data=request.form)
            
    return render_template('index.html', form_data={})

if __name__ == '__main__':
    app.run(debug=True)
