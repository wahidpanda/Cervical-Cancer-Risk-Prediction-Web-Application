from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model pipeline
MODEL_PATH = 'D:\Pendrive Data\Research Work\cervical cancer\web\model\cervical_cancer_model_20250413_142242.pkl' 
with open(MODEL_PATH, 'rb') as f:
    pipeline = pickle.load(f)

@app.route('/')
def home():
    """Render the main page with the prediction form"""
    return render_template('index.html', features=pipeline['metadata']['selected_features'])

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from form and create DataFrame with correct feature order
        data = request.form.to_dict()
        
        # Create input DataFrame with all selected features
        input_data = pd.DataFrame(columns=pipeline['metadata']['selected_features'])
        for feature in pipeline['metadata']['selected_features']:
            input_data[feature] = [float(data.get(feature, 0))]
        
        # Preprocessing pipeline
        scaled_data = pipeline['scaler'].transform(input_data.values)
        ica_data = pipeline['ica'].transform(scaled_data)
        pca_data = pipeline['pca'].transform(ica_data)
        
        # Get base model predictions
        base_probas = np.column_stack([
            model.predict_proba(pca_data)[:, 1] 
            for model in pipeline['bayesian_fusion']['base_models']
        ])
        
        # Get final prediction
        prediction = pipeline['bayesian_fusion']['model'].predict(base_probas)
        probability = pipeline['bayesian_fusion']['model'].predict_proba(base_probas)[:, 1][0]
        
        # Prepare response
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'risk_level': 'High' if prediction[0] == 1 else 'Low',
            'message': 'Further medical consultation recommended.' if prediction[0] == 1 
                      else 'No significant risk detected, but regular checkups are advised.'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)