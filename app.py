from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = Flask(__name__, template_folder='templates',static_folder='assets')
CORS(app)

# Configuration
app.config.update(
    MAX_CONTENT_LENGTH=32 * 1024 * 1024,
    UPLOAD_FOLDER='uploads',
    RESULTS_FOLDER='results',
    ALLOWED_EXTENSIONS={'csv'},
    SECRET_KEY=os.getenv('SECRET_KEY', 'default-secret-key')
)

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Define features and label mappings
LABEL_MAP = {
    'label': ['Anomaly', 'Normal'],
    'category': ['DoS', 'MITM ARP Spoofing', 'Mirai', 'Normal', 'Scan'],
    'subcategory': [
        "DoS-Synflooding", "MITM ARP Spoofing", "Mirai-Hostbruteforceg",
        "Normal", "Scan Port OS", "Scan Hostport", "Mirai-HTTP Flooding",
        "Mirai-Ackflooding", "Mirai-UDP Flooding"
    ]
}

MODEL_FEATURES = [
    'Src_Port', 'Dst_Port', 'Protocol', 'Flow_IAT_Mean', 'Flow_IAT_Min',
    'Bwd_IAT_Mean', 'Bwd_IAT_Min', 'Pkt_Len_Max', 'SYN_Flag_Cnt', 'ACK_Flag_Cnt'
]

def validate_input_data(df):
    """Validate and preprocess input data"""
    if df.empty:
        return pd.DataFrame(columns=MODEL_FEATURES)

    df_processed = df.copy()
    column_mapping = {
        'src_port': 'Src_Port', 'dst_port': 'Dst_Port', 'protocol': 'Protocol',
        'flowiatmean': 'Flow_IAT_Mean', 'flowiatmin': 'Flow_IAT_Min',
        'bwdiatmean': 'Bwd_IAT_Mean', 'bwdiatmin': 'Bwd_IAT_Min',
        'pktlenmax': 'Pkt_Len_Max', 'synflagcnt': 'SYN_Flag_Cnt',
        'ackflagcnt': 'ACK_Flag_Cnt'
    }
    
    df_processed = df_processed.rename(columns=column_mapping)
    for col in MODEL_FEATURES:
        if col not in df_processed.columns:
            df_processed[col] = 0.0
    df_processed[MODEL_FEATURES] = df_processed[MODEL_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return df_processed[MODEL_FEATURES]

def calculate_basic_stats(df):
    """Calculate basic statistics for numeric features"""
    stats = {}
    for feature in MODEL_FEATURES:
        stats[feature] = {
            'mean': float(df[feature].mean()),
            'min': float(df[feature].min()),
            'max': float(df[feature].max()),
            'median': float(df[feature].median())
        }
    return stats

def calculate_feature_distributions(df):
    """Calculate feature distributions for visualization"""
    return [{'feature': feature, 'data': df[feature].tolist()} for feature in MODEL_FEATURES]

def calculate_correlations(df):
    """Calculate correlation matrix for features"""
    corr_matrix = df[MODEL_FEATURES].corr().replace([np.inf, -np.inf], np.nan).fillna(0)
    return corr_matrix.values.tolist()

class ModelManager:
    def __init__(self):
        self.models = {'label': None, 'category': None, 'subcategory': None}
        self.load_models()

    def load_models(self):
        model_paths = {
            'label': 'Models/Label_Models/randomforest_label.pkl',
            'category': 'Models/Cat_Models/randomforest_Cat.pkl',
            'subcategory': 'Models/Sub_Cat_Models/randomforest_Sub_Cat.pkl'
        }
        
        for key, path in model_paths.items():
            try:
                if os.path.exists(path):
                    self.models[key] = joblib.load(path)
                    logging.info(f"Loaded model: {key}")
                else:
                    logging.warning(f"Model file not found: {path}")
            except Exception as e:
                logging.error(f"Error loading model '{key}': {str(e)}")

    def predict_with_confidence(self, X):
        predictions = {}
        for pred_type, model in self.models.items():
            if model is not None:
                try:
                    pred = model.predict(X)
                    proba = model.predict_proba(X)
                    confidence = np.max(proba, axis=1)
                    predictions[pred_type] = {
                        'predictions': [
                            {'label': LABEL_MAP[pred_type][p], 'confidence': float(c)}
                            for p, c in zip(pred, confidence)
                        ]
                    }
                except Exception as e:
                    logging.error(f"Prediction error for {pred_type}: {str(e)}")
        return predictions

model_manager = ModelManager()

@app.route('/')
def index():
    return render_template('index.html', MODEL_FEATURES=MODEL_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files and request.files['file'].filename:
            df = pd.read_csv(request.files['file'])
        else:
            form_data = request.form.to_dict()
            input_data = {k: [float(v)] for k, v in form_data.items() if v}
            df = pd.DataFrame(input_data)

        processed_df = validate_input_data(df)
        if processed_df.empty:
            return jsonify({'success': False, 'error': 'Invalid input data'}), 400

        predictions = model_manager.predict_with_confidence(processed_df)
        
        # Include input data in response
        input_data = {col: processed_df[col].tolist() for col in processed_df.columns}
        
        data_analysis = {
            'basic_stats': calculate_basic_stats(processed_df),
            'feature_distributions': calculate_feature_distributions(processed_df),
            'correlation_matrix': calculate_correlations(processed_df)
        }

        return jsonify({
            'success': True,
            'predictions': predictions,
            'input_data': input_data,
            'data_analysis': data_analysis
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)