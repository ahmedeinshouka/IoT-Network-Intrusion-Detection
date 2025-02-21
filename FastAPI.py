import os
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
from typing import Optional, Dict, List
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = FastAPI()

# Templates (no static mounting for templates directory)
templates = Jinja2Templates(directory="templates")

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

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

# Pydantic model for manual input
class ManualInput(BaseModel):
    src_port: float
    dst_port: float
    protocol: float
    flowiatmean: float
    flowiatmin: float
    bwdiatmean: float
    bwdiatmin: float
    pktlenmax: float
    synflagcnt: float
    ackflagcnt: float

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
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

def calculate_basic_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    for feature in MODEL_FEATURES:
        stats[feature] = {
            'mean': float(df[feature].mean()),
            'min': float(df[feature].min()),
            'max': float(df[feature].max()),
            'median': float(df[feature].median())
        }
    return stats

def calculate_feature_distributions(df: pd.DataFrame) -> List[Dict[str, any]]:
    return [{'feature': feature, 'data': df[feature].tolist()} for feature in MODEL_FEATURES]

def calculate_correlations(df: pd.DataFrame) -> List[List[float]]:
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

    def predict_with_confidence(self, X: pd.DataFrame) -> Dict[str, Dict[str, List[Dict[str, any]]]]:
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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "MODEL_FEATURES": MODEL_FEATURES})

@app.post("/predict")
async def predict(
    request: Request,
    file: Optional[UploadFile] = None,
    src_port: Optional[float] = Form(None),
    dst_port: Optional[float] = Form(None),
    protocol: Optional[float] = Form(None),
    flowiatmean: Optional[float] = Form(None),
    flowiatmin: Optional[float] = Form(None),
    bwdiatmean: Optional[float] = Form(None),
    bwdiatmin: Optional[float] = Form(None),
    pktlenmax: Optional[float] = Form(None),
    synflagcnt: Optional[float] = Form(None),
    ackflagcnt: Optional[float] = Form(None)
):
    try:
        if file and file.filename:
            contents = await file.read()
            df = pd.read_csv(pd.compat.StringIO(contents.decode('utf-8')))
        else:
            manual_input = {
                'src_port': [src_port], 'dst_port': [dst_port], 'protocol': [protocol],
                'flowiatmean': [flowiatmean], 'flowiatmin': [flowiatmin],
                'bwdiatmean': [bwdiatmean], 'bwdiatmin': [bwdiatmin],
                'pktlenmax': [pktlenmax], 'synflagcnt': [synflagcnt],
                'ackflagcnt': [ackflagcnt]
            }
            df = pd.DataFrame(manual_input)

        processed_df = validate_input_data(df)
        if processed_df.empty:
            raise HTTPException(status_code=400, detail="Invalid input data")

        predictions = model_manager.predict_with_confidence(processed_df)
        input_data = {col: processed_df[col].tolist() for col in processed_df.columns}
        data_analysis = {
            'basic_stats': calculate_basic_stats(processed_df),
            'feature_distributions': calculate_feature_distributions(processed_df),
            'correlation_matrix': calculate_correlations(processed_df)
        }

        return {
            'success': True,
            'predictions': predictions,
            'input_data': input_data,
            'data_analysis': data_analysis
        }

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)