import os
import pickle
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

class DualChannelModel(nn.Cell):
    def __init__(self, seq_len, num_time_features, num_global_features, lstm_hidden_size=64, fc_hidden_size=32):
        super(DualChannelModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_time_features,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.2
        )
        
        self.fc_global = nn.SequentialCell([
            nn.Dense(num_global_features, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Dense(fc_hidden_size, fc_hidden_size // 2),
            nn.ReLU()
        ])
        
        combined_size = lstm_hidden_size * 2 + fc_hidden_size // 2
        self.fc_combined = nn.SequentialCell([
            nn.Dense(combined_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, 1),
            nn.Sigmoid()
        ])

    def construct(self, time_series, global_features):
        output, _ = self.lstm(time_series)
        lstm_out = output[:, -1, :]
        
        global_out = self.fc_global(global_features)
        
        combined = ops.concat([lstm_out, global_out], axis=1)
        logits = self.fc_combined(combined)
        
        return logits

class PredictionRequest(BaseModel):
    features: List[float]
    threshold: Optional[float] = 0.5

class BatchPredictionRequest(BaseModel):
    features_list: List[List[float]]
    threshold: Optional[float] = 0.5

class PredictionResponse(BaseModel):
    is_buggy: bool
    probability: float
    threshold: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

app = FastAPI(title="代码缺陷预测API", description="用于CI/CD流水线集成的代码缺陷预测服务")

model = None
time_scaler = None
global_scaler = None
seq_len = 5
num_time_features = 6

def prepare_features(X, seq_len=5):
    n_samples = X.shape[0] if len(X.shape) > 1 else 1
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    n_features = X.shape[1]
    
    time_feature_count = seq_len * 6
    
    if n_features <= time_feature_count:
        time_features = X
        global_features = np.zeros((n_samples, 1))
    else:
        time_features = X[:, :time_feature_count]
        global_features = X[:, time_feature_count:]
    
    n_time_features = time_features.shape[1]
    
    padding = (seq_len * 6) - n_time_features
    if padding > 0:
        padding_matrix = np.zeros((n_samples, padding))
        time_features = np.concatenate([time_features, padding_matrix], axis=1)
    
    time_series = time_features.reshape(-1, seq_len, 6)
    
    return time_series, global_features

def load_model(dataset='la'):
    global model, time_scaler, global_scaler
    
    model_path = f'defect_prediction_model_{dataset}.ckpt'
    data_path = f'data/{dataset}'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    with open(os.path.join(data_path, 'features_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    if hasattr(train_data, 'values'):
        train_data = train_data.values
    
    X_train = train_data[:, :-1]
    
    for i in range(X_train.shape[1]):
        if X_train[:, i].dtype.kind in ['U', 'S', 'O']:
            unique_vals = np.unique(X_train[:, i])
            val_to_idx = {v: idx for idx, v in enumerate(unique_vals)}
            X_train[:, i] = np.array([val_to_idx.get(x, 0) for x in X_train[:, i]])
    
    X_train = X_train.astype(np.float32)
    
    time_series_train, global_features_train = prepare_features(X_train, seq_len)
    
    time_scaler = StandardScaler()
    time_series_train_2d = time_series_train.reshape(-1, time_series_train.shape[-1])
    time_scaler.fit(time_series_train_2d)
    
    global_scaler = StandardScaler()
    global_features_train = global_scaler.fit_transform(global_features_train)
    
    num_global_features = global_features_train.shape[1]
    
    model = DualChannelModel(seq_len, num_time_features, num_global_features)
    ms.load_checkpoint(model_path, model)
    model.set_train(False)
    
    print(f"模型加载成功: {model_path}")

@app.on_event("startup")
async def startup_event():
    load_model('la')

@app.get("/")
async def root():
    return {"message": "代码缺陷预测API服务运行中"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        X = np.array(request.features, dtype=np.float32).reshape(1, -1)
        
        time_series, global_features = prepare_features(X, seq_len)
        
        time_series_2d = time_series.reshape(-1, time_series.shape[-1])
        time_series = time_scaler.transform(time_series_2d).reshape(time_series.shape)
        
        global_features = global_scaler.transform(global_features)
        
        time_tensor = Tensor(time_series, ms.float32)
        global_tensor = Tensor(global_features, ms.float32)
        
        prediction = model(time_tensor, global_tensor).asnumpy()[0][0]
        
        is_buggy = prediction >= request.threshold
        
        return PredictionResponse(
            is_buggy=is_buggy,
            probability=float(prediction),
            threshold=request.threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        predictions = []
        
        for features in request.features_list:
            X = np.array(features, dtype=np.float32).reshape(1, -1)
            
            time_series, global_features = prepare_features(X, seq_len)
            
            time_series_2d = time_series.reshape(-1, time_series.shape[-1])
            time_series = time_scaler.transform(time_series_2d).reshape(time_series.shape)
            
            global_features = global_scaler.transform(global_features)
            
            time_tensor = Tensor(time_series, ms.float32)
            global_tensor = Tensor(global_features, ms.float32)
            
            prediction = model(time_tensor, global_tensor).asnumpy()[0][0]
            
            is_buggy = prediction >= request.threshold
            
            predictions.append(PredictionResponse(
                is_buggy=is_buggy,
                probability=float(prediction),
                threshold=request.threshold
            ))
        
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)