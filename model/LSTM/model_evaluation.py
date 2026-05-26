import os
import pickle
import argparse
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

class TextBasedModel(nn.Cell):
    def __init__(self, input_dim, hidden_size=128, num_layers=2):
        super(TextBasedModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        
        self.fc = nn.SequentialCell([
            nn.Dense(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, 1),
            nn.Sigmoid()
        ])
    
    def construct(self, x):
        output, _ = self.lstm(x)
        lstm_out = output[:, -1, :]
        logits = self.fc(lstm_out)
        return logits

def load_la_dataset(data_path):
    print(f"Loading LA dataset from {data_path}...")
    
    with open(os.path.join(data_path, 'features_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    with open(os.path.join(data_path, 'features_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    if hasattr(train_data, 'values'):
        train_data = train_data.values
    if hasattr(test_data, 'values'):
        test_data = test_data.values
    
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    
    for i in range(X_train.shape[1]):
        if X_train[:, i].dtype.kind in ['U', 'S', 'O']:
            unique_vals = np.unique(np.concatenate([X_train[:, i], X_test[:, i]]))
            val_to_idx = {v: idx for idx, v in enumerate(unique_vals)}
            X_train[:, i] = np.array([val_to_idx.get(x, 0) for x in X_train[:, i]])
            X_test[:, i] = np.array([val_to_idx.get(x, 0) for x in X_test[:, i]])
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    return X_train, X_test, y_train, y_test

def load_deeper_dataset(data_path):
    print(f"Loading Deeper dataset from {data_path}...")
    
    with open(os.path.join(data_path, 'features_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    with open(os.path.join(data_path, 'features_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    if hasattr(train_data, 'values'):
        train_data = train_data.values
    if hasattr(test_data, 'values'):
        test_data = test_data.values
    
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    
    for i in range(X_train.shape[1]):
        if X_train[:, i].dtype.kind in ['U', 'S', 'O']:
            unique_vals = np.unique(np.concatenate([X_train[:, i], X_test[:, i]]))
            val_to_idx = {v: idx for idx, v in enumerate(unique_vals)}
            X_train[:, i] = np.array([val_to_idx.get(x, 0) for x in X_train[:, i]])
            X_test[:, i] = np.array([val_to_idx.get(x, 0) for x in X_test[:, i]])
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    return X_train, X_test, y_train, y_test

def load_jitline_dataset(data_path):
    print(f"Loading JITLine dataset from {data_path}...")
    
    with open(os.path.join(data_path, 'features_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    with open(os.path.join(data_path, 'features_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    if hasattr(train_data, 'values'):
        train_data = train_data.values
    if hasattr(test_data, 'values'):
        test_data = test_data.values
    
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    
    for i in range(X_train.shape[1]):
        if X_train[:, i].dtype.kind in ['U', 'S', 'O']:
            unique_vals = np.unique(np.concatenate([X_train[:, i], X_test[:, i]]))
            val_to_idx = {v: idx for idx, v in enumerate(unique_vals)}
            X_train[:, i] = np.array([val_to_idx.get(x, 0) for x in X_train[:, i]])
            X_test[:, i] = np.array([val_to_idx.get(x, 0) for x in X_test[:, i]])
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    return X_train, X_test, y_train, y_test

def load_jitfine_dataset(data_path):
    print(f"Loading JITFine dataset from {data_path}...")
    
    with open(os.path.join(data_path, 'features_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    with open(os.path.join(data_path, 'features_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    if hasattr(train_data, 'values'):
        train_data = train_data.values
    if hasattr(test_data, 'values'):
        test_data = test_data.values
    
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    
    for i in range(X_train.shape[1]):
        if X_train[:, i].dtype.kind in ['U', 'S', 'O']:
            unique_vals = np.unique(np.concatenate([X_train[:, i], X_test[:, i]]))
            val_to_idx = {v: idx for idx, v in enumerate(unique_vals)}
            X_train[:, i] = np.array([val_to_idx.get(x, 0) for x in X_train[:, i]])
            X_test[:, i] = np.array([val_to_idx.get(x, 0) for x in X_test[:, i]])
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    return X_train, X_test, y_train, y_test

def prepare_features(X, seq_len=5):
    n_samples, n_features = X.shape
    
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

def evaluate_model(model, X_time, X_global, y_true, threshold=0.5):
    model.set_train(False)
    
    time_tensor = Tensor(X_time, ms.float32)
    global_tensor = Tensor(X_global, ms.float32)
    
    predictions = model(time_tensor, global_tensor).asnumpy()
    predictions = (predictions >= threshold).astype(int).flatten()
    
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    
    print("\n=== 评估指标 (阈值={}) ===".format(threshold))
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_with_multiple_thresholds(model, X_time, X_global, y_true):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    print("\n=== 多阈值评估 ===")
    for threshold in thresholds:
        metrics = evaluate_model(model, X_time, X_global, y_true, threshold)
        results.append({'threshold': threshold, **metrics})
    
    print("\n=== 阈值对比表 ===")
    print(f"{'阈值':<6} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8}")
    print("-" * 44)
    for result in results:
        print(f"{result['threshold']:<6} {result['accuracy']:<8.4f} {result['precision']:<8.4f} {result['recall']:<8.4f} {result['f1']:<8.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='代码缺陷预测模型评估（仅测试）')
    parser.add_argument('--dataset', type=str, default='la', 
                        choices=['la', 'deeper', 'jitline', 'jitfine'], 
                        help='选择数据集: la, deeper, jitline, jitfine')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
    args = parser.parse_args()
    
    data_path = f'data/{args.dataset}'
    model_path = f'defect_prediction_model_{args.dataset}.ckpt'
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    if args.dataset == 'la':
        X_train, X_test, y_train, y_test = load_la_dataset(data_path)
    elif args.dataset == 'deeper':
        X_train, X_test, y_train, y_test = load_deeper_dataset(data_path)
    elif args.dataset == 'jitline':
        X_train, X_test, y_train, y_test = load_jitline_dataset(data_path)
    elif args.dataset == 'jitfine':
        X_train, X_test, y_train, y_test = load_jitfine_dataset(data_path)
    else:
        print(f"未知数据集: {args.dataset}")
        return
    
    seq_len = 5
    time_series_train, global_features_train = prepare_features(X_train, seq_len)
    time_series_test, global_features_test = prepare_features(X_test, seq_len)
    
    time_scaler = StandardScaler()
    time_series_train_2d = time_series_train.reshape(-1, time_series_train.shape[-1])
    time_scaler.fit(time_series_train_2d)
    time_series_train = time_scaler.transform(time_series_train_2d).reshape(time_series_train.shape)
    time_series_test_2d = time_series_test.reshape(-1, time_series_test.shape[-1])
    time_series_test = time_scaler.transform(time_series_test_2d).reshape(time_series_test.shape)
    
    global_scaler = StandardScaler()
    global_features_train = global_scaler.fit_transform(global_features_train)
    global_features_test = global_scaler.transform(global_features_test)
    
    num_time_features = 6
    num_global_features = global_features_train.shape[1]
    
    print(f"\n时间特征数: {num_time_features}")
    print(f"全局特征数: {num_global_features}")
    print(f"加载模型: {model_path}")
    print(f"分类阈值: {args.threshold}")
    
    model = DualChannelModel(seq_len, num_time_features, num_global_features)
    ms.load_checkpoint(model_path, model)
    
    print("\n=== 训练集评估 ===")
    train_metrics = evaluate_model(model, time_series_train, global_features_train, y_train, threshold=args.threshold)
    
    print("\n=== 测试集评估 ===")
    test_metrics = evaluate_model(model, time_series_test, global_features_test, y_test, threshold=args.threshold)
    
    print("\n=== 测试集多阈值评估 ===")
    evaluate_with_multiple_thresholds(model, time_series_test, global_features_test, y_test)
    
    return train_metrics, test_metrics

if __name__ == '__main__':
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    main()