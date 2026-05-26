import os
import pickle
import argparse
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

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
    
    print_stats(X_train, y_train, X_test, y_test)
    return X_train, X_test, y_train, y_test

def load_cc2vec_dataset(data_path):
    print(f"Loading CC2Vec dataset from {data_path}...")
    
    with open(os.path.join(data_path, 'features_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    with open(os.path.join(data_path, 'features_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    if isinstance(train_data, dict):
        X_train = train_data['features']
        y_train = train_data['labels']
        X_test = test_data['features']
        y_test = test_data['labels']
    elif isinstance(train_data, list):
        X_train = np.array([item[:-1] for item in train_data])
        y_train = np.array([item[-1] for item in train_data])
        X_test = np.array([item[:-1] for item in test_data])
        y_test = np.array([item[-1] for item in test_data])
    else:
        if hasattr(train_data, 'values'):
            train_data = train_data.values
        if hasattr(test_data, 'values'):
            test_data = test_data.values
        
        X_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]
    
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)
    
    print_stats(X_train, y_train, X_test, y_test)
    return X_train, X_test, y_train, y_test

def load_deepjit_dataset(data_path):
    print(f"Loading DeepJIT dataset from {data_path}...")
    
    with open(os.path.join(data_path, 'features_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    with open(os.path.join(data_path, 'features_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    if isinstance(train_data, dict):
        X_train = train_data['features']
        y_train = train_data['labels']
        X_test = test_data['features']
        y_test = test_data['labels']
    else:
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
    
    print_stats(X_train, y_train, X_test, y_test)
    return X_train, X_test, y_train, y_test

def print_stats(X_train, y_train, X_test, y_test):
    print(f"Train features shape: {X_train.shape}, labels: {len(y_train)}")
    print(f"Test features shape: {X_test.shape}, labels: {len(y_test)}")
    print(f"Train label distribution: buggy={int(sum(y_train))}, clean={len(y_train)-int(sum(y_train))}")
    print(f"Test label distribution: buggy={int(sum(y_test))}, clean={len(y_test)-int(sum(y_test))}")

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
    
    print(f"Time series shape: {time_series.shape}")
    print(f"Global features shape: {global_features.shape}")
    
    return time_series, global_features

def apply_smote(X_time, X_global, y):
    n_samples = X_time.shape[0]
    seq_len = X_time.shape[1]
    n_time_features = X_time.shape[2]
    
    X_time_flat = X_time.reshape(n_samples, seq_len * n_time_features)
    X_combined = np.concatenate([X_time_flat, X_global], axis=1)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_combined, y)
    
    X_time_resampled = X_resampled[:, :seq_len * n_time_features].reshape(-1, seq_len, n_time_features)
    X_global_resampled = X_resampled[:, seq_len * n_time_features:]
    
    print(f"SMOTE后样本数: {len(y_resampled)}, buggy样本数: {int(sum(y_resampled))}")
    
    return X_time_resampled, X_global_resampled, y_resampled

def train_model(model, X_time_train, X_global_train, y_train, epochs=50, lr=0.001, batch_size=32, use_weighted_loss=False):
    n_positive = int(sum(y_train))
    n_negative = len(y_train) - n_positive
    pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    print(f"正样本权重: {pos_weight:.2f}")
    
    if use_weighted_loss:
        loss_fn = nn.BCELoss(reduction='none')
    else:
        loss_fn = nn.BCELoss(reduction='mean')
    
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)
    
    n_samples = len(y_train)
    
    print("开始训练模型...")
    for epoch in range(epochs):
        model.set_train(True)
        total_loss = 0
        step = 0
        
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            time_tensor = Tensor(X_time_train[batch_indices], ms.float32)
            global_tensor = Tensor(X_global_train[batch_indices], ms.float32)
            label_tensor = Tensor(y_train[batch_indices], ms.float32).reshape(-1, 1)
            
            def forward_fn():
                logits = model(time_tensor, global_tensor)
                if use_weighted_loss:
                    loss_unreduced = loss_fn(logits, label_tensor)
                    weights = label_tensor * pos_weight + (1 - label_tensor)
                    loss = ops.mean(loss_unreduced * weights)
                else:
                    loss = loss_fn(logits, label_tensor)
                return loss
            
            grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
            loss, grads = grad_fn()
            optimizer(grads)
            
            total_loss += loss.asnumpy()
            step += 1
        
        avg_loss = total_loss / step if step > 0 else 0
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return model

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
    
    print_stats(X_train, y_train, X_test, y_test)
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
    
    print_stats(X_train, y_train, X_test, y_test)
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
    
    print_stats(X_train, y_train, X_test, y_test)
    return X_train, X_test, y_train, y_test

def main():
    parser = argparse.ArgumentParser(description='代码缺陷预测模型训练')
    parser.add_argument('--dataset', type=str, default='la', 
                        choices=['la', 'deeper', 'jitline', 'jitfine'], 
                        help='选择数据集: la, deeper, jitline, jitfine')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--smote', action='store_true', help='使用SMOTE过采样')
    parser.add_argument('--weighted_loss', action='store_true', help='使用加权损失函数')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
    args = parser.parse_args()
    
    data_path = f'data/{args.dataset}'
    
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
    
    if args.smote:
        print("\n应用SMOTE过采样...")
        time_series_train, global_features_train, y_train = apply_smote(
            time_series_train, global_features_train, y_train
        )
    
    num_time_features = 6
    num_global_features = global_features_train.shape[1]
    
    print(f"\n时间特征数: {num_time_features}")
    print(f"全局特征数: {num_global_features}")
    print(f"使用加权损失: {args.weighted_loss}")
    print(f"分类阈值: {args.threshold}")
    
    model = DualChannelModel(seq_len, num_time_features, num_global_features)
    
    trained_model = train_model(model, time_series_train, global_features_train, y_train, 
                                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                                use_weighted_loss=args.weighted_loss)
    
    print("\n=== 训练集评估 ===")
    evaluate_model(trained_model, time_series_train, global_features_train, y_train, threshold=args.threshold)
    
    print("\n=== 测试集评估 ===")
    evaluate_model(trained_model, time_series_test, global_features_test, y_test, threshold=args.threshold)
    
    print("\n=== 测试集多阈值评估 ===")
    evaluate_with_multiple_thresholds(trained_model, time_series_test, global_features_test, y_test)
    
    ms.save_checkpoint(trained_model, f'defect_prediction_model_{args.dataset}.ckpt')
    print(f"\n模型已保存为: defect_prediction_model_{args.dataset}.ckpt")

if __name__ == '__main__':
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    main()