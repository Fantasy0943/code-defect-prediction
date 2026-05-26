import os
import pickle
import argparse
import re
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TextFeatureExtractor:
    def __init__(self, max_features=5000):
        self.tfidf_message = TfidfVectorizer(max_features=max_features)
        self.tfidf_code = TfidfVectorizer(max_features=max_features)
    
    def preprocess_code(self, code_changes):
        texts = []
        for change_list in code_changes:
            text = ''
            if isinstance(change_list, list):
                for change in change_list:
                    if isinstance(change, dict):
                        added = ' '.join(change.get('added_code', []))
                        removed = ' '.join(change.get('removed_code', []))
                        text += ' ' + added + ' ' + removed
            elif isinstance(change_list, dict):
                added = ' '.join(change_list.get('added_code', []))
                removed = ' '.join(change_list.get('removed_code', []))
                text = added + ' ' + removed
            
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            texts.append(text if text else 'empty')
        return texts
    
    def fit_transform(self, messages, code_changes):
        messages = [str(m) for m in messages]
        code_texts = self.preprocess_code(code_changes)
        
        msg_features = self.tfidf_message.fit_transform(messages).toarray()
        code_features = self.tfidf_code.fit_transform(code_texts).toarray()
        
        combined = np.concatenate([msg_features, code_features], axis=1)
        print(f"文本特征维度: {combined.shape}")
        
        return combined
    
    def transform(self, messages, code_changes):
        messages = [str(m) for m in messages]
        code_texts = self.preprocess_code(code_changes)
        
        msg_features = self.tfidf_message.transform(messages).toarray()
        code_features = self.tfidf_code.transform(code_texts).toarray()
        
        combined = np.concatenate([msg_features, code_features], axis=1)
        return combined

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

def load_cc2vec_dataset(data_path):
    print(f"Loading CC2Vec dataset from {data_path}...")
    
    with open(os.path.join(data_path, 'features_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    with open(os.path.join(data_path, 'features_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    train_hashes = train_data[0]
    train_labels = np.array(train_data[1], dtype=np.float32)
    train_messages = train_data[2]
    train_code = train_data[3]
    
    test_hashes = test_data[0]
    test_labels = np.array(test_data[1], dtype=np.float32)
    test_messages = test_data[2]
    test_code = test_data[3]
    
    print(f"Train samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print(f"Train label distribution: buggy={int(sum(train_labels))}, clean={len(train_labels)-int(sum(train_labels))}")
    print(f"Test label distribution: buggy={int(sum(test_labels))}, clean={len(test_labels)-int(sum(test_labels))}")
    
    return (train_messages, train_code, train_labels), (test_messages, test_code, test_labels)

def load_deepjit_dataset(data_path):
    print(f"Loading DeepJIT dataset from {data_path}...")
    
    with open(os.path.join(data_path, 'features_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    with open(os.path.join(data_path, 'features_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    train_hashes = train_data[0]
    train_labels = np.array(train_data[1], dtype=np.float32)
    train_messages = train_data[2]
    train_code = train_data[3]
    
    test_hashes = test_data[0]
    test_labels = np.array(test_data[1], dtype=np.float32)
    test_messages = test_data[2]
    test_code = test_data[3]
    
    print(f"Train samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print(f"Train label distribution: buggy={int(sum(train_labels))}, clean={len(train_labels)-int(sum(train_labels))}")
    print(f"Test label distribution: buggy={int(sum(test_labels))}, clean={len(test_labels)-int(sum(test_labels))}")
    
    return (train_messages, train_code, train_labels), (test_messages, test_code, test_labels)

def train_model(model, X_train, y_train, epochs=50, lr=0.001, batch_size=32):
    loss_fn = nn.BCELoss(reduction='mean')
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)
    
    n_samples = len(y_train)
    X_train = np.expand_dims(X_train, axis=1)
    
    print("开始训练模型...")
    for epoch in range(epochs):
        model.set_train(True)
        total_loss = 0
        step = 0
        
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            x_tensor = Tensor(X_train[batch_indices], ms.float32)
            y_tensor = Tensor(y_train[batch_indices], ms.float32).reshape(-1, 1)
            
            def forward_fn():
                logits = model(x_tensor)
                loss = loss_fn(logits, y_tensor)
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

def evaluate_model(model, X, y_true):
    model.set_train(False)
    
    X = np.expand_dims(X, axis=1)
    x_tensor = Tensor(X, ms.float32)
    
    predictions = model(x_tensor).asnumpy()
    predictions = (predictions >= 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    
    print("\n=== 评估指标 ===")
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

def load_ngram_dataset(data_path):
    print(f"Loading N-gram dataset from {data_path}...")
    
    train_file = os.path.join(data_path, 'train_data.txt')
    test_file = os.path.join(data_path, 'test_data_commit_onlyadds.txt')
    label_file = os.path.join(data_path, 'test_data_label_onlyadds.txt')
    
    with open(train_file, 'r', encoding='utf-8', errors='ignore') as f:
        train_texts = f.readlines()
    
    with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
        test_texts = f.readlines()
    
    with open(label_file, 'r') as f:
        test_labels = [int(float(line.strip())) for line in f.readlines()]
    
    train_labels = [0] * len(train_texts)
    
    train_labels = np.array(train_labels, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.float32)
    
    print(f"Train samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    
    return train_texts, test_texts, train_labels, test_labels

def main():
    parser = argparse.ArgumentParser(description='代码缺陷预测模型训练（文本向量化）')
    parser.add_argument('--dataset', type=str, default='cc2vec', 
                        choices=['cc2vec', 'deepjit', 'ngram'], 
                        help='选择数据集: cc2vec, deepjit, ngram')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--max_features', type=int, default=3000, help='TF-IDF最大特征数')
    args = parser.parse_args()
    
    data_path = f'data/{args.dataset}'
    
    if args.dataset == 'cc2vec':
        train_data, test_data = load_cc2vec_dataset(data_path)
        train_messages, train_code, train_labels = train_data
        test_messages, test_code, test_labels = test_data
        
        extractor = TextFeatureExtractor(max_features=args.max_features)
        X_train = extractor.fit_transform(train_messages, train_code)
        X_test = extractor.transform(test_messages, test_code)
    
    elif args.dataset == 'deepjit':
        train_data, test_data = load_deepjit_dataset(data_path)
        train_messages, train_code, train_labels = train_data
        test_messages, test_code, test_labels = test_data
        
        extractor = TextFeatureExtractor(max_features=args.max_features)
        X_train = extractor.fit_transform(train_messages, train_code)
        X_test = extractor.transform(test_messages, test_code)
    
    elif args.dataset == 'ngram':
        train_texts, test_texts, train_labels, test_labels = load_ngram_dataset(data_path)
        
        tfidf = TfidfVectorizer(max_features=args.max_features)
        X_train = tfidf.fit_transform(train_texts).toarray()
        X_test = tfidf.transform(test_texts).toarray()
        print(f"N-gram 特征维度: {X_train.shape}")
    
    else:
        print(f"未知数据集: {args.dataset}")
        return
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    input_dim = X_train.shape[1]
    print(f"\n输入特征维度: {input_dim}")
    
    model = TextBasedModel(input_dim=input_dim, hidden_size=128)
    
    trained_model = train_model(model, X_train, train_labels, 
                                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    
    print("\n=== 训练集评估 ===")
    evaluate_model(trained_model, X_train, train_labels)
    
    print("\n=== 测试集评估 ===")
    evaluate_model(trained_model, X_test, test_labels)
    
    ms.save_checkpoint(trained_model, f'defect_prediction_model_{args.dataset}_text.ckpt')
    print(f"\n模型已保存为: defect_prediction_model_{args.dataset}_text.ckpt")

if __name__ == '__main__':
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    main()