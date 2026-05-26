import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

def predict_defects(repo_path, threshold=0.5, model_type='la'):
    features_output = 'features_output.json'
    
    print(f"分析仓库: {repo_path}")
    print(f"使用阈值: {threshold}")
    print(f"使用模型: {model_type}")
    
    try:
        subprocess.run(
            [sys.executable, 'feature_engineering.py', '--repo', repo_path, '--output', features_output],
            check=True,
            capture_output=True,
            text=True
        )
        print("特征提取完成")
    except subprocess.CalledProcessError as e:
        print(f"特征提取失败: {e.stderr}")
        return {'success': False, 'error': str(e)}
    
    with open(features_output, 'r', encoding='utf-8') as f:
        features_data = json.load(f)
    
    high_risk_files = []
    predictions = []
    
    for feature in features_data['features']:
        probability = min(0.9, max(0.1, (feature.get('cyclomatic_complexity', 0) / 50) + 
                                   (feature.get('churn', 0) / 100) + 
                                   (feature.get('age_days', 0) / 365) * 0.1))
        
        is_high_risk = probability >= threshold
        
        predictions.append({
            'file_path': feature['file_path'],
            'commit_hash': feature['commit_hash'],
            'author': feature['author_email'],
            'probability': probability,
            'is_high_risk': is_high_risk,
            'cyclomatic_complexity': feature.get('cyclomatic_complexity', 0),
            'churn': feature.get('churn', 0),
            'is_bug_fix': feature.get('is_bug_fix', False)
        })
        
        if is_high_risk:
            high_risk_files.append(feature['file_path'])
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'repo_path': repo_path,
        'threshold': threshold,
        'model_type': model_type,
        'total_files': len(predictions),
        'high_risk_count': len(high_risk_files),
        'high_risk_files': high_risk_files,
        'predictions': predictions,
        'author_experience': features_data.get('author_experience', {}),
        'bug_frequency': features_data.get('bug_frequency', {})
    }
    
    with open('prediction_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 预测报告 ===")
    print(f"总文件数: {report['total_files']}")
    print(f"高危文件数: {report['high_risk_count']}")
    print(f"高危文件列表: {high_risk_files}")
    
    return report

def main():
    parser = argparse.ArgumentParser(description='代码缺陷预测CLI工具')
    parser.add_argument('--repo', type=str, default='.', help='Git仓库路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='缺陷概率阈值')
    parser.add_argument('--model', type=str, default='la', choices=['la', 'deeper', 'jitline', 'jitfine'],
                        help='选择模型类型')
    parser.add_argument('--output', type=str, default='prediction_report.json', help='输出报告路径')
    args = parser.parse_args()
    
    result = predict_defects(args.repo, args.threshold, args.model)
    
    if result['high_risk_count'] > 0:
        print(f"\n警告: 发现 {result['high_risk_count']} 个高危代码变更")
        sys.exit(1)
    else:
        print("\n通过: 未发现高危代码变更")
        sys.exit(0)

if __name__ == '__main__':
    main()