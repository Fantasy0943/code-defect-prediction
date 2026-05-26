import os
import re
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pydriller import Repository
from radon.complexity import cc_visit
from radon.metrics import mi_visit
import lizard

class CodeFeatureExtractor:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.current_time = datetime.now(timezone.utc)
        self.author_commits = defaultdict(int)
        self.author_prs = defaultdict(set)
    
    def extract_git_features(self):
        features = []
        
        for commit in Repository(self.repo_path).traverse_commits():
            for modified_file in commit.modified_files:
                churn = modified_file.added_lines + modified_file.deleted_lines
                age_days = (self.current_time - commit.author_date).days
                
                file_info = {
                    'commit_hash': commit.hash,
                    'author_name': commit.author.name,
                    'author_email': commit.author.email,
                    'commit_date': commit.author_date.isoformat(),
                    'file_path': modified_file.new_path or modified_file.old_path,
                    'added_lines': modified_file.added_lines,
                    'deleted_lines': modified_file.deleted_lines,
                    'churn': churn,
                    'age_days': age_days,
                    'message': commit.msg,
                    'is_bug_fix': self._is_bug_fix(commit.msg),
                    'file_type': self._get_file_type(modified_file.new_path or modified_file.old_path)
                }
                
                features.append(file_info)
                self.author_commits[commit.author.email] += 1
                
                pr_number = self._extract_pr_number(commit.msg)
                if pr_number:
                    self.author_prs[commit.author.email].add(pr_number)
        
        return features
    
    def _is_bug_fix(self, message):
        bug_patterns = [r'\bfix\b', r'\bbug\b', r'\bresolve\b', r'\bissue\b', r'\bbugfix\b']
        message_lower = message.lower()
        return any(re.search(pattern, message_lower) for pattern in bug_patterns)
    
    def _extract_pr_number(self, message):
        match = re.search(r'PR\s*#?(\d+)', message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def _get_file_type(self, file_path):
        if file_path:
            _, ext = os.path.splitext(file_path)
            return ext[1:] if ext else 'unknown'
        return 'unknown'
    
    def calculate_complexity(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            _, ext = os.path.splitext(file_path)
            
            if ext == '.py':
                cc_results = cc_visit(content)
                mi_results = mi_visit(content, multi=True)
                
                avg_cc = sum(r.complexity for r in cc_results) / len(cc_results) if cc_results else 0
                return {
                    'cyclomatic_complexity': avg_cc,
                    'maintainability_index': mi_results[0] if mi_results else 0,
                    'method_count': len(cc_results)
                }
            else:
                analysis = lizard.analyze_file(file_path)
                if analysis.function_list:
                    avg_cc = sum(f.cyclomatic_complexity for f in analysis.function_list) / len(analysis.function_list)
                    return {
                        'cyclomatic_complexity': avg_cc,
                        'maintainability_index': 0,
                        'method_count': len(analysis.function_list),
                        'nloc': analysis.nloc
                    }
                return {'cyclomatic_complexity': 0, 'maintainability_index': 0, 'method_count': 0}
        except Exception as e:
            return {'cyclomatic_complexity': 0, 'maintainability_index': 0, 'method_count': 0, 'error': str(e)}
    
    def get_author_experience(self):
        experience = {}
        for email, commits in self.author_commits.items():
            experience[email] = {
                'total_commits': commits,
                'unique_prs': len(self.author_prs[email])
            }
        return experience
    
    def get_bug_fix_frequency(self, features, months=3):
        cutoff_date = self.current_time - timedelta(days=months*30)
        bug_fixes = [f for f in features if f['is_bug_fix']]
        
        recent_bug_fixes = [f for f in bug_fixes if datetime.fromisoformat(f['commit_date']) >= cutoff_date]
        
        file_bug_counts = defaultdict(int)
        for bug in recent_bug_fixes:
            file_bug_counts[bug['file_path']] += 1
        
        return {
            'total_bug_fixes_last_3months': len(recent_bug_fixes),
            'file_bug_counts': dict(file_bug_counts)
        }

class FeaturePipeline:
    def __init__(self, repo_path):
        self.extractor = CodeFeatureExtractor(repo_path)
    
    def run(self):
        print("开始提取Git特征...")
        git_features = self.extractor.extract_git_features()
        print(f"提取了 {len(git_features)} 条Git特征记录")
        
        print("计算代码复杂度...")
        for feature in git_features:
            file_path = os.path.join(self.extractor.repo_path, feature['file_path'])
            if os.path.exists(file_path):
                complexity = self.extractor.calculate_complexity(file_path)
                feature.update(complexity)
            else:
                feature.update({'cyclomatic_complexity': 0, 'maintainability_index': 0, 'method_count': 0})
        
        print("获取作者经验值...")
        author_experience = self.extractor.get_author_experience()
        
        print("计算Bug修复频率...")
        bug_frequency = self.extractor.get_bug_fix_frequency(git_features)
        
        result = {
            'extraction_time': self.extractor.current_time.isoformat(),
            'total_records': len(git_features),
            'features': git_features,
            'author_experience': author_experience,
            'bug_frequency': bug_frequency
        }
        
        return result

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='代码缺陷预测模型特征工程')
    parser.add_argument('--repo', type=str, required=True, help='Git仓库路径')
    parser.add_argument('--output', type=str, default='features_output.json', help='输出文件路径')
    args = parser.parse_args()
    
    if not os.path.isdir(args.repo):
        print(f"错误: 仓库路径不存在: {args.repo}")
        return
    
    print(f"正在处理仓库: {args.repo}")
    pipeline = FeaturePipeline(args.repo)
    result = pipeline.run()
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"特征提取完成，结果已保存到: {args.output}")
    
    print("\n=== 统计摘要 ===")
    print(f"总记录数: {result['total_records']}")
    print(f"作者数量: {len(result['author_experience'])}")
    print(f"最近3个月Bug修复数: {result['bug_frequency']['total_bug_fixes_last_3months']}")
    
    print("\n=== 作者经验排名 ===")
    sorted_authors = sorted(result['author_experience'].items(), key=lambda x: x[1]['total_commits'], reverse=True)
    for email, stats in sorted_authors[:5]:
        print(f"  {email}: 提交数={stats['total_commits']}, PR数={stats['unique_prs']}")

if __name__ == '__main__':
    main()