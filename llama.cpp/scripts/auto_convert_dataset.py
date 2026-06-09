#!/usr/bin/env python3
"""
自动数据集转换工具
给定一个数据集目录，自动检测结构并转换，在目录中创建 converted 文件夹

支持的题目类型：
1. 多选题（Multiple Choice）: question + choices (列表) + answer (索引)
   - 标准格式: choices 为列表
   - GLUE 格式: premise/hypothesis + label
   - TruthfulQA 格式: mc1_targets/mc2_targets
   - ARC/OpenBookQA 格式: choices 为字典 {"text": [...], "label": [...]}
   - HellaSwag 格式: ctx + endings + label
2. 二选一（Binary Choice）: 
   - Winogrande: sentence + option1 + option2 + answer
   - PIQA: goal + sol1 + sol2 + label
3. 问答（Question Answering）: question + answer (文本，无选项)
4. 阅读理解（Reading Comprehension）: context + question + answers (文本和位置)
5. 数学推理（Math Reasoning）: question + answer (包含推理步骤)
6. 代码生成（Code Generation）: prompt + code + test

模块结构：
- 模块 1: 工具函数（文件操作、数据加载、配置检测）
- 模块 2: 类型检测（自动识别数据集格式）
- 模块 3: 数据解析辅助函数（字段提取、格式转换）
- 模块 4: 二进制打包（与 perplexity.cpp 兼容的二进制格式）
- 模块 5: 转换函数（按题目类型分类）
- 模块 6: 主流程（数据集转换入口）
"""

import os
import glob
import struct
import json
from datasets import load_dataset  # type: ignore
import argparse
from typing import Optional, Dict, Any

# ============================================================================
# 模块 1: 工具函数模块
# 职责: 文件操作、数据加载、配置检测、字段打印
# ============================================================================

def detect_fields(sample: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    自动检测数据集字段
    
    Args:
        sample: 数据集样本（字典）
    
    Returns:
        包含 question_field, choices_field, answer_field, title_field 的字典
    """
    # 常见的字段名模式
    QUESTION_KEYWORDS = ['question', 'ctx', 'context', 'prompt', 'sentence', 'text', 'input', 'query', 'premise', 'hypothesis', 'question_stem', 'goal']
    CHOICES_KEYWORDS = ['choices', 'endings', 'options', 'alternatives', 'candidates', 'options_list']
    ANSWER_KEYWORDS = ['answer', 'label', 'correct', 'target', 'gold', 'correct_answer', 'answerKey']
    TITLE_KEYWORDS = ['subject', 'category', 'activity_label', 'topic', 'domain']
    
    structure: Dict[str, Optional[str]] = {
        'question_field': None,
        'choices_field': None,
        'answer_field': None,
        'title_field': None
    }
    
    # 检测题目字段（大小写不敏感）
    for key in sample.keys():
        key_lower = key.lower()
        if any(kw in key_lower for kw in QUESTION_KEYWORDS):
            # 优先选择更短的、更精确的字段名
            # 对于 GPQA，优先选择 'Question' 而不是 'Pre-Revision Question'
            if not structure['question_field']:
                structure['question_field'] = key
            elif len(key) < len(structure['question_field']) or ('pre-revision' not in key_lower and 'pre-revision' in structure['question_field'].lower()):
                structure['question_field'] = key
    
    # 检测选项字段
    for key in sample.keys():
        key_lower = key.lower()
        if any(kw in key_lower for kw in CHOICES_KEYWORDS):
            value = sample[key]
            if hasattr(value, 'as_py'):
                value = value.as_py()
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                structure['choices_field'] = key
                break
            elif isinstance(value, dict) and 'text' in value:
                structure['choices_field'] = key
                break
    
    # 如果没有找到，检查是否有列表类型的字段
    if not structure['choices_field']:
        for key, value in sample.items():
            if hasattr(value, 'as_py'):
                value = value.as_py()
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                # 检查列表元素是否都是字符串
                if all(isinstance(item, str) or (hasattr(item, 'as_py') and isinstance(item.as_py(), str)) for item in value[:3]):
                    structure['choices_field'] = key
                    break
    
    # 如果没有找到，检查 Winogrande 格式（option1/option2）
    if not structure['choices_field']:
        fields_lower = {k.lower(): k for k in sample.keys()}
        if 'option1' in fields_lower and 'option2' in fields_lower:
            # Winogrande 格式：使用 option1 和 option2 作为选项字段
            structure['choices_field'] = 'option1_option2'  # 标记为特殊格式
            # 确保 sentence 被识别为题目字段（如果还没有）
            if not structure['question_field'] and 'sentence' in fields_lower:
                structure['question_field'] = fields_lower['sentence']
    
    # 如果没有找到，检查 PIQA 格式（sol1/sol2）
    if not structure['choices_field']:
        fields_lower = {k.lower(): k for k in sample.keys()}
        if 'sol1' in fields_lower and 'sol2' in fields_lower:
            # PIQA 格式：使用 sol1 和 sol2 作为选项字段
            structure['choices_field'] = 'sol1_sol2'  # 标记为特殊格式
            # 确保 goal 被识别为题目字段（如果还没有）
            if not structure['question_field'] and 'goal' in fields_lower:
                structure['question_field'] = fields_lower['goal']
    
    # 如果没有找到，检查 GPQA 格式（Incorrect Answer 1/2/3）
    if not structure['choices_field']:
        incorrect_answer_fields = [f for f in sample.keys() if 'incorrect' in f.lower() and 'answer' in f.lower()]
        if len(incorrect_answer_fields) >= 1:
            # GPQA 格式：有 Correct Answer 和 Incorrect Answer 1/2/3
            # 这些字段会被组合成 choices
            structure['choices_field'] = 'incorrect_answer_fields'  # 标记为特殊格式
    
    # 检测答案字段（大小写不敏感）
    # 优先选择更精确的字段名（如 'label' 优先于 'activity_label'）
    answer_candidates = []
    for key in sample.keys():
        key_lower = key.lower()
        if any(kw in key_lower for kw in ANSWER_KEYWORDS):
            # 排除标题字段（如 activity_label 可能是标题而不是答案）
            if key_lower not in ['activity_label', 'subject', 'category', 'topic']:
                # 对于 GPQA，优先选择 'Correct Answer' 而不是 'Pre-Revision Correct Answer'
                priority = 0
                if 'pre-revision' not in key_lower:
                    priority = 1  # 非 Pre-Revision 字段优先级更高
                answer_candidates.append((key, len(key), priority))
    
    if answer_candidates:
        # 先按优先级排序，再按长度排序（优先选择非 Pre-Revision 的、更短的字段名）
        answer_candidates.sort(key=lambda x: (-x[2], x[1]))
        structure['answer_field'] = answer_candidates[0][0]
    
    # 检测标题字段（可选）
    for key in sample.keys():
        key_lower = key.lower()
        if any(kw in key_lower for kw in TITLE_KEYWORDS):
            structure['title_field'] = key
            break
    
    return structure

def find_data_files(dataset_dir):
    """在目录中查找数据文件"""
    data_files = {}
    
    # 查找常见的 split 目录
    split_dirs = {
        'train': ['train_dataset', 'train', 'training'],
        'validation': ['validation_dataset', 'validation', 'val', 'dev'],
        'test': ['test_dataset', 'test', 'testing']
    }
    
    for split_name, dir_names in split_dirs.items():
        for dir_name in dir_names:
            dir_path = os.path.join(dataset_dir, dir_name)
            if os.path.isdir(dir_path):
                # 查找 parquet 文件
                parquet_files = glob.glob(os.path.join(dir_path, '*.parquet'))
                if parquet_files:
                    data_files[split_name] = parquet_files
                    break
    
    # 如果没有找到 split 目录，直接在根目录查找
    if not data_files:
        parquet_files = glob.glob(os.path.join(dataset_dir, '**/*.parquet'), recursive=True)
        if parquet_files:
            # 尝试从文件名推断 split
            for file in parquet_files:
                if 'train' in file.lower():
                    data_files.setdefault('train', []).append(file)
                elif 'val' in file.lower() or 'dev' in file.lower():
                    data_files.setdefault('validation', []).append(file)
                elif 'test' in file.lower():
                    data_files.setdefault('test', []).append(file)
    
    return data_files

def detect_configs_from_readme(dataset_dir):
    """从 README.md 检测可用的 configs"""
    readme_path = os.path.join(dataset_dir, 'README.md')
    if not os.path.exists(readme_path):
        return None
    
    try:
        import yaml
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取 YAML 前导块
        if content.startswith('---'):
            yaml_end = content.find('---', 3)
            if yaml_end > 0:
                yaml_content = content[3:yaml_end]
                data = yaml.safe_load(yaml_content)
                
                # 查找 configs
                if 'configs' in data:
                    configs = []
                    for config in data['configs']:
                        if isinstance(config, dict) and 'config_name' in config:
                            configs.append(config['config_name'])
                        elif isinstance(config, str):
                            configs.append(config)
                    return configs if configs else None
    except Exception:
        pass
    
    return None

def print_dataset_fields(sample, indent="    "):
    """打印数据集的所有字段及其类型和示例值"""
    all_fields = list(sample.keys())
    print(f"{indent}数据集字段列表 ({len(all_fields)} 个字段):")
    for field in all_fields:
        field_value = sample.get(field)
        field_type = type(field_value).__name__
        # 显示字段类型和简要内容
        if isinstance(field_value, str):
            preview = field_value[:50] + "..." if len(field_value) > 50 else field_value
            print(f"{indent}  - {field}: {field_type} = {repr(preview)}")
        elif isinstance(field_value, (list, tuple)):
            if len(field_value) > 0:
                first_item = field_value[0]
                if isinstance(first_item, str):
                    preview = first_item[:30] + "..." if len(first_item) > 30 else first_item
                    print(f"{indent}  - {field}: {field_type}[{len(field_value)}] = [{repr(preview)}, ...]")
                else:
                    print(f"{indent}  - {field}: {field_type}[{len(field_value)}] = [{type(first_item).__name__}, ...]")
            else:
                print(f"{indent}  - {field}: {field_type}[] = []")
        elif isinstance(field_value, dict):
            dict_keys = list(field_value.keys())[:5]
            if len(dict_keys) > 0:
                first_key = dict_keys[0]
                first_val = field_value[first_key]
                if isinstance(first_val, str):
                    preview = first_val[:30] + "..." if len(first_val) > 30 else first_val
                    print(f"{indent}  - {field}: {field_type} = {{'{first_key}': {repr(preview)}, ...}}")
                else:
                    print(f"{indent}  - {field}: {field_type} = {{keys: {dict_keys}...}}")
            else:
                print(f"{indent}  - {field}: {field_type} = {{}}")
        else:
            preview = str(field_value)[:50]
            print(f"{indent}  - {field}: {field_type} = {repr(preview)}")
    print()  # 空行分隔

# ============================================================================
# 模块 2: 类型检测模块
# 职责: 自动识别数据集的题目类型和子类型
# ============================================================================

def detect_task_type(sample):
    """
    检测题目类型（支持大小写不敏感的字段名匹配）
    
    返回:
        task_type: 'multiple_choice', 'binary_choice', 'qa', 'reading_comprehension', 
                   'math_reasoning', 'code_generation', 'hellaswag', 'unknown'
        task_subtype: 子类型，如 'winogrande', 'piqa', 'squad', 'gsm8k', 'glue', 'truthful_qa', 'hellaswag' 等
    """
    fields = list(sample.keys())
    # 创建字段名到小写的映射（用于大小写不敏感匹配）
    fields_lower_map = {field.lower(): field for field in fields}
    
    def has_field(field_name):
        """检查字段是否存在（大小写不敏感）"""
        return field_name.lower() in fields_lower_map
    
    def get_field(field_name):
        """获取字段值（大小写不敏感）"""
        field_key = fields_lower_map.get(field_name.lower())
        return sample.get(field_key) if field_key else None
    
    # 1. 代码生成
    # 支持 prompt + code/canonical_solution（标准格式）
    if has_field('prompt') and (has_field('code') or has_field('canonical_solution')):
        return 'code_generation', None
    # 支持 text + code + test_list（mbpp full config 格式）
    if has_field('text') and has_field('code') and has_field('test_list'):
        return 'code_generation', None
    
    # 2. 数学推理（有 question 和 answer，answer 通常包含推理步骤）
    if has_field('question') and has_field('answer'):
        answer = get_field('answer')
        if answer is None:
            answer = ''
        if isinstance(answer, str) and ('<<' in answer or '####' in answer or '\n' in answer):
            return 'math_reasoning', 'gsm8k'
    
    # 3. 阅读理解（有 context 和 question）
    if has_field('context') and has_field('question'):
        if has_field('answers') or has_field('answer'):
            return 'reading_comprehension', 'squad'
    
    # 4. HellaSwag 格式（ctx + endings + label）
    if has_field('ctx') and has_field('endings') and has_field('label'):
        endings = get_field('endings')
        if endings is not None:
            if hasattr(endings, 'as_py'):
                endings = endings.as_py()
            if isinstance(endings, (list, tuple)) and len(endings) >= 4:
                return 'multiple_choice', 'hellaswag'
    
    # 5. 二选一类型
    if has_field('option1') and has_field('option2'):
        return 'binary_choice', 'winogrande'
    if has_field('sol1') and has_field('sol2'):
        return 'binary_choice', 'piqa'
    
    # 6. GLUE 格式（支持多种字段组合）
    # 标准格式：premise/hypothesis 或 sentence1/sentence2
    # 变体格式：sentence (cola/sst2), question1/question2 (qqp), question/sentence (qnli), text1/text2 (MNLI)
    glue_patterns = [
        ('premise', 'hypothesis'),           # 标准 GLUE
        ('sentence1', 'sentence2'),          # 标准 GLUE (RTE, WNLI, MRPC, STSB)
        ('text1', 'text2'),                  # MNLI 直接加载格式
        ('question1', 'question2'),          # QQP 格式
        ('question', 'sentence'),            # QNLI 格式
    ]
    
    # 单句格式（需要 label）
    single_sentence_fields = ['sentence']    # CoLA, SST-2
    
    # 检查双字段格式
    for field1, field2 in glue_patterns:
        if has_field(field1) and has_field(field2):
            if has_field('label'):
                return 'multiple_choice', 'glue'
    
    # 检查单句格式（CoLA, SST-2）
    for field in single_sentence_fields:
        if has_field(field) and has_field('label'):
            # 确保不是其他类型（排除只有 text 字段的语言建模数据集）
            if not has_field('text') or len(fields) > 2:  # 如果有其他字段，可能是任务型
                return 'multiple_choice', 'glue'
    
    # 7. 多选题（有 choices 列表或字典）
    if has_field('choices'):
        choices = get_field('choices')
        if choices is None:
            choices = []
        if hasattr(choices, 'as_py') and callable(getattr(choices, 'as_py', None)):
            try:
                choices = choices.as_py()  # type: ignore
            except Exception:
                pass
        # 支持列表格式
        if isinstance(choices, (list, tuple)) and len(choices) >= 2:
            return 'multiple_choice', None
        # 支持字典格式（ARC/OpenBookQA: {"text": [...], "label": [...]}）
        if isinstance(choices, dict):
            if 'text' in choices:
                texts = choices.get('text', [])
                if hasattr(texts, 'as_py'):
                    texts = texts.as_py()
                if isinstance(texts, (list, tuple)) and len(texts) >= 2:
                    return 'multiple_choice', None
    
    # 8. TruthfulQA multiple_choice 格式（mc1_targets/mc2_targets）
    if has_field('mc1_targets') or has_field('mc2_targets'):
        return 'multiple_choice', 'truthful_qa'
    
    # 9. 问答类型（有 question 和 answer，但没有 choices）
    if has_field('question'):
        # 检查是否有 answer 相关字段
        if has_field('answer') or has_field('best_answer'):
            # 如果没有 choices 字段，或者只有 correct_answers/incorrect_answers（TruthfulQA generation）
            if not has_field('choices') and not has_field('mc1_targets'):
                # TruthfulQA generation 有 correct_answers 和 incorrect_answers，但这是问答类型
                if has_field('correct_answers') and has_field('best_answer'):
                    return 'qa', None
                elif has_field('answer'):
                    return 'qa', None
    
    # 10. 语言建模数据集（用于 perplexity 评估）
    # 只有 text 字段，没有 question/answer/choices 等任务型字段
    if has_field('text') and len(fields) <= 2:
        # 如果只有 text 字段，或者只有 text 和 domain 字段（Lambada plain_text）
        # 这是语言建模数据集，用于 perplexity 评估
        if not has_field('question') and not has_field('answer') and not has_field('choices'):
            return 'language_modeling', None
    
    # 11. GPQA 格式（Question + Correct Answer + Incorrect Answer 1/2/3）
    # 支持大小写不敏感的字段名匹配（注意：Correct Answer 是带空格的，不是下划线）
    if has_field('question'):
        # 检查是否有 Correct Answer 字段（支持大小写不敏感，支持空格/下划线变体）
        # 直接检查字段名中是否包含 'correct' 和 'answer'（忽略空格和下划线）
        correct_answer_field = None
        for field in fields:
            field_lower = field.lower()
            # 移除空格和下划线后检查
            field_normalized = field_lower.replace(' ', '').replace('_', '')
            if 'correct' in field_normalized and 'answer' in field_normalized:
                correct_answer_field = field
                break
        
        if correct_answer_field:
            # 检查是否有 Incorrect Answer 字段（支持大小写不敏感）
            incorrect_answer_fields = [f for f in fields if 'incorrect' in f.lower() and 'answer' in f.lower()]
            if len(incorrect_answer_fields) >= 1:
                return 'multiple_choice', None
    
    return 'unknown', None

def try_load_dataset(dataset_dir, config=None):
    """尝试多种方式加载数据集"""
    print(f"\n尝试加载数据集...")
    
    # 方法1: 直接使用目录路径（如果有 README.md）
    readme_path = os.path.join(dataset_dir, 'README.md')
    if os.path.exists(readme_path):
        try:
            print("  方法1: 从本地目录加载（使用 README.md）...")
            
            # 如果指定了 config，使用它
            if config:
                dataset = load_dataset(dataset_dir, config)
            else:
                # 尝试检测 configs
                configs = detect_configs_from_readme(dataset_dir)
                if configs:
                    print(f"  检测到 {len(configs)} 个 configs: {configs}")
                    if len(configs) == 1:
                        print(f"  使用唯一的 config: {configs[0]}")
                        dataset = load_dataset(dataset_dir, configs[0])
                    else:
                        # 尝试使用第一个 config
                        print(f"  尝试使用第一个 config: {configs[0]}")
                        dataset = load_dataset(dataset_dir, configs[0])
                else:
                    dataset = load_dataset(dataset_dir)
            
            print("  ✅ 成功!")
            return dataset, None
        except Exception as e:
            error_msg = str(e)
            # 检查是否需要指定 config
            if 'Config name is missing' in error_msg or 'pick one among' in error_msg:
                configs = detect_configs_from_readme(dataset_dir)
                if configs:
                    print(f"  ⚠️  需要指定 config，检测到: {configs}")
                    print(f"  尝试使用第一个 config: {configs[0]}")
                    try:
                        dataset = load_dataset(dataset_dir, configs[0])
                        print("  ✅ 成功!")
                        return dataset, configs[0]  # 返回使用的 config
                    except Exception as e2:
                        print(f"  ❌ 失败: {e2}")
                else:
                    print(f"  ❌ 失败: {e}")
            else:
                print(f"  ❌ 失败: {e}")
    
    # 方法2: 从 parquet 文件加载
    data_files = find_data_files(dataset_dir)
    if data_files:
        try:
            print("  方法2: 从 parquet 文件加载...")
            # 构建 data_files 字典
            parquet_data_files = {}
            for split, files in data_files.items():
                if len(files) == 1:
                    parquet_data_files[split] = files[0]
                else:
                    parquet_data_files[split] = files
            
            dataset = load_dataset('parquet', data_files=parquet_data_files)
            print("  ✅ 成功!")
            return dataset, None
        except Exception as e:
            print(f"  ❌ 失败: {e}")
    
    # 方法3: 尝试从 HuggingFace Hub 加载（使用目录名）
    dataset_name = os.path.basename(dataset_dir.rstrip('/'))
    try:
        print(f"  方法3: 尝试从 HuggingFace Hub 加载 '{dataset_name}'...")
        dataset = load_dataset(dataset_name)
        print("  ✅ 成功!")
        return dataset, None
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    
    return None, "无法加载数据集，请检查目录结构或网络连接"

# ============================================================================
# 模块 3: 数据解析辅助函数模块
# 职责: 字段提取、格式转换、数据标准化
# ============================================================================

def _extract_choices_from_dict(choices_dict):
    """从字典格式的 choices 中提取选项列表（ARC/OpenBookQA 格式）"""
    if not isinstance(choices_dict, dict):
        return None
    
    if 'text' in choices_dict:
        texts = choices_dict.get('text', [])
        labels = choices_dict.get('label', [])
        
        # 处理 pyarrow 类型
        if hasattr(texts, 'as_py'):
            texts = texts.as_py()
        if hasattr(labels, 'as_py'):
            labels = labels.as_py()
        
        if isinstance(texts, (list, tuple)):
            out = [str(x) if not hasattr(x, 'as_py') else str(x.as_py()) for x in texts]
            # 如果有 label，按 A,B,C,D 顺序排列
            if labels and len(labels) == len(out) and isinstance(labels, (list, tuple)):
                try:
                    order = [ord(str(l).upper()[0]) - ord('A') for l in labels]
                    if all(0 <= i < 4 for i in order):
                        sorted_pairs = sorted(zip(order, out))
                        out = [t for _, t in sorted_pairs]
                except:
                    pass
            return out[:4]
    return None

def _build_glue_question(sample, fields):
    """构建 GLUE 格式的问题（支持多种字段组合）"""
    # 尝试多种字段组合
    field1 = None
    field2 = None
    prefix1 = ""
    prefix2 = ""
    
    # 优先级顺序：标准格式 -> 变体格式 -> 单句格式
    if 'premise' in sample and 'hypothesis' in sample:
        field1 = sample.get('premise')
        field2 = sample.get('hypothesis')
        prefix1, prefix2 = "Premise:", "Hypothesis:"
    elif 'sentence1' in sample and 'sentence2' in sample:
        field1 = sample.get('sentence1')
        field2 = sample.get('sentence2')
        prefix1, prefix2 = "Sentence 1:", "Sentence 2:"
    elif 'text1' in sample and 'text2' in sample:
        field1 = sample.get('text1')
        field2 = sample.get('text2')
        prefix1, prefix2 = "Text 1:", "Text 2:"
    elif 'question1' in sample and 'question2' in sample:
        field1 = sample.get('question1')
        field2 = sample.get('question2')
        prefix1, prefix2 = "Question 1:", "Question 2:"
    elif 'question' in sample and 'sentence' in sample:
        field1 = sample.get('question')
        field2 = sample.get('sentence')
        prefix1, prefix2 = "Question:", "Sentence:"
    elif 'sentence' in sample:
        # 单句格式（CoLA, SST-2）
        field1 = sample.get('sentence')
        field2 = None
        prefix1 = "Sentence:"
        prefix2 = ""
    
    # 处理 pyarrow 类型
    if field1 is not None and hasattr(field1, 'as_py'):
        field1 = field1.as_py()
    if field2 is not None and hasattr(field2, 'as_py'):
        field2 = field2.as_py()
    
    field1 = str(field1).strip() if field1 else ""
    field2 = str(field2).strip() if field2 else ""
    
    # 构建问题
    if field1:
        if field2:
            # 双字段格式
            q = f"{prefix1} {field1} {prefix2} {field2}".strip()
        else:
            # 单句格式
            q = f"{prefix1} {field1}".strip()
        
        if not q.endswith("Answer:"):
            q = q.rstrip() + " Answer:"
        return q
    
    return None

def _get_glue_choices(dataset_kind='glue', sample=None):
    """获取 GLUE 格式的选项"""
    # 根据 dataset_kind 和样本字段判断选项
    dataset_kind_lower = dataset_kind.lower() if dataset_kind else ''
    
    # 2 选项任务
    if dataset_kind_lower in ('rte', 'qnli'):
        choices = ["entailment", "not entailment"]
    elif dataset_kind_lower == 'cola':
        choices = ["acceptable", "unacceptable"]
    elif dataset_kind_lower == 'sst2':
        choices = ["positive", "negative"]
    elif dataset_kind_lower == 'qqp':
        choices = ["duplicate", "not duplicate"]
    # 3 选项任务（MNLI）
    elif dataset_kind_lower in ('mnli', 'mnli_matched', 'mnli_mismatched'):
        # 检查是否有 label_text 字段来确定选项
        if sample and 'label_text' in sample:
            label_text = sample.get('label_text')
            if hasattr(label_text, 'as_py'):
                label_text = label_text.as_py()
            # 如果 label_text 是 "entailment", "neutral", "contradiction" 之一
            if str(label_text).lower() in ['entailment', 'neutral', 'contradiction']:
                choices = ["entailment", "neutral", "contradiction"]
            else:
                choices = ["entailment", "neutral", "contradiction"]  # 默认
        else:
            choices = ["entailment", "neutral", "contradiction"]
    # 默认 3 选项（其他 GLUE 任务）
    else:
        choices = ["entailment", "neutral", "contradiction"]
    
    # 补齐到 4 项（perplexity.cpp 要求至少 4 个选项）
    while len(choices) < 4:
        choices.append("")
    return choices[:4]

def _normalize_question(question_text):
    """标准化问题格式，确保以 Answer: 结尾"""
    question = str(question_text).strip()
    if not question:
        return None
    
    if not question.endswith("Answer:") and not question.endswith("Answer"):
        question = question.rstrip()
        if not question.endswith("?"):
            question += "?"
        question += " Answer:"
    
    return question

def _parse_answer_to_index(answer, valid_choices):
    """解析答案字符串为选项索引"""
    if answer is None or answer == '' or answer == -1:
        return 0, True  # 返回默认索引和是否为空答案
    
    if hasattr(answer, 'as_py'):
        answer = answer.as_py()
    
    answer_str = str(answer).strip()
    answer_upper = answer_str.upper()
    
    # 尝试解析答案
    if answer_upper in ('A', 'B', 'C', 'D'):
        answer_idx = ord(answer_upper) - ord('A')
    elif answer_str.isdigit():
        answer_idx = max(0, min(int(answer_str), len(valid_choices) - 1))
    else:
        # 尝试在选项中找到匹配
        answer_idx = 0
        for idx, choice in enumerate(valid_choices):
            if str(choice).strip() == answer_str or str(choice).strip().lower() == answer_str.lower():
                answer_idx = idx
                break
    
    return max(0, min(answer_idx, len(valid_choices) - 1)), False

def _print_conversion_stats(stats):
    """打印转换统计信息"""
    print(f"\n转换结果: 成功 {stats['success']}, 跳过 {stats['skipped']}")
    
    if stats.get('empty_answers', 0) > 0:
        empty_ratio = stats['empty_answers'] / stats['success'] * 100 if stats['success'] > 0 else 0
        if empty_ratio > 50:
            print(f"⚠️  注意: {stats['empty_answers']}/{stats['success']} ({empty_ratio:.1f}%) 的答案为空（可能是测试集）")
    
    if stats['warnings']:
        print(f"警告: {len(stats['warnings'])} 条")

# ============================================================================
# 模块 4: 二进制打包模块
# 职责: 将多选题数据打包为与 perplexity.cpp 兼容的二进制格式
# ============================================================================

def _bin_pack_string(s: str) -> bytes:
    """打包字符串为二进制格式"""
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b

def _bin_pack_mc(answers, labels):
    """打包多选题答案和标签"""
    data = struct.pack("<I", len(answers))
    for a in answers:
        data += _bin_pack_string(str(a))
    data += struct.pack("<" + "i" * len(labels), *labels)
    return data

def _bin_pack_task(question: str, answers, correct_idx: int) -> bytes:
    """打包单个多选题任务"""
    n = len(answers)
    if n == 0:
        return b""
    labels = [1 if i == correct_idx else 0 for i in range(n)]
    # mc2 留空（n=0），与 perplexity.cpp 格式一致
    return _bin_pack_string(question) + _bin_pack_mc(answers, labels) + _bin_pack_mc([], [])

def _write_binary_tasks(tasks, output_path):
    """将任务列表写入二进制文件（与 perplexity.cpp 兼容）"""
    if not tasks:
        return
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'wb') as f:
        # 写入任务数量和位置数组
        n_tasks = len(tasks)
        f.write(struct.pack("<I", n_tasks))
        
        # 先序列化所有任务，计算位置
        task_bytes_list = []
        for q, answers, idx in tasks:
            task_bytes_list.append(_bin_pack_task(q, answers, idx))
        
        # 计算位置
        header_size = 4 + n_tasks * 4
        task_pos = []
        pos = header_size
        for tb in task_bytes_list:
            task_pos.append(pos)
            pos += len(tb)
        
        # 写入位置数组
        f.write(struct.pack("<" + "I" * len(task_pos), *task_pos))
        
        # 写入任务数据
        for tb in task_bytes_list:
            f.write(tb)

# ============================================================================
# 模块 5: 转换函数模块
# 职责: 按题目类型分类的转换函数
# ============================================================================

# ----------------------------------------------------------------------------
# 5.1 多选题转换函数
# ----------------------------------------------------------------------------

def convert_single_config(dataset_dir, config_name, output_dir, splits=None, error_collector=None):
    """转换单个 config 的数据集
    
    Args:
        dataset_dir: 数据集目录
        config_name: config 名称
        output_dir: 输出目录
        splits: 要转换的 splits
        error_collector: 错误收集器字典，用于收集错误信息
    
    Returns:
        (success: bool, errors: list, conversion_info: dict) - 是否成功、错误列表和转换信息
        conversion_info 包含: task_type, task_subtype (如果成功转换)
    """
    errors = []
    conversion_info: Dict[str, Any] = {"task_type": None, "task_subtype": None}
    print(f"\n{'=' * 80}")
    print(f"转换 Config: {config_name}")
    print(f"{'=' * 80}")
    
    # 尝试加载该 config 的数据集
    dataset, error = try_load_dataset(dataset_dir, config=config_name)
    if dataset is None:
        error_msg = f"无法加载 config '{config_name}': {error}"
        print(f"  ❌ {error_msg}")
        errors.append(error_msg)
        if error_collector:
            error_collector.setdefault('configs', {}).setdefault(config_name, []).append(error_msg)
        return False, errors
    
    # 确定要转换的 splits
    if hasattr(dataset, 'keys'):
        available_splits = list(dataset.keys())
    else:
        available_splits = ['default']
        dataset = {'default': dataset}
    
    if splits is None:
        splits = available_splits
    else:
        splits = [s for s in splits if s in available_splits]
    
    if not splits:
        error_msg = f"Config '{config_name}' 没有可用的 splits"
        print(f"  ⚠️  {error_msg}")
        errors.append(error_msg)
        if error_collector:
            error_collector.setdefault('configs', {}).setdefault(config_name, []).append(error_msg)
        return False, errors
    
    print(f"  可用的 splits: {available_splits}")
    print(f"  将转换的 splits: {splits}")
    
    # 为这个 config 创建子目录
    config_output_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_output_dir, exist_ok=True)
    print(f"  输出目录: {config_output_dir}")
    
    # 转换每个 split
    all_success = True
    generated_files = []  # 记录生成的文件路径
    
    for split in splits:
        print(f"\n  {'-' * 78}")
        print(f"  转换 split: {split}")
        print(f"  {'-' * 78}")
        
        try:
            split_dataset = dataset[split] if hasattr(dataset, 'keys') else dataset
            
            if len(split_dataset) == 0:
                error_msg = f"Config '{config_name}' / Split '{split}' 为空，跳过"
                print(f"    ⚠️  {error_msg}")
                errors.append(error_msg)
                if error_collector:
                    error_collector.setdefault('configs', {}).setdefault(config_name, {}).setdefault('splits', {}).setdefault(split, []).append(error_msg)
                continue
            
            sample = split_dataset[0]  # type: ignore
            
            # 首先打印所有可用字段
            print_dataset_fields(sample, indent="    ")
            
            # 检测题目类型
            task_type, task_subtype = detect_task_type(sample)
            print(f"    检测到的题目类型: {task_type}" + (f" ({task_subtype})" if task_subtype else ""))
            
            # 保存转换信息（用于更新配置文件）
            if not conversion_info["task_type"]:
                conversion_info["task_type"] = task_type
                conversion_info["task_subtype"] = task_subtype
            
            fields = detect_fields(sample)
            print(f"    检测到的关键字段:")
            print(f"      题目字段: {fields['question_field'] or '❌ 未找到'}")
            print(f"      选项字段: {fields['choices_field'] or '❌ 未找到'}")
            print(f"      答案字段: {fields['answer_field'] or '❌ 未找到'}")
            
            # 根据题目类型选择转换方法
            # 多选题输出 .bin 格式（与 perplexity.cpp 兼容），其他类型输出 .txt
            if task_type == 'multiple_choice':
                output_path = os.path.join(config_output_dir, f"{split}.bin")
            else:
                output_path = os.path.join(config_output_dir, f"{split}.txt")
            
            if task_type == 'multiple_choice':
                # 多选题：根据 subtype 和 config 名称选择相应的转换函数
                if task_subtype == 'hellaswag':
                    # HellaSwag 格式（6行文本格式）
                    output_path = os.path.join(config_output_dir, f"{split}.txt")
                    convert_hellaswag_format(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        generated_files.append(output_path)
                elif task_subtype == 'glue':
                    # GLUE 格式（支持多种子任务）
                    dataset_kind = 'glue'
                    config_lower = config_name.lower()
                    dir_lower = dataset_dir.lower()
                    
                    # 根据 config 名称或目录名称判断子任务类型
                    if 'rte' in config_lower or 'rte' in dir_lower:
                        dataset_kind = 'rte'
                    elif 'qnli' in config_lower or 'qnli' in dir_lower:
                        dataset_kind = 'qnli'
                    elif 'mnli' in config_lower or 'mnli' in dir_lower:
                        dataset_kind = 'mnli'
                    elif 'cola' in config_lower or 'cola' in dir_lower:
                        dataset_kind = 'cola'
                    elif 'sst2' in config_lower or 'sst' in config_lower or 'sst-2' in config_lower:
                        dataset_kind = 'sst2'
                    elif 'qqp' in config_lower or 'qqp' in dir_lower:
                        dataset_kind = 'qqp'
                    
                    convert_multiple_choice_glue(split_dataset, output_path, fields, dataset_kind)
                    if os.path.exists(output_path):
                        generated_files.append(output_path)
                elif task_subtype == 'truthful_qa':
                    # TruthfulQA 格式（mc1_targets/mc2_targets）
                    convert_multiple_choice_truthful_qa(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        generated_files.append(output_path)
                elif 'arc' in config_name.lower() or 'arc' in dataset_dir.lower() or \
                     'openbookqa' in config_name.lower() or 'openbookqa' in dataset_dir.lower():
                    # ARC/OpenBookQA 格式（字典格式的 choices）
                    convert_multiple_choice_arc(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        generated_files.append(output_path)
                else:
                    # 标准多选题格式（列表格式的 choices）
                    convert_multiple_choice_standard(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        generated_files.append(output_path)
            
            elif task_type == 'binary_choice':
                # 二选一
                if task_subtype == 'winogrande':
                    convert_winogrande_format(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        generated_files.append(output_path)
                elif task_subtype == 'piqa':
                    convert_piqa_format(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        generated_files.append(output_path)
                else:
                    error_msg = f"Config '{config_name}' / Split '{split}': 未知的二选一格式"
                    print(f"    ❌ {error_msg}")
                    errors.append(error_msg)
                    all_success = False
                    if error_collector:
                        error_collector.setdefault('configs', {}).setdefault(config_name, {}).setdefault('splits', {}).setdefault(split, []).append(error_msg)
                    continue
                    
            elif task_type == 'qa':
                # 问答类型
                if not fields['question_field'] or not fields['answer_field']:
                    error_msg = f"Config '{config_name}' / Split '{split}': 问答类型缺少必需字段"
                    print(f"    ❌ {error_msg}")
                    errors.append(error_msg)
                    all_success = False
                    if error_collector:
                        error_collector.setdefault('configs', {}).setdefault(config_name, {}).setdefault('splits', {}).setdefault(split, []).append(error_msg)
                    continue
                convert_qa_format(split_dataset, output_path, fields)
                if os.path.exists(output_path):
                    generated_files.append(output_path)
                
            elif task_type == 'reading_comprehension':
                # 阅读理解
                convert_reading_comprehension_format(split_dataset, output_path, fields)
                if os.path.exists(output_path):
                    generated_files.append(output_path)
                
            elif task_type == 'math_reasoning':
                # 数学推理
                convert_math_reasoning_format(split_dataset, output_path, fields)
                if os.path.exists(output_path):
                    generated_files.append(output_path)
                
            elif task_type == 'code_generation':
                # 代码生成
                convert_code_generation_format(split_dataset, output_path, fields)
                if os.path.exists(output_path):
                    generated_files.append(output_path)
                
            elif task_type == 'language_modeling':
                # 语言建模数据集（用于 perplexity 评估）
                convert_language_modeling_format(split_dataset, output_path, fields)
                if os.path.exists(output_path):
                    generated_files.append(output_path)
                
            else:
                # 尝试兼容旧逻辑
                if not fields['choices_field']:
                    if 'option1' in sample and 'option2' in sample:
                        print(f"    检测到 Winogrande 格式（option1/option2）")
                        convert_winogrande_format(split_dataset, output_path, fields)
                        if os.path.exists(output_path):
                            generated_files.append(output_path)
                    elif 'sol1' in sample and 'sol2' in sample:
                        print(f"    检测到 PIQA 格式（sol1/sol2）")
                        convert_piqa_format(split_dataset, output_path, fields)
                        if os.path.exists(output_path):
                            generated_files.append(output_path)
                    else:
                        error_msg = f"Config '{config_name}' / Split '{split}': 无法识别题目类型（可用字段: {list(sample.keys())}）"
                        print(f"    ⚠️  无法识别题目类型")
                        print(f"    可用字段: {list(sample.keys())}")
                        print(f"    ❌ 跳过 split '{split}'（不支持此类型）")
                        errors.append(error_msg)
                        all_success = False
                        if error_collector:
                            error_collector.setdefault('configs', {}).setdefault(config_name, {}).setdefault('splits', {}).setdefault(split, []).append(error_msg)
                        continue
                elif not fields['question_field']:
                    error_msg = f"Config '{config_name}' / Split '{split}': 无法识别题目字段"
                    print(f"    ❌ {error_msg}")
                    errors.append(error_msg)
                    all_success = False
                    if error_collector:
                        error_collector.setdefault('configs', {}).setdefault(config_name, {}).setdefault('splits', {}).setdefault(split, []).append(error_msg)
                    continue
                else:
                    # 根据字段判断格式类型
                    sample_choices = sample.get('choices')
                    if isinstance(sample_choices, dict):
                        # ARC/OpenBookQA 格式（字典格式的 choices）
                        convert_multiple_choice_arc(split_dataset, output_path, fields)
                        if os.path.exists(output_path):
                            generated_files.append(output_path)
                    elif 'premise' in sample or 'hypothesis' in sample or 'sentence1' in sample or 'sentence2' in sample:
                        # GLUE 格式
                        dataset_kind = 'glue'
                        if 'rte' in config_name.lower() or 'rte' in dataset_dir.lower():
                            dataset_kind = 'rte'
                        elif 'qnli' in config_name.lower() or 'qnli' in dataset_dir.lower():
                            dataset_kind = 'qnli'
                        elif 'mnli' in config_name.lower() or 'mnli' in dataset_dir.lower():
                            dataset_kind = 'mnli'
                        convert_multiple_choice_glue(split_dataset, output_path, fields, dataset_kind)
                        if os.path.exists(output_path):
                            generated_files.append(output_path)
                    elif 'mc1_targets' in sample or 'mc2_targets' in sample:
                        # TruthfulQA 格式
                        convert_multiple_choice_truthful_qa(split_dataset, output_path, fields)
                        if os.path.exists(output_path):
                            generated_files.append(output_path)
                    else:
                        # 标准多选题格式
                        convert_multiple_choice_standard(split_dataset, output_path, fields)
                        if os.path.exists(output_path):
                            generated_files.append(output_path)
            
        except Exception as e:
            import traceback
            error_msg = f"Config '{config_name}' / Split '{split}' 转换时出错: {str(e)}"
            error_trace = traceback.format_exc()
            print(f"    ❌ {error_msg}")
            print(f"    详细错误:\n{error_trace}")
            errors.append(f"{error_msg}\n{error_trace}")
            all_success = False
            if error_collector:
                error_collector.setdefault('configs', {}).setdefault(config_name, {}).setdefault('splits', {}).setdefault(split, []).append(f"{error_msg}\n{error_trace}")
            continue
    
    # 将生成的文件路径添加到 conversion_info
    conversion_info['generated_files'] = generated_files
    
    return all_success, errors, conversion_info

def auto_convert_dataset(dataset_dir, output_subdir='converted', splits=None, config=None):
    """
    自动转换数据集
    
    Args:
        dataset_dir: 数据集目录路径
        output_subdir: 输出子目录名（默认 'converted'）
        splits: 要转换的 splits 列表（如果为 None，转换所有可用的 splits）
        config: 数据集 config 名称（如果指定，只转换该 config；如果为 None，转换所有 configs）
    """
    dataset_dir = os.path.abspath(dataset_dir)
    
    # 自动查找配置文件：在脚本所在目录查找 datasets_model_evaluation_config.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(script_dir, 'datasets_model_evaluation_config.json')
    
    if not os.path.isdir(dataset_dir):
        print(f"❌ 错误: 目录不存在: {dataset_dir}")
        return False
    
    print("=" * 80)
    print(f"自动转换数据集")
    print("=" * 80)
    print(f"数据集目录: {dataset_dir}")
    
    # 检测 configs
    configs = detect_configs_from_readme(dataset_dir)
    
    # 创建输出目录
    output_dir = os.path.join(dataset_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # 确定要转换的 configs
    if config:
        # 只转换指定的 config
        configs_to_convert = [config]
        print(f"将转换指定的 config: {config}")
    elif configs and len(configs) > 1:
        # 有多个 configs，转换所有
        configs_to_convert = configs
        print(f"\n检测到 {len(configs)} 个 configs: {configs}")
        print(f"将转换所有 configs（每个 config 将创建独立的子目录）")
    elif configs and len(configs) == 1:
        # 只有一个 config，如果名字是 'default'，直接放在 converted 目录下
        if configs[0] == 'default':
            configs_to_convert = None  # 标记为无 config
            print(f"\n检测到 1 个 config: {configs[0]}（将直接放在 converted 目录下）")
        else:
            configs_to_convert = configs
            print(f"\n检测到 1 个 config: {configs[0]}")
    else:
        # 没有 configs，尝试直接加载
        print(f"\n未检测到 configs，尝试直接加载数据集...")
        dataset, error = try_load_dataset(dataset_dir, config=None)
        if dataset is None:
            print(f"❌ {error}")
            return False
        
        # 确定要转换的 splits
        if hasattr(dataset, 'keys'):
            available_splits = list(dataset.keys())
        else:
            available_splits = ['default']
            dataset = {'default': dataset}
        
        if splits is None:
            splits = available_splits
        else:
            splits = [s for s in splits if s in available_splits]
        
        if not splits:
            print(f"❌ 错误: 没有可用的 splits 进行转换")
            return False
        
        print(f"可用的 splits: {available_splits}")
        print(f"将转换的 splits: {splits}")
        
        # 直接转换（没有 config）
        all_success = True
        conversion_info: Dict[str, Any] = {"task_type": None, "task_subtype": None, "generated_files": []}
        
        for split in splits:
            print(f"\n{'=' * 80}")
            print(f"转换 split: {split}")
            print(f"{'=' * 80}")
            
            try:
                split_dataset = dataset[split] if hasattr(dataset, 'keys') else dataset
                
                if len(split_dataset) == 0:
                    print(f"⚠️  Split '{split}' 为空，跳过")
                    continue
                
                sample = split_dataset[0]  # type: ignore
                
                # 首先打印所有可用字段
                print_dataset_fields(sample, indent="  ")
                
                # 检测题目类型
                task_type, task_subtype = detect_task_type(sample)
                print(f"检测到的题目类型: {task_type}" + (f" ({task_subtype})" if task_subtype else ""))
                
                # 保存转换信息（用于更新配置文件）
                if not conversion_info["task_type"]:
                    conversion_info["task_type"] = task_type
                    conversion_info["task_subtype"] = task_subtype
                
                fields = detect_fields(sample)
                print(f"检测到的关键字段:")
                print(f"  题目字段: {fields['question_field'] or '❌ 未找到'}")
                print(f"  选项字段: {fields['choices_field'] or '❌ 未找到'}")
                print(f"  答案字段: {fields['answer_field'] or '❌ 未找到'}")
                
                # 根据题目类型选择转换方法
                output_path = os.path.join(output_dir, f"{split}.txt")
                
                # 多选题输出 .bin 格式（与 perplexity.cpp 兼容），其他类型输出 .txt
                if task_type == 'multiple_choice':
                    # 根据 subtype 判断输出格式
                    if task_subtype == 'hellaswag':
                        # HellaSwag 格式（6行文本格式）
                        output_path = os.path.join(output_dir, f"{split}.txt")
                        convert_hellaswag_format(split_dataset, output_path, fields)
                        if os.path.exists(output_path):
                            conversion_info['generated_files'].append(output_path)
                    else:
                        output_path = os.path.join(output_dir, f"{split}.bin")
                        # 根据 subtype 和字段判断格式
                        if task_subtype == 'glue':
                            dataset_kind = 'glue'
                            dir_lower = dataset_dir.lower()
                            
                            # 根据目录名称判断子任务类型
                            if 'rte' in dir_lower:
                                dataset_kind = 'rte'
                            elif 'qnli' in dir_lower:
                                dataset_kind = 'qnli'
                            elif 'mnli' in dir_lower:
                                dataset_kind = 'mnli'
                            elif 'cola' in dir_lower:
                                dataset_kind = 'cola'
                            elif 'sst2' in dir_lower or 'sst' in dir_lower or 'sst-2' in dir_lower:
                                dataset_kind = 'sst2'
                            elif 'qqp' in dir_lower:
                                dataset_kind = 'qqp'
                            
                            convert_multiple_choice_glue(split_dataset, output_path, fields, dataset_kind)
                            if os.path.exists(output_path):
                                conversion_info['generated_files'].append(output_path)
                        elif task_subtype == 'truthful_qa':
                            convert_multiple_choice_truthful_qa(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info['generated_files'].append(output_path)
                        elif 'arc' in dataset_dir.lower() or 'openbookqa' in dataset_dir.lower():
                            convert_multiple_choice_arc(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info['generated_files'].append(output_path)
                        else:
                            # 检查 choices 是否为字典格式
                            sample_choices = sample.get('choices')
                            if isinstance(sample_choices, dict):
                                convert_multiple_choice_arc(split_dataset, output_path, fields)
                                if os.path.exists(output_path):
                                    conversion_info['generated_files'].append(output_path)
                            else:
                                convert_multiple_choice_standard(split_dataset, output_path, fields)
                                if os.path.exists(output_path):
                                    conversion_info['generated_files'].append(output_path)
                else:
                    output_path = os.path.join(output_dir, f"{split}.txt")
                
                if task_type == 'binary_choice':
                    if task_subtype == 'winogrande':
                        convert_winogrande_format(split_dataset, output_path, fields)
                        if os.path.exists(output_path):
                            conversion_info['generated_files'].append(output_path)
                    elif task_subtype == 'piqa':
                        convert_piqa_format(split_dataset, output_path, fields)
                        if os.path.exists(output_path):
                            conversion_info['generated_files'].append(output_path)
                    else:
                        print(f"❌ 未知的二选一格式，跳过 split '{split}'")
                        all_success = False
                        continue
                elif task_type == 'qa':
                    if not fields['question_field'] or not fields['answer_field']:
                        print(f"❌ 问答类型缺少必需字段，跳过 split '{split}'")
                        all_success = False
                        continue
                    convert_qa_format(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        conversion_info['generated_files'].append(output_path)
                elif task_type == 'reading_comprehension':
                    convert_reading_comprehension_format(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        conversion_info['generated_files'].append(output_path)
                elif task_type == 'math_reasoning':
                    convert_math_reasoning_format(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        conversion_info['generated_files'].append(output_path)
                elif task_type == 'code_generation':
                    convert_code_generation_format(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        conversion_info['generated_files'].append(output_path)
                elif task_type == 'language_modeling':
                    # 语言建模数据集（用于 perplexity 评估）
                    convert_language_modeling_format(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        conversion_info['generated_files'].append(output_path)
                elif task_type == 'unknown':
                    # 尝试兼容旧逻辑
                    if not fields['choices_field']:
                        if 'option1' in sample and 'option2' in sample:
                            print("检测到 Winogrande 格式（option1/option2）")
                            convert_winogrande_format(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info['generated_files'].append(output_path)
                        elif 'sol1' in sample and 'sol2' in sample:
                            print("检测到 PIQA 格式（sol1/sol2）")
                            convert_piqa_format(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info['generated_files'].append(output_path)
                        else:
                            print("⚠️  无法识别题目类型")
                            print(f"可用字段: {list(sample.keys())}")
                            print(f"❌ 跳过 split '{split}'（不支持此类型）")
                            all_success = False
                            continue
                    elif not fields['question_field']:
                        print(f"❌ 无法识别题目字段，跳过 split '{split}'")
                        all_success = False
                        continue
                    else:
                        # 根据字段判断格式类型
                        output_path = os.path.join(output_dir, f"{split}.bin")
                        sample_choices = sample.get('choices')
                        if isinstance(sample_choices, dict):
                            # ARC/OpenBookQA 格式（字典格式的 choices）
                            convert_multiple_choice_arc(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info['generated_files'].append(output_path)
                        elif 'premise' in sample or 'hypothesis' in sample or 'sentence1' in sample or 'sentence2' in sample:
                            # GLUE 格式
                            dataset_kind = 'glue'
                            if 'rte' in dataset_dir.lower():
                                dataset_kind = 'rte'
                            elif 'qnli' in dataset_dir.lower():
                                dataset_kind = 'qnli'
                            elif 'mnli' in dataset_dir.lower():
                                dataset_kind = 'mnli'
                            convert_multiple_choice_glue(split_dataset, output_path, fields, dataset_kind)
                            if os.path.exists(output_path):
                                conversion_info['generated_files'].append(output_path)
                        elif 'mc1_targets' in sample or 'mc2_targets' in sample:
                            # TruthfulQA 格式
                            convert_multiple_choice_truthful_qa(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info['generated_files'].append(output_path)
                        else:
                            # 标准多选题格式
                            convert_multiple_choice_standard(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info['generated_files'].append(output_path)
                
            except Exception as e:
                print(f"❌ 转换 split '{split}' 时出错: {e}")
                import traceback
                traceback.print_exc()
                all_success = False
                continue
        
        print(f"\n{'=' * 80}")
        if all_success:
            print("✅ 所有 splits 转换完成!")
        else:
            print("⚠️  部分 splits 转换失败，请查看上面的错误信息")
        print(f"{'=' * 80}")
        
        # 如果转换成功，自动更新配置文件
        if all_success and conversion_info.get("task_type"):
            print(f"\n{'=' * 80}")
            print("更新配置文件")
            print(f"{'=' * 80}")
            
            handle_config_update(
                config_file_path,
                dataset_dir,
                conversion_info['task_type'],
                conversion_info['task_subtype'],
                conversion_info.get('generated_files', [])
            )
        
        return all_success
    
    # 转换每个 config
    if configs_to_convert is None:
        # 没有 configs 或只有一个 'default' config，直接转换
        dataset, error = try_load_dataset(dataset_dir, config=None)
        if dataset is None:
            print(f"❌ {error}")
            return False
        
        # 确定要转换的 splits
        if hasattr(dataset, 'keys'):
            available_splits = list(dataset.keys())
        else:
            available_splits = ['default']
            dataset = {'default': dataset}
        
        if splits is None:
            splits = available_splits
        else:
            splits = [s for s in splits if s in available_splits]
        
        if not splits:
            print(f"❌ 错误: 没有可用的 splits 进行转换")
            return False
        
        print(f"可用的 splits: {available_splits}")
        print(f"将转换的 splits: {splits}")
        
        # 直接转换（没有 config 子目录）
        all_success = True
        # 注意：这里使用不同的变量名避免与上面的 conversion_info 遮蔽
        conversion_info_none: Dict[str, Any] = {"task_type": None, "task_subtype": None, "generated_files": []}
        
        for split in splits:
            print(f"\n{'=' * 80}")
            print(f"转换 split: {split}")
            print(f"{'=' * 80}")
            
            try:
                split_dataset = dataset[split] if hasattr(dataset, 'keys') else dataset
                
                if len(split_dataset) == 0:
                    print(f"⚠️  Split '{split}' 为空，跳过")
                    continue
                
                sample = split_dataset[0]  # type: ignore
                
                # 首先打印所有可用字段
                print_dataset_fields(sample, indent="  ")
                
                # 检测题目类型
                task_type, task_subtype = detect_task_type(sample)
                print(f"检测到的题目类型: {task_type}" + (f" ({task_subtype})" if task_subtype else ""))
                
                # 保存转换信息（用于更新配置文件）
                if not conversion_info_none["task_type"]:
                    conversion_info_none["task_type"] = task_type
                    conversion_info_none["task_subtype"] = task_subtype
                
                fields = detect_fields(sample)
                print(f"检测到的关键字段:")
                print(f"  题目字段: {fields['question_field'] or '❌ 未找到'}")
                print(f"  选项字段: {fields['choices_field'] or '❌ 未找到'}")
                print(f"  答案字段: {fields['answer_field'] or '❌ 未找到'}")
                
                # 根据题目类型选择转换方法
                output_path = os.path.join(output_dir, f"{split}.txt")
                
                # 多选题输出 .bin 格式（与 perplexity.cpp 兼容），其他类型输出 .txt
                if task_type == 'multiple_choice':
                    # 根据 subtype 判断输出格式
                    if task_subtype == 'hellaswag':
                        # HellaSwag 格式（6行文本格式）
                        output_path = os.path.join(output_dir, f"{split}.txt")
                        convert_hellaswag_format(split_dataset, output_path, fields)
                        if os.path.exists(output_path):
                            conversion_info_none['generated_files'].append(output_path)
                    else:
                        output_path = os.path.join(output_dir, f"{split}.bin")
                        # 根据 subtype 和字段判断格式
                        if task_subtype == 'glue':
                            dataset_kind = 'glue'
                            dir_lower = dataset_dir.lower()
                            
                            # 根据目录名称判断子任务类型
                            if 'rte' in dir_lower:
                                dataset_kind = 'rte'
                            elif 'qnli' in dir_lower:
                                dataset_kind = 'qnli'
                            elif 'mnli' in dir_lower:
                                dataset_kind = 'mnli'
                            elif 'cola' in dir_lower:
                                dataset_kind = 'cola'
                            elif 'sst2' in dir_lower or 'sst' in dir_lower or 'sst-2' in dir_lower:
                                dataset_kind = 'sst2'
                            elif 'qqp' in dir_lower:
                                dataset_kind = 'qqp'
                            
                            convert_multiple_choice_glue(split_dataset, output_path, fields, dataset_kind)
                            if os.path.exists(output_path):
                                conversion_info_none['generated_files'].append(output_path)
                        elif task_subtype == 'truthful_qa':
                            convert_multiple_choice_truthful_qa(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info_none['generated_files'].append(output_path)
                        elif 'arc' in dataset_dir.lower() or 'openbookqa' in dataset_dir.lower():
                            convert_multiple_choice_arc(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info_none['generated_files'].append(output_path)
                        else:
                            # 检查 choices 是否为字典格式
                            sample_choices = sample.get('choices')
                            if isinstance(sample_choices, dict):
                                convert_multiple_choice_arc(split_dataset, output_path, fields)
                                if os.path.exists(output_path):
                                    conversion_info_none['generated_files'].append(output_path)
                            else:
                                convert_multiple_choice_standard(split_dataset, output_path, fields)
                                if os.path.exists(output_path):
                                    conversion_info_none['generated_files'].append(output_path)
                else:
                    output_path = os.path.join(output_dir, f"{split}.txt")
                
                if task_type == 'binary_choice':
                    if task_subtype == 'winogrande':
                        convert_winogrande_format(split_dataset, output_path, fields)
                        if os.path.exists(output_path):
                            conversion_info_none['generated_files'].append(output_path)
                    elif task_subtype == 'piqa':
                        convert_piqa_format(split_dataset, output_path, fields)
                        if os.path.exists(output_path):
                            conversion_info_none['generated_files'].append(output_path)
                    else:
                        print(f"❌ 未知的二选一格式，跳过 split '{split}'")
                        all_success = False
                        continue
                elif task_type == 'qa':
                    if not fields['question_field'] or not fields['answer_field']:
                        print(f"❌ 问答类型缺少必需字段，跳过 split '{split}'")
                        all_success = False
                        continue
                    convert_qa_format(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        conversion_info_none['generated_files'].append(output_path)
                elif task_type == 'reading_comprehension':
                    convert_reading_comprehension_format(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        conversion_info_none['generated_files'].append(output_path)
                elif task_type == 'math_reasoning':
                    convert_math_reasoning_format(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        conversion_info_none['generated_files'].append(output_path)
                elif task_type == 'code_generation':
                    convert_code_generation_format(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        conversion_info_none['generated_files'].append(output_path)
                elif task_type == 'language_modeling':
                    # 语言建模数据集（用于 perplexity 评估）
                    convert_language_modeling_format(split_dataset, output_path, fields)
                    if os.path.exists(output_path):
                        conversion_info_none['generated_files'].append(output_path)
                elif task_type == 'unknown':
                    # 尝试兼容旧逻辑
                    if not fields['choices_field']:
                        if 'option1' in sample and 'option2' in sample:
                            print("检测到 Winogrande 格式（option1/option2）")
                            convert_winogrande_format(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info_none['generated_files'].append(output_path)
                        elif 'sol1' in sample and 'sol2' in sample:
                            print("检测到 PIQA 格式（sol1/sol2）")
                            convert_piqa_format(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info_none['generated_files'].append(output_path)
                        else:
                            print("⚠️  无法识别题目类型")
                            print(f"可用字段: {list(sample.keys())}")
                            print(f"❌ 跳过 split '{split}'（不支持此类型）")
                            all_success = False
                            continue
                    elif not fields['question_field']:
                        print(f"❌ 无法识别题目字段，跳过 split '{split}'")
                        all_success = False
                        continue
                    else:
                        # 根据字段判断格式类型
                        output_path = os.path.join(output_dir, f"{split}.bin")
                        sample_choices = sample.get('choices')
                        if isinstance(sample_choices, dict):
                            # ARC/OpenBookQA 格式（字典格式的 choices）
                            convert_multiple_choice_arc(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info_none['generated_files'].append(output_path)
                        elif 'premise' in sample or 'hypothesis' in sample or 'sentence1' in sample or 'sentence2' in sample:
                            # GLUE 格式
                            dataset_kind = 'glue'
                            if 'rte' in dataset_dir.lower():
                                dataset_kind = 'rte'
                            elif 'qnli' in dataset_dir.lower():
                                dataset_kind = 'qnli'
                            elif 'mnli' in dataset_dir.lower():
                                dataset_kind = 'mnli'
                            convert_multiple_choice_glue(split_dataset, output_path, fields, dataset_kind)
                            if os.path.exists(output_path):
                                conversion_info_none['generated_files'].append(output_path)
                        elif 'mc1_targets' in sample or 'mc2_targets' in sample:
                            # TruthfulQA 格式
                            convert_multiple_choice_truthful_qa(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info_none['generated_files'].append(output_path)
                        else:
                            # 标准多选题格式
                            convert_multiple_choice_standard(split_dataset, output_path, fields)
                            if os.path.exists(output_path):
                                conversion_info_none['generated_files'].append(output_path)
                
            except Exception as e:
                print(f"❌ 转换 split '{split}' 时出错: {e}")
                import traceback
                traceback.print_exc()
                all_success = False
                continue
        
        print(f"\n{'=' * 80}")
        if all_success:
            print("✅ 所有 splits 转换完成!")
        else:
            print("⚠️  部分 splits 转换失败，请查看上面的错误信息")
        print(f"{'=' * 80}")
        
        # 如果转换成功，自动更新配置文件
        if all_success and conversion_info_none.get("task_type"):
            print(f"\n{'=' * 80}")
            print("更新配置文件")
            print(f"{'=' * 80}")
            
            handle_config_update(
                config_file_path,
                dataset_dir,
                conversion_info_none['task_type'],
                conversion_info_none['task_subtype'],
                conversion_info_none.get('generated_files', [])
            )
        
        return all_success
    
    # 转换每个 config（有多个 configs）
    error_collector = {
        'configs': {},
        'summary': {
            'total_configs': len(configs_to_convert),
            'successful_configs': [],
            'failed_configs': [],
            'total_errors': 0
        }
    }
    
    all_configs_success = True
    successful_conversions = []  # 存储成功转换的信息
    
    for config_name in configs_to_convert:
        success, errors, conversion_info = convert_single_config(dataset_dir, config_name, output_dir, splits, error_collector)
        if not success:
            all_configs_success = False
            error_collector['summary']['failed_configs'].append(config_name)
            error_collector['summary']['total_errors'] += len(errors)
        else:
            error_collector['summary']['successful_configs'].append(config_name)
            # 保存成功转换的信息
            successful_conversions.append({
                'config_name': config_name,
                'task_type': conversion_info.get('task_type'),
                'task_subtype': conversion_info.get('task_subtype'),
                'generated_files': conversion_info.get('generated_files', [])
            })
    
    # 输出最终总结
    print_final_summary(error_collector, output_dir)
    
    # 如果转换成功，自动更新配置文件
    if successful_conversions:
        print(f"\n{'=' * 80}")
        print("更新配置文件")
        print(f"{'=' * 80}")
        
        # TODO: 目前使用第一个成功转换的信息（如果有多个 config，可能需要分别处理，为每个 config 创建独立的配置条目）
        conv_info = successful_conversions[0]
        handle_config_update(
            config_file_path,
            dataset_dir,
            conv_info['task_type'],
            conv_info['task_subtype'],
            conv_info.get('generated_files', [])
        )
    
    return all_configs_success

def print_final_summary(error_collector, output_dir):
    """打印最终转换总结"""
    print(f"\n{'=' * 80}")
    print("转换总结")
    print(f"{'=' * 80}")
    
    summary = error_collector.get('summary', {})
    total_configs = summary.get('total_configs', 0)
    successful_configs = summary.get('successful_configs', [])
    failed_configs = summary.get('failed_configs', [])
    total_errors = summary.get('total_errors', 0)
    
    print(f"\n总体统计:")
    print(f"  总 configs: {total_configs}")
    print(f"  ✅ 成功: {len(successful_configs)}")
    print(f"  ❌ 失败: {len(failed_configs)}")
    print(f"  总错误数: {total_errors}")
    
    if successful_configs:
        print(f"\n✅ 成功转换的 configs ({len(successful_configs)}):")
        for config in successful_configs:
            print(f"    - {config}")
    
    if failed_configs:
        print(f"\n❌ 转换失败的 configs ({len(failed_configs)}):")
        for config in failed_configs:
            print(f"    - {config}")
        
        # 详细错误信息
        print(f"\n详细错误信息:")
        configs_errors = error_collector.get('configs', {})
        for config_name in failed_configs:
            config_errors = configs_errors.get(config_name, {})
            if isinstance(config_errors, list):
                # 直接错误列表
                print(f"\n  Config: {config_name}")
                for i, error in enumerate(config_errors, 1):
                    print(f"    {i}. {error}")
            elif isinstance(config_errors, dict):
                # 有 splits 的错误
                print(f"\n  Config: {config_name}")
                splits_errors = config_errors.get('splits', {})
                for split_name, split_error_list in splits_errors.items():
                    print(f"    Split '{split_name}':")
                    for i, error in enumerate(split_error_list, 1):
                        # 只显示错误的第一行（避免太长）
                        error_first_line = error.split('\n')[0]
                        print(f"      {i}. {error_first_line}")
                        error_lines = error.split('\n')
                        if len(error_lines) > 1:
                            print(f"         (还有 {len(error_lines) - 1} 行详细信息)")
        
        # 保存错误信息到文件
        error_log_path = os.path.join(output_dir, 'conversion_errors.log')
        try:
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("数据集转换错误日志\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"总体统计:\n")
                f.write(f"  总 configs: {total_configs}\n")
                f.write(f"  成功: {len(successful_configs)}\n")
                f.write(f"  失败: {len(failed_configs)}\n")
                f.write(f"  总错误数: {total_errors}\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("详细错误信息\n")
                f.write("=" * 80 + "\n\n")
                
                for config_name in failed_configs:
                    config_errors = configs_errors.get(config_name, {})
                    f.write(f"\n{'=' * 80}\n")
                    f.write(f"Config: {config_name}\n")
                    f.write(f"{'=' * 80}\n\n")
                    
                    if isinstance(config_errors, list):
                        for i, error in enumerate(config_errors, 1):
                            f.write(f"{i}. {error}\n\n")
                    elif isinstance(config_errors, dict):
                        splits_errors = config_errors.get('splits', {})
                        for split_name, split_error_list in splits_errors.items():
                            f.write(f"Split '{split_name}':\n")
                            for i, error in enumerate(split_error_list, 1):
                                f.write(f"  {i}. {error}\n\n")
            
            print(f"\n💾 详细错误信息已保存到: {error_log_path}")
        except Exception as e:
            print(f"\n⚠️  保存错误日志失败: {e}")
    
    print(f"{'=' * 80}")

# ============================================================================
# 模块 5: 转换函数模块（续）
# ============================================================================

def convert_multiple_choice_glue(dataset, output_path, fields, dataset_kind='glue'):
    """转换 GLUE 格式的多选题（premise/hypothesis 或 sentence1/sentence2）"""
    stats = {'total': len(dataset), 'success': 0, 'skipped': 0, 'warnings': [], 'empty_answers': 0}
    tasks = []
    
    for i, example in enumerate(dataset):
        try:
            question = _build_glue_question(example, fields)
            if not question:
                stats['warnings'].append(f"样本 {i}: GLUE 格式缺少 premise/hypothesis")
                stats['skipped'] += 1
                continue
            
            # 传递第一个样本用于判断选项类型
            valid_choices = _get_glue_choices(dataset_kind, example if i == 0 else None)
            
            # 获取 label
            label = example.get('label')
            if hasattr(label, 'as_py'):
                label = label.as_py()
            
            if label is None:
                answer_idx = 0
                stats['empty_answers'] += 1
            else:
                answer_idx = int(label)
                answer_idx = max(0, min(answer_idx, len(valid_choices) - 1))
            
            tasks.append((question, valid_choices, answer_idx))
            stats['success'] += 1
        except Exception as e:
            stats['warnings'].append(f"样本 {i}: {str(e)}")
            stats['skipped'] += 1
    
    _write_binary_tasks(tasks, output_path)
    _print_conversion_stats(stats)

def convert_multiple_choice_truthful_qa(dataset, output_path, fields):
    """转换 TruthfulQA multiple_choice 格式（mc1_targets/mc2_targets）"""
    stats = {'total': len(dataset), 'success': 0, 'skipped': 0, 'warnings': []}
    tasks = []
    
    for i, example in enumerate(dataset):
        try:
            q_field = fields.get('question_field') or 'question'
            if q_field not in example:
                stats['warnings'].append(f"样本 {i}: 缺少题目字段")
                stats['skipped'] += 1
                continue
            
            question = _normalize_question(example[q_field])
            if not question:
                stats['warnings'].append(f"样本 {i}: 题目为空")
                stats['skipped'] += 1
                continue
            
            # 获取选项
            mc1 = example.get('mc1_targets', [])
            mc2 = example.get('mc2_targets', [])
            
            if hasattr(mc1, 'as_py'):
                mc1 = mc1.as_py()
            if hasattr(mc2, 'as_py'):
                mc2 = mc2.as_py()
            
            if not isinstance(mc1, (list, tuple)) or len(mc1) == 0:
                stats['warnings'].append(f"样本 {i}: TruthfulQA 格式缺少 mc1_targets")
                stats['skipped'] += 1
                continue
            
            valid_choices = []
            for x in mc1[:1]:
                if hasattr(x, 'as_py'):
                    valid_choices.append(str(x.as_py()))  # type: ignore
                else:
                    valid_choices.append(str(x))
            
            if isinstance(mc2, (list, tuple)):
                for x in mc2[:3]:
                    if hasattr(x, 'as_py'):
                        valid_choices.append(str(x.as_py()))  # type: ignore
                    else:
                        valid_choices.append(str(x))
            
            # 补齐到 4 项
            while len(valid_choices) < 4:
                valid_choices.append("")
            valid_choices = valid_choices[:4]
            
            # 正确答案是第一个（mc1_targets 的第一个）
            answer_idx = 0
            
            tasks.append((question, valid_choices, answer_idx))
            stats['success'] += 1
        except Exception as e:
            stats['warnings'].append(f"样本 {i}: {str(e)}")
            stats['skipped'] += 1
    
    _write_binary_tasks(tasks, output_path)
    _print_conversion_stats(stats)

def convert_multiple_choice_arc(dataset, output_path, fields):
    """转换 ARC/OpenBookQA 格式的多选题（字典格式的 choices）"""
    stats = {'total': len(dataset), 'success': 0, 'skipped': 0, 'warnings': [], 'empty_answers': 0}
    tasks = []
    
    for i, example in enumerate(dataset):
        try:
            # 获取题目（支持 question 或 question_stem）
            q_field = fields.get('question_field') or 'question'
            if q_field not in example:
                q_field = 'question_stem'
            if q_field not in example:
                stats['warnings'].append(f"样本 {i}: 缺少题目字段")
                stats['skipped'] += 1
                continue
            
            question = _normalize_question(example[q_field])
            if not question:
                stats['warnings'].append(f"样本 {i}: 题目为空")
                stats['skipped'] += 1
                continue
            
            # 获取选项（字典格式）
            choices_field = fields.get('choices_field') or 'choices'
            if choices_field not in example:
                stats['warnings'].append(f"样本 {i}: 缺少选项字段")
                stats['skipped'] += 1
                continue
            
            choices = example[choices_field]
            valid_choices = _extract_choices_from_dict(choices)
            
            if not valid_choices or len(valid_choices) < 2:
                stats['warnings'].append(f"样本 {i}: 字典格式选项无效")
                stats['skipped'] += 1
                continue
            
            # 补齐或截断为 4 项
            while len(valid_choices) < 4:
                valid_choices.append("")
            valid_choices = valid_choices[:4]
            
            # 获取答案
            answer_field = fields.get('answer_field') or 'answer'
            answer = example.get(answer_field) or example.get('answerKey') or example.get('Answer')
            
            answer_idx, is_empty = _parse_answer_to_index(answer, valid_choices)
            if is_empty:
                stats['empty_answers'] += 1
            
            tasks.append((question, valid_choices, answer_idx))
            stats['success'] += 1
        except Exception as e:
            stats['warnings'].append(f"样本 {i}: {str(e)}")
            stats['skipped'] += 1
    
    _write_binary_tasks(tasks, output_path)
    _print_conversion_stats(stats)

def _get_field_case_insensitive(example: Dict[str, Any], field_name: str) -> tuple[Optional[str], Any]:
    """
    大小写不敏感地获取字段名和值
    
    Returns:
        (actual_field_name, field_value) 或 (None, None)
    """
    field_lower = field_name.lower()
    for key in example.keys():
        if key.lower() == field_lower:
            return key, example[key]
    return None, None

def _extract_gpqa_choices(example: Dict[str, Any]) -> Optional[list]:
    """提取 GPQA 格式的选项（Correct Answer + Incorrect Answer 1/2/3）"""
    # 获取 Correct Answer（大小写不敏感）
    correct_field, correct_answer = _get_field_case_insensitive(example, 'Correct Answer')
    if correct_field is None:
        return None
    
    # 获取所有 Incorrect Answer 字段（大小写不敏感）
    incorrect_fields = []
    for i in range(1, 4):
        field_name = f'Incorrect Answer {i}'
        field_key, field_value = _get_field_case_insensitive(example, field_name)
        if field_key and field_value is not None:
            # 处理 pyarrow 类型
            if isinstance(field_value, (list, tuple)) and len(field_value) > 0:
                # 如果是列表，取第一个元素
                field_value = field_value[0]
            # 处理 pyarrow 类型（只有在不是 list/tuple 时才检查 as_py）
            if not isinstance(field_value, (list, tuple)):
                if hasattr(field_value, 'as_py') and callable(getattr(field_value, 'as_py', None)):
                    try:
                        field_value = field_value.as_py()  # type: ignore
                    except Exception:
                        pass
            field_str = str(field_value).strip()
            if field_str and field_str.lower() != 'none':
                incorrect_fields.append((i, field_str))
    
    if not incorrect_fields:
        return None
    
    # 构建选项列表：Correct Answer 作为第一个选项，然后是不正确答案
    choices = []
    if correct_answer is not None:
        # 处理 pyarrow 类型
        if isinstance(correct_answer, (list, tuple)) and len(correct_answer) > 0:
            correct_answer = correct_answer[0]
        # 处理 pyarrow 类型（只有在不是 list/tuple 时才检查 as_py）
        if not isinstance(correct_answer, (list, tuple)):
            if hasattr(correct_answer, 'as_py') and callable(getattr(correct_answer, 'as_py', None)):
                try:
                    correct_answer = correct_answer.as_py()  # type: ignore
                except Exception:
                    pass
        correct_str = str(correct_answer).strip()
        if correct_str and correct_str.lower() != 'none':
            choices.append(correct_str)
    
    # 按序号排序并添加不正确答案
    incorrect_fields.sort(key=lambda x: x[0])
    for _, answer in incorrect_fields:
        choices.append(answer)
    
    return choices if len(choices) >= 2 else None

def convert_multiple_choice_standard(dataset, output_path, fields):
    """转换标准多选题格式（列表格式的 choices 或 GPQA 格式）"""
    stats = {'total': len(dataset), 'success': 0, 'skipped': 0, 'warnings': [], 'empty_answers': 0}
    tasks = []
    
    for i, example in enumerate(dataset):
        try:
            # 获取题目（大小写不敏感）
            q_field = fields.get('question_field') or 'question'
            actual_q_field, question_value = _get_field_case_insensitive(example, q_field)
            if actual_q_field is None:
                stats['warnings'].append(f"样本 {i}: 缺少题目字段")
                stats['skipped'] += 1
                continue
            
            question = _normalize_question(question_value)
            if not question:
                stats['warnings'].append(f"样本 {i}: 题目为空")
                stats['skipped'] += 1
                continue
            
            # 获取选项
            choices_field = fields.get('choices_field') or 'choices'
            valid_choices = None
            
            # 首先尝试 GPQA 格式（Correct Answer + Incorrect Answer 1/2/3）
            if choices_field == 'incorrect_answer_fields' or _get_field_case_insensitive(example, 'Correct Answer')[0]:
                valid_choices = _extract_gpqa_choices(example)
            
            # 如果不是 GPQA 格式，尝试标准格式
            if valid_choices is None:
                actual_choices_field, choices = _get_field_case_insensitive(example, choices_field)
                if actual_choices_field is None:
                    stats['warnings'].append(f"样本 {i}: 缺少选项字段")
                    stats['skipped'] += 1
                    continue
                
                # 处理列表格式
                if hasattr(choices, 'as_py'):
                    choices = choices.as_py()  # type: ignore
                
                if not isinstance(choices, (list, tuple)):
                    stats['warnings'].append(f"样本 {i}: 选项格式无效（期望列表）")
                    stats['skipped'] += 1
                    continue
                
                valid_choices = []
                for c in choices:
                    if c is None:
                        continue
                    if hasattr(c, 'as_py'):
                        valid_choices.append(str(c.as_py()))  # type: ignore
                    else:
                        valid_choices.append(str(c))
                
                valid_choices = [c.strip() for c in valid_choices if c.strip()]
            
            if not valid_choices or len(valid_choices) < 2:
                stats['warnings'].append(f"样本 {i}: 有效选项不足")
                stats['skipped'] += 1
                continue
            
            # 补齐或截断为 4 项
            while len(valid_choices) < 4:
                valid_choices.append("")
            valid_choices = valid_choices[:4]
            
            # 获取答案（大小写不敏感）
            answer_field = fields.get('answer_field') or 'answer'
            actual_answer_field, answer = _get_field_case_insensitive(example, answer_field)
            
            # 如果是 GPQA 格式，正确答案是第一个选项（Correct Answer）
            if choices_field == 'incorrect_answer_fields' or _get_field_case_insensitive(example, 'Correct Answer')[0]:
                answer_idx = 0  # GPQA 格式：Correct Answer 是第一个选项
                is_empty = False
            else:
                # 标准格式：从字段中获取答案
                if actual_answer_field is None:
                    # 尝试其他可能的答案字段名
                    for alt_field in ['answerKey', 'Answer', 'label']:
                        alt_key, alt_value = _get_field_case_insensitive(example, alt_field)
                        if alt_key:
                            answer = alt_value
                            break
                
                answer_idx, is_empty = _parse_answer_to_index(answer, valid_choices)
            
            if is_empty:
                stats['empty_answers'] += 1
            
            tasks.append((question, valid_choices, answer_idx))
            stats['success'] += 1
        except Exception as e:
            stats['warnings'].append(f"样本 {i}: {str(e)}")
            stats['skipped'] += 1
    
    _write_binary_tasks(tasks, output_path)
    _print_conversion_stats(stats)

# ----------------------------------------------------------------------------
# 5.2 二选一转换函数
# ----------------------------------------------------------------------------

def convert_winogrande_format(dataset, output_path, fields):
    """转换 Winogrande 格式（二选一）。
    输出 1) 4 行/样本的 .txt（与 llama.cpp datasets 一致）；
    输出 2) 同目录下的 winogrande-debiased-eval-llama.csv，供 llama-perplexity 使用（每行 index,sentence,option1,option2,answer，answer 为 1 或 2）。
    """
    stats = {'total': len(dataset), 'success': 0, 'skipped': 0, 'warnings': []}
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    csv_rows = []  # 用于生成 llama-perplexity 用 CSV（仅含有效答案的样本）
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            try:
                sentence = str(example.get('sentence', '')).strip()
                option1 = str(example.get('option1', '')).strip()
                option2 = str(example.get('option2', '')).strip()
                answer = example.get('answer', None)
                
                # 检查必需字段
                if not sentence:
                    stats['warnings'].append(f"样本 {i}: sentence 为空")
                    stats['skipped'] += 1
                    continue
                
                if not option1:
                    stats['warnings'].append(f"样本 {i}: option1 为空")
                    stats['skipped'] += 1
                    continue
                
                if not option2:
                    stats['warnings'].append(f"样本 {i}: option2 为空")
                    stats['skipped'] += 1
                    continue
                
                # 处理答案（test split 可能没有答案）
                if answer is None or answer == '' or answer == -1:
                    # 测试集没有答案，使用 -1 作为占位符
                    answer_idx = -1
                else:
                    # 答案转换（1-based 转 0-based）
                    try:
                        answer_idx = int(answer) - 1
                        if not (0 <= answer_idx < 2):
                            stats['warnings'].append(f"样本 {i}: 答案索引无效 ({answer} -> {answer_idx})")
                            answer_idx = -1  # 使用 -1 作为占位符
                    except (ValueError, TypeError):
                        stats['warnings'].append(f"样本 {i}: 答案格式错误 ({answer!r})")
                        answer_idx = -1  # 使用 -1 作为占位符
                
                f.write(f"{sentence}\n")
                f.write(f"{answer_idx}\n")
                f.write(f"{option1}\n")
                f.write(f"{option2}\n")
                
                stats['success'] += 1
                
                # 仅对有有效答案的样本写入 CSV（llama-perplexity 用）
                if answer_idx in (0, 1) and '_' in sentence:
                    sent_safe = sentence.replace(",", " ")
                    csv_answer = str(answer_idx + 1)  # 1 或 2
                    csv_rows.append((len(csv_rows), sent_safe, option1, option2, csv_answer))
            except Exception as e:
                stats['warnings'].append(f"样本 {i}: {str(e)}")
                stats['skipped'] += 1
                import traceback
                traceback.print_exc()
    
    # 同目录下写入 llama-perplexity 用 CSV（与 .txt 同名仅改扩展为 .csv）
    if csv_rows:
        out_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]  # 如 validation
        csv_path = os.path.join(out_dir, base_name + '.csv') if out_dir else (base_name + '.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as cf:
            for idx, sent, o1, o2, ans in csv_rows:
                cf.write(f"{idx},{sent},{o1},{o2},{ans}\n")
        print(f"  已生成 llama-perplexity 用 CSV: {csv_path} ({len(csv_rows)} 行)")
    
    empty_answers = stats.get('empty_answers', 0)
    print(f"\n转换结果: 成功 {stats['success']}, 跳过 {stats['skipped']}")
    
    # 如果有很多空答案，说明可能是测试集
    if empty_answers > 0:
        empty_ratio = empty_answers / stats['success'] * 100 if stats['success'] > 0 else 0
        if empty_ratio > 50:
            print(f"⚠️  注意: {empty_answers}/{stats['success']} ({empty_ratio:.1f}%) 的答案为空（可能是测试集）")
    if stats['warnings']:
        print(f"警告数量: {len(stats['warnings'])}")
        if len(stats['warnings']) <= 10:
            for w in stats['warnings']:
                print(f"  - {w}")

# ----------------------------------------------------------------------------
# 5.3 问答转换函数
# ----------------------------------------------------------------------------

def convert_piqa_format(dataset, output_path, fields):
    """转换 PIQA 格式（二选一）"""
    stats = {'total': len(dataset), 'success': 0, 'skipped': 0, 'warnings': []}
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            try:
                goal = str(example.get('goal', '')).strip()
                sol1 = str(example.get('sol1', '')).strip()
                sol2 = str(example.get('sol2', '')).strip()
                label = example.get('label', '')
                
                if not all([goal, sol1, sol2]):
                    stats['warnings'].append(f"样本 {i}: 字段缺失")
                    stats['skipped'] += 1
                    continue
                
                # 答案转换
                try:
                    answer_idx = int(label)
                    if not (0 <= answer_idx < 2):
                        stats['warnings'].append(f"样本 {i}: 答案索引无效")
                        stats['skipped'] += 1
                        continue
                except (ValueError, TypeError):
                    stats['warnings'].append(f"样本 {i}: 答案格式错误")
                    stats['skipped'] += 1
                    continue
                
                f.write(f"{goal}\n")
                f.write(f"{answer_idx}\n")
                f.write(f"{sol1}\n")
                f.write(f"{sol2}\n")
                
                stats['success'] += 1
            except Exception as e:
                stats['warnings'].append(f"样本 {i}: {str(e)}")
                stats['skipped'] += 1
    
    empty_answers = stats.get('empty_answers', 0)
    print(f"\n转换结果: 成功 {stats['success']}, 跳过 {stats['skipped']}")
    
    # 如果有很多空答案，说明可能是测试集
    if empty_answers > 0:
        empty_ratio = empty_answers / stats['success'] * 100 if stats['success'] > 0 else 0
        if empty_ratio > 50:
            print(f"⚠️  注意: {empty_answers}/{stats['success']} ({empty_ratio:.1f}%) 的答案为空（可能是测试集）")

# ----------------------------------------------------------------------------
# 5.4 阅读理解转换函数
# ----------------------------------------------------------------------------

def convert_qa_format(dataset, output_path, fields):
    """转换问答格式（Question Answering）。
    输出 1) validation.txt（question\\nanswer\\n）；
    若有有效答案，同时输出 2) 同名 validation.bin（单选项多选题格式），供 llama-perplexity --multiple-choice 使用（TriviaQA/TruthfulQA generation 等）。
    """
    stats = {'total': len(dataset), 'success': 0, 'skipped': 0, 'warnings': [], 'empty_answers': 0}
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    question_field = fields.get('question_field', 'question')
    answer_field = fields.get('answer_field', 'answer')
    tasks_bin = []  # 用于生成同名 .bin：(question, [answer], correct_idx)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            try:
                question = str(example.get(question_field, '')).strip()
                answer = example.get(answer_field, None)
                
                if not question:
                    stats['warnings'].append(f"样本 {i}: 问题为空")
                    stats['skipped'] += 1
                    continue
                
                # 处理答案（可能是字符串、列表或字典）
                if answer is None:
                    answer_text = ""
                elif isinstance(answer, dict):
                    # 处理字典格式的答案
                    # TriviaQA 格式: {'value': '...', 'normalized_value': '...', 'aliases': [...], ...}
                    # SQuAD 格式: {'text': [...], 'answer_start': [...]}
                    # 注意：测试集的答案可能是 '<unk>'，这是正常的（测试集通常不提供答案）
                    
                    # 优先使用 value 字段
                    if 'value' in answer:
                        value = answer['value']
                        if value and str(value).strip() and str(value).strip() != '<unk>':
                            answer_text = str(value).strip()
                        elif 'normalized_value' in answer:
                            norm_value = answer['normalized_value']
                            if norm_value and str(norm_value).strip() and str(norm_value).strip() != '<unk>':
                                answer_text = str(norm_value).strip()
                            else:
                                answer_text = ""
                        else:
                            answer_text = ""
                    # 其次尝试 normalized_value
                    elif 'normalized_value' in answer:
                        norm_value = answer['normalized_value']
                        if norm_value and str(norm_value).strip() and str(norm_value).strip() != '<unk>':
                            answer_text = str(norm_value).strip()
                        else:
                            answer_text = ""
                    # SQuAD 格式
                    elif 'text' in answer:
                        text_list = answer['text']
                        if isinstance(text_list, list) and len(text_list) > 0:
                            answer_text = str(text_list[0]).strip()
                        else:
                            answer_text = str(text_list).strip()
                    # 尝试使用别名
                    elif 'aliases' in answer and isinstance(answer['aliases'], list) and len(answer['aliases']) > 0:
                        # 使用第一个非空别名
                        for alias in answer['aliases']:
                            alias_str = str(alias).strip()
                            if alias_str and alias_str != '<unk>':
                                answer_text = alias_str
                                break
                        else:
                            answer_text = ""
                    elif 'normalized_aliases' in answer and isinstance(answer['normalized_aliases'], list) and len(answer['normalized_aliases']) > 0:
                        # 使用第一个非空标准化别名
                        for alias in answer['normalized_aliases']:
                            alias_str = str(alias).strip()
                            if alias_str and alias_str != '<unk>':
                                answer_text = alias_str
                                break
                        else:
                            answer_text = ""
                    elif 'matched_wiki_entity_name' in answer and answer['matched_wiki_entity_name'] and str(answer['matched_wiki_entity_name']).strip() != '<unk>':
                        answer_text = str(answer['matched_wiki_entity_name']).strip()
                    else:
                        # 如果所有字段都无效，使用空字符串
                        answer_text = ""
                elif isinstance(answer, list):
                    if len(answer) > 0:
                        # 如果列表中的元素是字典，尝试提取值
                        first_item = answer[0]
                        if isinstance(first_item, dict):
                            if 'value' in first_item:
                                answer_text = str(first_item['value']).strip()
                            elif 'normalized_value' in first_item:
                                answer_text = str(first_item['normalized_value']).strip()
                            else:
                                answer_text = str(first_item).strip()
                        else:
                            answer_text = str(first_item).strip()
                    else:
                        answer_text = ""
                else:
                    answer_text = str(answer).strip()
                
                # 写入格式: question\nanswer\n
                f.write(f"{question}\n")
                f.write(f"{answer_text}\n")
                
                # 有有效答案时加入 .bin 任务（单选项，correct_idx=0，供 llama-perplexity --multiple-choice）
                if answer_text:
                    tasks_bin.append((question, [answer_text], 0))
                
                # 统计空答案（可能是测试集，这是正常的）
                if not answer_text:
                    stats.setdefault('empty_answers', 0)
                    stats['empty_answers'] = stats.get('empty_answers', 0) + 1
                
                stats['success'] += 1
            except Exception as e:
                stats['warnings'].append(f"样本 {i}: {str(e)}")
                stats['skipped'] += 1
    
    # 同名 .bin（与 output_path 仅扩展名不同），供 lm_evaluator / llama-perplexity 使用
    if tasks_bin:
        bin_path = os.path.splitext(output_path)[0] + '.bin'
        _write_binary_tasks(tasks_bin, bin_path)
        print(f"  已生成 lm_evaluator 用 validation.bin: {os.path.basename(bin_path)} ({len(tasks_bin)} 题)")
    
    empty_answers = stats.get('empty_answers', 0)
    print(f"\n转换结果: 成功 {stats['success']}, 跳过 {stats['skipped']}")
    
    # 如果有很多空答案，说明可能是测试集
    if empty_answers > 0:
        empty_ratio = empty_answers / stats['success'] * 100 if stats['success'] > 0 else 0
        if empty_ratio > 50:
            print(f"\n⚠️  注意: {empty_answers}/{stats['success']} ({empty_ratio:.1f}%) 的答案为空")
            print(f"   这可能是测试集（test split），答案被隐藏用于防止过拟合")
            print(f"   测试集评估方式:")
            print(f"     1. 模型生成答案（不看到标准答案）")
            print(f"     2. 评估系统将模型答案与隐藏的标准答案对比")
            print(f"     3. 计算准确率、F1等指标")
            print(f"   建议: 使用 validation split 进行本地评估和调试")
    
    if stats['warnings'] and len(stats['warnings']) <= 10:
        for w in stats['warnings']:
            print(f"  - {w}")
    elif stats['warnings']:
        print(f"警告数量: {len(stats['warnings'])} (仅显示前10条)")
        for w in stats['warnings'][:10]:
            print(f"  - {w}")

# ----------------------------------------------------------------------------
# 5.5 数学推理转换函数
# ----------------------------------------------------------------------------

def convert_reading_comprehension_format(dataset, output_path, fields):
    """转换阅读理解格式（Reading Comprehension，如 SQuAD）"""
    stats = {'total': len(dataset), 'success': 0, 'skipped': 0, 'warnings': [], 'empty_answers': 0}
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            try:
                context = str(example.get('context', '')).strip()
                question = str(example.get('question', '')).strip()
                answers = example.get('answers', None)
                
                if not context or not question:
                    stats['warnings'].append(f"样本 {i}: context 或 question 为空")
                    stats['skipped'] += 1
                    continue
                
                # 处理答案
                if answers is None:
                    answer_text = ""
                elif isinstance(answers, dict):
                    # SQuAD 格式: {'text': [...], 'answer_start': [...]}
                    # TriviaQA 格式: {'value': '...', 'normalized_value': '...', ...}
                    if 'text' in answers:
                        text_list = answers['text']
                        if isinstance(text_list, list) and len(text_list) > 0:
                            answer_text = str(text_list[0]).strip()
                        else:
                            answer_text = str(text_list).strip()
                    elif 'value' in answers and answers['value'] and answers['value'] != '<unk>':
                        answer_text = str(answers['value']).strip()
                    elif 'normalized_value' in answers and answers['normalized_value'] and answers['normalized_value'] != '<unk>':
                        answer_text = str(answers['normalized_value']).strip()
                    else:
                        answer_text = ""
                elif isinstance(answers, list):
                    if len(answers) > 0:
                        first_item = answers[0]
                        if isinstance(first_item, dict):
                            if 'value' in first_item:
                                answer_text = str(first_item['value']).strip()
                            elif 'normalized_value' in first_item:
                                answer_text = str(first_item['normalized_value']).strip()
                            elif 'text' in first_item:
                                answer_text = str(first_item['text']).strip()
                            else:
                                answer_text = str(first_item).strip()
                        else:
                            answer_text = str(first_item).strip()
                    else:
                        answer_text = ""
                else:
                    answer_text = str(answers).strip()
                
                # 写入格式: context\nquestion\nanswer\n
                f.write(f"{context}\n")
                f.write(f"{question}\n")
                f.write(f"{answer_text}\n")
                
                # 统计空答案
                if not answer_text:
                    stats['empty_answers'] += 1
                
                stats['success'] += 1
            except Exception as e:
                stats['warnings'].append(f"样本 {i}: {str(e)}")
                stats['skipped'] += 1
    
    empty_answers = stats.get('empty_answers', 0)
    print(f"\n转换结果: 成功 {stats['success']}, 跳过 {stats['skipped']}")
    
    # 如果有很多空答案，说明可能是测试集
    if empty_answers > 0:
        empty_ratio = empty_answers / stats['success'] * 100 if stats['success'] > 0 else 0
        if empty_ratio > 50:
            print(f"⚠️  注意: {empty_answers}/{stats['success']} ({empty_ratio:.1f}%) 的答案为空（可能是测试集）")

# ----------------------------------------------------------------------------
# 5.6 代码生成转换函数
# ----------------------------------------------------------------------------

def convert_math_reasoning_format(dataset, output_path, fields):
    """转换数学推理格式（Math Reasoning，如 GSM8K）"""
    stats = {'total': len(dataset), 'success': 0, 'skipped': 0, 'warnings': []}
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    question_field = fields.get('question_field', 'question')
    answer_field = fields.get('answer_field', 'answer')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            try:
                question = str(example.get(question_field, '')).strip()
                answer = str(example.get(answer_field, '')).strip()
                
                if not question:
                    stats['warnings'].append(f"样本 {i}: 问题为空")
                    stats['skipped'] += 1
                    continue
                
                # 写入格式: question\nanswer\n
                f.write(f"{question}\n")
                f.write(f"{answer}\n")
                
                stats['success'] += 1
            except Exception as e:
                stats['warnings'].append(f"样本 {i}: {str(e)}")
                stats['skipped'] += 1
    
    empty_answers = stats.get('empty_answers', 0)
    print(f"\n转换结果: 成功 {stats['success']}, 跳过 {stats['skipped']}")
    
    # 如果有很多空答案，说明可能是测试集
    if empty_answers > 0:
        empty_ratio = empty_answers / stats['success'] * 100 if stats['success'] > 0 else 0
        if empty_ratio > 50:
            print(f"⚠️  注意: {empty_answers}/{stats['success']} ({empty_ratio:.1f}%) 的答案为空（可能是测试集）")

# ----------------------------------------------------------------------------
# 5.7 HellaSwag 转换函数（特殊格式：6行文本）
# ----------------------------------------------------------------------------

def convert_hellaswag_format(dataset, output_path, fields):
    """转换 HellaSwag 格式（6行文本格式，与 perplexity.cpp 兼容）"""
    stats = {'total': len(dataset), 'success': 0, 'skipped': 0, 'warnings': []}
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            try:
                # 获取字段
                ctx = example.get('ctx') or example.get('context', '')
                activity_label = example.get('activity_label') or example.get('activity', '')
                endings = example.get('endings', [])
                label = example.get('label')
                
                # 处理 pyarrow 类型
                if hasattr(ctx, 'as_py'):
                    ctx = ctx.as_py()
                if hasattr(activity_label, 'as_py'):
                    activity_label = activity_label.as_py()
                if hasattr(endings, 'as_py'):
                    endings = endings.as_py()
                if hasattr(label, 'as_py'):
                    label = label.as_py()
                
                ctx = str(ctx).strip()
                activity_label = str(activity_label).strip() if activity_label else ''
                
                if not ctx:
                    stats['warnings'].append(f"样本 {i}: ctx 为空")
                    stats['skipped'] += 1
                    continue
                
                if not isinstance(endings, (list, tuple)) or len(endings) < 4:
                    stats['warnings'].append(f"样本 {i}: endings 格式无效或数量不足")
                    stats['skipped'] += 1
                    continue
                
                # 构建第1行：activity_label + ": " + ctx
                context_line = f"{activity_label}: {ctx}" if activity_label else ctx
                
                # 处理答案（label 可能是字符串 "0", "1", "2", "3"）
                try:
                    answer_idx = int(label) if label is not None else -1
                    if not (0 <= answer_idx < 4):
                        answer_idx = -1
                except (ValueError, TypeError):
                    answer_idx = -1
                
                # 写入6行格式
                f.write(f"{context_line}\n")  # 第1行
                f.write(f"{answer_idx}\n")    # 第2行
                for j in range(4):            # 第3-6行
                    ending = endings[j] if j < len(endings) else ""
                    if ending and hasattr(ending, 'as_py'):
                        ending = ending.as_py()  # type: ignore
                    f.write(f"{str(ending).strip()}\n")
                
                stats['success'] += 1
            except Exception as e:
                stats['warnings'].append(f"样本 {i}: {str(e)}")
                stats['skipped'] += 1
    
    _print_conversion_stats(stats)

def convert_language_modeling_format(dataset, output_path, fields):
    """
    转换语言建模数据集格式（用于 perplexity 评估）
    
    根据 perplexity.cpp 的实现，输入应该是纯文本文件，所有文本直接拼接。
    参考 evaluate_model.py 的做法，每个文本之间用两个换行符分隔。
    """
    stats = {'total': len(dataset), 'success': 0, 'skipped': 0, 'warnings': []}
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            try:
                # 获取 text 字段
                text = example.get('text')
                if text is None:
                    stats['warnings'].append(f"样本 {i}: 缺少 text 字段")
                    stats['skipped'] += 1
                    continue
                
                # 处理 pyarrow 类型
                if hasattr(text, 'as_py'):
                    text = text.as_py()
                
                text = str(text).strip()
                
                if not text:
                    stats['warnings'].append(f"样本 {i}: text 字段为空")
                    stats['skipped'] += 1
                    continue
                
                # 写入文本，每个文本之间用两个换行符分隔（参考 evaluate_model.py）
                f.write(text)
                f.write('\n\n')
                
                stats['success'] += 1
            except Exception as e:
                stats['warnings'].append(f"样本 {i}: {str(e)}")
                stats['skipped'] += 1
    
    _print_conversion_stats(stats)

def convert_code_generation_format(dataset, output_path, fields):
    """转换代码生成格式（Code Generation，如 HumanEval, MBPP）"""
    stats = {'total': len(dataset), 'success': 0, 'skipped': 0, 'warnings': []}
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            try:
                prompt = str(example.get('prompt', example.get('text', ''))).strip()
                code = str(example.get('code', example.get('canonical_solution', ''))).strip()
                test = example.get('test', example.get('test_list', None))
                
                if not prompt:
                    stats['warnings'].append(f"样本 {i}: prompt 为空")
                    stats['skipped'] += 1
                    continue
                
                # 处理测试用例
                if test is None:
                    test_text = ""
                elif isinstance(test, list):
                    test_text = '\n'.join([str(t).strip() for t in test if t])
                else:
                    test_text = str(test).strip()
                
                # 写入格式: prompt\ncode\ntest\n
                f.write(f"{prompt}\n")
                f.write(f"{code}\n")
                f.write(f"{test_text}\n")
                
                stats['success'] += 1
            except Exception as e:
                stats['warnings'].append(f"样本 {i}: {str(e)}")
                stats['skipped'] += 1
    
    empty_answers = stats.get('empty_answers', 0)
    print(f"\n转换结果: 成功 {stats['success']}, 跳过 {stats['skipped']}")
    
    # 如果有很多空答案，说明可能是测试集
    if empty_answers > 0:
        empty_ratio = empty_answers / stats['success'] * 100 if stats['success'] > 0 else 0
        if empty_ratio > 50:
            print(f"⚠️  注意: {empty_answers}/{stats['success']} ({empty_ratio:.1f}%) 的答案为空（可能是测试集）")

# ============================================================================
# 模块 6: 配置文件更新模块
# 职责: 更新和维护 datasets_model_evaluation_config.json 配置文件
# ============================================================================

# ----------------------------------------------------------------------------
# 6.1 路径和类型推断函数
# ----------------------------------------------------------------------------

def infer_category_from_path(dataset_path: str, base_path: Optional[str] = None, task_type: Optional[str] = None) -> tuple[str, str]:
    """
    从任务类型推断 category ID 和名称（不进行路径判断）
    
    Args:
        dataset_path: 数据集路径（保留参数以兼容现有调用）
        base_path: 基础路径（保留参数以兼容现有调用）
        task_type: 任务类型（必需）
    
    Returns:
        (category_id, category_name)
    """
    # task_type 到 category 的映射（直接映射，不依赖路径）
    task_type_to_category = {
        "code_generation": ("code_generation", "代码生成"),
        "math_reasoning": ("math_reasoning", "数学推理"),
        "language_modeling": ("language_modeling", "语言建模"),
        "reading_comprehension": ("reading_comprehension", "阅读理解"),
        "commonsense_reasoning": ("commonsense_reasoning", "常识推理"),
        "knowledge_evaluation": ("knowledge_evaluation", "知识评估"),
        "general_evaluation": ("general_evaluation", "综合评估"),
        "multiple_choice": ("general_evaluation", "综合评估"),
        "binary_choice": ("commonsense_reasoning", "常识推理"),
        "qa": ("knowledge_evaluation", "知识评估"),
    }
    
    # 直接从 task_type 推断 category
    if task_type and task_type in task_type_to_category:
        return task_type_to_category[task_type]
    
    # 如果 task_type 不在映射中，使用 task_type 本身作为 category_id
    if task_type:
        category_name = task_type.replace('_', ' ').title()  # 将下划线替换为空格并首字母大写
        return (task_type, category_name)
    
    # 如果 task_type 也没有，使用 'other'
    return ("other", "其他")

# ----------------------------------------------------------------------------
# 6.2 评估类型和参数推断函数
# ----------------------------------------------------------------------------

def infer_evaluation_type_and_type(task_type: str, task_subtype: Optional[str], dataset_path: str) -> tuple[str, str]:
    """
    从任务类型推断 evaluation_type 和 type/benchmark
    
    Returns:
        (evaluation_type, type_or_benchmark)
    """
    # 首先根据路径强制设置 evaluation_type（优先级最高）
    path_lower = dataset_path.lower()
    if 'code_generation' in path_lower:
        evaluation_type = 'generative'
    elif 'math_reasoning' in path_lower:
        evaluation_type = 'generative'
    elif 'language_modeling' in path_lower:
        evaluation_type = 'perplexity'
    # 然后根据任务类型确定 evaluation_type
    elif task_type in ('math_reasoning', 'code_generation'):
        evaluation_type = 'generative'
    elif task_type == 'language_modeling':
        evaluation_type = 'perplexity'
    else:
        evaluation_type = 'perplexity'
    
    # 根据 task_type 和 task_subtype 确定 type/benchmark
    if task_type == 'math_reasoning':
        type_or_benchmark = task_subtype or 'gsm8k'
    elif task_type == 'code_generation':
        # 根据路径推断 benchmark（mbpp 或 humaneval）
        if 'mbpp' in path_lower:
            type_or_benchmark = 'mbpp'
        else:
            type_or_benchmark = task_subtype or 'humaneval'
    elif task_type == 'language_modeling':
        type_or_benchmark = 'ppl'  # perplexity
    elif task_subtype == 'hellaswag':
        type_or_benchmark = 'hellaswag'
    elif task_subtype == 'winogrande':
        type_or_benchmark = 'winogrande'
    elif task_subtype == 'glue':
        type_or_benchmark = 'multiple_choice'
    elif task_subtype == 'truthful_qa':
        type_or_benchmark = 'multiple_choice'
    elif task_type == 'multiple_choice':
        # 从路径推断（path_lower 已在上面定义）
        if 'arc' in path_lower:
            type_or_benchmark = 'multiple_choice'
        elif 'openbookqa' in path_lower:
            type_or_benchmark = 'multiple_choice'
        else:
            type_or_benchmark = 'multiple_choice'
    elif task_type == 'binary_choice':
        type_or_benchmark = task_subtype or 'binary_choice'
    elif task_type == 'qa':
        type_or_benchmark = 'qa'
    elif task_type == 'reading_comprehension':
        type_or_benchmark = 'multiple_choice'  # SQuAD 等作为多选题处理
    else:
        type_or_benchmark = 'unknown'
    
    return evaluation_type, type_or_benchmark

def get_default_params(evaluation_type: str, type_or_benchmark: str) -> Dict[str, Any]:
    """获取默认参数"""
    if evaluation_type == 'perplexity':
        if type_or_benchmark == 'hellaswag':
            return {"ctx_size": 512, "hellaswag_tasks": 100}
        elif type_or_benchmark == 'winogrande':
            return {"ctx_size": 512, "winogrande_tasks": 100}
        elif type_or_benchmark == 'ppl':
            return {"ctx_size": 512, "batch_size": 512}
        else:
            return {"ctx_size": 512, "multiple_choice_tasks": 0}
    else:  # generative
        if type_or_benchmark == 'gsm8k':
            return {"n_shot": 0, "max_samples": None, "timeout_per_sample": 120}
        elif type_or_benchmark in ('humaneval', 'mbpp'):
            return {"max_samples": None, "timeout_per_sample": 60}
        else:
            return {"max_samples": None, "timeout_per_sample": 120}

# ----------------------------------------------------------------------------
# 6.3 路径计算函数
# ----------------------------------------------------------------------------

def _normalize_path(path: str) -> str:
    """
    标准化路径（转换为绝对路径并统一使用正斜杠）
    
    Args:
        path: 路径字符串
    
    Returns:
        标准化后的路径
    """
    return os.path.abspath(path).replace('\\', '/')

def _extract_model_evaluation_path(path: str) -> Optional[str]:
    """
    从路径中提取 model_evaluation 之后的部分
    
    Args:
        path: 标准化后的路径
    
    Returns:
        相对路径，如果无法提取则返回 None
    """
    parts = path.split('/')
    if 'model_evaluation' in parts:
        idx = parts.index('model_evaluation')
        if idx + 1 < len(parts):
            return '/'.join(parts[idx + 1:])
    return None

def calculate_relative_path(dataset_dir: str, base_path: Optional[str] = None) -> str:
    """
    计算数据集相对于 base_path 的相对路径
    
    Args:
        dataset_dir: 数据集目录的绝对路径
        base_path: 基础路径（如 datasets/model_evaluation），如果为 None 则尝试从路径推断
    
    Returns:
        相对路径（如 "knowledge_evaluation/mmlu"）
    """
    dataset_path = _normalize_path(dataset_dir)
    
    # 优先使用 base_path
    if base_path:
        base = _normalize_path(base_path)
        if dataset_path.startswith(base):
            return dataset_path[len(base):].lstrip('/')
    
    # 尝试从路径中提取 model_evaluation 之后的部分
    rel_path = _extract_model_evaluation_path(dataset_path)
    if rel_path:
        return rel_path
    
    # 如果无法推断，使用目录名
    return os.path.basename(dataset_dir)

def select_best_file(generated_files: Optional[list]) -> Optional[str]:
    """
    从生成的文件列表中选择最合适的文件
    
    优先级：validation > test > train > 其他
    
    Args:
        generated_files: 生成的文件路径列表
    
    Returns:
        选中的文件路径，如果没有则返回 None
    """
    if not generated_files:
        return None
    
    # 优先级顺序：validation > test > train
    split_priority = ['validation', 'test', 'train']
    
    for split in split_priority:
        for file_path in generated_files:
            # 检查文件名中是否包含该 split
            if f'/{split}.' in file_path or f'\\{split}.' in file_path:
                return file_path
    
    # 如果没找到，使用第一个文件
    return generated_files[0] if generated_files else None

def calculate_file_relative_path(
    file_path: str,
    dataset_path: str,
    base_path: Optional[str] = None
) -> str:
    """
    计算文件相对于 base_path 的相对路径
    
    统一处理文件路径计算，简化 update_config_file 中的逻辑
    
    Args:
        file_path: 文件的绝对路径
        dataset_path: 数据集目录的绝对路径
        base_path: 基础路径（如 datasets/model_evaluation）
    
    Returns:
        文件的相对路径
    """
    file_abs = _normalize_path(file_path)
    
    # 优先使用 base_path
    if base_path:
        base = _normalize_path(base_path)
        if file_abs.startswith(base):
            return file_abs[len(base):].lstrip('/')
    
    # 尝试从 dataset_path 计算
    dataset_abs = _normalize_path(dataset_path)
    if file_abs.startswith(dataset_abs):
        dataset_rel = calculate_relative_path(dataset_path, base_path)
        file_rel_to_dataset = file_abs[len(dataset_abs):].lstrip('/')
        return f"{dataset_rel}/{file_rel_to_dataset}" if dataset_rel else file_rel_to_dataset
    
    # 回退到 model_evaluation 提取逻辑
    rel_path = _extract_model_evaluation_path(file_abs)
    if rel_path:
        return rel_path
    
    # 最后回退到数据集目录的相对路径
    return calculate_relative_path(dataset_path, base_path)

# ----------------------------------------------------------------------------
# 6.4 数据集名称推断函数
# ----------------------------------------------------------------------------

def infer_dataset_name(dataset_dir: str) -> str:
    """
    从数据集目录路径推断数据集名称
    
    Args:
        dataset_dir: 数据集目录路径
    
    Returns:
        数据集名称
    """
    dataset_name = os.path.basename(os.path.abspath(dataset_dir))
    # 如果目录名是通用名称，尝试从路径中提取更有意义的名称
    if dataset_name in ('test_dataset', 'validation_dataset', 'converted'):
        # 尝试从父目录获取名称
        parent_dir = os.path.basename(os.path.dirname(os.path.abspath(dataset_dir)))
        if parent_dir and parent_dir not in ('test_dataset', 'validation_dataset', 'converted'):
            dataset_name = parent_dir
    return dataset_name

def infer_base_path(dataset_dir: str) -> Optional[str]:
    """
    从数据集目录路径推断 base_path（model_evaluation 目录）
    
    Args:
        dataset_dir: 数据集目录路径
    
    Returns:
        base_path 或 None
    """
    parts = os.path.abspath(dataset_dir).split(os.sep)
    if 'model_evaluation' in parts:
        idx = parts.index('model_evaluation')
        return os.sep.join(parts[:idx+1])
    return None

# ----------------------------------------------------------------------------
# 6.5 配置文件读写函数
# ----------------------------------------------------------------------------

def load_config_file(config_file_path: str) -> Dict[str, Any]:
    """
    加载配置文件，如果不存在或为空则创建默认配置
    
    Args:
        config_file_path: 配置文件路径
    
    Returns:
        配置字典
    """
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # 如果文件为空，返回默认配置
                if not content:
                    return {
                        "description": "与 llama.cpp datasets/model_evaluation 目录结构对齐；path 为相对 base_path 的子路径；测试数据通常在 test_dataset 或 validation_dataset 目录下。",
                        "base_path": "${LLAMA_CPP_ROOT}/datasets/model_evaluation",
                        "categories": []
                    }
                return json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            # 如果 JSON 解析失败，打印警告并返回默认配置
            print(f"  ⚠️  警告: 配置文件 {config_file_path} 格式无效，将使用默认配置")
            print(f"     错误信息: {e}")
            return {
                "description": "与 llama.cpp datasets/model_evaluation 目录结构对齐；path 为相对 base_path 的子路径；测试数据通常在 test_dataset 或 validation_dataset 目录下。",
                "base_path": "${LLAMA_CPP_ROOT}/datasets/model_evaluation",
                "categories": []
            }
    else:
        # 创建新配置
        return {
            "description": "与 llama.cpp datasets/model_evaluation 目录结构对齐；path 为相对 base_path 的子路径；测试数据通常在 test_dataset 或 validation_dataset 目录下。",
            "base_path": "${LLAMA_CPP_ROOT}/datasets/model_evaluation",
            "categories": []
        }

def save_config_file(config: Dict[str, Any], config_file_path: str) -> bool:
    """
    保存配置文件
    
    Args:
        config: 配置字典
        config_file_path: 配置文件路径
    
    Returns:
        是否成功保存
    """
    try:
        config_file_dir = os.path.dirname(config_file_path)
        if config_file_dir and not os.path.exists(config_file_dir):
            os.makedirs(config_file_dir, exist_ok=True)
        
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"  ⚠️  保存配置文件失败: {e}")
        return False

# ----------------------------------------------------------------------------
# 6.6 数据集条目管理函数
# ----------------------------------------------------------------------------

def check_dataset_exists(config: Dict[str, Any], rel_path: str) -> bool:
    """
    检查配置文件中是否已存在指定路径的数据集
    
    Args:
        config: 配置字典
        rel_path: 相对路径
    
    Returns:
        是否存在
    """
    for cat in config.get("categories", []):
        for ds in cat.get("datasets", []):
            if ds.get("path") == rel_path:
                return True
    return False

def find_or_create_category(config: Dict[str, Any], category_id: str, category_name: str, evaluation_type: str) -> Dict[str, Any]:
    """
    查找或创建 category
    
    Args:
        config: 配置字典
        category_id: category ID
        category_name: category 名称
        evaluation_type: 评估类型
    
    Returns:
        category 字典
    """
    # 根据 category_id 强制设置正确的 evaluation_type（优先级最高）
    category_evaluation_type_map = {
        'code_generation': 'generative',
        'math_reasoning': 'generative',
        'language_modeling': 'perplexity',
        'commonsense_reasoning': 'perplexity',
        'knowledge_evaluation': 'perplexity',
        'reading_comprehension': 'perplexity',
        'general_evaluation': 'perplexity',
    }
    if category_id in category_evaluation_type_map:
        correct_evaluation_type = category_evaluation_type_map[category_id]
        if evaluation_type != correct_evaluation_type:
            print(f"  ⚠️  警告: category {category_id} 的 evaluation_type 被强制设置为 {correct_evaluation_type}（原值: {evaluation_type}）")
            evaluation_type = correct_evaluation_type
    # 如果 category_id 不在映射中，使用传入的 evaluation_type（已经通过 task_type 推断）
    # 这样可以支持新的 category，而不是强制使用默认值
    
    # 查找现有 category
    for cat in config.get("categories", []):
        if cat.get("id") == category_id:
            # 确保 evaluation_type 一致
            if cat.get("evaluation_type") != evaluation_type:
                print(f"  ⚠️  警告: category {category_id} 的 evaluation_type 不一致（现有: {cat.get('evaluation_type')}, 新: {evaluation_type}）")
                cat["evaluation_type"] = evaluation_type
            return cat
    
    # 创建新 category
    category = {
        "id": category_id,
        "name": category_name,
        "evaluation_type": evaluation_type,
        "datasets": []
    }
    config.setdefault("categories", []).append(category)
    return category

def create_dataset_entry(
    dataset_name: str,
    rel_path: str,
    evaluation_type: str,
    type_or_benchmark: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    创建数据集条目
    
    Args:
        dataset_name: 数据集名称
        rel_path: 相对路径
        evaluation_type: 评估类型
        type_or_benchmark: 类型或基准
        params: 参数
    
    Returns:
        数据集条目字典
    """
    dataset_entry: Dict[str, Any] = {
        "name": dataset_name,
        "path": rel_path,
        "params": params
    }
    
    # 根据 evaluation_type 设置 type 或 benchmark
    if evaluation_type == "generative":
        dataset_entry["benchmark"] = type_or_benchmark
    else:
        dataset_entry["type"] = type_or_benchmark
    
    return dataset_entry

# ----------------------------------------------------------------------------
# 6.7 配置文件更新主函数
# ----------------------------------------------------------------------------

def update_config_file(
    config_file_path: str,
    dataset_name: str,
    dataset_path: str,
    task_type: str,
    task_subtype: Optional[str],
    base_path: Optional[str] = None,
    generated_files: Optional[list] = None
) -> bool:
    """
    更新或创建 datasets_model_evaluation_config.json 文件
    
    Args:
        config_file_path: 配置文件路径
        dataset_name: 数据集名称
        dataset_path: 数据集目录路径
        task_type: 检测到的任务类型
        task_subtype: 检测到的任务子类型
        base_path: 基础路径（用于计算相对路径）
    
    Returns:
        是否成功更新
    """
    try:
        # 选择最合适的文件并计算相对路径
        selected_file = select_best_file(generated_files)
        
        if selected_file:
            # 使用文件路径计算相对路径
            rel_path = calculate_file_relative_path(selected_file, dataset_path, base_path)
        else:
            # 没有生成文件，使用目录路径
            rel_path = calculate_relative_path(dataset_path, base_path)
        
        # 推断 evaluation_type 和 type/benchmark（先推断，因为 category 推断可能需要 task_type）
        evaluation_type, type_or_benchmark = infer_evaluation_type_and_type(task_type, task_subtype, rel_path)
        
        # 推断 category（传入 task_type 用于回退推断）
        category_id, category_name = infer_category_from_path(rel_path, base_path, task_type)
        
        # 获取默认参数
        params = get_default_params(evaluation_type, type_or_benchmark)
        
        # 加载配置
        config = load_config_file(config_file_path)
        
        # 检查是否已存在相同路径的数据集
        if check_dataset_exists(config, rel_path):
            print(f"  ℹ️  数据集 {dataset_name} (路径: {rel_path}) 已存在于配置文件中，跳过更新")
            return True
        
        # 查找或创建 category
        category = find_or_create_category(config, category_id, category_name, evaluation_type)
        
        # 创建数据集条目
        dataset_entry = create_dataset_entry(
            dataset_name, rel_path, evaluation_type, type_or_benchmark, params
        )
        
        # 添加到 category
        category["datasets"].append(dataset_entry)
        
        # 保存配置文件
        if save_config_file(config, config_file_path):
            print(f"  ✅ 已更新配置文件: {config_file_path}")
            print(f"     添加数据集: {dataset_name} ({category_id})")
            print(f"     路径: {rel_path}")
            print(f"     类型: {type_or_benchmark} ({evaluation_type})")
            return True
        else:
            return False
        
    except Exception as e:
        print(f"  ⚠️  更新配置文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ----------------------------------------------------------------------------
# 6.8 配置更新流程控制函数
# ----------------------------------------------------------------------------

def handle_config_update(
    update_config_file_path: str,
    dataset_dir: str,
    task_type: str,
    task_subtype: Optional[str],
    generated_files: Optional[list] = None
) -> bool:
    """
    处理配置文件更新的完整流程
    
    这是一个统一的入口函数，封装了配置更新的所有逻辑：
    - 推断数据集名称
    - 推断 base_path
    - 调用 update_config_file 更新配置
    
    Args:
        update_config_file_path: 配置文件路径
        dataset_dir: 数据集目录路径
        task_type: 任务类型
        task_subtype: 任务子类型
        generated_files: 生成的文件路径列表
    
    Returns:
        是否成功更新
    """
    # 推断数据集名称
    dataset_name = infer_dataset_name(dataset_dir)
    
    # 推断 base_path
    base_path = infer_base_path(dataset_dir)
    
    # 更新配置文件
    return update_config_file(
        update_config_file_path,
        dataset_name,
        dataset_dir,
        task_type,
        task_subtype,
        base_path,
        generated_files
    )

# ============================================================================
# 模块 7: 主流程模块
# 职责: 数据集转换的主入口和流程控制
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='自动转换数据集 - 在数据集目录中创建 converted 文件夹',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换整个数据集目录
  python auto_convert_dataset.py /path/to/dataset
  
  # 只转换特定的 splits
  python auto_convert_dataset.py /path/to/dataset --splits validation test
        """
    )
    
    parser.add_argument('dataset_dir', help='数据集目录路径')
    parser.add_argument('--output-subdir', default='converted', help='输出子目录名（默认: converted）')
    parser.add_argument('--splits', nargs='+', default=None, help='要转换的 splits（默认: 所有可用的）')
    parser.add_argument('--config', default=None, help='数据集 config 名称（如果有多个 config）')
    
    args = parser.parse_args()
    
    success = auto_convert_dataset(
        dataset_dir=args.dataset_dir,
        output_subdir=args.output_subdir,
        splits=args.splits,
        config=args.config
    )
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
