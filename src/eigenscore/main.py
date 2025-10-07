#!/usr/bin/env python3
"""
EigenScore implementation for low-resource languages
Based on "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection"

This implementation follows the exact methodology from the paper:
1. Extract internal states from LLM during generation (not post-hoc encoding)
2. Compute covariance matrix of internal state representations
3. Calculate eigenvalues and apply the EigenScore formula
4. Use eigenvalue distribution to detect hallucinations
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
# EigenScore使用LLM内部状态，不需要外部句子编码器
import torch.nn.functional as F

# Ensure repository paths are available regardless of execution location
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
REPO_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATASETS_DIR = REPO_ROOT / "datasets"
MODELS_DIR = REPO_ROOT / "models"
BLEURT_DIR = MODELS_DIR / "BLEURT-20"

# Import local modules
from haloscope.utils import seed_everything, get_measures, print_measures
from tokenization_utils import preprocess_text_for_language, dataset_to_language
from multi_gpu_utils import (
    setup_multi_gpu, load_model_multi_gpu, generate_with_multi_gpu,
    load_bleurt_multi_gpu, compute_bleurt_score_multi_gpu,
    cleanup_gpu_memory, print_gpu_memory_usage, MultiGPUConfig
)

# Model path candidates for each supported checkpoint
MODEL_DIR_CANDIDATES = {
    'llama2_7B': ['llama', 'Llama-2-7b-hf'],
    'llama3_2_1B': ['llama3_2_1B', 'Llama-3.2-1B'],
    'opt_6_7b': ['opt-6.7b', 'opt'],
    'opt_1_3b': ['opt_1_3b', 'opt-1.3b'],
}

def resolve_model_path(model_name):
    """Return a usable model directory for the requested checkpoint."""
    for candidate in MODEL_DIR_CANDIDATES[model_name]:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = MODELS_DIR / candidate_path
        if candidate_path.exists():
            return str(candidate_path)
    raise FileNotFoundError(
        f"Could not locate local weights for '{model_name}'. Checked: "
        + ", ".join(str(MODELS_DIR / c) for c in MODEL_DIR_CANDIDATES[model_name])
    )

class EigenScore:
    """
    EigenScore: 使用LLM内部状态协方差矩阵的特征值来检测幻觉    
    包含两个核心组件：
    1. EigenScore metric: 基于协方差矩阵特征值的一致性度量
    2. Feature clipping: 测试时特征裁剪，减少过度自信的生成
    """
    def __init__(self, min_eigenvalue=1e-3, feature_clipping=True, clipping_threshold=3.0):
        """
        初始化EigenScore检测器
        
        Args:
            min_eigenvalue: 最小特征值阈值（论文中设为1e-3）
            feature_clipping: 是否启用特征裁剪（论文中的第二个组件）
            clipping_threshold: 特征裁剪阈值（论文中用于截断极端激活）
        """
        self.min_eigenvalue = min_eigenvalue
        self.feature_clipping = feature_clipping
        self.clipping_threshold = clipping_threshold
        print(f"EigenScore initialized with min_eigenvalue: {min_eigenvalue}")
        if feature_clipping:
            print(f"Feature clipping enabled with threshold: {clipping_threshold}")
        
    def extract_internal_states(self, model, tokenizer, responses, dataset_language):
        """
        从LLM的内部状态提取特征表示（论文核心方法）
        
        Args:
            model: 生成模型（LLaMA或OPT）
            tokenizer: 对应的tokenizer
            responses: 生成的回答列表
            dataset_language: 数据集语言
            
        Returns:
            numpy array: 每个回答的内部状态表示
        """
        internal_states = []
        
        with torch.no_grad():
            for response in responses:
                # 构造完整的输入（问题+回答），模拟生成过程
                # 这样可以获得模型生成该回答时的内部状态
                input_text = response  # 简化处理，直接使用回答文本
                
                # 使用与生成时相同的tokenization方法
                if 'opt' in str(model.__class__).lower():
                    from tokenization_utils import dataset_to_language
                    language = dataset_to_language(dataset_language)
                    inputs = safe_tokenize_for_opt(tokenizer, input_text, language).to(model.device)
                else:
                    inputs = tokenizer(input_text, return_tensors='pt').input_ids.to(model.device)
                
                # 获取模型的内部隐藏状态
                outputs = model(inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # 所有层的隐藏状态
                
                # 使用最后一层的最后一个token作为句子表示（论文中的做法）
                # 这代表了模型生成该回答时的内部状态
                last_layer_states = hidden_states[-1]  # 最后一层 [batch_size, seq_len, hidden_size]
                sentence_repr = last_layer_states[0, -1, :].cpu().numpy()  # 最后一个token的表示
                
                internal_states.append(sentence_repr)
        
        return np.array(internal_states)
    
    def apply_feature_clipping(self, internal_states):
        """
        应用特征裁剪（论文中的第二个核心组件）
        
        论文中提到：从自一致幻觉检测的角度，探索了一种测试时特征裁剪方法
        来截断内部状态中的极端激活，这减少了过度自信的生成
        
        Args:
            internal_states: numpy array, 内部状态表示
            
        Returns:
            clipped_states: 裁剪后的内部状态
        """
        if not self.feature_clipping:
            return internal_states
            
        # 计算每个特征维度的统计信息
        mean = np.mean(internal_states, axis=0)
        std = np.std(internal_states, axis=0)
        
        # 应用特征裁剪：截断超过阈值的极端值
        # 这有助于减少过度自信的生成
        clipped_states = np.copy(internal_states)
        
        # 对于每个样本的每个特征维度
        for i in range(len(internal_states)):
            for j in range(len(mean)):
                # 计算标准化值
                z_score = abs((internal_states[i, j] - mean[j]) / (std[j] + 1e-8))
                
                # 如果超过阈值，进行裁剪
                if z_score > self.clipping_threshold:
                    # 裁剪到阈值范围内
                    sign = np.sign(internal_states[i, j] - mean[j])
                    clipped_states[i, j] = mean[j] + sign * self.clipping_threshold * std[j]
        
        return clipped_states
        
    def compute_eigenscore(self, internal_states):
        """
        基于内部状态计算EigenScore
        
        包含两个步骤：
        1. 特征裁剪（如果启用）- 减少过度自信的生成
        2. EigenScore计算 - 基于协方差矩阵特征值
        
        Args:
            internal_states: numpy array，LLM的内部状态表示 [num_responses, hidden_size]
            
        Returns:
            eigenscore: float，EigenScore值（越负表示越一致）
            eigenvalues: numpy array，协方差矩阵的特征值
        """
        if len(internal_states) < 2:
            return 0.0, np.array([1.0])
            
        embeddings = np.array(internal_states)
        
        # 步骤1: 应用特征裁剪（论文中的第二个核心组件）
        embeddings = self.apply_feature_clipping(embeddings)
        
        # 中心化嵌入（减去均值）
        centered_embeddings = embeddings - np.mean(embeddings, axis=0)
        
        # 计算协方差矩阵
        # 使用 (n-1) 作为分母进行无偏估计
        n_samples = len(embeddings)
        if n_samples <= 1:
            return 0.0, np.array([1.0])
            
        # 计算协方差矩阵的特征值
        # 优化：对于高维数据，只计算前k个最大特征值（更快）
        # 协方差矩阵是对称正定的，使用SVD更稳定
        if n_samples < centered_embeddings.shape[1]:
            # 当样本数 < 特征数时，使用SVD更高效
            # C = X^T X / (n-1), 其特征值 = SVD(X)的奇异值^2 / (n-1)
            from scipy.linalg import svd
            try:
                # 只计算前n_samples个奇异值（这是最多有意义的特征值数量）
                _, singular_values, _ = svd(centered_embeddings, full_matrices=False)
                eigenvalues = (singular_values ** 2) / (n_samples - 1)
                eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列
            except Exception as e:
                # Fallback to regular covariance if SVD fails
                print(f"SVD failed, using regular covariance: {e}")
                covariance_matrix = np.cov(centered_embeddings, rowvar=False, bias=False)
                eigenvalues = np.linalg.eigvals(covariance_matrix)
                eigenvalues = np.real(eigenvalues)
                eigenvalues = np.sort(eigenvalues)[::-1]
        else:
            # 标准情况：样本数 >= 特征数
            covariance_matrix = np.cov(centered_embeddings, rowvar=False, bias=False)
            eigenvalues = np.linalg.eigvals(covariance_matrix)
            eigenvalues = np.real(eigenvalues)  # 取实部
            eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列
        
        # 应用最小特征值阈值（论文中的做法）
        eigenvalues = np.maximum(eigenvalues, self.min_eigenvalue)
        
        # 计算EigenScore
        # EigenScore = log(λ₁) - log(mean(λ₂, λ₃, ..., λₖ))
        if len(eigenvalues) > 1:
            max_eigenvalue = eigenvalues[0]
            mean_other_eigenvalues = np.mean(eigenvalues[1:])
            eigenscore = np.log(max_eigenvalue) - np.log(mean_other_eigenvalues)
        else:
            # 如果只有一个特征值，使用简化公式
            eigenscore = np.log(eigenvalues[0])
            
        return float(eigenscore), eigenvalues

def load_low_resource_dataset(dataset_name):
    """
    Load and format low-resource language datasets with proper train/validation/test splits
    """
    if dataset_name == 'armenian':
        # Load SynDARin Armenian dataset
        armenian_dir = DATASETS_DIR / 'armenian'
        train_df = pd.read_csv(armenian_dir / 'SynDARin_Arm_train.csv')
        test_df = pd.read_csv(armenian_dir / 'SynDARin_Arm_test.csv')
        
        # Convert to standard format
        train_data = []
        for _, row in train_df.iterrows():
            train_data.append({
                'question': row['question'],
                'answer': row.get('correct_answer', row.get('answer', '')),
                'id': f"armenian_train_{len(train_data)}",
                'split': 'train'
            })
            
        test_data = []
        for _, row in test_df.iterrows():
            test_data.append({
                'question': row['question'],
                'answer': row.get('correct_answer', row.get('answer', '')),
                'id': f"armenian_test_{len(test_data)}",
                'split': 'test'
            })
            
        # For Armenian, create validation split from train data (20% of train)
        val_size = len(train_data) // 5
        val_data = train_data[-val_size:]
        train_data = train_data[:-val_size]
        
        # Update split labels
        for item in val_data:
            item['split'] = 'validation'
            item['id'] = item['id'].replace('train', 'val')
        
        dataset = {'train': train_data, 'validation': val_data, 'test': test_data}
                
    elif dataset_name == 'basque':
        # Load ElkarHizketak Basque dataset with all splits
        import pyarrow.parquet as pq
        
        basque_dir = DATASETS_DIR / 'basque'
        train_table = pq.read_table(basque_dir / 'train-00000-of-00001.parquet')
        val_table = pq.read_table(basque_dir / 'validation-00000-of-00001.parquet')
        test_table = pq.read_table(basque_dir / 'test-00000-of-00001.parquet')
        
        def convert_basque_split(table, split_name):
            df = table.to_pandas()
            data = []
            for _, row in df.iterrows():
                # Extract answer from dictionary structure (orig_answer has 'text' field)
                answer_dict = row.get('orig_answer', {})
                answer_text = answer_dict.get('text', '') if isinstance(answer_dict, dict) else ''
                
                data.append({
                    'question': row.get('question', row.get('input', '')),
                    'answer': answer_text,
                    'id': f"basque_{split_name}_{len(data)}",
                    'split': split_name
                })
            return data
            
        dataset = {
            'train': convert_basque_split(train_table, 'train'),
            'validation': convert_basque_split(val_table, 'validation'),
            'test': convert_basque_split(test_table, 'test')
        }
            
    elif dataset_name == 'tigrinya':
        # Load TigQA dataset with all splits
        def load_tigrinya_split(filename, split_name):
            tigrinya_dir = DATASETS_DIR / 'tigrinya'
            with open(tigrinya_dir / filename, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
                
            data = []
            for article in split_data['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        data.append({
                            'question': qa['question'],
                            'answer': qa['answers'][0]['text'] if qa['answers'] else '',
                            'context': context,
                            'id': qa['id'],
                            'split': split_name
                        })
            return data
        
        dataset = {
            'train': load_tigrinya_split('train.json', 'train'),
            'validation': load_tigrinya_split('dev.json', 'validation'),
            'test': load_tigrinya_split('test.json', 'test')
        }
    
    return dataset

def safe_tokenize_for_opt(tokenizer, text, dataset_language, max_length=512):
    """
    Safely tokenize text for OPT models by handling out-of-vocabulary characters
    """
    language = dataset_to_language(dataset_language)

    candidate_texts = []
    processed_text = preprocess_text_for_language(text, language)
    candidate_texts.append(processed_text)
    if processed_text != text:
        candidate_texts.append(text)

    last_error = None
    for candidate in candidate_texts:
        try:
            tokens = tokenizer(candidate, return_tensors='pt', truncation=True, max_length=max_length)

            if hasattr(tokenizer, 'vocab_size'):
                max_token_id = tokens.input_ids.max().item()
                if max_token_id >= tokenizer.vocab_size:
                    raise ValueError(
                        f"Token ID {max_token_id} exceeds vocab size {tokenizer.vocab_size}"
                    )

            return tokens.input_ids

        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            print(f"Tokenization failed for language={language}: {exc}")

    if language not in {'tigrinya', 'armenian'}:
        try:
            ascii_text = ''.join(char if ord(char) < 128 else ' ' for char in processed_text)
            ascii_text = re.sub(r'\s+', ' ', ascii_text).strip()
            tokens = tokenizer(ascii_text, return_tensors='pt', truncation=True, max_length=max_length)
            print(f"Using ASCII fallback: '{ascii_text[:50]}...'")
            return tokens.input_ids
        except Exception as exc:  # pylint: disable=broad-except
            print(f"ASCII fallback failed: {exc}")

    try:
        fallback_text = "Answer the question: "
        tokens = tokenizer(fallback_text, return_tensors='pt')
        print(f"Using generic fallback: '{fallback_text}'")
        return tokens.input_ids
    except Exception as exc:  # pylint: disable=broad-except
        print(f"All tokenization attempts failed (last error: {last_error}): {exc}")
        return tokenizer("", return_tensors='pt').input_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B', 
                       choices=['llama2_7B', 'llama3_2_1B', 'opt_6_7b', 'opt_1_3b'])
    parser.add_argument('--dataset_name', type=str, default='armenian',
                       choices=['armenian', 'basque', 'tigrinya'])
    parser.add_argument('--num_gene', type=int, default=5, help='Number of generations per question (论文中使用9个)')
    parser.add_argument('--gene', type=int, default=0, help='Generate responses (1) or analyze existing (0)')
    parser.add_argument('--reuse_lexical', type=str, default='', 
                       help='Reuse samples from lexical similarity (provide path to lexical results)')
    parser.add_argument('--generate_gt', type=int, default=0, help='Generate ground truth labels')
    parser.add_argument('--use_rouge', type=int, default=1, help='Use ROUGE instead of BLEURT')
    parser.add_argument('--thres_gt', type=float, default=0.5, help='Ground truth threshold')
    parser.add_argument('--use_percentile_threshold', type=int, default=1, help='Use percentile-based threshold instead of fixed threshold')
    parser.add_argument('--positive_percentage', type=float, default=0.3, help='Percentage of samples to label as positive (hallucinations)')
    parser.add_argument('--most_likely', type=int, default=1, help='Use greedy sampling for main response')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    # 移除sentence_model参数，因为EigenScore使用LLM内部状态
    # parser.add_argument('--sentence_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', 
    #                    help='Sentence transformer model for embeddings')
    parser.add_argument('--min_eigenvalue', type=float, default=1e-3, 
                       help='Minimum eigenvalue threshold (论文中设为1e-3)')
    parser.add_argument('--feature_clipping', type=int, default=0,
                       help='Enable feature clipping (论文中的第二个核心组件，默认关闭以加快速度)')
    parser.add_argument('--clipping_threshold', type=float, default=3.0,
                       help='Feature clipping threshold for extreme activations')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use (default: 4)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for generation (auto-determined if not specified)')
    parser.add_argument('--disable_multi_gpu', action='store_true', help='Disable multi-GPU and use single GPU')
    parser.add_argument('--gpu_id', type=int, default=None, help='Specific GPU ID to use (0, 1, 2, 3). If set, will use only this GPU')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    seed_everything(41)
    
    # Setup GPU environment
    if args.gpu_id is not None:
        # Use specific GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        num_gpus = 1
        print(f"Using specific GPU {args.gpu_id}")
    elif args.disable_multi_gpu:
        num_gpus = 1
        print("Using single GPU mode")
    else:
        num_gpus, primary_device = setup_multi_gpu()
        print(f"Using multi-GPU mode with {num_gpus} GPUs")
    
    # Initialize GPU configuration
    gpu_config = MultiGPUConfig(
        num_gpus=1 if args.gpu_id is not None else num_gpus,
        batch_size=args.batch_size,
        model_name=args.model_name
    )
    print(f"GPU Configuration: {gpu_config}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/eigenscore', exist_ok=True)
    os.makedirs(f'{args.output_dir}/eigenscore/{args.dataset_name}', exist_ok=True)
    os.makedirs(f'{args.output_dir}/eigenscore/{args.dataset_name}/answers', exist_ok=True)
    
    print(f"EigenScore Low-Resource Language Evaluation")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {'Generation' if args.gene else 'Ground Truth' if args.generate_gt else 'Analysis'}")
    
    # Check model-dataset compatibility
    if 'opt' in args.model_name.lower() and args.dataset_name == 'tigrinya':
        print("WARNING: OPT models have limited compatibility with Tigrinya (Ge'ez script)")
        print("This may result in tokenization errors or poor performance.")
        print("Consider using LLaMA models for Tigrinya, or Armenian/Basque datasets for OPT.")
        print("Continuing with fallback text processing...")
    
    # Load dataset with proper splits
    dataset_splits = load_low_resource_dataset(args.dataset_name)
    train_data = dataset_splits['train']
    val_data = dataset_splits['validation'] 
    test_data = dataset_splits['test']
    
    print(f"Loaded {args.dataset_name} dataset:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples") 
    print(f"  Test: {len(test_data)} samples")
    
    # Combine all data for processing (following EigenScore methodology)
    all_data = train_data + val_data + test_data
    
    if args.gene:
        # Generation phase - generate responses and extract internal states simultaneously
        generate_responses_with_internal_states(all_data, args, gpu_config)
    elif args.generate_gt:
        # Ground truth generation phase  
        generate_ground_truth(all_data, args, gpu_config)
    else:
        # Analysis phase - run EigenScore analysis
        run_eigenscore_analysis(all_data, args, gpu_config)

def generate_responses_with_internal_states(dataset, args, gpu_config):
    """
    Generate model responses AND extract internal states simultaneously
    This is the correct approach as per INSIDE paper - capture states during generation
    """
    MODEL = resolve_model_path(args.model_name)
    
    # Load model with GPU support
    if args.gpu_id is not None:
        # For specific GPU, use single GPU mode
        model, tokenizer, primary_device = load_model_multi_gpu(
            MODEL, args.model_name, num_gpus=1
        )
    else:
        # Use multi-GPU or single GPU based on configuration
        model, tokenizer, primary_device = load_model_multi_gpu(
            MODEL, args.model_name, gpu_config.num_gpus
        )
    
    print(f"Generating responses with {args.model_name} for EigenScore...")
    language = dataset_to_language(args.dataset_name)
    
    for i in tqdm(range(len(dataset)), desc="Generating"):
        question = dataset[i]['question']
        
        # Format prompt exactly like other methods
        if args.dataset_name == 'tigrinya' and 'context' in dataset[i]:
            # Truncate very long contexts to avoid OOM
            context = dataset[i]['context']
            if len(context) > 1000:  # Limit context to reasonable size
                context = context[:1000] + "..."
            prompt_text = "Concisely answer the following question based on the information in the given passage: \n" + \
                " Passage: " + context + " \n Q: " + question + " \n A:"
        else:
            prompt_text = f"Answer the question concisely. Q: {question}" + " A:"
        
        prompt_text = preprocess_text_for_language(prompt_text, language)

        # Use safe tokenization for OPT models with attention mask
        if 'opt' in args.model_name.lower():
            prompt_inputs = safe_tokenize_for_opt(tokenizer, prompt_text, language)
            prompt = prompt_inputs.to(primary_device)
            # Create attention mask to avoid warnings
            attention_mask = torch.ones_like(prompt).to(primary_device)
        else:
            prompt_inputs = tokenizer(prompt_text, return_tensors='pt', padding=True, truncation=True)
            prompt = prompt_inputs['input_ids'].to(primary_device)
            attention_mask = prompt_inputs['attention_mask'].to(primary_device)
        
        # Generate main response while capturing internal states
        main_answer = None
        main_internal_states = None
        torch.cuda.empty_cache()
        
        # Generate main response while capturing internal states
        generation_kwargs = {
            'attention_mask': attention_mask,
            'num_beams': 5,
            'num_return_sequences': 1,
            'do_sample': False,
            'max_new_tokens': 64,
            'output_hidden_states': True,  # 关键：获取内部状态
            'return_dict_in_generate': True,  # 返回详细信息
        }
        generated = generate_with_multi_gpu(model, tokenizer, prompt, generation_kwargs, primary_device)
            
        # 提取生成的文本
        generated_ids = generated.sequences
        decoded = tokenizer.decode(
            generated_ids[0, prompt.shape[-1]:],
            skip_special_tokens=True
        )
        
        # 提取内部状态（遵循INSIDE论文方法）
        if hasattr(generated, 'hidden_states') and generated.hidden_states:
            # hidden_states是一个tuple，每个元素对应一个生成步骤
            # 根据论文，使用生成过程中的内部状态
            last_step_states = generated.hidden_states[-1]  # 最后一步
            if isinstance(last_step_states, tuple):
                # 可能需要多层融合，这里先使用最后一层
                # 根据原始代码验证是否需要多层融合
                last_layer_states = last_step_states[-1]  # 最后一层
                
                # 使用最后一个生成token的状态（代表整个回答的语义）
                # 这是论文中提到的"internal states"的具体体现
                main_internal_states = last_layer_states[0, -1, :].cpu().numpy()
                
                print(f"Extracted internal state shape: {main_internal_states.shape}")
        
        # Clean up exactly like original
        if 'Answer the question concisely' in decoded:
            decoded = decoded.split('Answer the question concisely')[0]
            
        decoded = decoded.strip()
        if '\n' in decoded:
            decoded = decoded.split('\n')[0]
            
        main_answer = decoded
        print(f"Main answer: {decoded}")
        
        # Generate sampled responses while capturing internal states
        sampled_answers = []
        sampled_internal_states = []
        
        for gen_iter in range(args.num_gene):
            cleanup_gpu_memory()
            
            # Generate sampled response while capturing internal states
            generation_kwargs = {
                'attention_mask': attention_mask,
                'do_sample': True,
                'num_return_sequences': 1,
                'num_beams': 1,
                'max_new_tokens': 64,
                'temperature': 0.8,  # 较高温度以获得更多样化的回答
                'top_p': 0.95,
                'output_hidden_states': True,  # 关键：获取内部状态
                'return_dict_in_generate': True,  # 返回详细信息
            }
            generated = generate_with_multi_gpu(model, tokenizer, prompt, generation_kwargs, primary_device)
                
            # 提取生成的文本
            generated_ids = generated.sequences
            decoded = tokenizer.decode(
                generated_ids[0, prompt.shape[-1]:],
                skip_special_tokens=True
            )
            
            # 提取内部状态（与主回答相同的方法）
            sample_internal_states = None
            if hasattr(generated, 'hidden_states') and generated.hidden_states:
                last_step_states = generated.hidden_states[-1]  # 最后一步
                if isinstance(last_step_states, tuple):
                    last_layer_states = last_step_states[-1]  # 最后一层
                    sample_internal_states = last_layer_states[0, -1, :].cpu().numpy()
            
            # Clean up exactly like original
            if 'Answer the question concisely' in decoded:
                decoded = decoded.split('Answer the question concisely')[0]
                
            decoded = decoded.strip()
            if '\n' in decoded:
                decoded = decoded.split('\n')[0]
                
            sampled_answers.append(decoded)
            if sample_internal_states is not None:
                sampled_internal_states.append(sample_internal_states)
                
            if gen_iter < 3:  # Print first few for debugging
                print(f"Sample {gen_iter}: {decoded}")
            
        # Save main answer and its internal states
        np.save(
            f'{args.output_dir}/eigenscore/{args.dataset_name}/answers/main_answer_{args.model_name}_{args.dataset_name}_index_{i}.npy',
            [main_answer]
        )
        
        # Save sampled answers
        np.save(
            f'{args.output_dir}/eigenscore/{args.dataset_name}/answers/sampled_answers_{args.model_name}_{args.dataset_name}_index_{i}.npy',
            sampled_answers
        )
        
        # Save internal states (核心改进：保存生成时的真实内部状态)
        all_internal_states = []
        if main_internal_states is not None:
            all_internal_states.append(main_internal_states)
        all_internal_states.extend(sampled_internal_states)
        
        if len(all_internal_states) > 0:
            np.save(
                f'{args.output_dir}/eigenscore/{args.dataset_name}/answers/internal_states_{args.model_name}_{args.dataset_name}_index_{i}.npy',
                np.array(all_internal_states)
            )
            print(f"Saved {len(all_internal_states)} internal states")
        
    print(f"Generated responses saved to {args.output_dir}/eigenscore/{args.dataset_name}/answers/")

def generate_ground_truth(dataset, args, gpu_config):
    """
    Generate ground truth labels using BLEURT or ROUGE similarity scoring
    """
    print("Generating ground truth labels for EigenScore...")
    
    if args.use_rouge:
        # Use ROUGE scoring
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        gts = []
        for i in tqdm(range(len(dataset)), desc="Computing GT with ROUGE"):
            try:
                main_answer = np.load(
                    f'{args.output_dir}/eigenscore/{args.dataset_name}/answers/main_answer_{args.model_name}_{args.dataset_name}_index_{i}.npy'
                )[0]
                
                # Get reference answer
                reference = dataset[i]['answer']
                
                if reference and main_answer:
                    score = scorer.score(reference, main_answer)['rougeL'].fmeasure
                    gts.append(score)
                else:
                    gts.append(0.0)
                
            except FileNotFoundError:
                print(f"Main answer file not found for index {i}. Run generation first.")
                return
        
        gts = np.array(gts)
        suffix = 'rouge_score'
        
    else:
        # Use BLEURT scoring with multi-GPU support
        model, tokenizer = load_bleurt_multi_gpu(BLEURT_DIR, gpu_config.num_gpus)
        
        if model is None:
            print("Failed to load BLEURT model, falling back to ROUGE scoring...")
            args.use_rouge = 1
            return generate_ground_truth(dataset, args, gpu_config)
        
        # Collect all predictions and references for batch processing
        predictions = []
        references = []
        
        for i in tqdm(range(len(dataset)), desc="Loading data for BLEURT"):
            try:
                main_answer = np.load(
                    f'{args.output_dir}/eigenscore/{args.dataset_name}/answers/main_answer_{args.model_name}_{args.dataset_name}_index_{i}.npy'
                )[0]
                
                # Get reference answer
                reference = dataset[i]['answer']
                
                predictions.append(str(main_answer))
                references.append(str(reference))
                
            except FileNotFoundError:
                print(f"Main answer file not found for index {i}. Run generation first.")
                return
        
        # Compute BLEURT scores with multi-GPU support
        print("Computing BLEURT scores with multi-GPU...")
        gts = compute_bleurt_score_multi_gpu(model, tokenizer, predictions, references)
        suffix = 'bleurt_score'
    
    # Save ground truth scores
    filename = f'{args.output_dir}/eigenscore/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_{suffix}.npy'
    np.save(filename, gts)
    print(f"Ground truth scores saved to {filename}")
    
    # Print score statistics to help with threshold selection
    print(f"\nGround truth score statistics:")
    print(f"Min: {np.min(gts):.4f}")
    print(f"Max: {np.max(gts):.4f}")
    print(f"Mean: {np.mean(gts):.4f}")
    print(f"Median: {np.median(gts):.4f}")
    print(f"25th percentile: {np.percentile(gts, 25):.4f}")
    print(f"50th percentile: {np.percentile(gts, 50):.4f}")
    print(f"75th percentile: {np.percentile(gts, 75):.4f}")
    print(f"90th percentile: {np.percentile(gts, 90):.4f}")
    print(f"95th percentile: {np.percentile(gts, 95):.4f}")

def run_eigenscore_analysis(dataset, args, gpu_config):
    """
    Run EigenScore analysis using LLM internal states
    """
    print("Running EigenScore analysis with LLM internal states...")
    
    # Initialize EigenScore detector
    eigenscore_detector = EigenScore(
        min_eigenvalue=args.min_eigenvalue,
        feature_clipping=bool(args.feature_clipping),
        clipping_threshold=args.clipping_threshold
    )
    
    # Check if all internal states files exist to avoid unnecessary model loading
    print("Checking for pre-computed internal states...")
    all_internal_states_exist = True
    missing_files = []
    
    for i in range(len(dataset)):
        internal_states_file = f'{args.output_dir}/eigenscore/{args.dataset_name}/answers/internal_states_{args.model_name}_{args.dataset_name}_index_{i}.npy'
        if not os.path.exists(internal_states_file):
            all_internal_states_exist = False
            missing_files.append(i)
            if len(missing_files) <= 5:  # Only show first 5 missing files
                print(f"Missing internal states file for sample {i}")
    
    # Only load model if we need fallback method
    model = None
    tokenizer = None
    primary_device = None
    
    if not all_internal_states_exist:
        print(f"Found {len(missing_files)} missing internal states files. Loading model for fallback method...")
        MODEL = resolve_model_path(args.model_name)
        
        # Load model with GPU support
        if args.gpu_id is not None:
            # For specific GPU, use single GPU mode
            model, tokenizer, primary_device = load_model_multi_gpu(
                MODEL, args.model_name, num_gpus=1
            )
        else:
            # Use multi-GPU or single GPU based on configuration
            model, tokenizer, primary_device = load_model_multi_gpu(
                MODEL, args.model_name, gpu_config.num_gpus
            )
    else:
        print("All internal states files found! Skipping model loading for faster processing.")
    
    # Load ground truth labels
    print("Loading ground truth labels...")
    try:
        if args.use_rouge:
            gts = np.load(f'{args.output_dir}/eigenscore/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_rouge_score.npy')
            print("Loaded ROUGE-based ground truth scores")
        else:
            gts = np.load(f'{args.output_dir}/eigenscore/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_bleurt_score.npy')
            print("Loaded BLEURT-based ground truth scores")
    except FileNotFoundError as e:
        print(f"Ground truth file not found: {e}")
        print("Please run ground truth generation first with --generate_gt 1")
        return
        
    # Create binary labels with balanced approach
    if args.use_percentile_threshold:
        # Use percentile-based threshold to ensure balanced labels
        print(f"Using percentile-based threshold with {args.positive_percentage*100:.1f}% positive labels")
        
        # Calculate threshold based on desired percentage of positive labels
        # Note: Lower percentile for lower scores (worse answers = hallucinations)
        threshold_percentile = args.positive_percentage * 100
        dynamic_threshold = np.percentile(gts, threshold_percentile)
        
        print(f"Dynamic threshold (P{threshold_percentile:.1f}): {dynamic_threshold:.4f}")
        print(f"Fixed threshold would be: {args.thres_gt:.4f}")
        
        # Create labels: samples BELOW threshold are labeled as hallucinations (1)
        # Low ROUGE/BLEURT scores = dissimilar to reference = likely hallucination
        gt_label = np.asarray(gts < dynamic_threshold, dtype=np.int32)
        
        # Ensure we have at least some positive and negative samples
        n_positive = np.sum(gt_label)
        n_total = len(gt_label)
        
        if n_positive == 0:
            # If no positives, label the bottom few samples (worst scores) as positive
            n_force_positive = max(1, int(0.05 * n_total))  # At least 5% positive
            bottom_indices = np.argsort(gts)[:n_force_positive]  # Lowest scores = hallucinations
            gt_label[bottom_indices] = 1
            print(f"Forced {n_force_positive} samples to be positive (worst/lowest scores)")
        elif n_positive == n_total:
            # If all positive, label the top few samples (best scores) as negative
            n_force_negative = max(1, int(0.05 * n_total))  # At least 5% negative
            top_indices = np.argsort(gts)[-n_force_negative:]  # Highest scores = good answers
            gt_label[top_indices] = 0
            print(f"Forced {n_force_negative} samples to be negative (best/highest scores)")
            
        actual_threshold = dynamic_threshold
    else:
        # Use fixed threshold (original behavior)
        print(f"Using fixed threshold: {args.thres_gt:.4f}")
        # Samples BELOW threshold are hallucinations (low similarity = bad)
        gt_label = np.asarray(gts < args.thres_gt, dtype=np.int32)
        actual_threshold = args.thres_gt
    
    # Print label distribution
    n_positive = np.sum(gt_label)
    n_negative = len(gt_label) - n_positive
    print(f"\nLabel distribution:")
    print(f"Positive (hallucination): {n_positive} ({n_positive/len(gt_label)*100:.1f}%)")
    print(f"Negative (good): {n_negative} ({n_negative/len(gt_label)*100:.1f}%)")
    print(f"Threshold used: {actual_threshold:.4f}")
    
    # Print score ranges for each class
    if n_positive > 0 and n_negative > 0:
        pos_scores = gts[gt_label == 1]
        neg_scores = gts[gt_label == 0]
        print(f"Positive class scores: {np.min(pos_scores):.4f} - {np.max(pos_scores):.4f} (mean: {np.mean(pos_scores):.4f})")
        print(f"Negative class scores: {np.min(neg_scores):.4f} - {np.max(neg_scores):.4f} (mean: {np.mean(neg_scores):.4f})")
    
    print("Computing EigenScore scores...")
    all_scores = []
    all_eigenvalues = []
    
    for i in tqdm(range(len(dataset)), desc="EigenScore Analysis"):
        try:
            # 直接加载生成时保存的内部状态（论文正确方法）
            internal_states_file = f'{args.output_dir}/eigenscore/{args.dataset_name}/answers/internal_states_{args.model_name}_{args.dataset_name}_index_{i}.npy'
            
            if os.path.exists(internal_states_file):
                # 使用生成时的真实内部状态
                internal_states = np.load(internal_states_file)
                # Removed verbose print for faster processing
            else:
                # 如果没有内部状态文件，回退到重新编码（不推荐）
                if i < 5:  # Only print warnings for first few samples to avoid spam
                    print(f"Warning: Internal states file not found for sample {i}, using fallback method")
                
                # Load answers for fallback
                main_answer = np.load(
                    f'{args.output_dir}/eigenscore/{args.dataset_name}/answers/main_answer_{args.model_name}_{args.dataset_name}_index_{i}.npy'
                )[0]
                
                sampled_answers = np.load(
                    f'{args.output_dir}/eigenscore/{args.dataset_name}/answers/sampled_answers_{args.model_name}_{args.dataset_name}_index_{i}.npy'
                )
                
                # Convert numpy strings to regular Python strings
                main_answer = str(main_answer)
                sampled_answers = [str(answer) for answer in sampled_answers]
                all_responses = [main_answer] + sampled_answers
                
                # 使用fallback方法重新编码（不是论文的正确方法）
                internal_states = eigenscore_detector.extract_internal_states(
                    model, tokenizer, all_responses, args.dataset_name
                )
            
            # 基于内部状态计算EigenScore
            eigenscore, eigenvalues = eigenscore_detector.compute_eigenscore(internal_states)
            
            all_scores.append(eigenscore)
            all_eigenvalues.append(eigenvalues)
            
        except FileNotFoundError:
            print(f"Answer files not found for index {i}. Run generation first.")
            return
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            all_scores.append(0.0)  # Default score on error
            all_eigenvalues.append(np.array([1.0]))
    
    all_scores = np.array(all_scores)
    
    # EigenScore的解释：负值表示更一致（可能是幻觉），正值表示更多样化
    # 为了与其他方法保持一致（高分表示更可能是幻觉），我们使用负的EigenScore
    uncertainty_scores = -all_scores
    
    # Evaluate performance
    if len(gt_label[gt_label == 1]) > 0 and len(gt_label[gt_label == 0]) > 0:
        measures = get_measures(uncertainty_scores[gt_label == 1], uncertainty_scores[gt_label == 0])
        auroc, auprc, fpr95 = measures
        print(f"\nEigenScore Results for {args.dataset_name} ({args.model_name}):")
        print(f"AUROC: {auroc:.4f}")
        print_measures(auroc, auprc, fpr95, 'EigenScore')
    else:
        auroc = 0.5
        print("Warning: Ground truth has only one class, AUROC set to 0.5")
    
    # Save results
    results = {
        'auroc': auroc,
        'scores': uncertainty_scores.tolist(),
        'raw_eigenscores': all_scores.tolist(),
        'gt_labels': gt_label.tolist(),
        'gt_scores': gts.tolist(),  # Include raw ground truth scores
        'dataset': args.dataset_name,
        'model': args.model_name,
        'method': 'eigenscore_internal_states',  # 明确标注使用内部状态
        'num_samples': args.num_gene,
        'threshold': actual_threshold if 'actual_threshold' in locals() else args.thres_gt,
        'use_percentile_threshold': bool(args.use_percentile_threshold),
        'positive_percentage': args.positive_percentage,
        'label_distribution': {
            'positive': int(n_positive),
            'negative': int(n_negative),
            'positive_pct': float(n_positive/len(gt_label)*100)
        },
        'min_eigenvalue': args.min_eigenvalue,
        'note': 'Uses LLM internal states as per INSIDE paper'
    }
    
    results_file = f"{args.output_dir}/eigenscore/eigenscore_results_{args.dataset_name}_{args.model_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    # Save scores for further analysis
    scores_file = f"{args.output_dir}/eigenscore/{args.dataset_name}/scores_{args.model_name}_{args.dataset_name}.npy"
    np.save(scores_file, uncertainty_scores)
    print(f"Scores saved to {scores_file}")
    
    # Save eigenvalues for analysis
    eigenvalues_file = f"{args.output_dir}/eigenscore/{args.dataset_name}/eigenvalues_{args.model_name}_{args.dataset_name}.pkl"
    import pickle
    with open(eigenvalues_file, 'wb') as f:
        pickle.dump(all_eigenvalues, f)
    print(f"Eigenvalues saved to {eigenvalues_file}")

if __name__ == '__main__':
    main()
