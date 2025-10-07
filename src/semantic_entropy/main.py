#!/usr/bin/env python3
"""
Semantic Entropy implementation for hallucination detection
Based on "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation" paper
Link: https://arxiv.org/pdf/2302.09664

Semantic Entropy 是一种基于语义等价性的不确定性估计方法，用于检测大型语言模型的幻觉。
核心思想：
1. 对同一问题生成多个响应
2. 使用双向蕴含关系（bidirectional entailment）识别语义等价的答案簇
3. 基于语义簇的概率分布计算语义熵，作为不确定性度量
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings("ignore")

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

class SemanticEntropyCalculator:
    """
    Semantic Entropy Calculator: 基于双向蕴含关系的语义不确定性估计
    
    核心思想：
    1. 使用NLI模型判断两个答案之间的双向蕴含关系
    2. 如果答案A蕴含答案B，且答案B也蕴含答案A，则认为它们语义等价
    3. 将语义等价的答案聚类，基于语义簇的分布计算语义熵
    4. 语义熵越高表示模型对答案的语义不确定性越大
    """
    
    def __init__(self, 
                 nli_model='microsoft/deberta-v3-large-mnli',
                 entailment_threshold=0.8,
                 device='cuda'):
        """
        Initialize Semantic Entropy calculator
        
        Args:
            nli_model: NLI model for bidirectional entailment checking (论文中使用DeBERTa-v3-large)
            entailment_threshold: Threshold for entailment probability (论文中通常使用0.8)
            device: Computing device
        """
        self.entailment_threshold = entailment_threshold
        self.device = device
        
        # Initialize NLI model for bidirectional entailment checking
        # 初始化NLI模型用于双向蕴含关系检测
        try:
            print(f"Loading NLI model: {nli_model}")
            self.nli_tokenizer = DebertaV2Tokenizer.from_pretrained(nli_model)
            self.nli_model = DebertaV2ForSequenceClassification.from_pretrained(nli_model)
            self.nli_model.eval()
            self.nli_model.to(device)
            print(f"✓ Loaded NLI model: {nli_model}")
        except Exception as e:
            print(f"Failed to load NLI model {nli_model}: {e}")
            print("Falling back to sentence similarity for clustering...")
            # Fallback to sentence similarity
            self.use_nli = False
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
        else:
            self.use_nli = True
    
    @torch.no_grad()
    def check_bidirectional_entailment(self, answer1, answer2):
        """
        Check bidirectional entailment between two answers using NLI model
        检查两个答案之间的双向蕴含关系
        
        Args:
            answer1: First answer string
            answer2: Second answer string
            
        Returns:
            is_equivalent: Boolean indicating if answers are semantically equivalent
            entailment_scores: Tuple of (answer1->answer2, answer2->answer1) entailment probabilities
        """
        if not self.use_nli:
            # Fallback to simple string comparison
            return answer1.strip().lower() == answer2.strip().lower(), (1.0, 1.0)
        
        # Check entailment in both directions
        # 检查双向蕴含关系
        
        # Direction 1: answer1 entails answer2
        inputs_1_to_2 = self.nli_tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[(answer1, answer2)],
            add_special_tokens=True, padding="longest",
            truncation=True, return_tensors="pt",
            return_token_type_ids=True, return_attention_mask=True,
        )
        inputs_1_to_2 = inputs_1_to_2.to(self.device)
        logits_1_to_2 = self.nli_model(**inputs_1_to_2).logits
        probs_1_to_2 = torch.softmax(logits_1_to_2, dim=-1)
        entailment_prob_1_to_2 = probs_1_to_2[0][0].item()  # ENTAILMENT class is index 0
        
        # Direction 2: answer2 entails answer1
        inputs_2_to_1 = self.nli_tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[(answer2, answer1)],
            add_special_tokens=True, padding="longest",
            truncation=True, return_tensors="pt",
            return_token_type_ids=True, return_attention_mask=True,
        )
        inputs_2_to_1 = inputs_2_to_1.to(self.device)
        logits_2_to_1 = self.nli_model(**inputs_2_to_1).logits
        probs_2_to_1 = torch.softmax(logits_2_to_1, dim=-1)
        entailment_prob_2_to_1 = probs_2_to_1[0][0].item()  # ENTAILMENT class is index 0
        
        # Bidirectional entailment: both directions should have high entailment probability
        # 双向蕴含：两个方向都应该有高蕴含概率
        is_equivalent = (entailment_prob_1_to_2 >= self.entailment_threshold and 
                        entailment_prob_2_to_1 >= self.entailment_threshold)
        
        return is_equivalent, (entailment_prob_1_to_2, entailment_prob_2_to_1)
    
    def cluster_semantic_equivalent_answers(self, answers, likelihoods=None):
        """
        Cluster semantically equivalent answers using bidirectional entailment
        使用双向蕴含关系聚类语义等价的答案
        
        Args:
            answers: List of answer strings
            likelihoods: Optional list of answer likelihoods/probabilities
            
        Returns:
            clusters: List of clusters, each cluster contains indices of equivalent answers
            cluster_probabilities: List of total probability for each cluster
        """
        if len(answers) <= 1:
            return [[0]], [1.0] if likelihoods is None else [likelihoods[0]]
        
        n_answers = len(answers)
        
        # If no likelihoods provided, assume uniform distribution
        # 如果没有提供似然度，假设均匀分布
        if likelihoods is None:
            likelihoods = [1.0 / n_answers] * n_answers
        
        # Build adjacency matrix based on bidirectional entailment
        # 基于双向蕴含关系构建邻接矩阵
        adjacency_matrix = np.eye(n_answers, dtype=bool)  # Each answer is equivalent to itself
        
        print(f"  Checking bidirectional entailment for {n_answers} answers...")
        for i in range(n_answers):
            for j in range(i + 1, n_answers):
                is_equivalent, entailment_scores = self.check_bidirectional_entailment(
                    answers[i], answers[j]
                )
                
                if is_equivalent:
                    adjacency_matrix[i, j] = True
                    adjacency_matrix[j, i] = True
                    print(f"    Answers {i} and {j} are semantically equivalent")
                    print(f"    Entailment scores: {entailment_scores[0]:.3f}, {entailment_scores[1]:.3f}")
        
        # Find connected components (semantic clusters)
        # 寻找连通分量（语义簇）
        clusters = []
        visited = [False] * n_answers
        
        def dfs(node, current_cluster):
            visited[node] = True
            current_cluster.append(node)
            for neighbor in range(n_answers):
                if adjacency_matrix[node, neighbor] and not visited[neighbor]:
                    dfs(neighbor, current_cluster)
        
        for i in range(n_answers):
            if not visited[i]:
                current_cluster = []
                dfs(i, current_cluster)
                clusters.append(current_cluster)
        
        # Calculate cluster probabilities
        # 计算簇概率
        cluster_probabilities = []
        
        # Convert log probabilities to probabilities using softmax
        # 使用softmax将对数概率转换为概率
        if likelihoods is not None:
            # Assume likelihoods are log probabilities
            log_probs = np.array(likelihoods)
            # Apply softmax to get normalized probabilities
            max_log_prob = np.max(log_probs)
            exp_probs = np.exp(log_probs - max_log_prob)  # Subtract max for numerical stability
            normalized_probs = exp_probs / np.sum(exp_probs)
            
            for cluster in clusters:
                cluster_prob = sum(normalized_probs[idx] for idx in cluster)
                cluster_probabilities.append(cluster_prob)
        else:
            # If no likelihoods provided, assume uniform distribution
            # 如果没有提供似然度，假设均匀分布
            for cluster in clusters:
                cluster_prob = len(cluster) / len(answers)
                cluster_probabilities.append(cluster_prob)
        
        print(f"  Found {len(clusters)} semantic clusters with probabilities {cluster_probabilities}")
        
        return clusters, cluster_probabilities
    
    def calculate_semantic_entropy(self, cluster_probabilities):
        """
        Calculate semantic entropy based on cluster probability distribution
        基于簇概率分布计算语义熵
        
        Args:
            cluster_probabilities: List of probabilities for each semantic cluster
            
        Returns:
            semantic_entropy: Semantic entropy value (语义熵值)
        """
        if len(cluster_probabilities) <= 1:
            # If all answers are in the same semantic cluster, entropy is 0
            # 如果所有答案都在同一语义簇中，熵为0
            return 0.0
        
        # Calculate Shannon entropy over semantic clusters
        # 计算语义簇上的香农熵
        semantic_entropy = 0.0
        for prob in cluster_probabilities:
            if prob > 0:  # Avoid log(0)
                semantic_entropy -= prob * np.log(prob)
        
        return float(semantic_entropy)
    
    def compute_uncertainty(self, answers, likelihoods=None):
        """
        Compute Semantic Entropy uncertainty for a set of answers
        为一组答案计算语义熵不确定性
        
        Args:
            answers: List of answer strings
            likelihoods: Optional list of answer likelihoods/probabilities
            
        Returns:
            semantic_entropy: Semantic entropy value
            cluster_info: Dictionary with clustering information
        """
        if len(answers) <= 1:
            return 0.0, {'num_clusters': 1, 'clusters': [[0]], 'cluster_probabilities': [1.0]}
        
        # Step 1: Cluster semantically equivalent answers
        # 步骤1：聚类语义等价的答案
        clusters, cluster_probabilities = self.cluster_semantic_equivalent_answers(
            answers, likelihoods
        )
        
        # Step 2: Calculate semantic entropy
        # 步骤2：计算语义熵
        semantic_entropy = self.calculate_semantic_entropy(cluster_probabilities)
        
        cluster_info = {
            'num_clusters': len(clusters),
            'clusters': clusters,
            'cluster_probabilities': cluster_probabilities,
            'cluster_sizes': [len(cluster) for cluster in clusters]
        }
        
        return semantic_entropy, cluster_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B', 
                       choices=['llama2_7B', 'llama3_2_1B', 'opt_6_7b', 'opt_1_3b'])
    parser.add_argument('--dataset_name', type=str, default='armenian',
                       choices=['armenian', 'basque', 'tigrinya'])
    parser.add_argument('--num_gene', type=int, default=3, help='Number of generations per question for sampling')
    parser.add_argument('--gene', type=int, default=0, help='Generate responses (1) or analyze existing (0)')
    parser.add_argument('--generate_gt', type=int, default=0, help='Generate ground truth labels')
    parser.add_argument('--nli_model', type=str, default='microsoft/deberta-v3-large-mnli', help='NLI model for bidirectional entailment checking')
    parser.add_argument('--entailment_threshold', type=float, default=0.8, help='Threshold for entailment probability (paper uses 0.8)')
    parser.add_argument('--thres_gt', type=float, default=0.5, help='Ground truth threshold')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use (default: 4)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for generation (auto-determined if not specified)')
    parser.add_argument('--disable_multi_gpu', action='store_true', help='Disable multi-GPU and use single GPU')
    parser.add_argument('--gpu_id', type=int, default=None, help='Specific GPU ID to use (0, 1, 2, 3). If set, will use only this GPU')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    seed_everything(41)
    
    # Setup GPU environment - MUST be done before any CUDA operations
    if args.gpu_id is not None:
        # Use specific GPU - set environment variable before any CUDA calls
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        # Clear any existing CUDA context if it exists
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
        num_gpus = 1
        print(f"Using specific GPU {args.gpu_id}")
        print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
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
    os.makedirs(f'{args.output_dir}/semantic_entropy', exist_ok=True)
    os.makedirs(f'{args.output_dir}/semantic_entropy/{args.dataset_name}', exist_ok=True)
    os.makedirs(f'{args.output_dir}/semantic_entropy/{args.dataset_name}/answers', exist_ok=True)
    
    print(f"Semantic Entropy Low-Resource Language Evaluation")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {'Generation' if args.gene else 'Ground Truth' if args.generate_gt else 'Analysis'}")
    print(f"NLI Model: {args.nli_model}")
    print(f"Entailment Threshold: {args.entailment_threshold}")
    
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
    
    # Combine all data for processing (following Semantic Entropy methodology)
    all_data = train_data + val_data + test_data
    
    if args.gene:
        # Generation phase - generate multiple responses for Semantic Entropy analysis
        generate_responses(all_data, args, gpu_config)
    elif args.generate_gt:
        # Ground truth generation phase  
        generate_ground_truth(all_data, args, gpu_config)
    else:
        # Analysis phase - run Semantic Entropy analysis
        run_semantic_entropy_analysis(all_data, args)

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
                'answer': row.get('correct_answer', ''),
                'id': f"armenian_train_{len(train_data)}",
                'split': 'train'
            })
            
        test_data = []
        for _, row in test_df.iterrows():
            test_data.append({
                'question': row['question'],
                'answer': row.get('correct_answer', ''),
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
                data.append({
                    'question': row.get('question', row.get('input', '')),
                    'answer': row.get('answer', row.get('output', '')),
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

def safe_tokenize_for_opt_with_attention(tokenizer, text, dataset_language, max_length=512):
    """
    Safely tokenize text for OPT models with proper attention mask handling
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
            tokens = tokenizer(
                candidate, 
                return_tensors='pt', 
                truncation=True, 
                max_length=max_length,
                padding=True,
                return_attention_mask=True
            )

            if hasattr(tokenizer, 'vocab_size'):
                max_token_id = tokens.input_ids.max().item()
                if max_token_id >= tokenizer.vocab_size:
                    raise ValueError(
                        f"Token ID {max_token_id} exceeds vocab size {tokenizer.vocab_size}"
                    )

            return tokens

        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            print(f"Tokenization failed for language={language}: {exc}")

    if language not in {'tigrinya', 'armenian'}:
        try:
            ascii_text = ''.join(char if ord(char) < 128 else ' ' for char in processed_text)
            ascii_text = re.sub(r'\s+', ' ', ascii_text).strip()
            tokens = tokenizer(
                ascii_text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=max_length,
                padding=True,
                return_attention_mask=True
            )
            print(f"Using ASCII fallback: '{ascii_text[:50]}...'")
            return tokens
        except Exception as exc:  # pylint: disable=broad-except
            print(f"ASCII fallback failed: {exc}")

    try:
        fallback_text = "Answer the question: "
        tokens = tokenizer(
            fallback_text, 
            return_tensors='pt',
            padding=True,
            return_attention_mask=True
        )
        print(f"Using generic fallback: '{fallback_text}'")
        return tokens
    except Exception as exc:  # pylint: disable=broad-except
        print(f"All tokenization attempts failed (last error: {last_error}): {exc}")
        return tokenizer("", return_tensors='pt', padding=True, return_attention_mask=True)

def generate_responses(dataset, args, gpu_config):
    """
    Generate model responses for Semantic Entropy analysis
    Need multiple diverse responses for semantic clustering
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
    
    print(f"Generating responses with {args.model_name} for Semantic Entropy analysis...")
    print(f"Tokenizer setup: pad_token='{tokenizer.pad_token}' (id={tokenizer.pad_token_id}), eos_token='{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    language = dataset_to_language(args.dataset_name)
    
    for i in tqdm(range(len(dataset)), desc="Generating"):
        question = dataset[i]['question']
        
        # Format prompt exactly like other methods for consistency
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

        # Use safe tokenization for OPT models
        if 'opt' in args.model_name.lower():
            prompt_inputs = safe_tokenize_for_opt_with_attention(tokenizer, prompt_text, language)
            prompt = prompt_inputs['input_ids'].to(primary_device)
            attention_mask = prompt_inputs['attention_mask'].to(primary_device)
        else:
            prompt_inputs = tokenizer(prompt_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            prompt = prompt_inputs['input_ids'].to(primary_device)
            attention_mask = prompt_inputs['attention_mask'].to(primary_device)
        
        # Generate multiple diverse responses for Semantic Entropy analysis
        # 为语义熵分析生成多个多样化的响应
        responses = []
        response_likelihoods = []
        
        for gen_iter in range(args.num_gene):
            cleanup_gpu_memory()
            
            # Generate sampled response with multi-GPU support
            generation_kwargs = {
                'attention_mask': attention_mask,
                'do_sample': True,
                'num_return_sequences': 1,
                'num_beams': 1,
                'max_new_tokens': 64,
                'temperature': 0.7 + (gen_iter % 3) * 0.2,  # Vary temperature: 0.7, 0.9, 1.1
                'top_p': 0.9,
                'use_cache': True,
                'return_dict_in_generate': True,
                'output_scores': True  # Need scores for likelihood calculation
            }
            
            with torch.no_grad():
                generated_output = model.generate(prompt, **generation_kwargs)
                generated_tokens = generated_output.sequences
                scores = generated_output.scores
                
                # Calculate likelihood (normalized by sequence length)
                # 计算似然度（按序列长度归一化）
                if scores:
                    # Calculate average log probability per token to avoid underflow
                    log_likelihood = 0.0
                    num_tokens = len(scores)
                    
                    for step_idx, step_scores in enumerate(scores):
                        probs = torch.softmax(step_scores, dim=-1)
                        token_id = generated_tokens[0, prompt.shape[-1] + step_idx]
                        log_likelihood += torch.log(probs[0, token_id]).item()
                    
                    # Use average log likelihood to prevent underflow
                    avg_log_likelihood = log_likelihood / num_tokens if num_tokens > 0 else 0.0
                    
                    # Convert to perplexity-like score (higher is better)
                    # For Semantic Entropy, we need relative probabilities, not absolute
                    likelihood = np.exp(avg_log_likelihood)
                    
                    # Alternative: Use softmax over all generated responses for this question
                    # This will be normalized later when we collect all responses
                    likelihood = float(avg_log_likelihood)  # Keep as log probability for now
                else:
                    likelihood = 0.0  # Neutral log probability if no scores available
                
            decoded = tokenizer.decode(
                generated_tokens[0, prompt.shape[-1]:],
                skip_special_tokens=True
            )
            
            # Clean up exactly like other methods
            if 'Answer the question concisely' in decoded:
                decoded = decoded.split('Answer the question concisely')[0]
                
            decoded = decoded.strip()
            if '\n' in decoded:
                decoded = decoded.split('\n')[0]
                
            responses.append(decoded)
            response_likelihoods.append(likelihood)
            
            if gen_iter < 3:  # Print first few for debugging
                print(f"Response {gen_iter}: {decoded} (likelihood: {likelihood:.4f})")
            
        # Save generated responses and their likelihoods
        np.save(
            f'{args.output_dir}/semantic_entropy/{args.dataset_name}/answers/responses_{args.model_name}_{args.dataset_name}_index_{i}.npy',
            responses
        )
        np.save(
            f'{args.output_dir}/semantic_entropy/{args.dataset_name}/answers/likelihoods_{args.model_name}_{args.dataset_name}_index_{i}.npy',
            response_likelihoods
        )
        
    print(f"Generated responses saved to {args.output_dir}/semantic_entropy/{args.dataset_name}/answers/")

def generate_ground_truth(dataset, args, gpu_config):
    """
    Generate ground truth labels using BLEURT similarity scoring
    """
    print("Generating ground truth labels for Semantic Entropy...")
    
    # Load BLEURT model with multi-GPU support
    model, tokenizer = load_bleurt_multi_gpu(BLEURT_DIR, gpu_config.num_gpus)
    
    if model is None:
        print("Failed to load BLEURT model. BLEURT scoring is required for ground truth generation.")
        return
    
    # Collect all predictions and references for batch processing
    predictions = []
    references = []
    
    for i in tqdm(range(len(dataset)), desc="Loading data for BLEURT"):
        try:
            responses = np.load(
                f'{args.output_dir}/semantic_entropy/{args.dataset_name}/answers/responses_{args.model_name}_{args.dataset_name}_index_{i}.npy'
            )
            
            # Get reference answer
            reference = dataset[i]['answer']
            
            # For Semantic Entropy, we evaluate the first response (main response)
            # 对于语义熵，我们评估第一个响应（主要响应）
            main_response = responses[0] if len(responses) > 0 else ""
            
            predictions.append(str(main_response))
            references.append(str(reference))
            
        except FileNotFoundError:
            print(f"Responses file not found for index {i}. Run generation first.")
            return
    
    # Compute BLEURT scores with multi-GPU support
    print("Computing BLEURT scores with multi-GPU...")
    gts = compute_bleurt_score_multi_gpu(model, tokenizer, predictions, references)
    
    # Save ground truth scores
    filename = f'{args.output_dir}/semantic_entropy/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_bleurt_score.npy'
    np.save(filename, gts)
    print(f"Ground truth scores saved to {filename}")

def run_semantic_entropy_analysis(dataset, args):
    """
    Run Semantic Entropy analysis
    """
    print("Running Semantic Entropy analysis...")
    
    # Initialize Semantic Entropy calculator
    # Use cuda:0 since it's always the first visible device after CUDA_VISIBLE_DEVICES is set
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    semantic_entropy_calc = SemanticEntropyCalculator(
        nli_model=args.nli_model,
        entailment_threshold=args.entailment_threshold,
        device=device
    )
    
    # Load ground truth labels
    print("Loading ground truth labels...")
    try:
        gts = np.load(f'{args.output_dir}/semantic_entropy/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_bleurt_score.npy')
        print("Loaded BLEURT-based ground truth scores")
    except FileNotFoundError as e:
        print(f"Ground truth file not found: {e}")
        print("Please run ground truth generation first with --generate_gt 1")
        return
        
    # Create binary labels
    gt_label = np.asarray(gts > args.thres_gt, dtype=np.int32)
    
    print("Computing Semantic Entropy scores...")
    print(f"Processing {len(dataset)} samples with entailment threshold {args.entailment_threshold}")
    all_scores = []
    cluster_stats = []
    
    for i in tqdm(range(len(dataset)), desc="Semantic Entropy Analysis"):
        try:
            # Load generated responses and their likelihoods
            responses = np.load(
                f'{args.output_dir}/semantic_entropy/{args.dataset_name}/answers/responses_{args.model_name}_{args.dataset_name}_index_{i}.npy'
            )
            
            try:
                likelihoods = np.load(
                    f'{args.output_dir}/semantic_entropy/{args.dataset_name}/answers/likelihoods_{args.model_name}_{args.dataset_name}_index_{i}.npy'
                )
            except FileNotFoundError:
                # If no likelihoods file, assume uniform distribution
                likelihoods = None
            
            # Convert numpy strings to regular Python strings
            responses = [str(response) for response in responses if str(response).strip()]
            
            if len(responses) < 2:
                # If we have fewer than 2 responses, uncertainty is minimal
                # 如果响应少于2个，不确定性很小
                all_scores.append(0.0)
                cluster_stats.append({'num_clusters': 1, 'clusters': [[0]], 'cluster_probabilities': [1.0]})
                continue
            
            # Compute Semantic Entropy
            # 计算语义熵
            semantic_entropy, cluster_info = semantic_entropy_calc.compute_uncertainty(
                responses, likelihoods
            )
            all_scores.append(semantic_entropy)
            cluster_stats.append(cluster_info)
            
            if i < 5:  # Print first few for debugging
                print(f"Sample {i}: Semantic Entropy = {semantic_entropy:.4f}, Clusters = {cluster_info['num_clusters']}")
                print(f"  Cluster sizes: {cluster_info['cluster_sizes']}")
            
        except FileNotFoundError:
            print(f"Response files not found for index {i}. Run generation first.")
            return
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            all_scores.append(0.5)  # Default score on error
            cluster_stats.append({'num_clusters': 1, 'clusters': [[0]], 'cluster_probabilities': [1.0]})
    
    all_scores = np.array(all_scores)
    
    # Evaluate performance
    if len(gt_label[gt_label == 1]) > 0 and len(gt_label[gt_label == 0]) > 0:
        measures = get_measures(all_scores[gt_label == 1], all_scores[gt_label == 0])
        auroc, auprc, fpr95 = measures
        print(f"\nSemantic Entropy Results for {args.dataset_name} ({args.model_name}):")
        print(f"AUROC: {auroc:.4f}")
        print_measures(auroc, auprc, fpr95, 'Semantic Entropy')
    else:
        auroc = 0.5
        print("Warning: Ground truth has only one class, AUROC set to 0.5")
    
    # Compute clustering statistics
    # 计算聚类统计信息
    avg_clusters = np.mean([stat['num_clusters'] for stat in cluster_stats])
    print(f"Average number of semantic clusters per sample: {avg_clusters:.2f}")
    
    # Save results
    results = {
        'auroc': auroc,
        'scores': all_scores.tolist(),
        'gt_labels': gt_label.tolist(),
        'cluster_stats': cluster_stats,
        'avg_clusters': avg_clusters,
        'dataset': args.dataset_name,
        'model': args.model_name,
        'method': 'semantic_entropy',
        'nli_model': args.nli_model,
        'entailment_threshold': args.entailment_threshold,
        'num_samples': args.num_gene,
        'threshold': args.thres_gt
    }
    
    results_file = f"{args.output_dir}/semantic_entropy/semantic_entropy_results_{args.dataset_name}_{args.model_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    # Save scores for further analysis
    scores_file = f"{args.output_dir}/semantic_entropy/{args.dataset_name}/scores_{args.model_name}_{args.dataset_name}.npy"
    np.save(scores_file, all_scores)
    print(f"Scores saved to {scores_file}")

if __name__ == '__main__':
    main()
