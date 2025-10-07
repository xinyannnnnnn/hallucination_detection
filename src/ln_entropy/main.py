#!/usr/bin/env python3
"""
LN Entropy (Log-Normal Entropy) implementation for low-resource languages
Based on "Uncertainty Estimation in Autoregressive Structured Prediction" paper
Link: https://openreview.net/pdf?id=jN5y-zb5Q7m

LN Entropy 是一种基于序列级别不确定性估计的幻觉检测方法。
主要思路：
1. 对同一问题生成多个响应
2. 使用语义嵌入对响应进行聚类
3. 基于聚类分布计算对数正态熵，作为不确定性度量
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

class LNEntropyCalculator:
    """
    LN Entropy Calculator: 基于语义聚类的序列级不确定性估计
    
    核心思想：
    1. 对多个生成的响应进行语义聚类
    2. 基于聚类分布计算对数正态熵
    3. 熵值越高表示响应间语义差异越大，不确定性越高
    """
    
    def __init__(self, 
                 embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
                 similarity_threshold=0.7,
                 min_cluster_size=1,
                 device='cuda'):
        """
        Initialize LN Entropy calculator
        
        Args:
            embedding_model: Sentence embedding model name (使用轻量级模型以适应低资源语言)
            similarity_threshold: Clustering similarity threshold (聚类相似度阈值，论文中通常使用0.7)
            min_cluster_size: Minimum cluster size (最小簇大小)
            device: Computing device
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.device = device
        
        # Initialize sentence embedding model
        # 使用sentence-transformers库，支持多语言语义嵌入
        try:
            # For sentence-transformers, use device string format
            device_str = str(device) if isinstance(device, torch.device) else device
            self.embedding_model = SentenceTransformer(embedding_model, device=device_str)
            print(f"✓ Loaded sentence embedding model: {embedding_model}")
            print(f"  Model dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
            self.use_sentence_transformers = True
        except Exception as e:
            print(f"Failed to load sentence embedding model {embedding_model}: {e}")
            print("Falling back to basic BERT model...")
            # Fallback to basic BERT model
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.bert_model = AutoModel.from_pretrained('bert-base-multilingual-cased').to(device)
            self.embedding_model = None
            self.use_sentence_transformers = False
    
    def get_sentence_embeddings(self, responses):
        """
        Get semantic embeddings for responses
        获取响应的语义嵌入表示
        
        Args:
            responses: List of response strings
            
        Returns:
            embeddings: numpy array of shape [num_responses, embedding_dim]
        """
        if self.use_sentence_transformers:
            # Use sentence-transformers (preferred method)
            # 使用sentence-transformers获取语义嵌入（推荐方法）
            embeddings = self.embedding_model.encode(
                responses, 
                convert_to_numpy=True,
                show_progress_bar=False,  # Disable progress bar for cleaner output
                batch_size=32  # Process in batches for better memory efficiency
            )
        else:
            # Fallback to BERT embeddings
            # 回退到BERT嵌入方法
            embeddings = []
            for response in responses:
                inputs = self.tokenizer(response, return_tensors='pt', 
                                      truncation=True, max_length=512, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # Use [CLS] token embedding as sentence representation
                    # 使用[CLS]标记的嵌入作为句子表示
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(embedding[0])
            
            embeddings = np.array(embeddings)
        
        return embeddings
    
    def cluster_responses(self, embeddings):
        """
        Cluster responses based on semantic similarity
        基于语义相似度对响应进行聚类
        
        Args:
            embeddings: numpy array of embeddings [num_responses, embedding_dim]
            
        Returns:
            cluster_labels: array of cluster labels for each response
            num_clusters: number of clusters found
        """
        if len(embeddings) <= 1:
            return np.array([0]), 1
        
        # Calculate cosine similarity matrix
        # 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(embeddings)
        
        # Convert similarity to distance for clustering
        # 将相似度转换为距离用于聚类
        distance_matrix = 1 - similarity_matrix
        
        # Use Agglomerative Clustering with distance threshold
        # 使用层次聚类，基于距离阈值
        distance_threshold = 1 - self.similarity_threshold  # Convert similarity to distance
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        num_clusters = len(np.unique(cluster_labels))
        
        # Log clustering information for debugging
        if num_clusters > 1:
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            cluster_info = dict(zip(unique_labels, counts))
            print(f"  Clustering result: {num_clusters} clusters with distribution {cluster_info}")
        else:
            print(f"  Clustering result: All responses in single cluster (high similarity)")
        
        return cluster_labels, num_clusters
    
    def calculate_ln_entropy(self, cluster_labels):
        """
        Calculate Log-Normal Entropy based on cluster distribution
        基于聚类分布计算对数正态熵
        
        Args:
            cluster_labels: array of cluster labels
            
        Returns:
            ln_entropy: Log-Normal entropy value (对数正态熵值)
        """
        # Get cluster distribution
        # 获取聚类分布
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        if len(unique_labels) <= 1:
            # If all responses are in the same cluster, entropy is 0
            # 如果所有响应都在同一簇中，熵为0
            return 0.0
        
        # Calculate probabilities
        # 计算概率分布
        total_responses = len(cluster_labels)
        probabilities = counts / total_responses
        
        # Calculate Shannon entropy (base formulation)
        # 计算香农熵（基础公式）
        shannon_entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        
        # Apply log-normal transformation as per paper methodology
        # 根据论文方法应用对数正态变换
        # LN-Entropy = log(1 + exp(Shannon_entropy))
        # 这确保了熵值始终为正，并且对小的不确定性更敏感
        ln_entropy = np.log(1 + np.exp(shannon_entropy))
        
        return float(ln_entropy)
    
    def compute_uncertainty(self, responses):
        """
        Compute LN Entropy uncertainty for a set of responses
        为一组响应计算LN熵不确定性
        
        Args:
            responses: List of response strings
            
        Returns:
            ln_entropy: LN entropy value
            cluster_info: Dictionary with clustering information
        """
        if len(responses) <= 1:
            return 0.0, {'num_clusters': 1, 'cluster_labels': [0]}
        
        # Step 1: Get semantic embeddings
        # 步骤1：获取语义嵌入
        embeddings = self.get_sentence_embeddings(responses)
        
        # Step 2: Cluster responses
        # 步骤2：聚类响应
        cluster_labels, num_clusters = self.cluster_responses(embeddings)
        
        # Step 3: Calculate LN entropy
        # 步骤3：计算LN熵
        ln_entropy = self.calculate_ln_entropy(cluster_labels)
        
        cluster_info = {
            'num_clusters': num_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'cluster_distribution': {str(k): int(v) for k, v in dict(zip(*np.unique(cluster_labels, return_counts=True))).items()}
        }
        
        return ln_entropy, cluster_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B', 
                       choices=['llama2_7B', 'llama3_2_1B', 'opt_6_7b', 'opt_1_3b'])
    parser.add_argument('--dataset_name', type=str, default='armenian',
                       choices=['armenian', 'basque', 'tigrinya'])
    parser.add_argument('--num_gene', type=int, default=10, help='Number of generations per question for sampling')
    parser.add_argument('--gene', type=int, default=0, help='Generate responses (1) or analyze existing (0)')
    parser.add_argument('--generate_gt', type=int, default=0, help='Generate ground truth labels')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, help='Clustering similarity threshold')
    parser.add_argument('--embedding_model', type=str, default='paraphrase-multilingual-MiniLM-L12-v2', help='Sentence embedding model (multilingual for low-resource languages)')
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
    os.makedirs(f'{args.output_dir}/ln_entropy', exist_ok=True)
    os.makedirs(f'{args.output_dir}/ln_entropy/{args.dataset_name}', exist_ok=True)
    os.makedirs(f'{args.output_dir}/ln_entropy/{args.dataset_name}/answers', exist_ok=True)
    
    print(f"LN Entropy Low-Resource Language Evaluation")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {'Generation' if args.gene else 'Ground Truth' if args.generate_gt else 'Analysis'}")
    print(f"Similarity Threshold: {args.similarity_threshold}")
    
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
    
    # Combine all data for processing (following LN Entropy methodology)
    all_data = train_data + val_data + test_data
    
    if args.gene:
        # Generation phase - generate multiple responses for LN Entropy analysis
        generate_responses(all_data, args, gpu_config)
    elif args.generate_gt:
        # Ground truth generation phase  
        generate_ground_truth(all_data, args, gpu_config)
    else:
        # Analysis phase - run LN Entropy analysis
        run_ln_entropy_analysis(all_data, args)

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
                # Extract answer from nested structure
                answer = ''
                if 'orig_answer' in row and isinstance(row['orig_answer'], dict):
                    answer = row['orig_answer'].get('text', '')
                elif 'answers' in row and isinstance(row['answers'], dict):
                    if 'text' in row['answers'] and len(row['answers']['text']) > 0:
                        answer = row['answers']['text'][0]
                
                data.append({
                    'question': row.get('question', ''),
                    'answer': answer,
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
    Generate model responses for LN Entropy analysis
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
    
    print(f"Generating responses with {args.model_name} for LN Entropy analysis...")
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
        
        # Generate multiple diverse responses for LN Entropy analysis
        # 为LN熵分析生成多个多样化的响应
        responses = []
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
                'use_cache': True
            }
            generated = generate_with_multi_gpu(model, tokenizer, prompt, generation_kwargs, primary_device)
                
            decoded = tokenizer.decode(
                generated[0, prompt.shape[-1]:],
                skip_special_tokens=True
            )
            
            # Clean up exactly like other methods
            if 'Answer the question concisely' in decoded:
                decoded = decoded.split('Answer the question concisely')[0]
                
            decoded = decoded.strip()
            if '\n' in decoded:
                decoded = decoded.split('\n')[0]
                
            responses.append(decoded)
            if gen_iter < 3:  # Print first few for debugging
                print(f"Response {gen_iter}: {decoded}")
            
        # Save generated responses
        np.save(
            f'{args.output_dir}/ln_entropy/{args.dataset_name}/answers/responses_{args.model_name}_{args.dataset_name}_index_{i}.npy',
            responses
        )
        
    print(f"Generated responses saved to {args.output_dir}/ln_entropy/{args.dataset_name}/answers/")

def generate_ground_truth(dataset, args, gpu_config):
    """
    Generate ground truth labels using BLEURT similarity scoring
    """
    print("Generating ground truth labels for LN Entropy...")
    
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
                f'{args.output_dir}/ln_entropy/{args.dataset_name}/answers/responses_{args.model_name}_{args.dataset_name}_index_{i}.npy'
            )
            
            # Get reference answer
            reference = dataset[i]['answer']
            
            # For LN Entropy, we evaluate the semantic consistency of the first response
            # 对于LN熵，我们评估第一个响应的语义一致性
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
    filename = f'{args.output_dir}/ln_entropy/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_bleurt_score.npy'
    np.save(filename, gts)
    print(f"Ground truth scores saved to {filename}")

def run_ln_entropy_analysis(dataset, args):
    """
    Run LN Entropy analysis
    """
    print("Running LN Entropy analysis...")
    
    # Initialize LN Entropy calculator
    # Use cuda:0 since it's always the first visible device after CUDA_VISIBLE_DEVICES is set
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ln_entropy_calc = LNEntropyCalculator(
        embedding_model=args.embedding_model,
        similarity_threshold=args.similarity_threshold,
        device=device
    )
    
    # Load ground truth labels
    print("Loading ground truth labels...")
    try:
        gts = np.load(f'{args.output_dir}/ln_entropy/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_bleurt_score.npy')
        print("Loaded BLEURT-based ground truth scores")
    except FileNotFoundError as e:
        print(f"Ground truth file not found: {e}")
        print("Please run ground truth generation first with --generate_gt 1")
        return
        
    # Create binary labels
    gt_label = np.asarray(gts > args.thres_gt, dtype=np.int32)
    
    print("Computing LN Entropy scores...")
    print(f"Processing {len(dataset)} samples with similarity threshold {args.similarity_threshold}")
    all_scores = []
    cluster_stats = []
    
    for i in tqdm(range(len(dataset)), desc="LN Entropy Analysis"):
        try:
            # Load generated responses
            responses = np.load(
                f'{args.output_dir}/ln_entropy/{args.dataset_name}/answers/responses_{args.model_name}_{args.dataset_name}_index_{i}.npy'
            )
            
            # Convert numpy strings to regular Python strings
            responses = [str(response) for response in responses if str(response).strip()]
            
            if len(responses) < 2:
                # If we have fewer than 2 responses, uncertainty is minimal
                # 如果响应少于2个，不确定性很小
                all_scores.append(0.0)
                cluster_stats.append({'num_clusters': 1, 'cluster_labels': [0]})
                continue
            
            # Compute LN Entropy
            # 计算LN熵
            ln_entropy, cluster_info = ln_entropy_calc.compute_uncertainty(responses)
            all_scores.append(ln_entropy)
            cluster_stats.append(cluster_info)
            
            if i < 5:  # Print first few for debugging
                print(f"Sample {i}: LN Entropy = {ln_entropy:.4f}, Clusters = {cluster_info['num_clusters']}")
            
        except FileNotFoundError:
            print(f"Response files not found for index {i}. Run generation first.")
            return
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            all_scores.append(0.5)  # Default score on error
            cluster_stats.append({'num_clusters': 1, 'cluster_labels': [0]})
    
    all_scores = np.array(all_scores)
    
    # Evaluate performance
    if len(gt_label[gt_label == 1]) > 0 and len(gt_label[gt_label == 0]) > 0:
        measures = get_measures(all_scores[gt_label == 1], all_scores[gt_label == 0])
        auroc, auprc, fpr95 = measures
        print(f"\nLN Entropy Results for {args.dataset_name} ({args.model_name}):")
        print(f"AUROC: {auroc:.4f}")
        print_measures(auroc, auprc, fpr95, 'LN Entropy')
    else:
        auroc = 0.5
        print("Warning: Ground truth has only one class, AUROC set to 0.5")
    
    # Compute clustering statistics
    # 计算聚类统计信息
    avg_clusters = np.mean([stat['num_clusters'] for stat in cluster_stats])
    print(f"Average number of clusters per sample: {avg_clusters:.2f}")
    
    # Save results
    results = {
        'auroc': auroc,
        'scores': all_scores.tolist(),
        'gt_labels': gt_label.tolist(),
        'cluster_stats': cluster_stats,
        'avg_clusters': avg_clusters,
        'dataset': args.dataset_name,
        'model': args.model_name,
        'method': 'ln_entropy',
        'similarity_threshold': args.similarity_threshold,
        'embedding_model': args.embedding_model,
        'num_samples': args.num_gene,
        'threshold': args.thres_gt
    }
    
    results_file = f"{args.output_dir}/ln_entropy/ln_entropy_results_{args.dataset_name}_{args.model_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    # Save scores for further analysis
    scores_file = f"{args.output_dir}/ln_entropy/{args.dataset_name}/scores_{args.model_name}_{args.dataset_name}.npy"
    np.save(scores_file, all_scores)
    print(f"Scores saved to {scores_file}")

if __name__ == '__main__':
    main()
