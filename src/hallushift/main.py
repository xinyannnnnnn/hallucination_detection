#!/usr/bin/env python3
"""
HalluShift: Measuring Distribution Shifts towards Hallucination Detection in LLMs
Based on the paper: https://arxiv.org/pdf/2504.09482

HalluShift 通过分析LLM内部状态的分布偏移和token概率来检测幻觉。
核心思想：幻觉源于LLM的内部动态变化，在生成过程中模型会从事实准确的状态"偏移"到产生幻觉的状态。

主要步骤：
1. 生成响应并提取内部状态（hidden states）和token概率
2. 计算分布偏移特征（层间状态差异）
3. 计算token概率特征（熵、不确定性等）
4. 使用分类器综合特征进行幻觉检测
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# Ensure repository paths are available
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
    setup_multi_gpu, load_model_multi_gpu, cleanup_gpu_memory,
    load_bleurt_multi_gpu, compute_bleurt_score_multi_gpu, MultiGPUConfig
)

# Model path candidates
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

class HalluShiftDetector:
    """
    HalluShift幻觉检测器
    
    核心功能：
    1. 提取生成过程中的内部状态和token概率
    2. 计算分布偏移特征
    3. 计算token概率特征
    4. 综合特征进行分类
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.scaler = StandardScaler()
        self.classifier = None
    
    def extract_distribution_shift_features(self, hidden_states):
        """
        提取分布偏移特征
        
        根据论文，分析内部状态在不同层之间的分布变化
        
        Args:
            hidden_states: Tuple of hidden states from all layers
                          Each element: [batch_size, seq_len, hidden_dim]
        
        Returns:
            features: Distribution shift features (numpy array)
        """
        features = []
        
        # Convert to numpy and get last token representation for each layer
        layer_representations = []
        for layer_hidden in hidden_states:
            # Get last token: [batch_size, hidden_dim]
            last_token = layer_hidden[:, -1, :].detach().cpu().numpy()
            layer_representations.append(last_token[0])  # [hidden_dim]
        
        layer_representations = np.array(layer_representations)  # [num_layers, hidden_dim]
        
        # Feature 1: Layer-wise L2 norm shifts (相邻层之间的L2距离)
        layer_shifts = []
        for i in range(1, len(layer_representations)):
            shift = np.linalg.norm(layer_representations[i] - layer_representations[i-1])
            layer_shifts.append(shift)
        
        # Feature 2: Statistical measures of shifts (偏移的统计量)
        if layer_shifts:
            features.extend([
                np.mean(layer_shifts),      # 平均偏移
                np.std(layer_shifts),       # 偏移标准差
                np.max(layer_shifts),       # 最大偏移
                np.min(layer_shifts),       # 最小偏移
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Feature 3: First half vs second half shift difference (前半层vs后半层偏移差异)
        mid_point = len(layer_shifts) // 2
        if mid_point > 0:
            first_half_shift = np.mean(layer_shifts[:mid_point])
            second_half_shift = np.mean(layer_shifts[mid_point:])
            features.append(second_half_shift - first_half_shift)
        else:
            features.append(0.0)
        
        # Feature 4: Gradient of shifts (偏移的梯度/趋势)
        if len(layer_shifts) > 1:
            shift_gradient = np.gradient(layer_shifts)
            features.extend([
                np.mean(shift_gradient),
                np.std(shift_gradient),
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Feature 5: Representation consistency (表示一致性)
        # 计算所有层表示的平均余弦相似度
        similarities = []
        for i in range(len(layer_representations) - 1):
            sim = np.dot(layer_representations[i], layer_representations[i+1]) / (
                np.linalg.norm(layer_representations[i]) * np.linalg.norm(layer_representations[i+1]) + 1e-8
            )
            similarities.append(sim)
        
        if similarities:
            features.extend([
                np.mean(similarities),
                np.std(similarities),
            ])
        else:
            features.extend([0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def extract_token_probability_features(self, scores, generated_ids, prompt_length):
        """
        提取token概率特征
        
        根据论文，分析生成token的概率分布和不确定性
        
        Args:
            scores: Tuple of score tensors from generation (one per generated token)
                   Each element: [batch_size, vocab_size]
            generated_ids: Generated token IDs [batch_size, seq_len]
            prompt_length: Length of the prompt (to skip prompt tokens)
        
        Returns:
            features: Token probability features (numpy array)
        """
        features = []
        
        if not scores or len(scores) == 0:
            # If no scores available, return zero features
            return np.zeros(10, dtype=np.float32)
        
        token_probs = []
        token_entropies = []
        
        # Calculate actual number of generated tokens to avoid index out of bounds
        num_generated_tokens = generated_ids.shape[1] - prompt_length
        
        for step_idx, step_scores in enumerate(scores):
            # Skip if we've gone beyond the actual generated tokens
            if step_idx >= num_generated_tokens:
                break
            
            # Get probability distribution: [batch_size, vocab_size]
            probs = torch.softmax(step_scores, dim=-1)
            
            # Get probability of the actual generated token
            token_idx = generated_ids[0, prompt_length + step_idx]
            token_prob = probs[0, token_idx].item()
            token_probs.append(token_prob)
            
            # Calculate entropy for this token
            probs_np = probs[0].cpu().numpy()
            entropy = -np.sum(probs_np * np.log(probs_np + 1e-10))
            token_entropies.append(entropy)
        
        token_probs = np.array(token_probs)
        token_entropies = np.array(token_entropies)
        
        # Safety check: if no valid tokens were processed, return zero features
        if len(token_probs) == 0:
            return np.zeros(10, dtype=np.float32)
        
        # Feature 1: Token probability statistics (token概率统计)
        features.extend([
            np.mean(token_probs),       # 平均token概率
            np.std(token_probs),        # token概率标准差
            np.min(token_probs),        # 最小token概率
            np.max(token_probs),        # 最大token概率
        ])
        
        # Feature 2: Token entropy statistics (token熵统计)
        features.extend([
            np.mean(token_entropies),   # 平均熵
            np.std(token_entropies),    # 熵标准差
            np.max(token_entropies),    # 最大熵
        ])
        
        # Feature 3: Low probability token ratio (低概率token比例)
        low_prob_ratio = np.sum(token_probs < 0.1) / len(token_probs)
        features.append(low_prob_ratio)
        
        # Feature 4: Probability trend (概率趋势)
        # 检查概率是否随着生成逐渐降低（可能表示模型不确定性增加）
        if len(token_probs) > 1:
            prob_gradient = np.gradient(token_probs)
            features.extend([
                np.mean(prob_gradient),  # 概率变化趋势
                np.std(prob_gradient),   # 概率变化波动
            ])
        else:
            features.extend([0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def extract_combined_features(self, hidden_states, scores, generated_ids, prompt_length):
        """
        综合提取所有特征
        
        Args:
            hidden_states: Hidden states from all layers
            scores: Token scores from generation
            generated_ids: Generated token IDs
            prompt_length: Prompt length
        
        Returns:
            features: Combined feature vector
        """
        # Extract distribution shift features
        dist_features = self.extract_distribution_shift_features(hidden_states)
        
        # Extract token probability features
        prob_features = self.extract_token_probability_features(scores, generated_ids, prompt_length)
        
        # Combine features
        combined_features = np.concatenate([dist_features, prob_features])
        
        # Clean invalid values (inf, -inf, nan)
        combined_features = self._clean_features(combined_features)
        
        return combined_features
    
    def _clean_features(self, features):
        """
        Clean features by replacing inf and nan values with safe alternatives
        
        Args:
            features: Feature vector (numpy array)
        
        Returns:
            cleaned_features: Feature vector with inf/nan replaced
        """
        features = np.array(features, dtype=np.float32)
        
        # Replace inf with large finite values
        features[np.isposinf(features)] = 1e6
        features[np.isneginf(features)] = -1e6
        
        # Replace nan with 0
        features[np.isnan(features)] = 0.0
        
        return features
    
    def train(self, X_train, y_train):
        """
        训练幻觉检测分类器
        
        Args:
            X_train: Training features [num_samples, num_features]
            y_train: Training labels [num_samples] (0=non-hallucination, 1=hallucination)
        """
        # Clean training features
        X_train = self._clean_features(X_train)
        
        # Validate features
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print(f"⚠️  Warning: Training features still contain invalid values after cleaning")
            print(f"   NaN count: {np.sum(np.isnan(X_train))}")
            print(f"   Inf count: {np.sum(np.isinf(X_train))}")
            # Replace with zeros as last resort
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Check if we have at least 2 classes
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            print(f"⚠️  Warning: Training data contains only one class ({unique_classes[0]})")
            print(f"   Cannot train binary classifier. Using unsupervised fallback.")
            # Fit scaler but don't train classifier
            self.scaler.fit(X_train)
            self.classifier = None  # Will use unsupervised scoring
            return
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train logistic regression classifier (following paper)
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        self.classifier.fit(X_train_scaled, y_train)
        
        print(f"✓ Trained HalluShift classifier with {X_train.shape[0]} samples")
        print(f"  Feature dimension: {X_train.shape[1]}")
    
    def predict(self, X_test):
        """
        预测幻觉分数
        
        Args:
            X_test: Test features [num_samples, num_features]
        
        Returns:
            scores: Hallucination scores [num_samples]
        """
        # Clean test features
        X_test = self._clean_features(X_test)
        
        # Standardize features
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.classifier is None:
            # Unsupervised fallback: use feature-based scoring
            # Combine multiple feature dimensions into a hallucination score
            # Higher mean distribution shift + lower token probability = higher hallucination score
            
            # Distribution shift features are in first 9 dimensions (higher = more shift)
            dist_shift_scores = np.mean(X_test_scaled[:, :9], axis=1)
            
            # Token probability features are in dimensions 9-19 (lower prob = more hallucination)
            # So we negate the mean token probability
            if X_test_scaled.shape[1] > 9:
                token_prob_scores = -np.mean(X_test_scaled[:, 9:13], axis=1)  # First 4 are prob stats
            else:
                token_prob_scores = 0
            
            # Combine (normalize to [0, 1] range)
            combined = dist_shift_scores + token_prob_scores
            scores = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
            
            return scores
        
        # Predict probability of hallucination (class 1)
        scores = self.classifier.predict_proba(X_test_scaled)[:, 1]
        
        return scores

def load_low_resource_dataset(dataset_name):
    """
    Load and format low-resource language datasets with proper train/validation/test splits
    """
    if dataset_name == 'armenian':
        armenian_dir = DATASETS_DIR / 'armenian'
        train_df = pd.read_csv(armenian_dir / 'SynDARin_Arm_train.csv')
        test_df = pd.read_csv(armenian_dir / 'SynDARin_Arm_test.csv')
        
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
            
        val_size = len(train_data) // 5
        val_data = train_data[-val_size:]
        train_data = train_data[:-val_size]
        
        for item in val_data:
            item['split'] = 'validation'
            item['id'] = item['id'].replace('train', 'val')
        
        dataset = {'train': train_data, 'validation': val_data, 'test': test_data}
                
    elif dataset_name == 'basque':
        import pyarrow.parquet as pq
        
        basque_dir = DATASETS_DIR / 'basque'
        train_table = pq.read_table(basque_dir / 'train-00000-of-00001.parquet')
        val_table = pq.read_table(basque_dir / 'validation-00000-of-00001.parquet')
        test_table = pq.read_table(basque_dir / 'test-00000-of-00001.parquet')
        
        def convert_basque_split(table, split_name):
            df = table.to_pandas()
            data = []
            for _, row in df.iterrows():
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
    """Safely tokenize text for OPT models"""
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
                    raise ValueError(f"Token ID {max_token_id} exceeds vocab size")
            
            return tokens
        except Exception as exc:
            last_error = exc
    
    # ASCII fallback
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
            return tokens
        except Exception:
            pass
    
    # Final fallback
    return tokenizer(
        "Answer the question: ",
        return_tensors='pt',
        padding=True,
        return_attention_mask=True
    )

def generate_responses_with_features(dataset, args, gpu_config):
    """
    生成响应并提取HalluShift特征
    
    这是HalluShift的核心步骤：同时记录生成过程中的内部状态和token概率
    """
    MODEL = resolve_model_path(args.model_name)
    
    # Load model
    if args.gpu_id is not None:
        model, tokenizer, primary_device = load_model_multi_gpu(MODEL, args.model_name, num_gpus=1)
    else:
        model, tokenizer, primary_device = load_model_multi_gpu(MODEL, args.model_name, gpu_config.num_gpus)
    
    model.eval()
    
    print(f"Generating responses with HalluShift feature extraction...")
    print(f"Model: {args.model_name}")
    language = dataset_to_language(args.dataset_name)
    
    for i in tqdm(range(len(dataset)), desc="Generating with HalluShift"):
        question = dataset[i]['question']
        
        # Format prompt
        if args.dataset_name == 'tigrinya' and 'context' in dataset[i]:
            context = dataset[i]['context']
            if len(context) > 1000:
                context = context[:1000] + "..."
            prompt_text = "Concisely answer the following question based on the information in the given passage: \n" + \
                " Passage: " + context + " \n Q: " + question + " \n A:"
        else:
            prompt_text = f"Answer the question concisely. Q: {question}" + " A:"
        
        prompt_text = preprocess_text_for_language(prompt_text, language)
        
        # Tokenize
        if 'opt' in args.model_name.lower():
            prompt_inputs = safe_tokenize_for_opt(tokenizer, prompt_text, language)
            prompt = prompt_inputs['input_ids'].to(primary_device)
            attention_mask = prompt_inputs['attention_mask'].to(primary_device)
        else:
            prompt_inputs = tokenizer(prompt_text, return_tensors='pt', padding=True, truncation=True)
            prompt = prompt_inputs['input_ids'].to(primary_device)
            attention_mask = prompt_inputs['attention_mask'].to(primary_device)
        
        prompt_length = prompt.shape[-1]
        
        # Generate response with hidden states and scores
        cleanup_gpu_memory()
        
        with torch.no_grad():
            generation_output = model.generate(
                prompt,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False,  # Greedy decoding for main response
                num_beams=5,
                num_return_sequences=1,
                output_hidden_states=True,  # Key: Extract hidden states
                output_scores=True,          # Key: Extract token scores
                return_dict_in_generate=True # Key: Return detailed output
            )
        
        # Extract generated text
        generated_ids = generation_output.sequences
        decoded = tokenizer.decode(
            generated_ids[0, prompt_length:],
            skip_special_tokens=True
        )
        
        # Clean up text
        if 'Answer the question concisely' in decoded:
            decoded = decoded.split('Answer the question concisely')[0]
        decoded = decoded.strip()
        if '\n' in decoded:
            decoded = decoded.split('\n')[0]
        
        # Extract hidden states and scores
        # hidden_states: tuple of length (num_generated_tokens)
        #   each element: tuple of (num_layers) tensors [batch_size, seq_len, hidden_dim]
        # scores: tuple of length (num_generated_tokens)
        #   each element: tensor [batch_size, vocab_size]
        
        hidden_states = generation_output.hidden_states
        scores = generation_output.scores
        
        # For HalluShift, we use the hidden states from the last generated token
        # This represents the final internal state after generating the complete response
        if hidden_states and len(hidden_states) > 0:
            # Get hidden states from the last generation step
            last_step_hidden = hidden_states[-1]  # Tuple of layer states
            
            # Convert to format expected by feature extractor
            # Stack all layers: [num_layers, batch_size, seq_len, hidden_dim]
            all_layers = torch.stack(last_step_hidden, dim=0)
            
            # We only need the last token from each layer
            # Result: [num_layers, batch_size, hidden_dim]
            last_token_states = all_layers[:, :, -1, :]
            
            # Transpose to [batch_size, num_layers, hidden_dim] and convert to tuple of tensors
            # to match expected format
            layer_tuple = tuple(last_token_states[layer_idx:layer_idx+1, :, :].permute(1, 0, 2) 
                              for layer_idx in range(last_token_states.shape[0]))
        else:
            layer_tuple = None
        
        # Save response and raw data for feature extraction
        save_data = {
            'question': question,
            'response': decoded,
            'hidden_states_available': layer_tuple is not None,
            'scores_available': scores is not None and len(scores) > 0,
        }
        
        # Save response text
        np.save(
            f'{args.output_dir}/hallushift/{args.dataset_name}/answers/response_{args.model_name}_{args.dataset_name}_index_{i}.npy',
            [decoded]
        )
        
        # Extract and save features
        if layer_tuple is not None and scores is not None:
            detector = HalluShiftDetector(device=primary_device)
            features = detector.extract_combined_features(
                layer_tuple, scores, generated_ids, prompt_length
            )
            
            # Save features
            np.save(
                f'{args.output_dir}/hallushift/{args.dataset_name}/features/features_{args.model_name}_{args.dataset_name}_index_{i}.npy',
                features
            )
            
            if i < 3:  # Debug: print first few
                print(f"Sample {i}: Response: {decoded[:50]}...")
                print(f"  Feature shape: {features.shape}")
                print(f"  Feature sample: {features[:5]}")
    
    print(f"✓ Generated responses and extracted features saved")

def generate_ground_truth(dataset, args, gpu_config):
    """生成ground truth标签"""
    print("Generating ground truth labels...")
    
    # Load BLEURT model
    model, tokenizer = load_bleurt_multi_gpu(BLEURT_DIR, gpu_config.num_gpus)
    
    if model is None:
        print("Failed to load BLEURT model.")
        return
    
    predictions = []
    references = []
    
    for i in tqdm(range(len(dataset)), desc="Loading responses"):
        try:
            response = np.load(
                f'{args.output_dir}/hallushift/{args.dataset_name}/answers/response_{args.model_name}_{args.dataset_name}_index_{i}.npy'
            )[0]
            
            reference = dataset[i]['answer']
            
            predictions.append(str(response))
            references.append(str(reference))
        except FileNotFoundError:
            print(f"Response file not found for index {i}. Run generation first.")
            return
    
    # Compute BLEURT scores
    print("Computing BLEURT scores...")
    gts = compute_bleurt_score_multi_gpu(model, tokenizer, predictions, references)
    
    # Save ground truth
    filename = f'{args.output_dir}/hallushift/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_bleurt.npy'
    np.save(filename, gts)
    print(f"✓ Ground truth saved to {filename}")

def run_hallushift_analysis(dataset, args):
    """
    运行HalluShift分析
    
    使用train set训练分类器，在validation/test set上评估
    """
    print("Running HalluShift analysis...")
    
    # Load ground truth
    try:
        gts = np.load(
            f'{args.output_dir}/hallushift/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_bleurt.npy'
        )
    except FileNotFoundError:
        print("Ground truth file not found. Run --generate_gt 1 first.")
        return
    
    # Create binary labels
    gt_labels = np.asarray(gts > args.thres_gt, dtype=np.int32)
    
    print(f"Ground truth statistics:")
    print(f"  Positive (non-hallucination): {np.sum(gt_labels == 1)}")
    print(f"  Negative (hallucination): {np.sum(gt_labels == 0)}")
    
    # Provide helpful information about BLEURT scores
    print(f"BLEURT score statistics:")
    print(f"  Min: {np.min(gts):.4f}, Max: {np.max(gts):.4f}, Mean: {np.mean(gts):.4f}")
    print(f"  Threshold: {args.thres_gt}")
    
    # Warn if all samples are in one class
    if np.sum(gt_labels == 1) == 0:
        print(f"⚠️  Warning: All samples labeled as hallucinations (BLEURT <= {args.thres_gt})")
        print(f"   Consider using a lower threshold, e.g., --thres_gt {np.mean(gts):.2f}")
    elif np.sum(gt_labels == 0) == 0:
        print(f"⚠️  Warning: All samples labeled as non-hallucinations (BLEURT > {args.thres_gt})")
        print(f"   Consider using a higher threshold, e.g., --thres_gt {np.mean(gts):.2f}")
    
    # Load features
    all_features = []
    for i in range(len(dataset)):
        try:
            features = np.load(
                f'{args.output_dir}/hallushift/{args.dataset_name}/features/features_{args.model_name}_{args.dataset_name}_index_{i}.npy'
            )
            all_features.append(features)
        except FileNotFoundError:
            print(f"Features not found for index {i}. Run generation first.")
            return
    
    all_features = np.array(all_features)
    print(f"Loaded features: {all_features.shape}")
    
    # Split data according to dataset splits
    train_indices = [i for i, item in enumerate(dataset) if item['split'] == 'train']
    val_indices = [i for i, item in enumerate(dataset) if item['split'] == 'validation']
    test_indices = [i for i, item in enumerate(dataset) if item['split'] == 'test']
    
    print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Train on train set
    X_train = all_features[train_indices]
    y_train = gt_labels[train_indices]
    
    detector = HalluShiftDetector()
    detector.train(X_train, y_train)
    
    # Initialize metrics
    auroc_val = None
    auroc_test = None
    
    # Evaluate on validation set
    if len(val_indices) > 0:
        X_val = all_features[val_indices]
        y_val = gt_labels[val_indices]
        
        val_scores = detector.predict(X_val)
        
        # Compute metrics
        val_pos = val_scores[y_val == 0]  # Hallucinations (higher score = more likely hallucination)
        val_neg = val_scores[y_val == 1]  # Non-hallucinations
        
        if len(val_pos) > 0 and len(val_neg) > 0:
            auroc_val, auprc_val, fpr95_val = get_measures(val_pos, val_neg)
            print(f"\nValidation Set Results:")
            print(f"AUROC: {auroc_val:.4f}")
            print_measures(auroc_val, auprc_val, fpr95_val, 'HalluShift')
        else:
            auroc_val = 0.5
            print("Warning: Validation set has only one class")
    
    # Evaluate on test set
    test_scores = np.array([])
    y_test = np.array([])
    
    if len(test_indices) > 0:
        X_test = all_features[test_indices]
        y_test = gt_labels[test_indices]
        
        test_scores = detector.predict(X_test)
        
        # Compute metrics
        test_pos = test_scores[y_test == 0]  # Hallucinations
        test_neg = test_scores[y_test == 1]  # Non-hallucinations
        
        if len(test_pos) > 0 and len(test_neg) > 0:
            auroc_test, auprc_test, fpr95_test = get_measures(test_pos, test_neg)
            print(f"\nTest Set Results:")
            print(f"AUROC: {auroc_test:.4f}")
            print_measures(auroc_test, auprc_test, fpr95_test, 'HalluShift')
        else:
            auroc_test = 0.5
            print("Warning: Test set has only one class")
    else:
        auroc_test = 0.0
    
    # Save results
    results = {
        'dataset': args.dataset_name,
        'model': args.model_name,
        'method': 'hallushift',
        'auroc_test': float(auroc_test) if len(test_indices) > 0 else None,
        'auroc_val': float(auroc_val) if len(val_indices) > 0 else None,
        'test_scores': test_scores.tolist() if len(test_indices) > 0 else [],
        'test_labels': y_test.tolist() if len(test_indices) > 0 else [],
        'num_train': len(train_indices),
        'num_val': len(val_indices),
        'num_test': len(test_indices),
        'threshold': args.thres_gt,
    }
    
    results_file = f"{args.output_dir}/hallushift/hallushift_results_{args.dataset_name}_{args.model_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {results_file}")

def main():
    parser = argparse.ArgumentParser(description='HalluShift: Distribution Shift-based Hallucination Detection')
    parser.add_argument('--model_name', type=str, default='llama2_7B',
                       choices=['llama2_7B', 'llama3_2_1B', 'opt_6_7b', 'opt_1_3b'])
    parser.add_argument('--dataset_name', type=str, default='armenian',
                       choices=['armenian', 'basque', 'tigrinya'])
    parser.add_argument('--gene', type=int, default=0, help='Generate responses (1) or analyze (0)')
    parser.add_argument('--generate_gt', type=int, default=0, help='Generate ground truth labels (1) or not (0)')
    parser.add_argument('--thres_gt', type=float, default=0.5, help='Ground truth threshold for BLEURT')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--disable_multi_gpu', action='store_true', help='Disable multi-GPU')
    parser.add_argument('--gpu_id', type=int, default=None, help='Specific GPU ID to use')
    
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(41)
    
    # Setup GPU
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        num_gpus = 1
        print(f"Using GPU {args.gpu_id}")
    elif args.disable_multi_gpu:
        num_gpus = 1
        print("Using single GPU")
    else:
        num_gpus, _ = setup_multi_gpu()
        print(f"Using {num_gpus} GPUs")
    
    gpu_config = MultiGPUConfig(
        num_gpus=1 if args.gpu_id is not None else num_gpus,
        batch_size=args.batch_size,
        model_name=args.model_name
    )
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/hallushift', exist_ok=True)
    os.makedirs(f'{args.output_dir}/hallushift/{args.dataset_name}', exist_ok=True)
    os.makedirs(f'{args.output_dir}/hallushift/{args.dataset_name}/answers', exist_ok=True)
    os.makedirs(f'{args.output_dir}/hallushift/{args.dataset_name}/features', exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"HalluShift: Distribution Shift-based Hallucination Detection")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {'Generation' if args.gene else 'Ground Truth' if args.generate_gt else 'Analysis'}")
    print(f"{'='*60}\n")
    
    # Load dataset
    dataset_splits = load_low_resource_dataset(args.dataset_name)
    train_data = dataset_splits['train']
    val_data = dataset_splits['validation']
    test_data = dataset_splits['test']
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_data)}")
    print(f"  Validation: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    
    # Combine all splits for processing
    all_data = train_data + val_data + test_data
    
    if args.gene:
        generate_responses_with_features(all_data, args, gpu_config)
    elif args.generate_gt:
        generate_ground_truth(all_data, args, gpu_config)
    else:
        run_hallushift_analysis(all_data, args)

if __name__ == '__main__':
    main()

