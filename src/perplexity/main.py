#!/usr/bin/env python3
"""
Perplexity-based hallucination detection using RMD (Relative Mahalanobis Distance)
Based on "Out-of-Distribution Detection and Selective Generation for Conditional Language Models"
Paper: https://openreview.net/pdf?id=kJUS5nD0vPB

This implementation uses RMD on model embeddings rather than pure perplexity scores,
as the paper shows that perplexity alone is not effective for OOD detection.
The RMD method calculates the relative distance between in-domain and out-of-domain
distributions in the embedding space to detect hallucinations.

Follows the exact experimental setup from HaloScope paper Section 4.
Uses BLEURT for ground truth labeling and supports language-specific preprocessing.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from typing import Dict, Any, Tuple
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

# Import RMD utilities
from rmd_utils import (RMDCalculator, generate_and_evaluate_responses, prepare_embeddings_with_labels)
from evaluation_utils import comprehensive_evaluation

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

def calculate_perplexity(
    model,
    tokenizer,
    text: str,
    device: torch.device
) -> float:
    """
    Calculate perplexity for a given text using the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text
        device: torch device
        
    Returns:
        Perplexity score
    """
    # Tokenize input
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    with torch.no_grad():
        # Get model outputs
        outputs = model(**encodings, labels=encodings['input_ids'])
        
        # Calculate perplexity
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return perplexity

def calculate_token_level_perplexity(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device: torch.device
) -> Dict[str, float]:
    """
    Calculate perplexity at token level, separating prompt and response.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        response: Generated response
        device: torch device
        
    Returns:
        Dictionary with various perplexity metrics
    """
    # Combine prompt and response
    full_text = prompt + response
    
    # Tokenize
    prompt_encodings = tokenizer(prompt, return_tensors='pt', truncation=True)
    full_encodings = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=2048)
    
    prompt_length = prompt_encodings['input_ids'].shape[1]
    
    # Move to device
    full_encodings = {k: v.to(device) for k, v in full_encodings.items()}
    
    with torch.no_grad():
        # Get model outputs
        outputs = model(**full_encodings)
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = full_encodings['input_ids'][..., 1:].contiguous()
        
        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Calculate perplexity for full sequence
        full_loss = token_losses.mean()
        full_perplexity = torch.exp(full_loss).item()
        
        # Calculate perplexity for response only
        if prompt_length < len(token_losses):
            response_losses = token_losses[prompt_length:]
            response_loss = response_losses.mean()
            response_perplexity = torch.exp(response_loss).item()
            
            # Calculate max and mean token perplexity for response
            max_token_perplexity = torch.exp(response_losses.max()).item()
            token_perplexities = torch.exp(response_losses)
            mean_token_perplexity = token_perplexities.mean().item()
        else:
            response_perplexity = full_perplexity
            max_token_perplexity = full_perplexity
            mean_token_perplexity = full_perplexity
    
    return {
        'full_perplexity': full_perplexity,
        'response_perplexity': response_perplexity,
        'max_token_perplexity': max_token_perplexity,
        'mean_token_perplexity': mean_token_perplexity
    }

def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.7
) -> Tuple[str, Dict[str, float]]:
    """
    Generate response and calculate perplexity metrics.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        device: torch device
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Tuple of (generated_text, perplexity_metrics)
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1500)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode - use token-level slicing like hallushift
    response = tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Calculate perplexity metrics
    perplexity_metrics = calculate_token_level_perplexity(
        model, tokenizer, prompt, response, device
    )
    
    return response, perplexity_metrics

def evaluate_hallucination(perplexity_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Evaluate hallucination based on perplexity thresholds.
    
    Args:
        perplexity_metrics: Dictionary of perplexity scores
        
    Returns:
        Hallucination evaluation results
    """
    # Define thresholds (can be tuned based on validation data)
    thresholds = {
        'response_perplexity': 10.0,
        'max_token_perplexity': 50.0,
        'mean_token_perplexity': 15.0
    }
    
    # Calculate hallucination scores
    hallucination_scores = {}
    for metric_name, threshold in thresholds.items():
        if metric_name in perplexity_metrics:
            score = perplexity_metrics[metric_name] / threshold
            hallucination_scores[f'{metric_name}_score'] = min(score, 10.0)  # Cap at 10
    
    # Combined hallucination score (weighted average)
    weights = {'response_perplexity_score': 0.5, 
               'max_token_perplexity_score': 0.3,
               'mean_token_perplexity_score': 0.2}
    
    combined_score = sum(
        hallucination_scores.get(k, 0) * v 
        for k, v in weights.items()
    )
    
    # Binary classification
    is_hallucination = combined_score > 1.0
    
    return {
        'hallucination_scores': hallucination_scores,
        'combined_score': combined_score,
        'is_hallucination': is_hallucination,
        'confidence': min(combined_score, 1.0) if is_hallucination else max(0, 1.0 - combined_score)
    }

def process_dataset_rmd(
    dataset,
    model,
    tokenizer,
    device,
    dataset_name: str,
    max_samples: int = None,
    layer_idx: int = -1,
    train_ratio: float = 0.5,
    bleurt_threshold: float = 0.5,
    output_dir: str = "results/perplexity"
) -> pd.DataFrame:
    """
    Process entire dataset using RMD method with BLEURT-based ground truth labels.
    Uses the same strategy as hallushift.
    
    Args:
        dataset: The dataset to process (list of dictionaries)
        model: The language model
        tokenizer: The tokenizer
        device: torch device
        dataset_name: Name of the dataset
        max_samples: Maximum number of samples to process
        layer_idx: Layer to extract embeddings from
        train_ratio: Ratio of data to use for training RMD
        bleurt_threshold: BLEURT threshold for hallucination detection
        output_dir: Output directory for temporary files
        
    Returns:
        DataFrame with results
    """
    # Limit samples if specified
    if max_samples:
        dataset = dataset[:min(max_samples, len(dataset))]
    
    # Split dataset for training and testing
    total_samples = len(dataset)
    train_size = int(total_samples * train_ratio)
    
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    print(f"Dataset split: {train_size} train, {len(test_dataset)} test")
    
    # Generate responses and evaluate with BLEURT for training set
    print("Generating training responses and evaluating with BLEURT...")
    train_response_data, train_bleurt_df = generate_and_evaluate_responses(
        train_dataset, model, tokenizer, device, dataset_name, bleurt_threshold, output_dir
    )
    
    # Prepare embeddings with ground truth labels for training
    print("Preparing training embeddings with labels...")
    language = dataset_to_language(dataset_name)
    all_embeddings, labels, bleurt_scores = prepare_embeddings_with_labels(
        train_response_data, train_bleurt_df, model, tokenizer, device, layer_idx, language
    )
    
    # Split into hallucinations (positive) and non-hallucinations (negative)
    hallucination_mask = labels == 1
    non_hallucination_mask = labels == 0
    
    hallucination_embeddings = all_embeddings[hallucination_mask]
    non_hallucination_embeddings = all_embeddings[non_hallucination_mask]
    
    print(f"Training data: {len(hallucination_embeddings)} hallucinations, {len(non_hallucination_embeddings)} non-hallucinations")
    
    if len(hallucination_embeddings) == 0 or len(non_hallucination_embeddings) == 0:
        print("Warning: Need both hallucination and non-hallucination examples for RMD training")
        print(f"BLEURT score statistics: min={bleurt_scores.min():.4f}, max={bleurt_scores.max():.4f}, mean={bleurt_scores.mean():.4f}")
        print(f"Using percentile-based split instead of threshold {bleurt_threshold}")
        
        # Fallback: Use percentile-based splitting
        # Sort by BLEURT scores and split at median
        sorted_indices = np.argsort(bleurt_scores)
        median_idx = len(sorted_indices) // 2
        
        # Bottom half = hallucinations (low BLEURT scores)
        # Top half = non-hallucinations (high BLEURT scores)
        hallucination_indices = sorted_indices[:median_idx]
        non_hallucination_indices = sorted_indices[median_idx:]
        
        hallucination_embeddings = all_embeddings[hallucination_indices]
        non_hallucination_embeddings = all_embeddings[non_hallucination_indices]
        
        # Update labels for consistency
        labels = np.zeros(len(bleurt_scores), dtype=int)
        labels[hallucination_indices] = 1
        
        print(f"After percentile split: {len(hallucination_embeddings)} hallucinations, {len(non_hallucination_embeddings)} non-hallucinations")
    
    # Fit RMD distributions (in-domain = non-hallucination, out-domain = hallucination)
    print("Fitting RMD distributions...")
    rmd_calc = RMDCalculator(device)
    rmd_calc.fit_gaussian_distributions(non_hallucination_embeddings, hallucination_embeddings)
    
    # Generate responses and evaluate test set
    print("Processing test samples...")
    test_response_data, test_bleurt_df = generate_and_evaluate_responses(
        test_dataset, model, tokenizer, device, dataset_name, bleurt_threshold, output_dir
    )
    
    # Create ground truth mapping
    id_to_bleurt = dict(zip(test_bleurt_df['id'].astype(str), test_bleurt_df['bleurt_score'].astype(float)))
    id_to_ground_truth = dict(zip(test_bleurt_df['id'].astype(str), test_bleurt_df['hallucination']))
    
    # Step 1: Extract embeddings for all test samples
    print("Extracting embeddings for all test samples...")
    all_test_embeddings = []
    test_prompts = [item['prompt_with_answer'] for item in test_response_data]
    
    for idx, prompt in enumerate(tqdm(test_prompts, desc="Extracting embeddings")):
        if idx % 50 == 0:
            print(f"Processing test sample {idx+1}/{len(test_prompts)}")
        
        embedding = rmd_calc.extract_embeddings(
            model, tokenizer, [prompt], layer_idx, language
        )
        all_test_embeddings.append(embedding[0])
    
    all_test_embeddings = np.array(all_test_embeddings)
    
    # Step 2: Calculate RMD scores for all samples at once
    print("Calculating RMD scores for all test samples...")
    all_rmd_scores = rmd_calc.calculate_rmd_scores(all_test_embeddings)
    
    # Step 3: Apply dynamic threshold on all scores to ensure 2 classes
    print("Applying dynamic threshold for binary classification...")
    rmd_scores = all_rmd_scores['rmd_scores']
    threshold = np.median(rmd_scores)  # Use median to ensure ~50/50 split
    print(f"Using dynamic threshold (median): {threshold:.4f}")
    
    predictions = rmd_scores > threshold
    rmd_std = np.std(rmd_scores)
    if rmd_std > 0:
        confidence = np.abs(rmd_scores - threshold) / rmd_std
        confidence = np.clip(confidence, 0, 1)
    else:
        confidence = np.ones_like(rmd_scores) * 0.5
    
    # Step 4: Create results with predictions
    results = []
    for idx, item in enumerate(test_response_data):
        sample_id = str(item['id'])
        
        # Get ground truth from BLEURT
        bleurt_score = id_to_bleurt.get(sample_id, 0.5)
        ground_truth_hallucination = id_to_ground_truth.get(sample_id, 0)
        
        # Store results
        result = {
            'id': sample_id,
            'question': item['question'],
            'context': item['context'], 
            'generated_answer': item['generated_answer'],
            'reference_answer': item['reference_answer'],
            'bleurt_score': bleurt_score,
            'ground_truth_hallucination': ground_truth_hallucination,
            'rmd_score': float(rmd_scores[idx]),
            'in_domain_distance': float(all_rmd_scores['in_domain_distances'][idx]),
            'out_domain_distance': float(all_rmd_scores['out_domain_distances'][idx]),
            'rmd_prediction': bool(predictions[idx]),
            'rmd_confidence': float(confidence[idx])
        }
        results.append(result)
    
    print(f"Processed {len(results)} test samples")
    print(f"RMD prediction distribution:")
    print(f"  Hallucinations (True): {predictions.sum()}")
    print(f"  Non-hallucinations (False): {(~predictions).sum()}")
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Perplexity-based (RMD) hallucination detection')
    parser.add_argument('--model_name', type=str, default='opt_6_7b', 
                        choices=['llama2_7B', 'llama3_2_1B', 'opt_6_7b', 'opt_1_3b'],
                        help='Model to use')
    parser.add_argument('--dataset_name', type=str, default='tigrinya',
                        choices=['tigrinya', 'armenian', 'basque'],
                        help='Dataset to evaluate')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None for all samples)')
    parser.add_argument('--layer_idx', type=int, default=-1,
                        help='Layer index to extract embeddings from (-1 for last layer)')
    parser.add_argument('--train_ratio', type=float, default=0.5,
                        help='Ratio of data to use for training RMD')
    parser.add_argument('--bleurt_threshold', type=float, default=0.5,
                        help='BLEURT threshold for ground truth labeling')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='Specific GPU ID to use (0, 1, 2, 3). If not set, uses device_map="auto"')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use (default: 4)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for generation (auto-determined if not specified)')
    parser.add_argument('--disable_multi_gpu', action='store_true', help='Disable multi-GPU and use single GPU')
    
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
    
    # Check model-dataset compatibility
    if 'opt' in args.model_name.lower() and args.dataset_name == 'tigrinya':
        print("WARNING: OPT models have limited compatibility with Tigrinya (Ge'ez script)")
        print("This may result in tokenization errors or poor performance.")
        print("Consider using LLaMA models for Tigrinya, or Armenian/Basque datasets for OPT.")
        print("Continuing with fallback text processing...")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model_path = resolve_model_path(args.model_name)
    
    # Load model with GPU support
    if args.gpu_id is not None:
        # For specific GPU, use single GPU mode
        model, tokenizer, primary_device = load_model_multi_gpu(
            model_path, args.model_name, num_gpus=1
        )
    else:
        # Use multi-GPU or single GPU based on configuration
        model, tokenizer, primary_device = load_model_multi_gpu(
            model_path, args.model_name, gpu_config.num_gpus
        )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    device = primary_device
    
    # Create output directory
    perplexity_dir = Path(args.output_dir) / "perplexity" / args.dataset_name
    perplexity_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset_splits = load_low_resource_dataset(args.dataset_name)
    train_data = dataset_splits['train']
    val_data = dataset_splits['validation'] 
    test_data = dataset_splits['test']
    
    print(f"Loaded {args.dataset_name} dataset:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples") 
    print(f"  Test: {len(test_data)} samples")
    
    # Combine all data for processing (following other methods' methodology)
    all_data = train_data + val_data + test_data
    dataset = all_data
    
    # Process dataset using RMD with BLEURT evaluation
    print("Processing dataset with RMD method using BLEURT labels...")
    results_df = process_dataset_rmd(
        dataset, model, tokenizer, device, 
        args.dataset_name, args.max_samples, args.layer_idx, args.train_ratio,
        args.bleurt_threshold, str(perplexity_dir)
    )
    
    # Save detailed results
    output_path = perplexity_dir / f"rmd_{args.model_name}_{args.dataset_name}_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # Perform comprehensive evaluation with AUC-ROC and other metrics
    evaluation_metrics = comprehensive_evaluation(
        results_df, str(perplexity_dir), args.model_name, args.dataset_name, "RMD"
    )
    
    # Generate and save summary statistics
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("RMD HALLUCINATION DETECTION SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Model: {args.model_name}")
    summary_lines.append(f"Dataset: {args.dataset_name}")
    summary_lines.append(f"Total samples processed: {len(results_df)}")
    summary_lines.append(f"Train/Test split: {args.train_ratio:.1%}/{1-args.train_ratio:.1%}")
    summary_lines.append(f"Layer used: {args.layer_idx}")
    summary_lines.append(f"BLEURT threshold: {args.bleurt_threshold}")
    summary_lines.append("")
    
    summary_lines.append("RMD Score Statistics:")
    summary_lines.append(f"  Mean RMD score: {results_df['rmd_score'].mean():.4f}")
    summary_lines.append(f"  Std RMD score: {results_df['rmd_score'].std():.4f}")
    summary_lines.append(f"  Min RMD score: {results_df['rmd_score'].min():.4f}")
    summary_lines.append(f"  Max RMD score: {results_df['rmd_score'].max():.4f}")
    
    # Add diagnostic info if RMD scores are all zeros
    if results_df['rmd_score'].std() < 1e-6:
        summary_lines.append("")
        summary_lines.append("  ⚠ WARNING: All RMD scores are near zero!")
        summary_lines.append("  This indicates the in-domain and out-domain distributions are identical.")
        summary_lines.append("  Possible causes:")
        summary_lines.append("    - Training set has only hallucinations (or only non-hallucinations)")
        summary_lines.append("    - BLEURT threshold is not appropriate for this dataset")
        summary_lines.append("    - Model embeddings are not discriminative enough")
        summary_lines.append("  Try adjusting --bleurt_threshold or using more diverse training samples.")
    
    summary_lines.append("")
    
    summary_lines.append("Distance Metrics:")
    summary_lines.append(f"  Mean in-domain distance: {results_df['in_domain_distance'].mean():.4f}")
    summary_lines.append(f"  Mean out-domain distance: {results_df['out_domain_distance'].mean():.4f}")
    summary_lines.append(f"  Mean RMD confidence: {results_df['rmd_confidence'].mean():.4f}")
    summary_lines.append("")
    
    summary_lines.append("BLEURT Evaluation:")
    summary_lines.append(f"  Mean BLEURT score: {results_df['bleurt_score'].mean():.4f}")
    summary_lines.append(f"  Std BLEURT score: {results_df['bleurt_score'].std():.4f}")
    summary_lines.append(f"  Min BLEURT score: {results_df['bleurt_score'].min():.4f}")
    summary_lines.append(f"  Max BLEURT score: {results_df['bleurt_score'].max():.4f}")
    summary_lines.append(f"  Ground truth hallucinations: {results_df['ground_truth_hallucination'].sum()}")
    summary_lines.append(f"  Ground truth hallucination rate: {results_df['ground_truth_hallucination'].mean():.2%}")
    summary_lines.append("")
    
    summary_lines.append("RMD Predictions:")
    summary_lines.append(f"  RMD predicted hallucinations: {results_df['rmd_prediction'].sum()}")
    summary_lines.append(f"  RMD predicted hallucination rate: {results_df['rmd_prediction'].mean():.2%}")
    summary_lines.append("")
    
    # Calculate performance metrics
    performance_stats = {}
    if len(results_df) > 0:
        # Check for single-class scenario
        n_classes_true = len(set(results_df['ground_truth_hallucination']))
        n_classes_pred = len(set(results_df['rmd_prediction']))
        
        if n_classes_true < 2:
            print(f"\nWarning: Only {n_classes_true} class in ground truth labels.")
            print(f"Ground truth distribution: {results_df['ground_truth_hallucination'].value_counts().to_dict()}")
            print("This indicates all samples are being labeled the same way by BLEURT.")
            print("Metrics requiring both classes (AUC-ROC, AUC-PR, Precision, Recall, F1) cannot be calculated.")
        
        if n_classes_pred < 2:
            print(f"\nWarning: Only {n_classes_pred} class in RMD predictions.")
            print(f"Prediction distribution: {results_df['rmd_prediction'].value_counts().to_dict()}")
            print("This may indicate an issue with the RMD calculation or thresholding.")
        
        # Calculate accuracy of RMD predictions vs BLEURT ground truth
        accuracy = (results_df['rmd_prediction'] == results_df['ground_truth_hallucination']).mean()
        performance_stats['accuracy'] = float(accuracy)
        
        # Calculate confusion matrix with proper handling
        try:
            true_pos = ((results_df['rmd_prediction'] == 1) & (results_df['ground_truth_hallucination'] == 1)).sum()
            true_neg = ((results_df['rmd_prediction'] == 0) & (results_df['ground_truth_hallucination'] == 0)).sum()
            false_pos = ((results_df['rmd_prediction'] == 1) & (results_df['ground_truth_hallucination'] == 0)).sum()
            false_neg = ((results_df['rmd_prediction'] == 0) & (results_df['ground_truth_hallucination'] == 1)).sum()
            
            performance_stats.update({
                'true_positives': int(true_pos),
                'true_negatives': int(true_neg),
                'false_positives': int(false_pos),
                'false_negatives': int(false_neg)
            })
        except Exception as e:
            print(f"Warning: Could not calculate confusion matrix: {e}")
            performance_stats.update({
                'true_positives': 0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0
            })
        
        # Calculate precision, recall, F1 with proper handling
        precision = recall = f1 = 0.0
        
        try:
            if true_pos + false_pos > 0:
                precision = true_pos / (true_pos + false_pos)
                performance_stats['precision'] = float(precision)
            else:
                performance_stats['precision'] = 0.0
        except:
            performance_stats['precision'] = 0.0
        
        try:
            if true_pos + false_neg > 0:
                recall = true_pos / (true_pos + false_neg)
                performance_stats['recall'] = float(recall)
            else:
                performance_stats['recall'] = 0.0
        except:
            performance_stats['recall'] = 0.0
        
        try:
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                performance_stats['f1_score'] = float(f1)
            else:
                performance_stats['f1_score'] = 0.0
        except:
            performance_stats['f1_score'] = 0.0
        
        # Calculate AUC-ROC and AUC-PR metrics only if both classes present
        auc_roc = auc_pr = None
        if n_classes_true > 1 and n_classes_pred > 1:
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                # Use RMD scores as the prediction scores
                auc_roc = roc_auc_score(results_df['ground_truth_hallucination'], results_df['rmd_score'])
                auc_pr = average_precision_score(results_df['ground_truth_hallucination'], results_df['rmd_score'])
                performance_stats['auc_roc'] = float(auc_roc)
                performance_stats['auc_pr'] = float(auc_pr)
            except Exception as e:
                print(f"Warning: Could not calculate AUC metrics: {e}")
                performance_stats['auc_roc'] = None
                performance_stats['auc_pr'] = None
        else:
            performance_stats['auc_roc'] = None
            performance_stats['auc_pr'] = None
        
        summary_lines.append("Performance vs BLEURT Ground Truth:")
        summary_lines.append(f"  Accuracy: {accuracy:.4f}")
        if auc_roc is not None:
            summary_lines.append(f"  AUC-ROC: {auc_roc:.4f}")
        else:
            summary_lines.append("  AUC-ROC: N/A (single class or error)")
        if auc_pr is not None:
            summary_lines.append(f"  AUC-PR: {auc_pr:.4f}")
        else:
            summary_lines.append("  AUC-PR: N/A (single class or error)")
        summary_lines.append(f"  Precision: {precision:.4f}")
        summary_lines.append(f"  Recall: {recall:.4f}")
        summary_lines.append(f"  F1-Score: {f1:.4f}")
        summary_lines.append(f"  True Positives: {true_pos}")
        summary_lines.append(f"  True Negatives: {true_neg}")  
        summary_lines.append(f"  False Positives: {false_pos}")
        summary_lines.append(f"  False Negatives: {false_neg}")
    
    # Print to console
    print("\n" + "\n".join(summary_lines))
    
    # Save summary to file
    summary_file = perplexity_dir / f"rmd_{args.model_name}_{args.dataset_name}_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
    print(f"\nSummary saved to: {summary_file}")
    
    # Print evaluation metrics if available
    if evaluation_metrics:
        print(f"\nEvaluation Metrics:")
        if 'auc_roc' in evaluation_metrics and evaluation_metrics['auc_roc'] is not None:
            print(f"  AUC-ROC: {evaluation_metrics['auc_roc']:.4f}")
        if 'auc_pr' in evaluation_metrics and evaluation_metrics['auc_pr'] is not None:
            print(f"  AUC-PR: {evaluation_metrics['auc_pr']:.4f}")
        if 'accuracy' in evaluation_metrics:
            print(f"  Accuracy: {evaluation_metrics['accuracy']:.4f}")
        if 'f1_score' in evaluation_metrics:
            print(f"  F1-Score: {evaluation_metrics['f1_score']:.4f}")
        if 'precision' in evaluation_metrics:
            print(f"  Precision: {evaluation_metrics['precision']:.4f}")
        if 'recall' in evaluation_metrics:
            print(f"  Recall: {evaluation_metrics['recall']:.4f}")
    
    # Calculate and save final metrics
    if len(results_df) > 0 and 'ground_truth_hallucination' in results_df.columns:
        # Check if we have both classes
        n_classes = len(set(results_df['ground_truth_hallucination']))
        
        if n_classes > 1:
            # Calculate AUROC using RMD scores vs ground truth
            try:
                from sklearn.metrics import roc_auc_score
                auroc = roc_auc_score(results_df['ground_truth_hallucination'], results_df['rmd_score'])
                print(f"\nFinal AUROC: {auroc:.4f}")
                
                # Use haloscope utils for consistent reporting
                positive_scores = results_df[results_df['ground_truth_hallucination'] == 1]['rmd_score'].values
                negative_scores = results_df[results_df['ground_truth_hallucination'] == 0]['rmd_score'].values
                
                if len(positive_scores) > 0 and len(negative_scores) > 0:
                    measures = get_measures(positive_scores, negative_scores)
                    auroc, auprc, fpr95 = measures
                    print(f"\nRMD Results for {args.dataset_name} ({args.model_name}):")
                    print_measures(auroc, auprc, fpr95, 'RMD')
                else:
                    print("\nCannot calculate measures: Missing positive or negative samples after filtering.")
            except Exception as e:
                print(f"Error calculating final metrics: {e}")
        else:
            print("\nSkipping final AUROC calculation: Only one class present in ground truth.")
            print("This is expected when all samples have similar BLEURT scores.")
    
    # Save final results summary
    final_results = {
        'dataset': args.dataset_name,
        'model': args.model_name,
        'method': 'perplexity_rmd',
        'total_samples': len(results_df),
        'train_ratio': args.train_ratio,
        'layer_idx': args.layer_idx,
        'bleurt_threshold': args.bleurt_threshold,
        'mean_rmd_score': float(results_df['rmd_score'].mean()) if len(results_df) > 0 else 0,
        'std_rmd_score': float(results_df['rmd_score'].std()) if len(results_df) > 0 else 0,
        'hallucination_rate': float(results_df['ground_truth_hallucination'].mean()) if len(results_df) > 0 else 0,
    }
    
    results_file = perplexity_dir / f"perplexity_results_{args.dataset_name}_{args.model_name}.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Final results saved to {results_file}")
    
    print(f"\nAll files saved in: {perplexity_dir}")
    print("="*60)

if __name__ == '__main__':
    main()