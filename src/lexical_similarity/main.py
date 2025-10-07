#!/usr/bin/env python3
"""
Lexical Similarity baseline for hallucination detection (consistency-based)
- Scores a response by 1 - average cosine similarity (TF-IDF) between the main answer and sampled answers.
- Shares CLI, dataset, GT, saving, and eval conventions with SelfCheckGPT (NLI) script.

Lexical Similarity baseline (ULexiSim) from UQ-NLG:
- For each input x with m sampled responses {s_j}, compute the average ROUGE-L F1
  across *all pairs* of responses (or between main and samples if desired).
- Report Uncertainty = 1 - mean_ROUGE_L  (higher => more uncertain), matching selective-NLG usage.

Reference: UQ-NLG evaluates 'lexical similarity' as the average ROUGE-L between responses.

Follows the exact experimental setup from HaloScope paper Section 4.
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

# ROUGE-L scoring
from rouge_score import rouge_scorer

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

def select_optimal_threshold(scores, method='balanced'):
    """
    Select optimal threshold based on different criteria
    
    Args:
        scores: array of ground truth scores
        method: 'balanced' (median), 'percentile_25', 'percentile_75', 'mean'
    
    Returns:
        optimal threshold value
    """
    if method == 'balanced':
        # Use a threshold slightly below median to ensure both classes exist
        # This handles cases where many scores are identical at the median
        median_val = np.median(scores)
        unique_scores = np.unique(scores)
        
        # If all scores are identical, use a threshold that creates some separation
        if len(unique_scores) == 1:
            return float(unique_scores[0]) - 1e-6
        
        # Find the largest score that is less than median
        below_median = unique_scores[unique_scores < median_val]
        if len(below_median) > 0:
            # Use a threshold between the highest below-median score and median
            return float((below_median[-1] + median_val) / 2)
        else:
            # All unique scores are >= median, use median minus small epsilon
            return float(median_val - 1e-6)
            
    elif method == 'percentile_25':
        return np.percentile(scores, 25)
    elif method == 'percentile_75':
        return np.percentile(scores, 75)
    elif method == 'mean':
        return np.mean(scores)
    else:
        return np.median(scores)  # Default fallback

def pairwise_mean_rougel(responses):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    n = len(responses)
    tot = 0.0; cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            r = scorer.score(responses[i], responses[j])['rougeL'].fmeasure
            tot += r; cnt += 1
    return (tot / cnt) if cnt else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B', 
                       choices=['llama2_7B', 'llama3_2_1B', 'opt_6_7b', 'opt_1_3b'])
    parser.add_argument('--dataset_name', type=str, default='armenian',
                       choices=['armenian', 'basque', 'tigrinya'])
    parser.add_argument('--num_gene', type=int, default=5, help='Number of generations per question for sampling')
    parser.add_argument('--gene', type=int, default=0, help='Generate responses (1) or analyze existing (0)')
    parser.add_argument('--generate_gt', type=int, default=0, help='Generate ground truth labels')
    parser.add_argument('--use_rouge', type=int, default=1, help='Use ROUGE instead of BLEURT')
    parser.add_argument('--thres_gt', type=float, default=0.7, help='Ground truth threshold')
    parser.add_argument('--auto_threshold', action='store_true', help='Automatically select optimal threshold for balanced classes')
    parser.add_argument('--most_likely', type=int, default=1, help='Use greedy sampling for main response')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    # compute option: pairwise among samples (default) or vs main only
    parser.add_argument('--vs_main_only', type=int, default=0, help='Compare vs main answer only (not pairwise)')
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
    os.makedirs(f'{args.output_dir}/lexical_similarity', exist_ok=True)
    os.makedirs(f'{args.output_dir}/lexical_similarity/{args.dataset_name}', exist_ok=True)
    os.makedirs(f'{args.output_dir}/lexical_similarity/{args.dataset_name}/answers', exist_ok=True)
    
    print(f"Lexical Similarity (ULexiSim) Low-Resource Language Evaluation")
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
    
    # Combine all data for processing (following SelfCheckGPT methodology)
    all_data = train_data + val_data + test_data
    
    if args.gene:
        # Generation phase - generate main responses and samples
        generate_responses(all_data, args, gpu_config)
    elif args.generate_gt:
        # Ground truth generation phase  
        generate_ground_truth(all_data, args, gpu_config)
    else:
        # Analysis phase - run Lexical Similarity analysis
        run_lexical_similarity_analysis(all_data, args)

def generate_responses(dataset, args, gpu_config):
    """
    Generate model responses for Lexical Similarity analysis
    Need both main response and multiple sampled responses for comparison
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
    
    print(f"Generating responses with {args.model_name} for Lexical Similarity...")
    language = dataset_to_language(args.dataset_name)
    
    for i in tqdm(range(len(dataset)), desc="Generating"):
        question = dataset[i]['question']
        
        # Format prompt exactly like SelfCheckGPT
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
        
        # Generate main response (most likely) 
        main_answer = None
        torch.cuda.empty_cache()
        
        # Generate main response with multi-GPU support
        generation_kwargs = {
            'attention_mask': attention_mask,
            'num_beams': 5,
            'num_return_sequences': 1,
            'do_sample': False,
            'max_new_tokens': 64,
        }
        generated = generate_with_multi_gpu(model, tokenizer, prompt, generation_kwargs, primary_device)
            
        decoded = tokenizer.decode(
            generated[0, prompt.shape[-1]:],
            skip_special_tokens=True
        )
        
        # Clean up exactly like original
        if 'Answer the question concisely' in decoded:
            decoded = decoded.split('Answer the question concisely')[0]
            
        decoded = decoded.strip()
        if '\n' in decoded:
            decoded = decoded.split('\n')[0]
            
        main_answer = decoded
        print(f"Main answer: {decoded}")
        
        # Generate sampled responses for comparison
        sampled_answers = []
        for gen_iter in range(args.num_gene):
            cleanup_gpu_memory()
            
            # Generate sampled response with multi-GPU support
            generation_kwargs = {
                'attention_mask': attention_mask,
                'do_sample': True,
                'num_return_sequences': 1,
                'num_beams': 1,
                'max_new_tokens': 64,
                'temperature': 0.8,  # Higher temperature for more diversity
                'top_p': 0.95
            }
            generated = generate_with_multi_gpu(model, tokenizer, prompt, generation_kwargs, primary_device)
                
            decoded = tokenizer.decode(
                generated[0, prompt.shape[-1]:],
                skip_special_tokens=True
            )
            
            # Clean up exactly like original
            if 'Answer the question concisely' in decoded:
                decoded = decoded.split('Answer the question concisely')[0]
                
            decoded = decoded.strip()
            if '\n' in decoded:
                decoded = decoded.split('\n')[0]
                
            sampled_answers.append(decoded)
            if gen_iter < 3:  # Print first few for debugging
                print(f"Sample {gen_iter}: {decoded}")
            
        # Save main answer
        np.save(
            f'{args.output_dir}/lexical_similarity/{args.dataset_name}/answers/main_answer_{args.model_name}_{args.dataset_name}_index_{i}.npy',
            [main_answer]
        )
        
        # Save sampled answers
        np.save(
            f'{args.output_dir}/lexical_similarity/{args.dataset_name}/answers/sampled_answers_{args.model_name}_{args.dataset_name}_index_{i}.npy',
            sampled_answers
        )
        
    print(f"Generated responses saved to {args.output_dir}/lexical_similarity/{args.dataset_name}/answers/")

def generate_ground_truth(dataset, args, gpu_config):
    """
    Generate ground truth labels using BLEURT or ROUGE similarity scoring
    """
    print("Generating ground truth labels for Lexical Similarity...")
    
    if args.use_rouge:
        # Use ROUGE scoring
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        gts = []
        for i in tqdm(range(len(dataset)), desc="Computing GT with ROUGE"):
            try:
                main_answer = np.load(
                    f'{args.output_dir}/lexical_similarity/{args.dataset_name}/answers/main_answer_{args.model_name}_{args.dataset_name}_index_{i}.npy'
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
                    f'{args.output_dir}/lexical_similarity/{args.dataset_name}/answers/main_answer_{args.model_name}_{args.dataset_name}_index_{i}.npy'
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
    filename = f'{args.output_dir}/lexical_similarity/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_{suffix}.npy'
    np.save(filename, gts)
    print(f"Ground truth scores saved to {filename}")

def run_lexical_similarity_analysis(dataset, args):
    """
    Run Lexical Similarity (ULexiSim) analysis
    """
    print("Running Lexical Similarity (ULexiSim) analysis...")
    
    # Load ground truth labels
    print("Loading ground truth labels...")
    try:
        if args.use_rouge:
            gts = np.load(f'{args.output_dir}/lexical_similarity/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_rouge_score.npy')
            print("Loaded ROUGE-based ground truth scores")
        else:
            gts = np.load(f'{args.output_dir}/lexical_similarity/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_bleurt_score.npy')
            print("Loaded BLEURT-based ground truth scores")
    except FileNotFoundError as e:
        print(f"Ground truth file not found: {e}")
        print("Please run ground truth generation first with --generate_gt 1")
        return
        
    # Create binary labels with intelligent threshold selection
    print(f"Ground truth score statistics:")
    print(f"  Min: {gts.min():.4f}, Max: {gts.max():.4f}")
    print(f"  Mean: {gts.mean():.4f}, Median: {np.median(gts):.4f}")
    print(f"  Std: {gts.std():.4f}")
    print(f"  Unique values: {len(np.unique(gts))}")

    # Smart threshold selection to ensure balanced classes
    def ensure_balanced_classes(scores, initial_threshold, method_name=""):
        """Helper function to ensure we get both positive and negative classes"""
        labels = np.asarray(scores > initial_threshold, dtype=np.int32)
        n_positive = len(labels[labels == 1])
        n_negative = len(labels[labels == 0])
        
        print(f"{method_name}threshold {initial_threshold:.4f}: {n_positive} positive, {n_negative} negative")
        
        # If we don't have both classes, try different thresholds
        if n_positive == 0 or n_negative == 0:
            print("→ Single class detected, trying alternative thresholds...")
            
            # Try different percentiles to find a balanced split
            for percentile in [50, 40, 60, 30, 70, 25, 75]:
                test_threshold = np.percentile(scores, percentile)
                test_labels = np.asarray(scores > test_threshold, dtype=np.int32)
                test_pos = len(test_labels[test_labels == 1])
                test_neg = len(test_labels[test_labels == 0])
                
                if test_pos > 0 and test_neg > 0:
                    print(f"→ Found balanced split at {percentile}th percentile ({test_threshold:.4f}): {test_pos} positive, {test_neg} negative")
                    return test_threshold, test_labels
            
            # If percentiles don't work, try finding a threshold between unique values
            unique_scores = np.unique(scores)
            if len(unique_scores) > 1:
                # Try threshold between consecutive unique values
                for i in range(len(unique_scores) - 1):
                    test_threshold = (unique_scores[i] + unique_scores[i+1]) / 2
                    test_labels = np.asarray(scores > test_threshold, dtype=np.int32)
                    test_pos = len(test_labels[test_labels == 1])
                    test_neg = len(test_labels[test_labels == 0])
                    
                    if test_pos > 0 and test_neg > 0:
                        print(f"→ Found balanced split between unique values ({test_threshold:.4f}): {test_pos} positive, {test_neg} negative")
                        return test_threshold, test_labels
            
            # Last resort: create artificial split
            print("→ Using artificial split as last resort")
            n_samples = len(scores)
            artificial_threshold = np.sort(scores)[n_samples // 2]  # Use value at middle position
            artificial_labels = np.asarray(scores > artificial_threshold, dtype=np.int32)
            return artificial_threshold, artificial_labels
        
        return initial_threshold, labels

    if args.auto_threshold:
        # Use balanced threshold selection when auto-threshold is enabled
        threshold = select_optimal_threshold(gts, 'balanced')
        threshold, gt_label = ensure_balanced_classes(gts, threshold, "Auto-")
    else:
        # Start with user-specified threshold but ensure balanced classes
        threshold, gt_label = ensure_balanced_classes(gts, args.thres_gt, "Initial ")
    
    final_positive = len(gt_label[gt_label == 1])
    final_negative = len(gt_label[gt_label == 0])
    print(f"Final result: {final_positive} positive, {final_negative} negative samples")
    
    ans_dir = f"{args.output_dir}/lexical_similarity/{args.dataset_name}/answers"
    scores = []

    print("Computing Lexical Similarity (ULexiSim) scores...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    for i in tqdm(range(len(dataset)), desc="ULexiSim Analysis"):
        try:
            sp = f"{ans_dir}/sampled_answers_{args.model_name}_{args.dataset_name}_index_{i}.npy"
            if not os.path.exists(sp):
                print(f"Missing samples at {i}; re-run generation with --gene 1.")
                return
            samples = [str(s) for s in np.load(sp) if str(s).strip()]
            if len(samples) < 2:
                scores.append(1.0)  # maximal uncertainty if we can't assess dispersion
                continue

            if args.vs_main_only:
                mp = f"{ans_dir}/main_answer_{args.model_name}_{args.dataset_name}_index_{i}.npy"
                if not os.path.exists(mp):
                    print(f"Missing main answer at {i}.")
                    return
                main = str(np.load(mp)[0])
                vals = []
                for s in samples:
                    vals.append(scorer.score(main, s)['rougeL'].fmeasure)
                mean_rougel = float(np.mean(vals)) if vals else 0.0
            else:
                mean_rougel = pairwise_mean_rougel(samples)

            # Uncertainty = 1 - mean lexical similarity
            u = 1.0 - float(mean_rougel)
            scores.append(u)
            
        except FileNotFoundError:
            print(f"Answer files not found for index {i}. Run generation first.")
            return
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            scores.append(0.5)  # Default score on error

    scores = np.asarray(scores, dtype=np.float32)

    # Evaluate performance
    if len(gt_label[gt_label == 1]) > 0 and len(gt_label[gt_label == 0]) > 0:
        auroc, auprc, fpr95 = get_measures(scores[gt_label==1], scores[gt_label==0])
        print(f"\nLexical Similarity Results for {args.dataset_name} ({args.model_name}):")
        print(f"AUROC: {auroc:.4f}")
        print_measures(auroc, auprc, fpr95, 'ULexiSim')
    else:
        auroc = 0.5
        print("Warning: Ground truth has only one class, AUROC set to 0.5")

    # Save results
    results = {
        'auroc': float(auroc),
        'scores': scores.tolist(),
        'gt_labels': gt_label.tolist(),
        'dataset': args.dataset_name,
        'model': args.model_name,
        'method': 'lexical_similarity_uq_nlg',
        'threshold': float(threshold),  # Save actual threshold used
        'vs_main_only': bool(args.vs_main_only)
    }
    
    results_file = f"{args.output_dir}/lexical_similarity/ulexisim_results_{args.dataset_name}_{args.model_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    # Save scores for further analysis
    scores_file = f"{args.output_dir}/lexical_similarity/{args.dataset_name}/scores_ulexisim_{args.model_name}_{args.dataset_name}.npy"
    np.save(scores_file, scores)
    print(f"Scores saved to {scores_file}")

if __name__ == '__main__':
    main()
