#!/usr/bin/env python3
"""
HaloScope implementation for low-resource languages
Based on methods/haloscope/hal_det_llama.py

Follows the exact experimental setup from HaloScope paper Section 4.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import argparse
import json
import pandas as pd

from datasets import load_dataset, Dataset
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict

# Import local modules
from utils import seed_everything, get_measures, print_measures
from linear_probe import get_linear_acc

# Ensure repository paths are available regardless of execution location
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
REPO_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATASETS_DIR = REPO_ROOT / "datasets"
MODELS_DIR = REPO_ROOT / "models"
BLEURT_DIR = MODELS_DIR / "BLEURT-20"

from tokenization_utils import preprocess_text_for_language, dataset_to_language
import re

# Model path candidates for each supported checkpoint
MODEL_DIR_CANDIDATES = {
    'llama2_7B': ['llama', 'Llama-2-7b-hf'],
    'llama3_2_1B': ['llama3_2_1B', 'Llama-3.2-1B'],
    'opt_6_7b': ['opt-6.7b', 'opt'],
    'opt_1_3b': ['opt_1_3b', 'opt-1.3b', 'opt'],
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B', 
                       choices=['llama2_7B', 'llama3_2_1B', 'opt_6_7b', 'opt_1_3b'])
    parser.add_argument('--dataset_name', type=str, default='armenian',
                       choices=['armenian', 'basque', 'tigrinya'])
    parser.add_argument('--num_gene', type=int, default=1, help='Number of generations per question')
    parser.add_argument('--gene', type=int, default=0, help='Generate responses (1) or analyze existing (0)')
    parser.add_argument('--generate_gt', type=int, default=0, help='Generate ground truth labels')
    parser.add_argument('--weighted_svd', type=int, default=1, help='Use weighted SVD')
    parser.add_argument('--feat_loc_svd', type=int, default=3, help='Feature location for SVD (1=heads, 2=mlp, 3=layers)')
    parser.add_argument('--wild_ratio', type=float, default=0.75, help='Ratio of data for unlabeled training')
    parser.add_argument('--thres_gt', type=float, default=0.5, help='Ground truth threshold')
    parser.add_argument('--most_likely', type=int, default=1, help='Use greedy sampling')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    seed_everything(41)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/{args.dataset_name}_hal_det', exist_ok=True)
    os.makedirs(f'{args.output_dir}/{args.dataset_name}_hal_det/answers', exist_ok=True)
    
    print(f"HaloScope Low-Resource Language Evaluation")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {'Generation' if args.gene else 'Analysis'}")
    
    # Check model-dataset compatibility
    if 'opt' in args.model_name.lower() and args.dataset_name == 'tigrinya':
        print("WARNING: OPT models have limited compatibility with Tigrinya (Ge'ez script)")
        print("This may result in tokenization errors or poor performance.")
        print("Consider using LLaMA models for Tigrinya, or Armenian/Basque datasets for OPT.")
        
        # For automated runs, continue with fallback handling
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
    
    # Combine all data for processing (following HaloScope methodology)
    # HaloScope uses all available data and creates its own train/val/test split
    all_data = train_data + val_data + test_data
    
    if args.gene:
        # Generation phase - generate on all data
        generate_responses(all_data, args)
    elif args.generate_gt:
        # Ground truth generation phase  
        generate_ground_truth(all_data, args)
    else:
        # Analysis phase - use actual dataset splits
        run_haloscope_analysis_with_splits(dataset_splits, args)

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
                'answer': row.get('answer', ''),
                'id': f"armenian_train_{len(train_data)}",
                'split': 'train'
            })
            
        test_data = []
        for _, row in test_df.iterrows():
            test_data.append({
                'question': row['question'],
                'answer': row.get('answer', ''),
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

def generate_responses(dataset, args):
    """
    Generate model responses for questions - following original exactly for each model type
    """
    MODEL = resolve_model_path(args.model_name)
    
    # Handle different model types with their specific configurations
    if 'opt' in args.model_name.lower():
        # OPT-specific loading (following hal_det_opt.py)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16
        ).cuda()
    else:
        # LLaMA-specific loading (following hal_det_llama.py)  
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Generating responses with {args.model_name}...")
    language = dataset_to_language(args.dataset_name)
    
    for i in tqdm(range(len(dataset)), desc="Generating"):
        answers = [None] * args.num_gene
        
        question = dataset[i]['question']
        
        # Format prompt exactly like original 
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
            prompt = safe_tokenize_for_opt(tokenizer, prompt_text, language).cuda()
        else:
            prompt = tokenizer(prompt_text, return_tensors='pt').input_ids.cuda()
        
        for gen_iter in range(args.num_gene):
            # Clear GPU cache before each generation to avoid OOM
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                if args.most_likely:
                    # Use exact parameters from original - num_beams=5
                    generated = model.generate(
                        prompt,
                        num_beams=5,
                        num_return_sequences=1,
                        do_sample=False,
                        max_new_tokens=64,
                    )
                else:
                    # Sampling exactly like original
                    generated = model.generate(
                        prompt,
                        do_sample=True,
                        num_return_sequences=1,
                        num_beams=1,
                        max_new_tokens=64,
                        temperature=0.5,
                        top_p=1.0
                    )
                
            decoded = tokenizer.decode(
                generated[0, prompt.shape[-1]:],
                skip_special_tokens=True
            )
            
            # Clean up exactly like original
            if 'Answer the question concisely' in decoded:
                print('#####error')
                print(decoded.split('Answer the question concisely')[1])
                print('#####error')
                decoded = decoded.split('Answer the question concisely')[0]
                
            decoded = decoded.strip()
            if '\n' in decoded:
                decoded = decoded.split('\n')[0]
                
            answers[gen_iter] = decoded
            print(f"Generated: {decoded}")
            
        # Save answers
        info = 'most_likely_' if args.most_likely else 'batch_generations_'
        np.save(
            f'{args.output_dir}/{args.dataset_name}_hal_det/answers/{info}hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy',
            answers
        )
        
    print(f"Generated responses saved to {args.output_dir}/{args.dataset_name}_hal_det/answers/")

def generate_ground_truth(dataset, args):
    """
    Generate ground truth labels using BLEURT similarity scoring
    Exactly as in original HaloScope implementation
    """
    print("Generating ground truth labels...")
    
    try:
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer

        model = BleurtForSequenceClassification.from_pretrained(str(BLEURT_DIR)).cuda()
        tokenizer = BleurtTokenizer.from_pretrained(str(BLEURT_DIR))
        model.eval()

        print("BLEURT model loaded successfully")

    except Exception as e:
        print(f"Error loading BLEURT model: {e}")
        print("BLEURT scoring is required for ground truth generation.")
        return

    gts = []
    for i in tqdm(range(len(dataset)), desc="Computing GT with BLEURT"):
        info = 'most_likely_' if args.most_likely else 'batch_generations_'
        try:
            answers = np.load(
                f'{args.output_dir}/{args.dataset_name}_hal_det/answers/{info}hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy'
            )

            # Get reference answers (can be multiple)
            reference = dataset[i]['answer']
            all_answers = [reference] if isinstance(reference, str) else reference

            # Compute BLEURT scores
            predictions = answers
            all_results = np.zeros((len(all_answers), len(predictions)))

            with torch.no_grad():
                for anw in range(len(all_answers)):
                    inputs = tokenizer(predictions.tolist(), [all_answers[anw]] * len(predictions),
                                     padding='longest', return_tensors='pt')
                    for key in list(inputs.keys()):
                        inputs[key] = inputs[key].cuda()
                    res = np.asarray(model(**inputs).logits.flatten().tolist())
                    all_results[anw] = res

            gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)

            if i % 10 == 0:
                print("samples passed: ", i)

        except FileNotFoundError:
            print(f"Answers file not found for index {i}. Run generation first.")
            return

    gts = np.array(gts)
    suffix = 'bleurt_score'
    
    # Save ground truth scores
    if args.most_likely:
        filename = f'{args.output_dir}/ml_{args.dataset_name}_{suffix}.npy'
    else:
        filename = f'{args.output_dir}/bg_{args.dataset_name}_{suffix}.npy'
        
    np.save(filename, gts)
    print(f"Ground truth scores saved to {filename}")
    
def run_haloscope_analysis_with_splits(dataset_splits, args):
    """
    Run HaloScope hallucination detection analysis using actual dataset splits
    """
    print("Running HaloScope analysis with proper dataset splits...")
    
    train_data = dataset_splits['train']
    val_data = dataset_splits['validation'] 
    test_data = dataset_splits['test']
    
    # For generation and ground truth, we still need the combined dataset
    all_data = train_data + val_data + test_data
    
    # Load model and tokenizer with model-specific configurations  
    MODEL = resolve_model_path(args.model_name)
    
    if 'opt' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16
        ).cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Extract embeddings from generated responses
    print("Extracting embeddings...")
    embed_generated = extract_embeddings(all_data, model, tokenizer, args)
    
    # Load ground truth labels
    print("Loading ground truth labels...")
    try:
        gts = np.load(f'{args.output_dir}/ml_{args.dataset_name}_bleurt_score.npy')
        print("Loaded BLEURT-based ground truth scores")
    except FileNotFoundError as e:
        print(f"Ground truth file not found: {e}")
        print("Please run ground truth generation first with --generate_gt 1")
        return
        
    # Create binary labels
    gt_label = np.asarray(gts > args.thres_gt, dtype=np.int32)
    
    # Use actual dataset splits instead of random splitting
    print("Using actual dataset splits for analysis:")
    print(f"  Training set: {len(train_data)} samples (for unlabeled learning)")
    print(f"  Validation set: {len(val_data)} samples (for hyperparameter tuning)")
    print(f"  Test set: {len(test_data)} samples (for final evaluation)")
    
    # Create index mappings for the splits
    train_indices = list(range(len(train_data)))
    val_indices = list(range(len(train_data), len(train_data) + len(val_data)))
    test_indices = list(range(len(train_data) + len(val_data), len(all_data)))
    
    # Split labels according to actual dataset splits
    gt_label_train = gt_label[train_indices]
    gt_label_val = gt_label[val_indices] 
    gt_label_test = gt_label[test_indices]
    
    # Run HaloScope SVD analysis
    print("Running HaloScope SVD analysis...")
    results = run_svd_analysis_with_actual_splits(
        embed_generated, gt_label_train, gt_label_val, gt_label_test,
        train_indices, val_indices, test_indices, args
    )
    
    print(f"\nHaloScope Results for {args.dataset_name} ({args.model_name}):")
    print(f"Test AUROC: {results['test_auroc']:.4f}")
    
    # Save results
    results_file = f"{args.output_dir}/haloscope_results_{args.dataset_name}_{args.model_name}_proper_splits.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

def run_haloscope_analysis(dataset, args):
    """
    Run HaloScope hallucination detection analysis
    """
    print("Running HaloScope analysis...")
    
    # Load model and tokenizer with model-specific configurations  
    MODEL = resolve_model_path(args.model_name)
    
    if 'opt' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16
        ).cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16, 
            device_map="auto"
        ).cuda()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Extract embeddings from generated responses
    print("Extracting embeddings...")
    embed_generated = extract_embeddings(dataset, model, tokenizer, args)
    
    # Load ground truth labels
    print("Loading ground truth labels...")
    try:
        gts = np.load(f'{args.output_dir}/ml_{args.dataset_name}_bleurt_score.npy')
        print("Loaded BLEURT-based ground truth scores")
    except FileNotFoundError as e:
        print(f"Ground truth file not found: {e}")
        print("Please run ground truth generation first with --generate_gt 1")
        return
        
    # Create binary labels
    gt_label = np.asarray(gts > args.thres_gt, dtype=np.int32)
    
    # Split data following paper methodology (adapted for small datasets)
    length = len(dataset)
    permuted_index = np.random.permutation(length)
    wild_q_indices = permuted_index[:int(args.wild_ratio * length)]
    
    # For small datasets, use smaller validation set
    val_size = min(100, max(10, len(wild_q_indices) // 4))  # At least 10, at most 100, or 1/4 of wild
    wild_q_indices1 = wild_q_indices[:len(wild_q_indices) - val_size]  # Training  
    wild_q_indices2 = wild_q_indices[len(wild_q_indices) - val_size:]   # Validation
    
    print(f"Data split: Total={length}, Wild={len(wild_q_indices)}, Wild_train={len(wild_q_indices1)}, Wild_val={len(wild_q_indices2)}, Test={length - len(wild_q_indices)}")
    
    # Split labels
    gt_label_test = []
    gt_label_wild = []
    gt_label_val = []
    
    for i in range(length):
        if i not in wild_q_indices:
            gt_label_test.extend(gt_label[i: i+1])
        elif i in wild_q_indices1:
            gt_label_wild.extend(gt_label[i: i+1])
        else:
            gt_label_val.extend(gt_label[i: i+1])
            
    gt_label_test = np.asarray(gt_label_test)
    gt_label_wild = np.asarray(gt_label_wild)
    gt_label_val = np.asarray(gt_label_val)
    
    # Run HaloScope SVD analysis
    print("Running HaloScope SVD analysis...")
    results = run_svd_analysis(embed_generated, gt_label_wild, gt_label_val, gt_label_test, wild_q_indices1, wild_q_indices2, args)
    
    print(f"\nHaloScope Results for {args.dataset_name} ({args.model_name}):")
    print(f"Test AUROC: {results['test_auroc']:.4f}")
    
    # Save results
    results_file = f"{args.output_dir}/haloscope_results_{args.dataset_name}_{args.model_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

def extract_embeddings(dataset, model, tokenizer, args):
    """
    Extract embeddings from model responses
    """
    embed_generated = []
    language = dataset_to_language(args.dataset_name)
    
    # Define layer hooks adapted for standard transformers
    if 'llama' in args.model_name.lower():
        HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
        MLPS = [f"model.layers.{i}.mlp.down_proj" for i in range(model.config.num_hidden_layers)]
    elif 'opt' in args.model_name.lower():
        HEADS = [f"model.decoder.layers.{i}.self_attn.out_proj" for i in range(model.config.num_hidden_layers)]
        MLPS = [f"model.decoder.layers.{i}.fc2" for i in range(model.config.num_hidden_layers)]
    
    for i in tqdm(range(len(dataset)), desc="Extracting embeddings"):
        question = dataset[i]['question']
        
        # Load generated answers
        info = 'most_likely_' if args.most_likely else 'batch_generations_'
        answers = np.load(
            f'{args.output_dir}/{args.dataset_name}_hal_det/answers/{info}hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy'
        )
        
        for answer in answers:
            # Format prompt with same context truncation
            if args.dataset_name == 'tigrinya' and 'context' in dataset[i]:
                # Truncate context for memory efficiency  
                context = dataset[i]['context']
                if len(context) > 1000:
                    context = context[:1000] + "..."
                prompt_text = "Concisely answer the following question based on the information in the given passage: \n" + \
                    " Passage: " + context + " \n Q: " + question + " \n A: " + answer
            else:
                prompt_text = f"Answer the question concisely. Q: {question}" + " A: " + answer
            
            prompt_text = preprocess_text_for_language(prompt_text, language)

            # Use safe tokenization for OPT models in embedding extraction too
            if 'opt' in args.model_name.lower():
                prompt = safe_tokenize_for_opt(tokenizer, prompt_text, language).cuda()
            else:
                prompt = tokenizer(prompt_text, return_tensors='pt').input_ids.cuda()
            
            with torch.no_grad():
                if args.feat_loc_svd == 3:  # Layer-wise features
                    hidden_states = model(prompt, output_hidden_states=True).hidden_states
                    hidden_states = torch.stack(hidden_states, dim=0).squeeze()
                    hidden_states = hidden_states.detach().cpu().numpy()[:, -1, :]
                    embed_generated.append(hidden_states)
                elif args.feat_loc_svd == 2:  # MLP features (using baukit)
                    with TraceDict(model, MLPS) as ret:
                        output = model(prompt, output_hidden_states=True)
                    mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
                    mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()
                    mlp_wise_hidden_states = mlp_wise_hidden_states[:, -1, :]
                    embed_generated.append(mlp_wise_hidden_states)
                elif args.feat_loc_svd == 1:  # Attention head features (using baukit)
                    with TraceDict(model, HEADS) as ret:
                        output = model(prompt, output_hidden_states=True)
                    head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
                    head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
                    head_wise_hidden_states = head_wise_hidden_states[:, -1, :]
                    embed_generated.append(head_wise_hidden_states)
                    
    embed_generated = np.asarray(np.stack(embed_generated), dtype=np.float32)
    
    # Save embeddings
    save_path = f'{args.output_dir}/{args.dataset_name}_hal_det/embeddings_{args.feat_loc_svd}_{args.model_name}.npy'
    np.save(save_path, embed_generated)
    print(f"Embeddings saved to {save_path}")
    
    return embed_generated

def run_svd_analysis_with_actual_splits(embed_generated, gt_label_train, gt_label_val, gt_label_test, train_indices, val_indices, test_indices, args):
    """
    Run SVD analysis using actual dataset splits instead of random splitting
    """
    # Split embeddings by actual dataset splits
    embed_generated_train = embed_generated[train_indices]
    embed_generated_val = embed_generated[val_indices]
    embed_generated_test = embed_generated[test_indices]
    
    print(f"Embedding shapes: Train={embed_generated_train.shape}, Val={embed_generated_val.shape}, Test={embed_generated_test.shape}")
    print(f"Label shapes: Train={len(gt_label_train)}, Val={len(gt_label_val)}, Test={len(gt_label_test)}")
    
    if args.feat_loc_svd == 3:
        embed_generated_train = embed_generated_train[:, 1:, :]  # Skip input embedding
        embed_generated_val = embed_generated_val[:, 1:, :]
        embed_generated_test = embed_generated_test[:, 1:, :]
    
    # SVD parameter search on validation set
    best_auroc = 0
    best_params = {}
    
    print("Searching for best SVD parameters on validation set...")
    for k in range(1, min(11, embed_generated_val.shape[0])):  # Limit k to available samples
        for layer in range(embed_generated_val.shape[1]):
            # Mean center embeddings on validation set
            centered = embed_generated_val[:, layer, :] - embed_generated_val[:, layer, :].mean(0)
            
            # SVD
            _, s, Vt = np.linalg.svd(centered, full_matrices=False)
            projection = Vt[:k, :].T
            
            if args.weighted_svd:
                projection = s[:k] * projection
                
            # Compute scores
            scores = np.mean(np.matmul(centered, projection), -1, keepdims=True)
            scores = np.sqrt(np.sum(np.square(scores), axis=1))
            
            # Test both directions for validation set
            if len(gt_label_val[gt_label_val == 1]) > 0 and len(gt_label_val[gt_label_val == 0]) > 0:
                measures1 = get_measures(scores[gt_label_val == 1], scores[gt_label_val == 0])
                measures2 = get_measures(-scores[gt_label_val == 1], -scores[gt_label_val == 0])
                
                if measures1[0] > measures2[0]:
                    measures = measures1
                    sign = 1
                else:
                    measures = measures2
                    sign = -1
                    
                if measures[0] > best_auroc:
                    best_auroc = measures[0]
                    best_params = {
                        'k': k,
                        'layer': layer,
                        'sign': sign,
                        'auroc': measures[0]
                    }
                    
    # Handle case where no valid parameters were found
    if not best_params:
        print("No valid parameters found during validation search. Using defaults.")
        best_params = {
            'k': 1,
            'layer': 0,
            'sign': 1,
            'auroc': 0.5
        }
        best_auroc = 0.5
    
    print(f"Best validation AUROC: {best_auroc:.4f} (k={best_params['k']}, layer={best_params['layer']})")
    
    # Train final model using best hyperparameters on training set
    layer = best_params['layer']
    k = best_params['k']
    sign = best_params['sign']
    
    # Fit on training data (unlabeled approach following HaloScope)
    centered_train = embed_generated_train[:, layer, :] - embed_generated_train[:, layer, :].mean(0)
    _, s, Vt = np.linalg.svd(centered_train, full_matrices=False)
    projection = Vt[:k, :].T
    
    if args.weighted_svd:
        projection = s[:k] * projection
        
    train_scores = np.mean(np.matmul(centered_train, projection), -1, keepdims=True)
    train_scores = np.sqrt(np.sum(np.square(train_scores), axis=1)) * sign
    
    # Linear probe training on training set
    from linear_probe import get_linear_acc
    
    # Use membership scores to create pseudo-labels for training
    threshold = np.percentile(train_scores, 50)  # Use median as threshold
    pseudo_labels = (train_scores > threshold).astype(int)
    
    # Train linear probe
    best_acc, final_acc, (clf, best_state, best_preds, preds, labels_val), losses_train = get_linear_acc(
        embed_generated_train[:, layer, :],
        pseudo_labels,
        embed_generated_train[:, layer, :],
        pseudo_labels,
        2,
        epochs=50,
        batch_size=min(512, len(embed_generated_train)),
        cosine=True,
        nonlinear=True,
        learning_rate=0.05,
        weight_decay=0.0003
    )
    
    # Evaluate on test set
    clf.eval()
    output = clf(torch.from_numpy(embed_generated_test[:, layer, :]).cuda())
    test_probs = torch.sigmoid(output).cpu().data.numpy()
    
    if len(gt_label_test[gt_label_test == 1]) > 0 and len(gt_label_test[gt_label_test == 0]) > 0:
        test_measures = get_measures(test_probs[gt_label_test == 1], test_probs[gt_label_test == 0])
        test_auroc = test_measures[0]
    else:
        test_auroc = 0.5
        print("Warning: Test set has only one class, AUROC set to 0.5")
    
    print(f"Test AUROC: {test_auroc:.4f}")
    
    return {
        'test_auroc': test_auroc,
        'best_val_auroc': best_auroc,
        'best_params': best_params,
        'dataset': args.dataset_name,
        'model': args.model_name,
        'split_method': 'actual_dataset_splits'
    }

def run_svd_analysis(embed_generated, gt_label_wild, gt_label_val, gt_label_test, wild_indices, val_indices, args):
    """
    Run SVD analysis following HaloScope methodology
    """
    # Split embeddings by indices
    feat_indices_wild = []
    feat_indices_val = []
    feat_indices_test = []
    
    for i in range(len(embed_generated)):
        if i in wild_indices:
            feat_indices_wild.append(i)
        elif i in val_indices:
            feat_indices_val.append(i)
        else:
            feat_indices_test.append(i)
            
    embed_generated_wild = embed_generated[feat_indices_wild]
    embed_generated_val = embed_generated[feat_indices_val]
    embed_generated_test = embed_generated[feat_indices_test]
    
    print(f"Embedding shapes: Wild={embed_generated_wild.shape}, Val={embed_generated_val.shape}, Test={embed_generated_test.shape}")
    print(f"Wild indices: {len(feat_indices_wild)}, Val indices: {len(feat_indices_val)}, Test indices: {len(feat_indices_test)}")
    
    if args.feat_loc_svd == 3:
        embed_generated_wild = embed_generated_wild[:, 1:, :]  # Skip input embedding
        embed_generated_val = embed_generated_val[:, 1:, :]
        embed_generated_test = embed_generated_test[:, 1:, :]
    
    # SVD parameter search on validation set
    best_auroc = 0
    best_params = {}
    
    for k in range(1, 11):  # Search k from 1 to 10
        for layer in range(len(embed_generated_val[0])):
            # Mean center embeddings
            centered = embed_generated_val[:, layer, :] - embed_generated_val[:, layer, :].mean(0)
            
            # SVD
            _, s, Vt = np.linalg.svd(centered, full_matrices=False)
            projection = Vt[:k, :].T
            
            if args.weighted_svd:
                projection = s[:k] * projection
                
            # Compute scores
            scores = np.mean(np.matmul(centered, projection), -1, keepdims=True)
            scores = np.sqrt(np.sum(np.square(scores), axis=1))
            
            # Test both directions
            measures1 = get_measures(scores[gt_label_val == 1], scores[gt_label_val == 0])
            measures2 = get_measures(-scores[gt_label_val == 1], -scores[gt_label_val == 0])
            
            if measures1[0] > measures2[0]:
                measures = measures1
                sign = 1
            else:
                measures = measures2
                sign = -1
                
            if measures[0] > best_auroc:
                best_auroc = measures[0]
                best_params = {
                    'k': k,
                    'layer': layer,
                    'sign': sign,
                    'auroc': measures[0]
                }
                
    print(f"Best validation AUROC: {best_auroc:.4f} (k={best_params['k']}, layer={best_params['layer']})")
    
    # Train final model using best hyperparameters
    layer = best_params['layer']
    k = best_params['k']
    sign = best_params['sign']
    
    # Fit on wild (unlabeled) data
    centered_wild = embed_generated_wild[:, layer, :] - embed_generated_wild[:, layer, :].mean(0)
    _, s, Vt = np.linalg.svd(centered_wild, full_matrices=False)
    projection = Vt[:k, :].T
    
    if args.weighted_svd:
        projection = s[:k] * projection
        
    wild_scores = np.mean(np.matmul(centered_wild, projection), -1, keepdims=True)
    wild_scores = np.sqrt(np.sum(np.square(wild_scores), axis=1)) * sign
    
    # Linear probe training as per paper
    from linear_probe import get_linear_acc
    
    # Use membership scores to create pseudo-labels
    threshold = np.percentile(wild_scores, 50)  # Use median as threshold
    pseudo_labels = (wild_scores > threshold).astype(int)
    
    # Train linear probe
    best_acc, final_acc, (clf, best_state, best_preds, preds, labels_val), losses_train = get_linear_acc(
        embed_generated_wild[:, layer, :],
        pseudo_labels,
        embed_generated_wild[:, layer, :],
        pseudo_labels,
        2,
        epochs=50,
        batch_size=512,
        cosine=True,
        nonlinear=True,
        learning_rate=0.05,
        weight_decay=0.0003
    )
    
    # Evaluate on test set
    clf.eval()
    output = clf(torch.from_numpy(embed_generated_test[:, layer, :]).cuda())
    test_probs = torch.sigmoid(output).cpu().data.numpy()
    
    test_measures = get_measures(test_probs[gt_label_test == 1], test_probs[gt_label_test == 0])
    print(f"Test AUROC: {test_measures[0]:.4f}")
    
    return {
        'test_auroc': test_measures[0],
        'best_val_auroc': best_auroc,
        'best_params': best_params,
        'dataset': args.dataset_name,
        'model': args.model_name
    }

if __name__ == '__main__':
    main()
