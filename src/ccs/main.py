#!/usr/bin/env python3
"""
CCS* implementation for low-resource languages
Based on the CCS* baseline described in HaloScope paper (Section 4).

Key differences from original CCS:
- Original CCS: Uses human-written answers, not LLM generations
- CCS*: Trains binary classifier using LLM generations without membership estimation

Following HaloScope paper description:
"CCS does not leverage LLM generations but instead human-written answers, 
and does not involve a membership estimation process. For a fair comparison, 
we implemented an improved version CCS*, which trains the binary classifier 
using the LLM generations."

Implementation: Extract hidden states from LLM-generated answers and train
an unsupervised CCS probe to detect hallucinations based on consistency patterns.
"""

import os
import sys
import torch
import numpy as np
import pickle
import argparse
import json
import pandas as pd
import re
from pathlib import Path

from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict

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
from ccs.ccs_probe import CCSHallucinationProbe
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B', 
                       choices=['llama2_7B', 'llama3_2_1B', 'opt_6_7b', 'opt_1_3b'])
    parser.add_argument('--dataset_name', type=str, default='armenian',
                       choices=['armenian', 'basque', 'tigrinya'])
    parser.add_argument('--num_gene', type=int, default=10, help='Number of generations per question for sampling')
    parser.add_argument('--gene', type=int, default=0, help='Generate responses (1) or analyze existing (0)')
    parser.add_argument('--generate_gt', type=int, default=0, help='Generate ground truth labels')
    parser.add_argument('--use_rouge', type=int, default=0, help='Use ROUGE instead of BLEURT')
    parser.add_argument('--thres_gt', type=float, default=0.5, help='Ground truth threshold')
    parser.add_argument('--most_likely', type=int, default=1, help='Use greedy sampling for main response')
    parser.add_argument('--layer', type=int, default=-1, help='Which layer to extract hidden states from')
    parser.add_argument('--all_layers', type=int, default=0, help='Extract from all layers')
    parser.add_argument('--token_idx', type=int, default=-1, help='Which token position to use')
    parser.add_argument('--ccs_nepochs', type=int, default=1000, help='CCS training epochs')
    parser.add_argument('--ccs_ntries', type=int, default=10, help='CCS random restarts')
    parser.add_argument('--ccs_lr', type=float, default=1e-3, help='CCS learning rate')
    parser.add_argument('--ccs_batch_size', type=int, default=-1, help='CCS batch size')
    parser.add_argument('--linear_probe', type=int, default=1, help='Use linear probe vs MLP')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
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
    os.makedirs(f'{args.output_dir}/ccs', exist_ok=True)
    os.makedirs(f'{args.output_dir}/ccs/{args.dataset_name}', exist_ok=True)
    os.makedirs(f'{args.output_dir}/ccs/{args.dataset_name}/answers', exist_ok=True)
    os.makedirs(f'{args.output_dir}/ccs/{args.dataset_name}/hidden_states', exist_ok=True)
    
    print(f"="*70)
    print(f"CCS* (Contrast-Consistent Search) - HaloScope Baseline")
    print(f"="*70)
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {'Generation' if args.gene else 'Ground Truth' if args.generate_gt else 'Analysis'}")
    print(f"\nMethod: CCS* uses LLM-generated response pairs (greedy vs sampled)")
    print(f"        to train an unsupervised probe for hallucination detection.")
    print(f"="*70)
    
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
    
    # Combine all data for processing
    all_data = train_data + val_data + test_data
    
    if args.gene:
        # Generation phase - generate main responses and samples
        generate_responses(all_data, args, gpu_config)
    elif args.generate_gt:
        # Ground truth generation phase  
        generate_ground_truth(all_data, args, gpu_config)
    else:
        # Analysis phase - run CCS
        run_ccs_analysis(all_data, args)

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

def generate_responses(dataset, args, gpu_config):
    """
    Generate model responses for CCS* (HaloScope Section 4)
    
    Following the experimental setup:
    1. Main response: Greedy decoding (beam search, most likely output)
    2. Sampled responses: Stochastic sampling with temperature for diversity
    
    These form the contrast pairs for CCS* training:
    - Consistent outputs (greedy ≈ sampled) → Low hallucination
    - Inconsistent outputs (greedy ≠ sampled) → High hallucination
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
    
    print(f"Generating responses with {args.model_name} for CCS...")
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
            prompt = safe_tokenize_for_opt(tokenizer, prompt_text, language).to(primary_device)
            # Create attention mask to avoid warnings
            attention_mask = torch.ones_like(prompt).to(primary_device)
        else:
            prompt_inputs = tokenizer(prompt_text, return_tensors='pt', padding=True, truncation=True)
            prompt = prompt_inputs['input_ids'].to(primary_device)
            attention_mask = prompt_inputs['attention_mask'].to(primary_device)
        
        # Generate main response (most likely) 
        main_answer = None
        cleanup_gpu_memory()
        
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
            f'{args.output_dir}/ccs/{args.dataset_name}/answers/main_answer_{args.model_name}_{args.dataset_name}_index_{i}.npy',
            [main_answer]
        )
        
        # Save sampled answers
        np.save(
            f'{args.output_dir}/ccs/{args.dataset_name}/answers/sampled_answers_{args.model_name}_{args.dataset_name}_index_{i}.npy',
            sampled_answers
        )
        
    print(f"Generated responses saved to {args.output_dir}/ccs/{args.dataset_name}/answers/")

def generate_ground_truth(dataset, args, gpu_config):
    """
    Generate ground truth labels using BLEURT similarity scoring
    """
    print("Generating ground truth labels for CCS...")
    
    if args.use_rouge:
        # Use ROUGE scoring
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        gts = []
        for i in tqdm(range(len(dataset)), desc="Computing GT with ROUGE"):
            try:
                main_answer = np.load(
                    f'{args.output_dir}/ccs/{args.dataset_name}/answers/main_answer_{args.model_name}_{args.dataset_name}_index_{i}.npy'
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
                    f'{args.output_dir}/ccs/{args.dataset_name}/answers/main_answer_{args.model_name}_{args.dataset_name}_index_{i}.npy'
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
    filename = f'{args.output_dir}/ccs/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_{suffix}.npy'
    np.save(filename, gts)
    print(f"Ground truth scores saved to {filename}")

def extract_hidden_states(dataset, model, tokenizer, args):
    """
    Extract hidden states for contrast pairs (main vs sampled responses)
    """
    print("Extracting hidden states for CCS...")
    language = dataset_to_language(args.dataset_name)
    
    # Determine device
    if args.gpu_id is not None:
        primary_device = torch.device("cuda:0")
    else:
        primary_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define layer hooks for different models
    if 'llama' in args.model_name.lower():
        if args.all_layers:
            LAYERS = [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
        else:
            layer_idx = args.layer if args.layer >= 0 else model.config.num_hidden_layers + args.layer
            LAYERS = [f"model.layers.{layer_idx}"]
    elif 'opt' in args.model_name.lower():
        if args.all_layers:
            LAYERS = [f"model.decoder.layers.{i}" for i in range(model.config.num_hidden_layers)]
        else:
            layer_idx = args.layer if args.layer >= 0 else model.config.num_hidden_layers + args.layer
            LAYERS = [f"model.decoder.layers.{layer_idx}"]
    
    main_states = []
    sampled_states = []
    
    for i in tqdm(range(len(dataset)), desc="Extracting hidden states"):
        question = dataset[i]['question']
        
        # Load generated answers
        main_answer = np.load(
            f'{args.output_dir}/ccs/{args.dataset_name}/answers/main_answer_{args.model_name}_{args.dataset_name}_index_{i}.npy'
        )[0]
        
        sampled_answers = np.load(
            f'{args.output_dir}/ccs/{args.dataset_name}/answers/sampled_answers_{args.model_name}_{args.dataset_name}_index_{i}.npy'
        )
        
        # Extract hidden states for main response
        if args.dataset_name == 'tigrinya' and 'context' in dataset[i]:
            context = dataset[i]['context']
            if len(context) > 1000:
                context = context[:1000] + "..."
            prompt_text = "Concisely answer the following question based on the information in the given passage: \n" + \
                " Passage: " + context + " \n Q: " + question + " \n A: " + str(main_answer)
        else:
            prompt_text = f"Answer the question concisely. Q: {question}" + " A: " + str(main_answer)
        
        prompt_text = preprocess_text_for_language(prompt_text, language)

        # Use safe tokenization for OPT models
        if 'opt' in args.model_name.lower():
            prompt = safe_tokenize_for_opt(tokenizer, prompt_text, language).to(primary_device)
        else:
            prompt = tokenizer(prompt_text, return_tensors='pt').input_ids.to(primary_device)
        
        with torch.no_grad():
            with TraceDict(model, LAYERS) as ret:
                output = model(prompt, output_hidden_states=True)
            
            # Extract and combine layer representations
            layer_states = []
            for layer in LAYERS:
                layer_output = ret[layer].output
                # Handle tuple output (transformers layers can return tuples)
                if isinstance(layer_output, tuple):
                    layer_output = layer_output[0]
                hidden_state = layer_output.squeeze().detach().cpu()
                # Get the last token representation
                if len(hidden_state.shape) == 2:  # [seq_len, hidden_dim]
                    token_state = hidden_state[args.token_idx, :]  # Last token by default
                else:  # [hidden_dim] if single token
                    token_state = hidden_state
                layer_states.append(token_state)
            
            # Concatenate or average layer states
            if len(layer_states) == 1:
                main_state = layer_states[0].numpy()
            else:
                main_state = torch.stack(layer_states, dim=0).mean(dim=0).numpy()
            
            main_states.append(main_state)
        
        # Extract hidden states for sampled responses (use first sample for contrast)
        # In CCS, we need pairs, so we'll use main vs first sampled response
        if len(sampled_answers) > 0:
            sampled_answer = str(sampled_answers[0])
            
            if args.dataset_name == 'tigrinya' and 'context' in dataset[i]:
                context = dataset[i]['context']
                if len(context) > 1000:
                    context = context[:1000] + "..."
                prompt_text = "Concisely answer the following question based on the information in the given passage: \n" + \
                    " Passage: " + context + " \n Q: " + question + " \n A: " + sampled_answer
            else:
                prompt_text = f"Answer the question concisely. Q: {question}" + " A: " + sampled_answer
            
            prompt_text = preprocess_text_for_language(prompt_text, language)

            if 'opt' in args.model_name.lower():
                prompt = safe_tokenize_for_opt(tokenizer, prompt_text, language).to(primary_device)
            else:
                prompt = tokenizer(prompt_text, return_tensors='pt').input_ids.to(primary_device)
            
            with torch.no_grad():
                with TraceDict(model, LAYERS) as ret:
                    output = model(prompt, output_hidden_states=True)
                
                # Extract and combine layer representations
                layer_states = []
                for layer in LAYERS:
                    layer_output = ret[layer].output
                    # Handle tuple output (transformers layers can return tuples)
                    if isinstance(layer_output, tuple):
                        layer_output = layer_output[0]
                    hidden_state = layer_output.squeeze().detach().cpu()
                    # Get the last token representation
                    if len(hidden_state.shape) == 2:  # [seq_len, hidden_dim]
                        token_state = hidden_state[args.token_idx, :]  # Last token by default
                    else:  # [hidden_dim] if single token
                        token_state = hidden_state
                    layer_states.append(token_state)
                
                # Concatenate or average layer states
                if len(layer_states) == 1:
                    sampled_state = layer_states[0].numpy()
                else:
                    sampled_state = torch.stack(layer_states, dim=0).mean(dim=0).numpy()
                
                sampled_states.append(sampled_state)
        else:
            # If no sampled answers, use same as main (will result in low inconsistency)
            sampled_states.append(main_state)
    
    main_states = np.array(main_states)
    sampled_states = np.array(sampled_states)
    
    # Save hidden states
    main_file = f'{args.output_dir}/ccs/{args.dataset_name}/hidden_states/main_states_{args.model_name}_{args.dataset_name}_layer_{args.layer}.npy'
    sampled_file = f'{args.output_dir}/ccs/{args.dataset_name}/hidden_states/sampled_states_{args.model_name}_{args.dataset_name}_layer_{args.layer}.npy'
    
    np.save(main_file, main_states)
    np.save(sampled_file, sampled_states)
    
    print(f"Hidden states saved to {args.output_dir}/ccs/{args.dataset_name}/hidden_states/")
    print(f"Main states shape: {main_states.shape}")
    print(f"Sampled states shape: {sampled_states.shape}")
    
    return main_states, sampled_states

def run_ccs_analysis(dataset, args):
    """
    Run CCS* analysis for hallucination detection (HaloScope baseline)
    
    Following HaloScope paper Section 4 experimental setup:
    1. Extract hidden states from LLM-generated response pairs (greedy vs sampled)
    2. Train unsupervised CCS probe on contrast pairs without labels
    3. Use inconsistency score to detect hallucinations at test time
    
    Key hyperparameters (matching paper):
    - nepochs=1000: Training epochs for CCS probe
    - ntries=10: Random restarts for robust training
    - lr=1e-3: Learning rate
    - linear_probe=True: Use linear probe (simpler than MLP)
    """
    print("Running CCS* analysis...")
    
    # Load model and tokenizer
    MODEL = resolve_model_path(args.model_name)
    
    # Determine device setup
    if args.gpu_id is not None:
        primary_device = torch.device("cuda:0")  # After CUDA_VISIBLE_DEVICES is set
    else:
        primary_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if 'opt' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16
        ).to(primary_device)
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
    
    # Extract hidden states
    main_states, sampled_states = extract_hidden_states(dataset, model, tokenizer, args)
    
    # Load ground truth labels
    print("Loading ground truth labels...")
    try:
        if args.use_rouge:
            gts = np.load(f'{args.output_dir}/ccs/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_rouge_score.npy')
            print("Loaded ROUGE-based ground truth scores")
        else:
            gts = np.load(f'{args.output_dir}/ccs/{args.dataset_name}/gt_{args.model_name}_{args.dataset_name}_bleurt_score.npy')
            print("Loaded BLEURT-based ground truth scores")
    except FileNotFoundError as e:
        print(f"Ground truth file not found: {e}")
        print("Please run ground truth generation first with --generate_gt 1")
        return
        
    # Create binary labels
    gt_label = np.asarray(gts > args.thres_gt, dtype=np.int32)
    
    # Split data for training and testing (following original CCS methodology)
    n_samples = len(main_states)
    train_size = n_samples // 2
    
    # Training set (unlabeled)
    main_train = main_states[:train_size]
    sampled_train = sampled_states[:train_size]
    
    # Test set
    main_test = main_states[train_size:]
    sampled_test = sampled_states[train_size:]
    gt_test = gt_label[train_size:]
    
    print(f"Training CCS with {train_size} samples...")
    print(f"Testing on {len(gt_test)} samples...")
    
    # Initialize and train CCS probe
    device = primary_device
    ccs = CCSHallucinationProbe(
        main_train, sampled_train,
        nepochs=args.ccs_nepochs,
        ntries=args.ccs_ntries,
        lr=args.ccs_lr,
        batch_size=args.ccs_batch_size,
        verbose=True,
        device=device,
        linear=bool(args.linear_probe),
        weight_decay=0.01,
        var_normalize=False
    )
    
    print("Training CCS probe...")
    best_loss = ccs.repeated_train()
    print(f"Best training loss: {best_loss:.6f}")
    
    # Evaluate on test set
    print("Evaluating CCS...")
    hallucination_scores = ccs.get_hallucination_scores(main_test, sampled_test)
    
    # Evaluate performance
    if len(gt_test[gt_test == 1]) > 0 and len(gt_test[gt_test == 0]) > 0:
        measures = get_measures(hallucination_scores[gt_test == 1], hallucination_scores[gt_test == 0])
        auroc, auprc, fpr95 = measures
        print(f"\nCCS Results for {args.dataset_name} ({args.model_name}):")
        print(f"AUROC: {auroc:.4f}")
        print_measures(auroc, auprc, fpr95, 'CCS*')
    else:
        auroc = 0.5
        auprc = 0.5
        fpr95 = 1.0
        print("Warning: Ground truth has only one class, AUROC set to 0.5")
    
    # Save results
    results = {
        'auroc': auroc,
        'auprc': auprc,
        'fpr95': fpr95,
        'scores': hallucination_scores.tolist(),
        'gt_labels': gt_test.tolist(),
        'dataset': args.dataset_name,
        'model': args.model_name,
        'method': 'ccs_star',
        'layer': args.layer,
        'best_loss': best_loss,
        'threshold': args.thres_gt,
        'ccs_params': {
            'nepochs': args.ccs_nepochs,
            'ntries': args.ccs_ntries,
            'lr': args.ccs_lr,
            'linear_probe': bool(args.linear_probe)
        }
    }
    
    results_file = f"{args.output_dir}/ccs/ccs_results_{args.dataset_name}_{args.model_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    # Save scores for further analysis
    scores_file = f"{args.output_dir}/ccs/{args.dataset_name}/scores_{args.model_name}_{args.dataset_name}.npy"
    np.save(scores_file, hallucination_scores)
    print(f"Scores saved to {scores_file}")

if __name__ == '__main__':
    main()
