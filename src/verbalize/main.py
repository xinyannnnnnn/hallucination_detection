#!/usr/bin/env python3
"""
Verbalize method implementation for hallucination detection
Based on the HaloScope paper Section 4.1 baseline implementation

Following the exact experimental setup from the HaloScope paper:
- Uses prompting-based strategy to ask models to express confidence in words
- Extracts confidence values (0-100) directly as uncertainty scores
- Implements single sampling approach (no multiple generations needed)
"""

import os, sys
import torch
import numpy as np
import argparse
import json
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure repository paths are available regardless of execution location
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
REPO_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATASETS_DIR = REPO_ROOT / "datasets"
MODELS_DIR = REPO_ROOT / "models"
BLEURT_DIR = MODELS_DIR / "BLEURT-20"

# Import from haloscope implementation
from haloscope.utils import seed_everything, get_measures, print_measures
from tokenization_utils import preprocess_text_for_language, dataset_to_language

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
    parser.add_argument('--num_gene', type=int, default=1, help='Number of generations per question')
    parser.add_argument('--gene', type=int, default=0, help='Generate responses (1) or analyze existing (0)')
    parser.add_argument('--generate_gt', type=int, default=0, help='Generate ground truth labels')
    parser.add_argument('--thres_gt', type=float, default=0.5, help='Ground truth threshold')
    parser.add_argument('--most_likely', type=int, default=1, help='Use greedy sampling')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--gpu_id', type=int, default=None, help='Specific GPU ID to use (0, 1, 2, 3). If set, will use only this GPU')
    
    args = parser.parse_args()
    
    # Setup GPU environment
    if args.gpu_id is not None:
        # Use specific GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        print(f"Using specific GPU {args.gpu_id}")
    
    # Set random seed for reproducibility (same as HaloScope)
    seed_everything(41)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/{args.dataset_name}_verbalize', exist_ok=True)
    os.makedirs(f'{args.output_dir}/{args.dataset_name}_verbalize/responses', exist_ok=True)
    
    print(f"Verbalize Hallucination Detection Evaluation")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {'Generation' if args.gene else 'Analysis'}")
    
    # Check model-dataset compatibility
    if 'llama' in args.model_name.lower() and args.dataset_name == 'tigrinya':
        print("WARNING: LLaMA models have limited support for Tigrinya (Ge'ez script)")
        print("Using English-based prompts for better compatibility.")
        print("Consider using OPT models for better Tigrinya support, or Armenian/Basque datasets for LLaMA.")
    
    # Load dataset with proper splits (using same loader as HaloScope)
    dataset_splits = load_low_resource_dataset(args.dataset_name)
    train_data = dataset_splits['train']
    val_data = dataset_splits['validation'] 
    test_data = dataset_splits['test']
    
    print(f"Loaded {args.dataset_name} dataset:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples") 
    print(f"  Test: {len(test_data)} samples")
    
    # Combine all data for processing (following HaloScope methodology)
    all_data = train_data + val_data + test_data
    
    if args.gene:
        # Generation phase - generate responses and confidence scores
        generate_responses_with_confidence(all_data, args)
    elif args.generate_gt:
        # Ground truth generation phase using BLEURT
        generate_ground_truth(all_data, args)
    else:
        # Analysis phase - evaluate confidence scores
        run_verbalize_analysis_with_splits(dataset_splits, args)


def load_low_resource_dataset(dataset_name):
    """
    Load and format low-resource language datasets with proper train/validation/test splits
    (Same as HaloScope implementation)
    """
    if dataset_name == 'armenian':
        # Load SynDARin Armenian dataset
        armenian_dir = DATASETS_DIR / 'armenian'
        train_df = pd.read_csv(armenian_dir / 'SynDARin_Arm_train.csv')
        test_df = pd.read_csv(armenian_dir / 'SynDARin_Arm_test.csv')
        
        # Convert to standard format
        train_data = []
        for _, row in train_df.iterrows():
            # Armenian dataset uses 'correct_answer' column and includes option numbers
            correct_answer = row.get('correct_answer', '')
            # Clean answer by removing option number (e.g., "1. Տապպուտի" -> "Տապպուտի")
            if correct_answer and '. ' in correct_answer:
                clean_answer = correct_answer.split('. ', 1)[1]
            else:
                clean_answer = correct_answer
                
            train_data.append({
                'question': row['question'],
                'answer': clean_answer,
                'id': f"armenian_train_{len(train_data)}",
                'split': 'train'
            })
            
        test_data = []
        for _, row in test_df.iterrows():
            # Armenian dataset uses 'correct_answer' column and includes option numbers
            correct_answer = row.get('correct_answer', '')
            # Clean answer by removing option number (e.g., "1. Տապպուտի" -> "Տապպուտի")
            if correct_answer and '. ' in correct_answer:
                clean_answer = correct_answer.split('. ', 1)[1]
            else:
                clean_answer = correct_answer
                
            test_data.append({
                'question': row['question'],
                'answer': clean_answer,
                'id': f"armenian_test_{len(test_data)}",
                'split': 'test'
            })
            
        # Create validation split from train data (20% of train)
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
    (Same as HaloScope implementation)
    """
    language = dataset_to_language(dataset_language)

    candidate_texts = []
    # processed_text = preprocess_text_for_language(text, language)
    processed_text = text
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


def create_confidence_prompt(question, answer, dataset_name, model_name):
    """
    Create confidence prompts following HaloScope paper methodology
    
    From the paper:
    - For open-book QA: "Q: [question] A:[answer]. The proposed answer is true with a confidence value (0-100) of ,"
    - For context-based QA: "Context: [Context] Q: [question] A:[answer]. The proposed answer is true with a confidence value (0-100) of ,"
    """
    if dataset_name == 'tigrinya':
        # For LLaMA models with Tigrinya, use English prompts
        if 'llama' in model_name.lower():
            return f"Given the context and question, rate the confidence of this answer from 0-100. Context: [Context] Question: {question} Answer: {answer} Confidence (0-100): "
        else:
            # Context-based prompt for Tigrinya with OPT
            return f"Context: [Context] Q: {question} A: {answer}. \\n The proposed answer is true with a confidence value (0-100) of "
    else:
        # Open-book QA prompt for Armenian and Basque
        return f"Q: {question} A: {answer}. \\n The proposed answer is true with a confidence value (0-100) of "


def generate_responses_with_confidence(dataset, args):
    """
    Generate model responses and confidence scores following Verbalize method
    
    Two-stage process:
    1. Generate answer to question
    2. Ask model for confidence score on the generated answer
    """
    MODEL = resolve_model_path(args.model_name)
    
    # Load model and tokenizer (same configurations as HaloScope)
    if 'opt' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
        if args.gpu_id is not None:
            # For specific GPU, load directly to that device
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.float16
            ).cuda()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.float16
            ).cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        if args.gpu_id is not None:
            # For specific GPU, load directly to that device
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float16
            ).cuda()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float16,
                device_map="auto"
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Generating responses and confidence scores with {args.model_name}...")
    language = dataset_to_language(args.dataset_name)
    
    for i in tqdm(range(len(dataset)), desc="Generating"):
        question = dataset[i]['question']
        
        # Stage 1: Generate answer to question (same as HaloScope)
        if args.dataset_name == 'tigrinya' and 'context' in dataset[i]:
            context = dataset[i]['context']
            if len(context) > 1000:
                context = context[:1000] + "..."
            
            # For LLaMA models with Tigrinya, use English prompts since LLaMA2 doesn't handle Ge'ez script well
            if 'llama' in args.model_name.lower():
                prompt_text = f"Based on the given context, answer the question concisely. Context: {context} Question: {question} Answer:"
            else:
                prompt_text = "Concisely answer the following question based on the information in the given passage: \\n" + \
                    " Passage: " + context + " \\n Q: " + question + " \\n A:"
        else:
            # For LLaMA models with Tigrinya, use simplified English prompts
            if args.dataset_name == 'tigrinya' and 'llama' in args.model_name.lower():
                prompt_text = f"Answer this question: {question} Answer:"
            else:
                prompt_text = f"Answer the question concisely. Q: {question}" + " A:"
        
        # Don't preprocess for better handling of low-resource languages
        # prompt_text = preprocess_text_for_language(prompt_text, language)

        # Generate answer using same tokenization as HaloScope
        if 'opt' in args.model_name.lower():
            prompt = safe_tokenize_for_opt(tokenizer, prompt_text, language).cuda()
            # Create attention mask to avoid warnings
            attention_mask = torch.ones_like(prompt).cuda()
        else:
            prompt_inputs = tokenizer(prompt_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            prompt = prompt_inputs['input_ids'].cuda()
            attention_mask = prompt_inputs['attention_mask'].cuda()
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            if args.most_likely:
                # Use exact parameters from HaloScope
                generated = model.generate(
                    prompt,
                    attention_mask=attention_mask,
                    num_beams=5,
                    num_return_sequences=1,
                    do_sample=False,
                    max_new_tokens=64,
                )
            else:
                generated = model.generate(
                    prompt,
                    attention_mask=attention_mask,
                    do_sample=True,
                    num_return_sequences=1,
                    num_beams=1,
                    max_new_tokens=64,
                    temperature=0.5,
                    top_p=1.0
                )
            
        answer = tokenizer.decode(
            generated[0, prompt.shape[-1]:],
            skip_special_tokens=True
        )
        
        # Clean up answer (same as HaloScope)
        if 'Answer the question concisely' in answer:
            print('#####error')
            print(answer.split('Answer the question concisely')[1])
            print('#####error')
            answer = answer.split('Answer the question concisely')[0]
            
        answer = answer.strip()
        if '\\n' in answer:
            answer = answer.split('\\n')[0]
            
        # Stage 2: Ask for confidence score on the generated answer
        confidence_prompt = create_confidence_prompt(question, answer, args.dataset_name, args.model_name)
        
        # Handle context for Tigrinya dataset
        if args.dataset_name == 'tigrinya' and 'context' in dataset[i]:
            context = dataset[i]['context']
            if len(context) > 1000:
                context = context[:1000] + "..."
            confidence_prompt = confidence_prompt.replace("[Context]", context)
        
        # Don't preprocess for better handling of low-resource languages
        # confidence_prompt = preprocess_text_for_language(confidence_prompt, language)

        # Tokenize confidence prompt
        if 'opt' in args.model_name.lower():
            conf_prompt = safe_tokenize_for_opt(tokenizer, confidence_prompt, language).cuda()
            # Create attention mask to avoid warnings
            conf_attention_mask = torch.ones_like(conf_prompt).cuda()
        else:
            conf_prompt_inputs = tokenizer(confidence_prompt, return_tensors='pt', padding=True, truncation=True)
            conf_prompt = conf_prompt_inputs['input_ids'].cuda()
            conf_attention_mask = conf_prompt_inputs['attention_mask'].cuda()
        
        torch.cuda.empty_cache()
        
        # Generate confidence score
        with torch.no_grad():
            conf_generated = model.generate(
                conf_prompt,
                attention_mask=conf_attention_mask,
                num_beams=1,
                num_return_sequences=1,
                do_sample=False,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id
            )
        
        conf_text = tokenizer.decode(
            conf_generated[0, conf_prompt.shape[-1]:],
            skip_special_tokens=True
        )
        
        # Extract confidence score
        confidence_score = extract_confidence_score(conf_text)
        
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Confidence: {confidence_score}")
        print("---")
        
        # Save results
        result = {
            'question': question,
            'answer': answer,
            'confidence_raw': conf_text,
            'confidence_score': confidence_score,
            'dataset_index': i
        }
        
        # Save individual result
        np.save(
            f'{args.output_dir}/{args.dataset_name}_verbalize/responses/verbalize_{args.model_name}_{args.dataset_name}_response_index_{i}.npy',
            result
        )
        
    print(f"Generated responses and confidence scores saved to {args.output_dir}/{args.dataset_name}_verbalize/responses/")


def extract_confidence_score(confidence_text):
    """
    Extract confidence score (0-100) from model's textual response
    
    The model should generate something like "95" or "95%" or "95 percent"
    """
    # Clean the text
    text = confidence_text.strip().lower()
    
    # Try to extract numeric value using regex
    # Look for patterns like "95", "95%", "95 percent", etc.
    patterns = [
        r'(\d+)%',           # "95%"
        r'(\d+)\s*percent',  # "95 percent"
        r'(\d+)\s*$',        # "95" at end of string
        r'^(\d+)',           # "95" at start of string
        r'(\d+)',            # any number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                score = int(match.group(1))
                # Ensure score is in valid range
                if 0 <= score <= 100:
                    return score
            except ValueError:
                continue
    
    # Fallback: look for confidence-related words (check more specific patterns first)
    if 'very uncertain' in text or 'not confident' in text or 'low confidence' in text:
        return 20
    elif 'uncertain' in text:  # Check uncertain before certain
        return 20
    elif 'very confident' in text or 'high confidence' in text or 'certain' in text:
        return 80
    elif 'medium' in text or 'moderate' in text or 'somewhat confident' in text:
        return 50
    elif 'low' in text:
        return 20
    elif 'high' in text or 'confident' in text:
        return 80
    
    # Default fallback
    return 50


def generate_ground_truth(dataset, args):
    """
    Generate ground truth labels using BLEURT similarity scoring
    Following HaloScope paper methodology
    """
    print("Generating ground truth labels using BLEURT...")
    
    # Load BLEURT model
    try:
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
        
        model = BleurtForSequenceClassification.from_pretrained(str(BLEURT_DIR)).cuda()
        tokenizer_bleurt = BleurtTokenizer.from_pretrained(str(BLEURT_DIR))
        model.eval()
        
        print("BLEURT model loaded successfully")
        
    except Exception as e:
        print(f"Error loading BLEURT model: {e}")
        print(f"Please ensure BLEURT model is available at {BLEURT_DIR}")
        return
    
    gts = []
    for i in tqdm(range(len(dataset)), desc="Computing GT with BLEURT"):
        try:
            result = np.load(
                f'{args.output_dir}/{args.dataset_name}_verbalize/responses/verbalize_{args.model_name}_{args.dataset_name}_response_index_{i}.npy',
                allow_pickle=True
            ).item()
            
            # Get reference answers
            reference = dataset[i]['answer']
            generated_answer = result['answer']
            all_answers = [reference] if isinstance(reference, str) else reference
            
            # Compute BLEURT scores
            predictions = [generated_answer]
            all_results = np.zeros((len(all_answers), len(predictions)))
            
            with torch.no_grad():
                for anw in range(len(all_answers)):
                    inputs = tokenizer_bleurt(predictions, [all_answers[anw]] * len(predictions),
                                           padding='longest', return_tensors='pt')
                    for key in list(inputs.keys()):
                        inputs[key] = inputs[key].cuda()
                    res = np.asarray(model(**inputs).logits.flatten().tolist())
                    all_results[anw] = res
            
            gts.extend(np.max(all_results, axis=0))
            
            if i % 10 == 0:
                print(f"samples passed: {i}")
                
        except FileNotFoundError:
            print(f"Response file not found for index {i}. Run generation first.")
            return
    
    gts = np.array(gts)
    
    # Save ground truth scores
    filename = f'{args.output_dir}/verbalize_{args.dataset_name}_bleurt_score.npy'
    np.save(filename, gts)
    print(f"Ground truth scores saved to {filename}")


def run_verbalize_analysis_with_splits(dataset_splits, args):
    """
    Run Verbalize hallucination detection analysis using actual dataset splits
    """
    print("Running Verbalize analysis with proper dataset splits...")
    
    train_data = dataset_splits['train']
    val_data = dataset_splits['validation'] 
    test_data = dataset_splits['test']
    
    # Combine all data to match file indexing
    all_data = train_data + val_data + test_data
    
    # Load ground truth labels
    print("Loading ground truth labels...")
    try:
        gts = np.load(f'{args.output_dir}/verbalize_{args.dataset_name}_bleurt_score.npy')
        print("Loaded BLEURT-based ground truth scores")
    except FileNotFoundError as e:
        print(f"Ground truth file not found: {e}")
        print("Please run ground truth generation first with --generate_gt 1")
        return
        
    # Create binary labels
    gt_label = np.asarray(gts > args.thres_gt, dtype=np.int32)
    
    # Load confidence scores
    print("Loading confidence scores...")
    confidence_scores = []
    
    for i in range(len(all_data)):
        try:
            result = np.load(
                f'{args.output_dir}/{args.dataset_name}_verbalize/responses/verbalize_{args.model_name}_{args.dataset_name}_response_index_{i}.npy',
                allow_pickle=True
            ).item()
            
            confidence_scores.append(result['confidence_score'])
            
        except FileNotFoundError:
            print(f"Response file not found for index {i}. Run generation first.")
            return
    
    confidence_scores = np.array(confidence_scores)
    
    # Split data according to actual dataset splits
    print("Using actual dataset splits for analysis:")
    print(f"  Training set: {len(train_data)} samples")
    print(f"  Validation set: {len(val_data)} samples")
    print(f"  Test set: {len(test_data)} samples")
    
    # Create index mappings for the splits
    train_indices = list(range(len(train_data)))
    val_indices = list(range(len(train_data), len(train_data) + len(val_data)))
    test_indices = list(range(len(train_data) + len(val_data), len(all_data)))
    
    # Split labels and scores according to actual dataset splits
    gt_label_test = gt_label[test_indices]
    confidence_scores_test = confidence_scores[test_indices]
    
    # Evaluate on test set using confidence scores directly
    # Higher confidence = more likely to be truthful
    # Invert scores for hallucination detection (lower confidence = more likely hallucinated)
    inverted_confidence = 100 - confidence_scores_test  # Convert to hallucination scores
    
    # Ensure we have both classes for evaluation
    if len(gt_label_test[gt_label_test == 1]) > 0 and len(gt_label_test[gt_label_test == 0]) > 0:
        # gt_label_test == 1 means truthful, gt_label_test == 0 means hallucinated
        # inverted_confidence: higher values = more likely hallucinated
        measures = get_measures(inverted_confidence[gt_label_test == 0], inverted_confidence[gt_label_test == 1])
        test_auroc = measures[0]
        test_auprc = measures[1]
        test_fpr95 = measures[2]
    else:
        test_auroc = 0.5
        test_auprc = 0.5
        test_fpr95 = 1.0
        print("Warning: Test set has only one class, metrics set to defaults")
    
    print(f"\\nVerbalize Results for {args.dataset_name} ({args.model_name}):")
    print_measures(test_auroc, test_auprc, test_fpr95, "Verbalize")
    
    # Analysis of confidence distribution
    print(f"\\nConfidence Score Analysis:")
    print(f"  Mean confidence: {confidence_scores_test.mean():.2f}")
    print(f"  Std confidence: {confidence_scores_test.std():.2f}")
    print(f"  Min confidence: {confidence_scores_test.min()}")
    print(f"  Max confidence: {confidence_scores_test.max()}")
    
    print(f"\\nConfidence by truthfulness:")
    truthful_conf = confidence_scores_test[gt_label_test == 1]
    hallucinated_conf = confidence_scores_test[gt_label_test == 0]
    print(f"  Truthful samples - Mean: {truthful_conf.mean():.2f}, Std: {truthful_conf.std():.2f}")
    print(f"  Hallucinated samples - Mean: {hallucinated_conf.mean():.2f}, Std: {hallucinated_conf.std():.2f}")
    
    # Save results
    results = {
        'test_auroc': float(test_auroc),
        'test_auprc': float(test_auprc),
        'test_fpr95': float(test_fpr95),
        'dataset': args.dataset_name,
        'model': args.model_name,
        'method': 'verbalize',
        'confidence_stats': {
            'mean': float(confidence_scores_test.mean()),
            'std': float(confidence_scores_test.std()),
            'min': int(confidence_scores_test.min()),
            'max': int(confidence_scores_test.max()),
            'truthful_mean': float(truthful_conf.mean()),
            'truthful_std': float(truthful_conf.std()),
            'hallucinated_mean': float(hallucinated_conf.mean()),
            'hallucinated_std': float(hallucinated_conf.std())
        }
    }
    
    results_file = f"{args.output_dir}/verbalize_results_{args.dataset_name}_{args.model_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
