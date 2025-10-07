#!/usr/bin/env python3
"""Self-evaluation baseline for low-resource languages following HaloScope."""

import os
import json
import re
import sys
import ctypes
from pathlib import Path
import argparse


# Work around sandbox environments that forbid POSIX shared memory (shm_open)
# by preloading libgomp before importing torch. This keeps imports deterministic
# even when /dev/shm access is restricted.
_LIBGOMP_PATH = Path('/usr/lib/x86_64-linux-gnu/libgomp.so.1')
if _LIBGOMP_PATH.exists():
    try:
        ctypes.CDLL(str(_LIBGOMP_PATH), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:  # pragma: no cover - best effort safeguard
        print(f"Warning: failed to preload { _LIBGOMP_PATH }: {exc}")


import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / 'src'
HALOSCOPE_ROOT = PROJECT_ROOT / 'methods' / 'haloscope'

if str(HALOSCOPE_ROOT) not in sys.path:
    sys.path.append(str(HALOSCOPE_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

import llama_iti  # noqa: E402  pylint: disable=wrong-import-position
from tokenization_utils import preprocess_text_for_language, dataset_to_language  # noqa: E402  pylint: disable=wrong-import-position


MODELS_ROOT = PROJECT_ROOT / 'models'
DATASETS_ROOT = PROJECT_ROOT / 'datasets'
HF_NAMES = {
    'llama2_7B': MODELS_ROOT / 'llama',
    'llama3_2_1B': MODELS_ROOT / 'Llama-3.2-1B',
    'opt_6_7b': MODELS_ROOT / 'opt',
    'opt_1_3b': MODELS_ROOT / 'opt-1.3b',
}


def language_aware_tokenize(tokenizer, text, language, max_length=512):
    """Tokenize text with language-specific normalization and fallbacks."""

    lang = dataset_to_language(language)
    candidate_texts = []
    processed_text = preprocess_text_for_language(text, lang)
    candidate_texts.append(processed_text)
    if processed_text != text:
        candidate_texts.append(text)

    last_error = None
    for candidate in candidate_texts:
        try:
            return tokenizer(candidate, return_tensors='pt', truncation=True, max_length=max_length)
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            print(f"Language-aware tokenization failed for language={lang}: {exc}")

    if lang not in {'tigrinya', 'armenian'}:
        try:
            ascii_text = ''.join(char if ord(char) < 128 else ' ' for char in processed_text)
            ascii_text = re.sub(r'\s+', ' ', ascii_text).strip()
            return tokenizer(ascii_text, return_tensors='pt', truncation=True, max_length=max_length)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"ASCII fallback failed for language={lang}: {exc}")

    print(f"Falling back to empty prompt (last error: {last_error})")
    return tokenizer("", return_tensors='pt', truncation=True, max_length=max_length)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

class LowResourceDatasetLoader:
    """Load and preprocess low-resource language datasets"""
    
    def __init__(self):
        self.datasets = {}

    @staticmethod
    def _records_to_dataset(records):
        if records:
            return Dataset.from_list(records)

        keys = ['id', 'context', 'question', 'answer', 'language', 'split']
        return Dataset.from_dict({key: [] for key in keys})

    def _ensure_cached(self, dataset_name, loader_fn):
        if dataset_name not in self.datasets:
            self.datasets[dataset_name] = loader_fn()
        return self.datasets[dataset_name]

    def load_armenian_dataset(self):
        """Load Armenian SynDARin dataset with train/validation/test splits."""

        def _load():
            train_path = DATASETS_ROOT / 'armenian' / 'SynDARin_Arm_train.csv'
            test_path = DATASETS_ROOT / 'armenian' / 'SynDARin_Arm_test.csv'

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_records = []
            for _, row in train_df.iterrows():
                train_records.append({
                    'id': f"armenian_train_{len(train_records)}",
                    'context': row.get('paragraph', ''),
                    'question': row['question'],
                    'answer': row['correct_answer'],
                    'language': 'armenian',
                    'split': 'train'
                })

            val_size = len(train_records) // 5
            if val_size > 0:
                val_records = train_records[-val_size:]
                train_records = train_records[:-val_size]
            else:
                val_records = []

            for idx, record in enumerate(val_records):
                record['split'] = 'validation'
                record['id'] = f"armenian_val_{idx}"

            train_records = [
                {**record, 'id': f"armenian_train_{idx}"}
                for idx, record in enumerate(train_records)
            ]

            test_records = []
            for _, row in test_df.iterrows():
                test_records.append({
                    'id': f"armenian_test_{len(test_records)}",
                    'context': row.get('paragraph', ''),
                    'question': row['question'],
                    'answer': row.get('correct_answer', row.get('answer', '')),
                    'language': 'armenian',
                    'split': 'test'
                })

            return {
                'train': self._records_to_dataset(train_records),
                'validation': self._records_to_dataset(val_records),
                'test': self._records_to_dataset(test_records)
            }

        return self._ensure_cached('armenian', _load)

    def load_basque_dataset(self):
        """Load Basque ElkarHizketak dataset with explicit splits."""

        def _load():
            basque_dir = DATASETS_ROOT / 'basque'

            def convert_split(filename, split_name):
                file_path = basque_dir / filename
                if not file_path.exists():
                    raise FileNotFoundError(f"Missing Basque split file: {file_path}")

                df = pd.read_parquet(file_path)
                records = []
                for _, row in df.iterrows():
                    records.append({
                        'id': f"basque_{split_name}_{len(records)}",
                        'context': row.get('context', row.get('prompt', '')),
                        'question': row.get('question', row.get('input', '')),
                        'answer': row.get('answer', row.get('output', '')),
                        'language': 'basque',
                        'split': split_name
                    })
                return self._records_to_dataset(records)

            return {
                'train': convert_split('train-00000-of-00001.parquet', 'train'),
                'validation': convert_split('validation-00000-of-00001.parquet', 'validation'),
                'test': convert_split('test-00000-of-00001.parquet', 'test')
            }

        return self._ensure_cached('basque', _load)

    def load_tigrinya_dataset(self):
        """Load TigQA dataset with train/dev/test splits."""

        def _load():
            tigrinya_dir = DATASETS_ROOT / 'tigrinya'

            def convert_split(filename, split_name):
                with open(tigrinya_dir / filename, 'r', encoding='utf-8') as f:
                    payload = json.load(f)

                records = []
                for article in payload['data']:
                    for paragraph in article['paragraphs']:
                        context = paragraph.get('context', '')
                        for qa in paragraph['qas']:
                            answer_text = ''
                            if qa.get('answers'):
                                answer_text = qa['answers'][0].get('text', '').strip()

                            records.append({
                                'id': f"tigrinya_{split_name}_{len(records)}",
                                'context': context,
                                'question': qa['question'].strip(),
                                'answer': answer_text,
                                'language': 'tigrinya',
                                'split': split_name
                            })

                return self._records_to_dataset(records)

            return {
                'train': convert_split('train.json', 'train'),
                'validation': convert_split('dev.json', 'validation'),
                'test': convert_split('test.json', 'test')
            }

        return self._ensure_cached('tigrinya', _load)

    def load_dataset(self, dataset_name):
        """Load specified dataset with train/validation/test splits."""

        if dataset_name == 'armenian':
            return self.load_armenian_dataset()
        if dataset_name == 'basque':
            return self.load_basque_dataset()
        if dataset_name == 'tigrinya':
            return self.load_tigrinya_dataset()

        raise ValueError(f"Unknown dataset: {dataset_name}")

class SelfEvaluationTemplate:
    """Generate self-evaluation prompts for different languages"""
    
    TEMPLATES = {
        'armenian': {
            'without_context': """Question: {question}
Proposed Answer: {answer}
Is the proposed answer correct?
(A) Yes
(B) No
The proposed answer is""",
            'with_context': """Context: {context}
Question: {question}
Proposed Answer: {answer}
Is the proposed answer correct?
(A) Yes
(B) No
The proposed answer is"""
        },
        'basque': {
            'without_context': """Question: {question}
Proposed Answer: {answer}
Is the proposed answer correct?
(A) Yes
(B) No
The proposed answer is""",
            'with_context': """Context: {context}
Question: {question}
Proposed Answer: {answer}
Is the proposed answer correct?
(A) Yes
(B) No
The proposed answer is"""
        },
        'tigrinya': {
            'without_context': """Question: {question}
Proposed Answer: {answer}
Is the proposed answer correct?
(A) Yes
(B) No
The proposed answer is""",
            'with_context': """Context: {context}
Question: {question}
Proposed Answer: {answer}
Is the proposed answer correct?
(A) Yes
(B) No
The proposed answer is"""
        }
    }
    
    # Token mappings for "True" in each language
    TRUE_TOKENS = {
        'armenian': 'A',  # Option A for "Yes"
        'basque': 'A',    # Option A for "Yes"
        'tigrinya': 'A'   # Option A for "Yes"
    }
    
    def generate_prompt(self, language, question, answer, context=None):
        """Generate self-evaluation prompt for given language"""
        if language not in self.TEMPLATES:
            raise ValueError(f"Language {language} not supported")
        
        template_key = 'with_context' if context else 'without_context'
        template = self.TEMPLATES[language][template_key]
        
        if context:
            return template.format(context=context, question=question, answer=answer)
        else:
            return template.format(question=question, answer=answer)
    
    def get_true_token(self, language):
        """Get the token representing 'True' for the given language"""
        return self.TRUE_TOKENS[language]

class SelfEvaluationExperiment:
    """Main class for running self-evaluation experiments"""
    
    def __init__(self, model_name, dataset_name, model_dir=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        if model_dir:
            self.model_path = Path(model_dir)
        else:
            try:
                self.model_path = Path(HF_NAMES[model_name])
            except KeyError as exc:  # pragma: no cover - defensive branch
                raise ValueError(f"Unknown model name: {model_name}") from exc

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path '{self.model_path}' does not exist")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.dataset_loader = LowResourceDatasetLoader()
        self.template_generator = SelfEvaluationTemplate()
        
        # Will be initialized when needed
        self.tokenizer = None
        self.model = None
        
    def initialize_model(self):
        """Initialize tokenizer and model"""
        print(f"Loading model: {self.model_path}")

        torch_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        device_map = 'auto' if self.device.type == 'cuda' else None
        model_kwargs = {
            'low_cpu_mem_usage': True,
            'torch_dtype': torch_dtype,
        }
        if device_map is not None:
            model_kwargs['device_map'] = device_map

        model_path_str = str(self.model_path)

        if 'opt' in self.model_name:
            # For OPT models, use transformers directly
            from transformers import AutoTokenizer, OPTForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(model_path_str, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = OPTForCausalLM.from_pretrained(model_path_str, **model_kwargs)
        else:
            # For Llama models, use the custom tokenizer shipped with HaloScope utilities
            self.tokenizer = llama_iti.LlamaTokenizer.from_pretrained(model_path_str, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = llama_iti.LlamaForCausalLM.from_pretrained(model_path_str, **model_kwargs)

        if device_map is None:
            self.model.to(self.device)

        self.model.eval()
        
    def generate_initial_answers(self, dataset, num_samples=None):
        """Generate initial answers for questions"""
        if not self.model:
            self.initialize_model()
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        answers = []
        for i, item in enumerate(tqdm(dataset, desc="Generating answers")):
            # Create generation prompt
            if item['context']:
                prompt = f"Answer these questions concisely based on the context: \\n Context: {item['context']} Q: {item['question']} A:"
            else:
                prompt = f"Answer the question concisely. Q: {item['question']} A:"
            
            language = item.get('language', self.dataset_name)
            inputs = language_aware_tokenize(self.tokenizer, prompt, language, max_length=512)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # Generate (greedy sampling following paper)
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=64,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode answer
            answer = self.tokenizer.decode(
                generated[0, input_ids.shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            answers.append({
                'id': item['id'],
                'question': item['question'],
                'context': item['context'],
                'ground_truth': item['answer'],
                'generated_answer': answer,
                'language': item['language']
            })
            
            # Clear GPU cache periodically
            if self.device.type == 'cuda' and i % 5 == 0:
                torch.cuda.empty_cache()
            
        return answers
    
    def run_self_evaluation(self, qa_pairs):
        """Run self-evaluation on question-answer pairs"""
        if not self.model:
            self.initialize_model()
        
        confidence_scores = []
        
        for qa in tqdm(qa_pairs, desc="Running self-evaluation"):
            # Generate self-evaluation prompt
            prompt = self.template_generator.generate_prompt(
                qa['language'],
                qa['question'], 
                qa['generated_answer'],
                qa['context']
            )
            
            inputs = language_aware_tokenize(self.tokenizer, prompt, qa.get('language', self.dataset_name), max_length=512)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Get log probability of "True" token
            true_token = self.template_generator.get_true_token(qa['language'])
            try:
                true_token_id = self.tokenizer.encode(true_token, add_special_tokens=False)[0]
                confidence_score = F.log_softmax(logits, dim=-1)[true_token_id].item()
            except (IndexError, KeyError):
                print(f"Warning: Could not find token '{true_token}' for language {qa['language']}")
                confidence_score = 0.0
            
            confidence_scores.append(confidence_score)
        
        return np.array(confidence_scores)
    
    def evaluate_with_bleurt(self, qa_pairs):
        """Evaluate ground truth using BleuRT"""
        print("Loading BleuRT model for evaluation...")
        try:
            from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
            bleurt_path = MODELS_ROOT / 'BLEURT-20'
            bleurt_model = BleurtForSequenceClassification.from_pretrained(str(bleurt_path))
            bleurt_model.to(self.device)
            bleurt_tokenizer = BleurtTokenizer.from_pretrained(str(bleurt_path)) 
            bleurt_model.eval()
            
            bleurt_scores = []
            
            for qa in tqdm(qa_pairs, desc="Computing BleuRT scores"):
                with torch.no_grad():
                    inputs = bleurt_tokenizer(
                        [qa['generated_answer']], 
                        [qa['ground_truth']], 
                        padding='longest', 
                        return_tensors='pt'
                    )
                    for key in inputs.keys():
                        inputs[key] = inputs[key].to(self.device)
                        
                    score = bleurt_model(**inputs).logits.item()
                    bleurt_scores.append(score)
            
            return np.array(bleurt_scores)
        except ImportError:
            print("BleuRT not available, using simple string similarity as fallback...")
            # Fallback to simple similarity
            scores = []
            for qa in qa_pairs:
                # Simple word overlap similarity as fallback
                gen_words = set(qa['generated_answer'].lower().split())
                gt_words = set(qa['ground_truth'].lower().split()) 
                if len(gen_words | gt_words) == 0:
                    similarity = 0.0
                else:
                    similarity = len(gen_words & gt_words) / len(gen_words | gt_words)
                scores.append(similarity)
            return np.array(scores)
    
    def run_experiment(self, num_samples=None, threshold=0.5):
        """Run complete self-evaluation experiment"""
        print(f"Running self-evaluation experiment on {self.dataset_name}")
        
        # Load dataset with train/validation/test splits
        dataset_splits = self.dataset_loader.load_dataset(self.dataset_name)
        for split_name, split_dataset in dataset_splits.items():
            print(f"Loaded {len(split_dataset)} {split_name} samples")

        non_empty_splits = [ds for ds in dataset_splits.values() if len(ds) > 0]
        if not non_empty_splits:
            raise RuntimeError(f"No data found for dataset '{self.dataset_name}'")

        if len(non_empty_splits) == 1:
            dataset = non_empty_splits[0]
        else:
            dataset = concatenate_datasets(non_empty_splits)

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            print(f"Using {len(dataset)} samples")
        
        # Generate initial answers
        qa_pairs = self.generate_initial_answers(dataset)
        
        # Run self-evaluation
        confidence_scores = self.run_self_evaluation(qa_pairs)
        
        # Evaluate with BleuRT
        bleurt_scores = self.evaluate_with_bleurt(qa_pairs)
        
        # Create ground truth labels (threshold=0.5 following paper)
        gt_labels = (bleurt_scores > threshold).astype(int)
        
        # Calculate AUROC
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(gt_labels)) > 1:
                auroc = roc_auc_score(gt_labels, confidence_scores)
            else:
                print(f"Warning: Only one class present in ground truth (all {gt_labels[0]}). Using accuracy instead.")
                predicted_labels = (confidence_scores > np.median(confidence_scores)).astype(int)
                auroc = np.mean(predicted_labels == gt_labels)
        except ImportError:
            print("sklearn not available, calculating simple accuracy...")
            # Simple accuracy as fallback
            predicted_labels = (confidence_scores > np.median(confidence_scores)).astype(int)
            auroc = np.mean(predicted_labels == gt_labels)
        
        print(f"Self-evaluation AUROC on {self.dataset_name}: {auroc:.4f}")
        
        # Save results
        split_sizes = {split: len(ds) for split, ds in dataset_splits.items()}

        results = {
            'dataset': self.dataset_name,
            'model': self.model_name,
            'auroc': auroc,
            'num_samples': len(qa_pairs),
            'confidence_scores': confidence_scores.tolist(),
            'bleurt_scores': bleurt_scores.tolist(),
            'gt_labels': gt_labels.tolist(),
            'split_sizes': split_sizes
        }
        
        os.makedirs(f'results/self_evaluation', exist_ok=True)
        with open(f'results/self_evaluation/{self.dataset_name}_{self.model_name}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return auroc, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B', 
                       choices=list(HF_NAMES.keys()))
    parser.add_argument('--dataset_name', type=str, required=True,
                       choices=['armenian', 'basque', 'tigrinya'])
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Local directory with model data')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='BleuRT threshold for ground truth (default: 0.5)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    seed_everything(41)
    
    # Run experiment
    experiment = SelfEvaluationExperiment(
        args.model_name, 
        args.dataset_name,
        args.model_dir
    )
    
    auroc, results = experiment.run_experiment(
        num_samples=args.num_samples,
        threshold=args.threshold
    )
    
    print(f"\nFinal Results:")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_name}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Number of samples: {results['num_samples']}")

if __name__ == '__main__':
    main()
