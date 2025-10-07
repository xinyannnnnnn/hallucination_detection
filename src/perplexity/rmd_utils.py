import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import subprocess
from typing import List, Dict, Tuple, Any
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class RMDCalculator:
    """
    Relative Mahalanobis Distance calculator for hallucination detection.
    Based on the approach described in the perplexity paper.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.in_domain_stats = None
        self.out_domain_stats = None
        self.fitted = False
    
    def extract_embeddings(
        self,
        model,
        tokenizer,
        texts: List[str],
        layer_idx: int = -1,
        language: str = None
    ) -> np.ndarray:
        """
        Extract embeddings from model hidden states with language-specific preprocessing.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            texts: List of input texts
            layer_idx: Which layer to extract from (-1 for last)
            language: Language for preprocessing (optional)
            
        Returns:
            Embeddings array of shape (n_texts, embedding_dim)
        """
        from tokenization_utils import preprocess_text_for_language
        
        embeddings = []
        
        for text in tqdm(texts, desc="Extracting embeddings", leave=False):
            # Apply language-specific preprocessing if specified
            if language:
                text = preprocess_text_for_language(text, language)
            
            # Tokenize
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx]  # Shape: (batch, seq_len, hidden_dim)
                
                # Average pool over sequence length
                embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def fit_gaussian_distributions(
        self,
        in_domain_embeddings: np.ndarray,
        out_domain_embeddings: np.ndarray = None
    ):
        """
        Fit Gaussian distributions to in-domain and out-of-domain embeddings.
        
        Args:
            in_domain_embeddings: In-domain embedding vectors
            out_domain_embeddings: Out-of-domain embedding vectors (optional)
        """
        # Fit in-domain distribution
        in_mean = np.mean(in_domain_embeddings, axis=0)
        in_cov = EmpiricalCovariance().fit(in_domain_embeddings)
        
        self.in_domain_stats = {
            'mean': in_mean,
            'precision': in_cov.precision_,
            'covariance': in_cov.covariance_
        }
        
        # Fit out-of-domain distribution (if provided)
        if out_domain_embeddings is not None:
            out_mean = np.mean(out_domain_embeddings, axis=0)
            out_cov = EmpiricalCovariance().fit(out_domain_embeddings)
            
            self.out_domain_stats = {
                'mean': out_mean,
                'precision': out_cov.precision_, 
                'covariance': out_cov.covariance_
            }
        else:
            # Use general/background distribution as approximation
            # This is a simplified approach when we don't have out-domain data
            self.out_domain_stats = {
                'mean': np.zeros_like(in_mean),
                'precision': np.eye(len(in_mean)) * 0.1,  # Weak precision
                'covariance': np.eye(len(in_mean)) * 10.0  # High variance
            }
        
        self.fitted = True
    
    def calculate_mahalanobis_distance(
        self,
        embeddings: np.ndarray,
        mean: np.ndarray,
        precision: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Mahalanobis distance for given embeddings.
        
        Args:
            embeddings: Input embeddings
            mean: Distribution mean
            precision: Precision matrix (inverse covariance)
            
        Returns:
            Mahalanobis distances
        """
        diff = embeddings - mean
        distances = np.sum(diff @ precision * diff, axis=1)
        return distances
    
    def calculate_rmd_scores(self, test_embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate RMD scores for test embeddings.
        
        Args:
            test_embeddings: Test embedding vectors
            
        Returns:
            Dictionary with RMD scores and component distances
        """
        if not self.fitted:
            raise ValueError("Must fit distributions first using fit_gaussian_distributions()")
        
        # Calculate Mahalanobis distance to in-domain distribution
        in_distances = self.calculate_mahalanobis_distance(
            test_embeddings,
            self.in_domain_stats['mean'],
            self.in_domain_stats['precision']
        )
        
        # Calculate Mahalanobis distance to out-of-domain distribution  
        out_distances = self.calculate_mahalanobis_distance(
            test_embeddings,
            self.out_domain_stats['mean'],
            self.out_domain_stats['precision']
        )
        
        # RMD score = distance_out - distance_in
        # Higher RMD means farther from in-domain, closer to out-domain (more likely hallucination)
        rmd_scores = out_distances - in_distances
        
        return {
            'rmd_scores': rmd_scores,
            'in_domain_distances': in_distances,
            'out_domain_distances': out_distances
        }
    
    def predict_hallucination(
        self,
        test_embeddings: np.ndarray,
        threshold: float = None
    ) -> Dict[str, Any]:
        """
        Predict hallucination based on RMD scores.
        
        Args:
            test_embeddings: Test embedding vectors
            threshold: RMD threshold for hallucination detection
                      If None, uses median of RMD scores (dynamic threshold)
            
        Returns:
            Dictionary with predictions and scores
        """
        scores = self.calculate_rmd_scores(test_embeddings)
        rmd_scores = scores['rmd_scores']
        
        # Use dynamic threshold if not specified
        # This ensures both classes are present for AUROC calculation
        if threshold is None:
            threshold = np.median(rmd_scores)
            print(f"Using dynamic threshold (median): {threshold:.4f}")
        
        # Predict hallucination (higher RMD = more likely hallucination)
        predictions = rmd_scores > threshold
        
        # Calculate confidence (normalized distance from threshold)
        confidence = np.abs(rmd_scores - threshold) / (np.std(rmd_scores) + 1e-8)
        confidence = np.clip(confidence, 0, 1)
        
        return {
            'predictions': predictions,
            'rmd_scores': rmd_scores,
            'confidence': confidence,
            **scores
        }

def prepare_embeddings_from_dataset(
    dataset,
    model,
    tokenizer,
    device: torch.device,
    dataset_name: str,
    max_samples: int = None,
    layer_idx: int = -1
) -> Tuple[List[str], np.ndarray]:
    """
    Prepare embeddings from dataset for RMD calculation.
    
    Args:
        dataset: The dataset
        model: Language model
        tokenizer: Tokenizer
        device: Torch device
        dataset_name: Dataset name
        max_samples: Maximum samples to process
        layer_idx: Layer index to extract embeddings from
        
    Returns:
        Tuple of (texts, embeddings)
    """
    # Determine context key
    if dataset_name == 'armenian':
        context_key = 'context' if 'context' in dataset.column_names else 'paragraph'
    else:
        context_key = 'context'
    
    # Limit samples
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Prepare texts for embedding extraction
    texts = []
    for sample in tqdm(dataset, desc="Preparing texts", leave=False):
        context = sample.get(context_key, '')
        question = sample.get('question', '')
        
        # Format as QA prompt
        text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        texts.append(text)
    
    # Get language for preprocessing
    from tokenization_utils import dataset_to_language
    language = dataset_to_language(dataset_name)
    
    # Extract embeddings
    rmd_calc = RMDCalculator(device)
    embeddings = rmd_calc.extract_embeddings(model, tokenizer, texts, layer_idx, language)
    
    return texts, embeddings

def generate_and_evaluate_responses(
    dataset,
    model,
    tokenizer,
    device: torch.device,
    dataset_name: str,
    bleurt_threshold: float = 0.5,
    output_dir: str = "results/perplexity"
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Generate model responses and evaluate them with BLEURT to create ground truth labels.
    Uses the same strategy as hallushift with language-specific preprocessing.
    
    Args:
        dataset: The dataset samples
        model: Language model
        tokenizer: Tokenizer
        device: Device
        dataset_name: Dataset name
        bleurt_threshold: BLEURT threshold for hallucination detection
        output_dir: Output directory for temporary files
        
    Returns:
        Tuple of (response_data, bleurt_df with ground truth labels)
    """
    from tokenization_utils import preprocess_text_for_language, dataset_to_language
    
    # Get REPO_ROOT from the main module's path structure
    from pathlib import Path
    CURRENT_DIR = Path(__file__).resolve().parent
    SRC_DIR = CURRENT_DIR.parent
    REPO_ROOT = SRC_DIR.parent
    
    # Determine context key based on dataset structure (works with list of dicts)
    context_key = 'context'
    if dataset_name == 'armenian' and len(dataset) > 0:
        # Check first sample to determine if we use 'context' or 'paragraph'
        if 'paragraph' in dataset[0] and 'context' not in dataset[0]:
            context_key = 'paragraph'
    
    # Get language for preprocessing
    language = dataset_to_language(dataset_name)
    print(f"Generating responses for {len(dataset)} samples with language: {language}")
    
    response_data = []
    references = []
    candidates = []
    ids = []
    
    for idx, sample in enumerate(tqdm(dataset, desc="Generating responses", disable=False)):
        if idx % 50 == 0:  # Print progress every 50 samples
            print(f"Processing sample {idx+1}/{len(dataset)}")
        
        context = sample.get(context_key, '')
        question = sample.get('question', '')
        reference_answer = sample.get('answer', sample.get('answers', ''))
        
        # Apply language-specific preprocessing
        context = preprocess_text_for_language(context, language)
        question = preprocess_text_for_language(question, language)
        
        # Create prompt
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1500)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Extract generated answer using proper token-level slicing
        generated_answer = tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        
        # Check if reference answer is valid before processing
        sample_id = str(sample.get('id', idx))
        if reference_answer is None or str(reference_answer).strip() == "" or str(reference_answer) == "None":
            print(f"Warning: Skipping sample {sample_id} due to empty reference answer")
            continue
        
        # Store data
        response_data.append({
            'id': sample_id,
            'context': context,
            'question': question,
            'generated_answer': generated_answer,
            'reference_answer': reference_answer,
            'prompt_with_answer': f"{prompt} {generated_answer}"
        })
        
        # Prepare for BLEURT evaluation - handle newlines to prevent file mismatch
        clean_reference = str(reference_answer).replace('\n', ' ').replace('\r', ' ').strip()
        clean_candidate = str(generated_answer).replace('\n', ' ').replace('\r', ' ').strip()
        
        references.append(clean_reference)
        candidates.append(clean_candidate)
        ids.append(sample_id)
    
    print("Running BLEURT evaluation...")
    
    # Validate we have data for BLEURT
    if not references or not candidates or not ids:
        raise ValueError("No valid samples found for BLEURT evaluation. All reference answers may be empty.")
    
    if len(references) != len(candidates) or len(references) != len(ids):
        raise ValueError(f"Mismatch in BLEURT input lengths: refs={len(references)}, cands={len(candidates)}, ids={len(ids)}")
    
    # Try using bleurt_pytorch first (preferred method used by other scripts)
    bleurt_scores = None
    try:
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
        from multi_gpu_utils import compute_bleurt_score_multi_gpu
        
        # Load BLEURT model
        bleurt_dir = REPO_ROOT / "models" / "BLEURT-20"
        print(f"Loading BLEURT model from {bleurt_dir}...")
        
        bleurt_model = BleurtForSequenceClassification.from_pretrained(str(bleurt_dir))
        bleurt_model.to(device)
        bleurt_tokenizer = BleurtTokenizer.from_pretrained(str(bleurt_dir))
        bleurt_model.eval()
        
        print("Computing BLEURT scores with bleurt_pytorch...")
        bleurt_scores = compute_bleurt_score_multi_gpu(bleurt_model, bleurt_tokenizer, candidates, references, device)
        
        print("BLEURT scoring completed successfully with bleurt_pytorch")
        
    except (ImportError, Exception) as e:
        print(f"BLEURT (bleurt_pytorch) not available: {e}")
        print("Falling back to ROUGE scoring...")
        
        # Fallback to ROUGE scoring
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            
            bleurt_scores = []
            for ref, cand in zip(references, candidates):
                score = scorer.score(ref, cand)['rougeL'].fmeasure
                bleurt_scores.append(score)
            
            bleurt_scores = np.array(bleurt_scores)
            print("ROUGE scoring completed successfully")
            
        except ImportError as rouge_error:
            print(f"ROUGE also not available: {rouge_error}")
            raise RuntimeError("Both BLEURT and ROUGE are unavailable. Please install one of them.")
    
    # Create DataFrame with BLEURT/ROUGE scores and hallucination labels
    bleurt_df = pd.DataFrame({
        'id': ids,
        'bleurt_score': bleurt_scores,
        'hallucination': (bleurt_scores < bleurt_threshold).astype(int)
    })
    
    print(f"Generated {len(response_data)} responses")
    print(f"Evaluation complete. Hallucination rate: {bleurt_df['hallucination'].mean():.2%}")
    
    return response_data, bleurt_df

def bleurt_processing(file1, file2, threshold=0.5):
    """
    Processes BLEURT scores to detect hallucinations (same as hallushift).
    
    Args:
        file1 (str): Path to the file containing IDs.
        file2 (str): Path to the file containing BLEURT scores.
        threshold (float): The threshold for BLEURT score.
        
    Returns:
        pandas.DataFrame: DataFrame with 'id', 'bleurt_score', and 'hallucination' columns.
    """
    try:
        with open(file1, 'r', encoding='utf-8') as f3:
            column1 = [line.strip() for line in f3.readlines()]
        with open(file2, 'r', encoding='utf-8') as f4:
            column2 = [line.strip() for line in f4.readlines()]

        if len(column1) == len(column2):
            df = pd.DataFrame({
                'id': column1,
                'bleurt_score': column2
            })
            df = df.groupby('id', as_index=False, sort=False)['bleurt_score'].max()
            df['hallucination'] = df['bleurt_score'].astype(float).apply(lambda x: 0 if x > threshold else 1)
            return df
        else:
            raise ValueError("All columns are not of same length during bleurt processing")
    except Exception as e:
        raise ValueError(f"An error occurred while bleurt processing: {e}")

def prepare_embeddings_with_labels(
    response_data: List[Dict],
    bleurt_df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    layer_idx: int = -1,
    language: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare embeddings with ground truth labels from BLEURT evaluation.
    
    Args:
        response_data: Generated response data
        bleurt_df: DataFrame with BLEURT scores and hallucination labels
        model: Language model
        tokenizer: Tokenizer
        device: Device
        layer_idx: Layer index to extract embeddings from
        
    Returns:
        Tuple of (all_embeddings, hallucination_labels, bleurt_scores)
    """
    # Create ID to label mapping
    id_to_label = dict(zip(bleurt_df['id'].astype(str), bleurt_df['hallucination']))
    id_to_score = dict(zip(bleurt_df['id'].astype(str), bleurt_df['bleurt_score'].astype(float)))
    
    # Extract embeddings for all responses
    texts = [item['prompt_with_answer'] for item in response_data]
    
    rmd_calc = RMDCalculator(device)
    embeddings = rmd_calc.extract_embeddings(model, tokenizer, texts, layer_idx, language)
    
    # Get labels for each sample
    labels = []
    scores = []
    for item in tqdm(response_data, desc="Preparing labels", leave=False):
        sample_id = str(item['id'])
        label = id_to_label.get(sample_id, 0)  # Default to non-hallucination
        score = id_to_score.get(sample_id, 0.5)  # Default score
        labels.append(label)
        scores.append(score)
    
    return embeddings, np.array(labels), np.array(scores)
