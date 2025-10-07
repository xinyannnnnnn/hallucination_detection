#!/usr/bin/env python3
"""
Utility functions for HaloScope low-resource implementation
Based on methods/haloscope/utils.py and methods/haloscope/metric_utils.py
"""

import os
import random
import numpy as np
import torch
from sklearn import metrics

def seed_everything(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_measures(in_scores, out_scores, plot=False):
    """
    Compute AUROC, AUPRC, and FPR95 metrics
    
    Args:
        in_scores: scores for positive class (hallucinations)
        out_scores: scores for negative class (non-hallucinations) 
        plot: whether to plot ROC curve (unused for now)
    
    Returns:
        Tuple of (auroc, auprc, fpr95)
    """
    if len(in_scores) == 0 or len(out_scores) == 0:
        return 0.5, 0.5, 1.0
        
    # Combine scores and create labels
    scores = np.concatenate([in_scores, out_scores])
    labels = np.concatenate([np.ones(len(in_scores)), np.zeros(len(out_scores))])
    
    # Compute AUROC
    auroc = metrics.roc_auc_score(labels, scores)
    
    # Compute AUPRC 
    auprc = metrics.average_precision_score(labels, scores)
    
    # Compute FPR95
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    
    # Find FPR when TPR is closest to 95%
    target_tpr = 0.95
    tpr_diff = np.abs(tpr - target_tpr)
    idx = np.argmin(tpr_diff)
    fpr95 = fpr[idx]
    
    return auroc, auprc, fpr95

def print_measures(auroc, auprc, fpr95, method_name=''):
    """
    Print evaluation metrics in a formatted way
    """
    print(f"\n{method_name} Results:" if method_name else "Results:")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"FPR95: {fpr95:.4f}")
    
def stable_rank_normalization(scores):
    """
    Normalize scores using rank-based normalization for stability
    """
    if len(scores) == 0:
        return scores
        
    ranks = np.argsort(np.argsort(scores))
    normalized = ranks / (len(scores) - 1)
    return normalized

def compute_calibration_error(predictions, labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Indices of predictions in this bin
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece
