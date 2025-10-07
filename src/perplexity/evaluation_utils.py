import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score
)
from typing import Dict, Any, List, Tuple
import json
import os

def calculate_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    method_name: str = "RMD"
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for hallucination detection.
    
    Args:
        y_true: True binary labels (1 for hallucination, 0 for non-hallucination)
        y_pred: Predicted binary labels
        y_scores: Prediction scores (continuous)
        method_name: Name of the method for reporting
        
    Returns:
        Dictionary with all evaluation metrics
    """
    # Handle edge cases
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    if len(unique_true) < 2:
        print(f"\n{'='*60}")
        print(f"Warning: Only one class present in y_true.")
        print(f"True labels present: {unique_true}")
        print(f"Distribution: {np.bincount(y_true.astype(int))}")
        print(f"Cannot calculate metrics requiring both classes:")
        print(f"  - AUC-ROC (requires positive and negative samples)")
        print(f"  - AUC-PR (requires positive and negative samples)")
        print(f"  - Precision (requires true positives or false positives)")
        print(f"  - Recall (requires true positives or false negatives)")
        print(f"  - F1-Score (requires both precision and recall)")
        print(f"{'='*60}\n")
        
        # Calculate basic stats that we can
        tn = fp = fn = tp = 0
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except ValueError:
            # Handle case where confusion matrix can't be calculated
            # Calculate manually
            for i in range(len(y_true)):
                if y_true[i] == 1 and y_pred[i] == 1:
                    tp += 1
                elif y_true[i] == 0 and y_pred[i] == 0:
                    tn += 1
                elif y_true[i] == 0 and y_pred[i] == 1:
                    fp += 1
                elif y_true[i] == 1 and y_pred[i] == 0:
                    fn += 1
        
        # Return default metrics with all required keys
        return {
            'method': method_name,
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'auc_roc': None,
            'auc_pr': None,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'specificity': 0.0,
            'false_positive_rate': 0.0,
            'true_negative_rate': 0.0,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(y_true),
            'positive_samples': int(np.sum(y_true)),
            'negative_samples': int(len(y_true) - np.sum(y_true))
        }
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC AUC
    try:
        auc_roc = roc_auc_score(y_true, y_scores)
    except ValueError as e:
        print(f"Warning: Could not calculate ROC AUC: {e}")
        auc_roc = np.nan
    
    # Precision-Recall AUC
    try:
        auc_pr = average_precision_score(y_true, y_scores)
    except ValueError as e:
        print(f"Warning: Could not calculate PR AUC: {e}")
        auc_pr = np.nan
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'method': method_name,
        'accuracy': float(accuracy),
        'auc_roc': float(auc_roc) if not np.isnan(auc_roc) else None,
        'auc_pr': float(auc_pr) if not np.isnan(auc_pr) else None,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'false_positive_rate': float(false_positive_rate),
        'true_negative_rate': float(true_negative_rate),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_samples': len(y_true),
        'positive_samples': int(np.sum(y_true)),
        'negative_samples': int(len(y_true) - np.sum(y_true))
    }

def create_ground_truth_labels(
    results_df: pd.DataFrame,
    method: str = "rmd_threshold"
) -> np.ndarray:
    """
    Create ground truth labels for evaluation.
    Since we don't have explicit ground truth, we use various heuristics.
    
    Args:
        results_df: Results DataFrame
        method: Method to create ground truth labels
        
    Returns:
        Binary ground truth array
    """
    if method == "rmd_threshold":
        # Use RMD score distribution to create labels
        # Assume top 30% of RMD scores are hallucinations
        threshold = np.percentile(results_df['rmd_score'], 70)
        return (results_df['rmd_score'] > threshold).astype(int)
        
    elif method == "confidence_threshold":
        # Use confidence scores - low confidence = hallucination
        threshold = np.percentile(results_df['rmd_confidence'], 30)
        return (results_df['rmd_confidence'] < threshold).astype(int)
        
    elif method == "distance_ratio":
        # Use ratio of in-domain to out-domain distance
        ratio = results_df['out_domain_distance'] / (results_df['in_domain_distance'] + 1e-8)
        threshold = np.percentile(ratio, 70)
        return (ratio > threshold).astype(int)
        
    else:
        # Default: use the model's predictions
        return results_df['rmd_prediction'].astype(int)

def calculate_curve_data(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    method_name: str = "RMD"
) -> Dict[str, Any]:
    """
    Calculate curve data without plotting.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        method_name: Method name
        
    Returns:
        Dictionary with curve data
    """
    if len(np.unique(y_true)) < 2:
        return {'error': 'Only one class present - cannot calculate curves'}
    
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        # ROC Curve data
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        auc_roc = roc_auc_score(y_true, y_scores)
        
        # Precision-Recall Curve data
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        auc_pr = average_precision_score(y_true, y_scores)
        
        return {
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist(),
                'auc': float(auc_roc)
            },
            'pr_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist(),
                'auc': float(auc_pr)
            }
        }
        
    except Exception as e:
        return {'error': f'Error calculating curves: {e}'}

def save_evaluation_results(
    results_df: pd.DataFrame,
    evaluation_metrics: Dict[str, Any],
    output_dir: str,
    model_name: str,
    dataset_name: str,
    method_name: str = "RMD"
):
    """
    Save comprehensive evaluation results to files.
    
    Args:
        results_df: Results DataFrame
        evaluation_metrics: Dictionary of evaluation metrics
        output_dir: Output directory
        model_name: Model name
        dataset_name: Dataset name
        method_name: Method name
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results CSV (already done in main)
    
    # Save evaluation metrics as JSON
    metrics_filename = f"{method_name.lower()}_{model_name}_{dataset_name}_metrics.json"
    metrics_path = os.path.join(output_dir, metrics_filename)
    
    with open(metrics_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    
    print(f"Evaluation metrics saved to: {metrics_path}")
    
    # Save summary statistics as text
    summary_filename = f"{method_name.lower()}_{model_name}_{dataset_name}_summary.txt"
    summary_path = os.path.join(output_dir, summary_filename)
    
    with open(summary_path, 'w') as f:
        f.write(f"Hallucination Detection Evaluation Summary\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total Samples: {evaluation_metrics['total_samples']}\n\n")
        
        f.write(f"Dataset Distribution:\n")
        f.write(f"- Positive samples (hallucinations): {evaluation_metrics['positive_samples']}\n")
        f.write(f"- Negative samples (non-hallucinations): {evaluation_metrics['negative_samples']}\n")
        f.write(f"- Positive rate: {evaluation_metrics['positive_samples']/evaluation_metrics['total_samples']:.2%}\n\n")
        
        f.write(f"Performance Metrics:\n")
        f.write(f"- Accuracy: {evaluation_metrics['accuracy']:.4f}\n")
        if evaluation_metrics['auc_roc'] is not None:
            f.write(f"- AUC-ROC: {evaluation_metrics['auc_roc']:.4f}\n")
        if evaluation_metrics['auc_pr'] is not None:
            f.write(f"- AUC-PR: {evaluation_metrics['auc_pr']:.4f}\n")
        f.write(f"- Precision: {evaluation_metrics['precision']:.4f}\n")
        f.write(f"- Recall: {evaluation_metrics['recall']:.4f}\n")
        f.write(f"- F1-Score: {evaluation_metrics['f1_score']:.4f}\n")
        f.write(f"- Specificity: {evaluation_metrics['specificity']:.4f}\n\n")
        
        f.write(f"Confusion Matrix:\n")
        f.write(f"- True Positives: {evaluation_metrics['true_positives']}\n")
        f.write(f"- True Negatives: {evaluation_metrics['true_negatives']}\n")
        f.write(f"- False Positives: {evaluation_metrics['false_positives']}\n")
        f.write(f"- False Negatives: {evaluation_metrics['false_negatives']}\n\n")
        
        # Add method-specific statistics
        if 'rmd_score' in results_df.columns:
            f.write(f"RMD Score Statistics:\n")
            f.write(f"- Mean: {results_df['rmd_score'].mean():.4f}\n")
            f.write(f"- Std: {results_df['rmd_score'].std():.4f}\n")
            f.write(f"- Min: {results_df['rmd_score'].min():.4f}\n")
            f.write(f"- Max: {results_df['rmd_score'].max():.4f}\n")
            f.write(f"- Median: {results_df['rmd_score'].median():.4f}\n\n")
        
        if 'rmd_confidence' in results_df.columns:
            f.write(f"RMD Confidence Statistics:\n")
            f.write(f"- Mean: {results_df['rmd_confidence'].mean():.4f}\n")
            f.write(f"- Std: {results_df['rmd_confidence'].std():.4f}\n")
            f.write(f"- Min: {results_df['rmd_confidence'].min():.4f}\n")
            f.write(f"- Max: {results_df['rmd_confidence'].max():.4f}\n")
    
    print(f"Summary statistics saved to: {summary_path}")

def comprehensive_evaluation(
    results_df: pd.DataFrame,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    method_name: str = "RMD"
) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation of hallucination detection results.
    
    Args:
        results_df: Results DataFrame
        output_dir: Output directory
        model_name: Model name
        dataset_name: Dataset name
        method_name: Method name
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nPerforming comprehensive evaluation...")
    
    # Since we don't have ground truth labels, we'll create pseudo-ground truth
    # using multiple methods and report metrics for each
    
    evaluation_results = {}
    
    # Method 1: Use RMD threshold-based ground truth
    try:
        y_true_rmd = create_ground_truth_labels(results_df, "rmd_threshold")
        y_pred = results_df['rmd_prediction'].astype(int)
        y_scores = results_df['rmd_score'].values
        
        metrics_rmd = calculate_evaluation_metrics(
            y_true_rmd, y_pred, y_scores, f"{method_name}_rmd_threshold"
        )
        evaluation_results['rmd_threshold'] = metrics_rmd
        
        # Calculate curve data for RMD threshold method
        curve_data = calculate_curve_data(y_true_rmd, y_scores, f"{method_name}_RMD")
        metrics_rmd['curve_data'] = curve_data
        
    except Exception as e:
        print(f"Warning: Could not evaluate with RMD threshold method: {e}")
    
    # Method 2: Use confidence-based ground truth
    try:
        y_true_conf = create_ground_truth_labels(results_df, "confidence_threshold")
        y_pred = results_df['rmd_prediction'].astype(int)
        y_scores = results_df['rmd_score'].values
        
        metrics_conf = calculate_evaluation_metrics(
            y_true_conf, y_pred, y_scores, f"{method_name}_confidence_threshold"
        )
        evaluation_results['confidence_threshold'] = metrics_conf
        
    except Exception as e:
        print(f"Warning: Could not evaluate with confidence threshold method: {e}")
    
    # Method 3: Use distance ratio ground truth
    try:
        y_true_ratio = create_ground_truth_labels(results_df, "distance_ratio")
        y_pred = results_df['rmd_prediction'].astype(int)
        y_scores = results_df['rmd_score'].values
        
        metrics_ratio = calculate_evaluation_metrics(
            y_true_ratio, y_pred, y_scores, f"{method_name}_distance_ratio"
        )
        evaluation_results['distance_ratio'] = metrics_ratio
        
    except Exception as e:
        print(f"Warning: Could not evaluate with distance ratio method: {e}")
    
    # Save all evaluation results
    if evaluation_results:
        # Use the first available evaluation as primary
        primary_eval = list(evaluation_results.values())[0]
        
        save_evaluation_results(
            results_df, primary_eval, output_dir, model_name, dataset_name, method_name
        )
        
        # Save all evaluation methods
        all_evals_path = os.path.join(output_dir, f"all_evaluations_{model_name}_{dataset_name}.json")
        with open(all_evals_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"All evaluation methods saved to: {all_evals_path}")
        
        return primary_eval
    else:
        print("Warning: No evaluation methods succeeded")
        return {}