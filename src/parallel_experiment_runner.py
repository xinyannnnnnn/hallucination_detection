#!/usr/bin/env python3
"""
Parallel Multi-GPU Experiment Runner
Runs multiple independent experiments in parallel, one per GPU
"""

import os
import sys
import subprocess
import multiprocessing as mp
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Add src directory to path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(SRC_DIR))

def run_single_experiment(experiment_config):
    """
    Run a single experiment on a specific GPU
    
    Args:
        experiment_config: Dictionary containing experiment parameters
    """
    gpu_id = experiment_config['gpu_id']
    method = experiment_config['method']
    dataset = experiment_config['dataset']
    model = experiment_config['model']
    phase = experiment_config['phase']
    additional_args = experiment_config.get('additional_args', [])
    
    # Set CUDA_VISIBLE_DEVICES to use only the specified GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Construct command
    script_path = SRC_DIR / method / "main.py"
    
    if not script_path.exists():
        return {
            'gpu_id': gpu_id,
            'method': method,
            'dataset': dataset,
            'model': model,
            'phase': phase,
            'status': 'failed',
            'error': f"Script not found: {script_path}"
        }
    
    cmd = [
        "python", str(script_path),
        "--dataset_name", dataset,
        "--model_name", model,
        "--num_gpus", "1",  # Use single GPU
        "--disable_multi_gpu",  # Disable multi-GPU mode
        "--output_dir", f"./results_parallel"
    ]
    
    # Add phase-specific arguments
    if phase in ['generate', 'all']:
        cmd.extend(['--gene', '1', '--num_gene', '10'])
    
    if phase in ['ground_truth', 'all']:
        cmd.extend(['--generate_gt', '1'])
    
    # Add method-specific arguments
    if method == 'ln_entropy':
        cmd.extend(['--similarity_threshold', '0.7'])
    elif method == 'eigenscore':
        cmd.extend(['--feature_clipping', '1', '--clipping_threshold', '3.0'])
    
    # Add additional arguments
    cmd.extend(additional_args)
    
    # Use ROUGE for faster processing
    cmd.extend(['--use_rouge', '1'])
    
    print(f"GPU {gpu_id}: Starting {method} on {dataset} with {model}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            env=env,
            capture_output=True, 
            text=True, 
            timeout=3600  # 1 hour timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"GPU {gpu_id}: ✅ Completed successfully in {duration:.1f}s")
            return {
                'gpu_id': gpu_id,
                'method': method,
                'dataset': dataset,
                'model': model,
                'phase': phase,
                'status': 'success',
                'duration': duration,
                'stdout': result.stdout[-500:],  # Last 500 chars
                'stderr': result.stderr[-500:] if result.stderr else None
            }
        else:
            print(f"GPU {gpu_id}: ❌ Failed with return code {result.returncode}")
            return {
                'gpu_id': gpu_id,
                'method': method,
                'dataset': dataset,
                'model': model,
                'phase': phase,
                'status': 'failed',
                'return_code': result.returncode,
                'duration': duration,
                'stdout': result.stdout[-500:],
                'stderr': result.stderr[-500:] if result.stderr else None
            }
            
    except subprocess.TimeoutExpired:
        print(f"GPU {gpu_id}: ⏰ Timeout after 1 hour")
        return {
            'gpu_id': gpu_id,
            'method': method,
            'dataset': dataset,
            'model': model,
            'phase': phase,
            'status': 'timeout',
            'duration': 3600
        }
    except Exception as e:
        print(f"GPU {gpu_id}: 💥 Exception: {e}")
        return {
            'gpu_id': gpu_id,
            'method': method,
            'dataset': dataset,
            'model': model,
            'phase': phase,
            'status': 'exception',
            'error': str(e)
        }

def run_parallel_experiments(experiment_configs, max_workers=None):
    """
    Run multiple experiments in parallel
    
    Args:
        experiment_configs: List of experiment configurations
        max_workers: Maximum number of parallel workers (default: number of GPUs)
    """
    if max_workers is None:
        max_workers = len(experiment_configs)
    
    print(f"Running {len(experiment_configs)} experiments in parallel")
    print(f"Using {max_workers} workers")
    print("=" * 60)
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_config = {
            executor.submit(run_single_experiment, config): config 
            for config in experiment_configs
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Exception in experiment {config}: {e}")
                results.append({
                    'gpu_id': config['gpu_id'],
                    'method': config['method'],
                    'dataset': config['dataset'],
                    'model': config['model'],
                    'phase': config['phase'],
                    'status': 'exception',
                    'error': str(e)
                })
    
    return results

def create_experiment_configs(mode='all_methods', custom_configs=None):
    """
    Create experiment configurations based on mode
    
    Args:
        mode: 'all_methods', 'all_datasets', 'all_models', 'custom'
        custom_configs: List of custom experiment configurations
    """
    if custom_configs:
        return custom_configs
    
    # Available options
    methods = ['lexical_similarity', 'ln_entropy', 'eigenscore']
    datasets = ['armenian', 'basque', 'tigrinya']
    models = ['llama2_7B', 'llama3_2_1B', 'opt_6_7b', 'opt_1_3b']
    phases = ['all']  # Can be 'generate', 'ground_truth', 'analysis', 'all'
    
    configs = []
    
    if mode == 'all_methods':
        # Test all methods on Armenian dataset with LLaMA2-7B
        for i, method in enumerate(methods):
            configs.append({
                'gpu_id': i,
                'method': method,
                'dataset': 'armenian',
                'model': 'llama2_7B',
                'phase': 'all',
                'additional_args': []
            })
    
    elif mode == 'all_datasets':
        # Test lexical_similarity on all datasets with LLaMA2-7B
        for i, dataset in enumerate(datasets):
            configs.append({
                'gpu_id': i,
                'method': 'lexical_similarity',
                'dataset': dataset,
                'model': 'llama2_7B',
                'phase': 'all',
                'additional_args': []
            })
    
    elif mode == 'all_models':
        # Test lexical_similarity on Armenian with all models
        for i, model in enumerate(models):
            configs.append({
                'gpu_id': i,
                'method': 'lexical_similarity',
                'dataset': 'armenian',
                'model': model,
                'phase': 'all',
                'additional_args': []
            })
    
    elif mode == 'comprehensive':
        # Run comprehensive experiments across all combinations
        # This will use more than 4 GPUs, so we'll cycle through them
        gpu_id = 0
        for method in methods:
            for dataset in datasets:
                for model in models:
                    configs.append({
                        'gpu_id': gpu_id % 4,  # Cycle through 4 GPUs
                        'method': method,
                        'dataset': dataset,
                        'model': model,
                        'phase': 'all',
                        'additional_args': []
                    })
                    gpu_id += 1
    
    return configs

def print_results_summary(results):
    """Print a summary of experiment results"""
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    
    total = len(results)
    successful = len([r for r in results if r['status'] == 'success'])
    failed = len([r for r in results if r['status'] == 'failed'])
    timeout = len([r for r in results if r['status'] == 'timeout'])
    exception = len([r for r in results if r['status'] == 'exception'])
    
    print(f"Total experiments: {total}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏰ Timeout: {timeout}")
    print(f"💥 Exception: {exception}")
    
    print(f"\nSuccess rate: {successful/total*100:.1f}%")
    
    # Print individual results
    print("\nDetailed Results:")
    print("-" * 60)
    for result in results:
        status_icon = {
            'success': '✅',
            'failed': '❌',
            'timeout': '⏰',
            'exception': '💥'
        }.get(result['status'], '❓')
        
        duration_str = f" ({result.get('duration', 0):.1f}s)" if 'duration' in result else ""
        print(f"{status_icon} GPU {result['gpu_id']}: {result['method']} on {result['dataset']} with {result['model']}{duration_str}")
        
        if result['status'] != 'success' and 'error' in result:
            print(f"    Error: {result['error']}")
    
    # Save results to file
    results_file = "parallel_experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Run parallel multi-GPU experiments')
    parser.add_argument('--mode', type=str, default='all_methods',
                       choices=['all_methods', 'all_datasets', 'all_models', 'comprehensive'],
                       help='Experiment mode')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum number of parallel workers (default: 4)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='JSON file with custom experiment configurations')
    
    args = parser.parse_args()
    
    print("Parallel Multi-GPU Experiment Runner")
    print("=" * 60)
    
    # Load custom configurations if provided
    if args.config_file:
        with open(args.config_file, 'r') as f:
            custom_configs = json.load(f)
        experiment_configs = custom_configs
        print(f"Loaded {len(experiment_configs)} custom experiments from {args.config_file}")
    else:
        experiment_configs = create_experiment_configs(args.mode)
        print(f"Created {len(experiment_configs)} experiments for mode: {args.mode}")
    
    # Limit to available GPUs
    if len(experiment_configs) > args.max_workers:
        print(f"Warning: {len(experiment_configs)} experiments but only {args.max_workers} workers")
        print("Some experiments will be queued")
    
    # Run experiments
    start_time = time.time()
    results = run_parallel_experiments(experiment_configs, args.max_workers)
    total_time = time.time() - start_time
    
    # Print summary
    print_results_summary(results)
    print(f"\nTotal execution time: {total_time:.1f}s")

if __name__ == '__main__':
    main()
