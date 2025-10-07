#!/usr/bin/env python3
"""
Multi-GPU utilities for hallucination detection experiments
Provides device management, model loading, and tensor operations for 4-card GPU setup
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from pathlib import Path
import warnings

def setup_multi_gpu():
    """
    Setup multi-GPU environment for distributed training/inference
    Returns the number of available GPUs and current device
    """
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return 0, torch.device('cpu')
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    
    # Set memory allocation strategy for better multi-GPU utilization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    return num_gpus, torch.device('cuda')

def get_device_for_model(model_name, num_gpus=4):
    """
    Determine the best device mapping strategy for different model types
    """
    if num_gpus == 0:
        return None, torch.device('cpu')
    
    # For multi-GPU setups, use device_map="auto" for automatic distribution
    if num_gpus >= 2:
        return "auto", torch.device('cuda:0')  # Primary device for operations
    else:
        return None, torch.device('cuda:0')

def load_model_multi_gpu(model_path, model_name, num_gpus=4, torch_dtype=torch.float16):
    """
    Load model with multi-GPU support
    """
    print(f"Loading {model_name} with {num_gpus} GPU(s)...")
    
    # Check actual available devices after CUDA_VISIBLE_DEVICES
    actual_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Actual CUDA devices available: {actual_device_count}")
    
    # Determine device mapping strategy
    device_map, primary_device = get_device_for_model(model_name, num_gpus)
    
    # Ensure primary_device is valid for current CUDA context
    if torch.cuda.is_available() and actual_device_count > 0:
        # Always use cuda:0 as primary device since it's the first visible device
        primary_device = torch.device('cuda:0')
        print(f"Using primary device: {primary_device}")
        
        # Override device_map if we only have 1 GPU available
        if actual_device_count == 1:
            device_map = None  # Force single GPU mode
            print("Forcing single GPU mode due to CUDA_VISIBLE_DEVICES restriction")
    
    # Handle different model types
    if 'opt' in model_name.lower():
        # OPT-specific loading with multi-GPU support
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        
        model_kwargs = {
            'torch_dtype': torch_dtype,
        }
        
        if device_map:
            model_kwargs['device_map'] = device_map
        else:
            model_kwargs['device_map'] = None
            
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        # If not using device_map, manually move to GPU
        if not device_map:
            model = model.to(primary_device)
            
    else:
        # LLaMA-specific loading with multi-GPU support
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        model_kwargs = {
            'low_cpu_mem_usage': True,
            'torch_dtype': torch_dtype,
        }
        
        if device_map:
            model_kwargs['device_map'] = device_map
        else:
            model_kwargs['device_map'] = None
            
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        # If not using device_map, manually move to GPU
        if not device_map:
            model = model.to(primary_device)
    
    # Set up pad token
    if tokenizer.pad_token is None:
        if 'opt' in model_name.lower():
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure pad_token_id is set correctly
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    print(f"Model loaded successfully on device(s)")
    if device_map == "auto":
        print("Using automatic device mapping across GPUs")
    else:
        print(f"Using single device: {primary_device}")
    
    return model, tokenizer, primary_device

def move_tensor_to_device(tensor, device):
    """
    Safely move tensor to specified device, handling multi-GPU scenarios
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensor.items()}
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)([move_tensor_to_device(t, device) for t in tensor])
    else:
        return tensor

def generate_with_multi_gpu(model, tokenizer, inputs, generation_kwargs, device):
    """
    Generate text with proper multi-GPU handling
    """
    # Ensure inputs are on the correct device
    if isinstance(inputs, dict):
        inputs = move_tensor_to_device(inputs, device)
    else:
        inputs = inputs.to(device)
    
    # Clear GPU cache before generation
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        # Add device-specific generation parameters
        generation_kwargs_copy = generation_kwargs.copy()
        
        # Ensure pad_token_id and eos_token_id are set
        if 'pad_token_id' not in generation_kwargs_copy:
            generation_kwargs_copy['pad_token_id'] = tokenizer.pad_token_id
        if 'eos_token_id' not in generation_kwargs_copy:
            generation_kwargs_copy['eos_token_id'] = tokenizer.eos_token_id
        
        # Generate with proper device handling
        if isinstance(inputs, dict):
            generated = model.generate(**inputs, **generation_kwargs_copy)
        else:
            generated = model.generate(inputs, **generation_kwargs_copy)
    
    return generated

def load_bleurt_multi_gpu(bleurt_dir, num_gpus=4):
    """
    Load BLEURT model with multi-GPU support
    """
    try:
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
        
        # Load tokenizer first
        tokenizer = BleurtTokenizer.from_pretrained(str(bleurt_dir))
        
        # Try multi-GPU first, fall back to single GPU if it fails
        model = None
        model_kwargs = {}
        
        if num_gpus >= 2:
            try:
                # Try multi-GPU loading first
                model_kwargs['device_map'] = "auto"
                model = BleurtForSequenceClassification.from_pretrained(str(bleurt_dir), **model_kwargs)
                print(f"BLEURT model loaded successfully with multi-GPU ({num_gpus} GPUs)")
            except Exception as e:
                print(f"Multi-GPU loading failed: {e}")
                print("Falling back to single GPU mode...")
                model = None
        
        if model is None:
            # Fall back to single GPU mode
            model_kwargs = {'device_map': None}
            model = BleurtForSequenceClassification.from_pretrained(str(bleurt_dir), **model_kwargs)
            model = model.cuda()
            print(f"BLEURT model loaded successfully with single GPU")
            
        model.eval()
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading BLEURT model: {e}")
        return None, None

def compute_bleurt_score_multi_gpu(model, tokenizer, predictions, references, device=None):
    """
    Compute BLEURT scores with multi-GPU support
    """
    if model is None or tokenizer is None:
        return np.zeros(len(predictions))
    
    # Auto-detect device from model if not provided
    if device is None:
        device = next(model.parameters()).device
    
    print(f"Computing BLEURT scores on device: {device}")
    
    scores = []
    batch_size = 8  # Process in batches to manage memory
    
    for i in range(0, len(predictions), batch_size):
        batch_preds = predictions[i:i+batch_size]
        batch_refs = references[i:i+batch_size]
        
        try:
            with torch.no_grad():
                inputs = tokenizer(batch_preds, batch_refs, 
                                 padding='longest', return_tensors='pt')
                
                # Ensure all inputs are on the correct device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                outputs = model(**inputs)
                batch_scores = outputs.logits.cpu().numpy().flatten()
                scores.extend(batch_scores)
                
        except Exception as e:
            print(f"Error computing BLEURT scores for batch {i//batch_size}: {e}")
            # Debug info
            if i == 0:  # Only print debug info for first batch
                print(f"  Model device: {next(model.parameters()).device}")
                print(f"  Target device: {device}")
                if 'inputs' in locals():
                    print(f"  Input devices: {[(k, v.device if isinstance(v, torch.Tensor) else 'N/A') for k, v in inputs.items()]}")
            scores.extend([0.0] * len(batch_preds))
    
    return np.array(scores)

def get_optimal_batch_size(model_name, num_gpus=4):
    """
    Get optimal batch size based on model and GPU configuration
    """
    if num_gpus == 0:
        return 1
    
    # Conservative batch sizes for different model sizes
    if '7B' in model_name or '6.7b' in model_name:
        if num_gpus >= 4:
            return 4
        elif num_gpus >= 2:
            return 2
        else:
            return 1
    elif '1.3b' in model_name or '1B' in model_name:
        if num_gpus >= 4:
            return 8
        elif num_gpus >= 2:
            return 4
        else:
            return 2
    else:
        return 1

def cleanup_gpu_memory():
    """
    Clean up GPU memory after operations
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_gpu_memory_usage():
    """
    Print current GPU memory usage for all GPUs
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("GPU Memory Usage:")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3     # GB
            print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

class MultiGPUConfig:
    """
    Configuration class for multi-GPU experiments
    """
    def __init__(self, num_gpus=4, batch_size=None, model_name='llama2_7B'):
        self.num_gpus = num_gpus
        self.model_name = model_name
        self.batch_size = batch_size or get_optimal_batch_size(model_name, num_gpus)
        self.device_map = "auto" if num_gpus >= 2 else None
        self.primary_device = torch.device('cuda:0') if num_gpus > 0 else torch.device('cpu')
        
    def __str__(self):
        return f"MultiGPUConfig(num_gpus={self.num_gpus}, batch_size={self.batch_size}, device_map={self.device_map})"
