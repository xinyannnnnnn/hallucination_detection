#!/usr/bin/env python3
"""
CCS Probe implementation for hallucination detection
Based on methods/CCS/utils.py but adapted for hallucination detection

Implements the Contrast-Consistent Search (CCS) method for discovering
latent knowledge in language models without supervision, specifically
adapted for hallucination detection tasks.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class MLPProbe(nn.Module):
    """
    Multi-layer perceptron probe for CCS
    """
    def __init__(self, d, hidden_dim=100):
        super().__init__()
        self.linear1 = nn.Linear(d, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)

class CCSHallucinationProbe:
    """
    CCS* method for hallucination detection (HaloScope baseline)
    
    Key adaptation from original CCS:
    - Original CCS: Uses human-written statement/negation pairs
    - CCS*: Uses LLM-generated response pairs (greedy vs sampled)
    
    Contrast pairs:
    - x0: Hidden states from main responses (greedy/beam search - most likely)
    - x1: Hidden states from sampled responses (stochastic sampling)
    
    Core insight: 
    - If the model internally "knows" the answer, greedy and sampled outputs 
      should be consistent (low inconsistency score)
    - If the model is hallucinating, outputs will be inconsistent 
      (high inconsistency score)
    
    This matches the CCS* baseline described in HaloScope paper (Section 4):
    "We implemented an improved version CCS*, which trains the binary classifier 
    using the LLM generations"
    """
    
    def __init__(self, x0, x1, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, 
                 var_normalize=False):
        """
        Initialize CCS probe for hallucination detection
        
        Args:
            x0: Hidden states from main responses [n_samples, hidden_dim]
            x1: Hidden states from sampled responses [n_samples, hidden_dim]  
            nepochs: Number of training epochs
            ntries: Number of random restarts
            lr: Learning rate
            batch_size: Batch size (-1 for full batch)
            verbose: Print training progress
            device: Device to use
            linear: Use linear probe vs MLP
            weight_decay: Weight decay for optimizer
            var_normalize: Normalize by standard deviation
        """
        # data
        self.var_normalize = var_normalize
        self.x0 = self.normalize(x0)  # main responses
        self.x1 = self.normalize(x1)  # sampled responses
        self.d = self.x0.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # probe
        self.linear = linear
        self.probe = self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)
        
    def initialize_probe(self):
        """Initialize the probe network"""
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device)
        return self.probe

    def normalize(self, x):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        normalized_x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            normalized_x /= normalized_x.std(axis=0, keepdims=True)
        return normalized_x
        
    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        return x0, x1

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        
        For hallucination detection:
        - Informative loss: Ensures the probe gives confident predictions
        - Consistent loss: Ensures main and sampled responses have consistent truth values
          when they should be consistent (non-hallucinated)
        """
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss

    def get_hallucination_scores(self, x0_test, x1_test):
        """
        Computes hallucination scores for the given test inputs
        Higher scores indicate higher likelihood of hallucination
        """
        x0 = torch.tensor(self.normalize(x0_test), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.normalize(x1_test), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        
        # Inconsistency score: how much the two responses disagree
        # Higher inconsistency suggests hallucination
        inconsistency_score = torch.abs(p0 - (1-p1)).detach().cpu().numpy().flatten()
        return inconsistency_score
    
    def get_acc(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        Adapted for hallucination detection evaluation
        """
        x0 = torch.tensor(self.normalize(x0_test), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.normalize(x1_test), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        
        # For hallucination detection, we use inconsistency as the signal
        inconsistency_scores = torch.abs(p0 - (1-p1)).detach().cpu().numpy().flatten()
        
        # Convert to binary predictions (threshold at median)
        threshold = np.median(inconsistency_scores)
        predictions = (inconsistency_scores > threshold).astype(int)
        
        # Calculate accuracy (take max of acc and 1-acc for label flipping)
        acc = (predictions == y_test).mean()
        acc = max(acc, 1 - acc)
        
        return acc
        
    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]
            
                # probe
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

                # get the corresponding loss
                loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        return loss.detach().cpu().item()
    
    def repeated_train(self):
        """
        Train with multiple random restarts and keep the best result
        """
        best_loss = np.inf
        for train_num in range(self.ntries):
            if self.verbose:
                print(f"Training attempt {train_num + 1}/{self.ntries}")
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss
                if self.verbose:
                    print(f"New best loss: {best_loss:.6f}")

        return best_loss