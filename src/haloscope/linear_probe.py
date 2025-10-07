#!/usr/bin/env python3
"""
Linear probe training for HaloScope
Adapted from methods/haloscope/linear_probe.py
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader

class ArrayDataset(Dataset):
    """Simple dataset wrapper for numpy arrays"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class NonLinearClassifier(nn.Module):
    """2-layer MLP classifier as specified in HaloScope paper"""
    def __init__(self, feat_dim, num_classes=2):
        super(NonLinearClassifier, self).__init__()
        self.fc1 = nn.Linear(feat_dim, 1024)  # Hidden layer with 1024 units
        self.fc2 = nn.Linear(1024, 1)         # Output layer
        
    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = self.fc2(x)
        return x

class LinearClassifier(nn.Module):
    """Simple linear classifier"""
    def __init__(self, feat_dim, num_classes=2):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, 1)
        
    def forward(self, features):
        return self.fc(features)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args):
    """Adjust learning rate with cosine annealing"""
    lr = args['learning_rate']
    if args.get('cosine', False):
        eta_min = lr * 0.001  # lr_decay_rate^3 
        lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / args['epochs'])) / 2
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_epoch(train_loader, classifier, optimizer, epoch, args):
    """Train for one epoch"""
    classifier.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    for idx, (features, labels) in enumerate(train_loader):
        features = features.cuda().float()
        labels = labels.cuda().long()
        bsz = labels.shape[0]
        
        optimizer.zero_grad()
        
        # Forward pass
        output = classifier(features)
        loss = F.binary_cross_entropy_with_logits(output.view(-1), labels.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        predictions = (torch.sigmoid(output) > 0.5).long().view(-1)
        correct = predictions.eq(labels).float().sum()
        accuracy = correct / bsz
        
        # Update meters
        losses.update(loss.item(), bsz)
        accuracies.update(accuracy.item(), bsz)
        
        if (idx + 1) % 50 == 0:
            print(f'Epoch [{epoch}][{idx+1}/{len(train_loader)}] '
                  f'Loss: {losses.avg:.4f} Acc: {accuracies.avg:.4f}')
                  
    return losses.avg, accuracies.avg

def validate_epoch(val_loader, classifier):
    """Validate for one epoch"""
    classifier.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.cuda().float()
            labels = labels.cuda().long()
            bsz = labels.shape[0]
            
            # Forward pass
            output = classifier(features)
            loss = F.binary_cross_entropy_with_logits(output.view(-1), labels.float())
            
            # Compute predictions and probabilities
            probs = torch.sigmoid(output)
            predictions = (probs > 0.5).long().view(-1)
            
            # Compute accuracy
            correct = predictions.eq(labels).float().sum()
            accuracy = correct / bsz
            
            # Update meters
            losses.update(loss.item(), bsz)
            accuracies.update(accuracy.item(), bsz)
            
            # Store predictions and labels
            all_preds.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
            
    return losses.avg, accuracies.avg, np.array(all_preds), np.array(all_labels)

def get_linear_acc(ftrain, ltrain, ftest, ltest, n_cls, 
                   epochs=50, learning_rate=0.05, weight_decay=0.0003,
                   batch_size=512, cosine=True, nonlinear=True, 
                   print_ret=True, **kwargs):
    """
    Train linear/non-linear probe following HaloScope paper specifications
    
    Args:
        ftrain: Training features
        ltrain: Training labels  
        ftest: Test features
        ltest: Test labels
        n_cls: Number of classes (unused, always binary)
        epochs: Number of training epochs (default 50)
        learning_rate: Learning rate (default 0.05)
        weight_decay: Weight decay (default 0.0003) 
        batch_size: Batch size (default 512)
        cosine: Use cosine annealing (default True)
        nonlinear: Use 2-layer MLP (default True)
        print_ret: Print training progress
        
    Returns:
        best_acc, final_acc, (classifier, best_state, best_preds, final_preds, labels), train_loss
    """
    
    # Convert labels to binary if needed
    unique_labels = np.unique(ltrain)
    if len(unique_labels) > 2:
        raise ValueError(f"Expected binary classification, got {len(unique_labels)} classes")
        
    label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
    ltrain_mapped = np.array([label_map[l] for l in ltrain])
    ltest_mapped = np.array([label_map[l] for l in ltest])
    
    # Create datasets and loaders
    train_dataset = ArrayDataset(ftrain, ltrain_mapped)
    test_dataset = ArrayDataset(ftest, ltest_mapped)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    feat_dim = ftrain.shape[1]
    if nonlinear:
        classifier = NonLinearClassifier(feat_dim).cuda()
    else:
        classifier = LinearClassifier(feat_dim).cuda()
        
    # Create optimizer (SGD as specified in paper)
    optimizer = optim.SGD(classifier.parameters(), 
                         lr=learning_rate,
                         momentum=0.9,
                         weight_decay=weight_decay)
    
    # Training arguments
    args = {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'cosine': cosine
    }
    
    best_acc = 0.0
    best_state = None
    best_preds = None
    
    if print_ret:
        print(f"Training {'nonlinear' if nonlinear else 'linear'} probe...")
        print(f"Features: {feat_dim}D, Samples: {len(ftrain)} train / {len(ftest)} test")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, args)
        
        # Train
        train_loss, train_acc = train_epoch(train_loader, classifier, optimizer, epoch, args)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(test_loader, classifier)
        
        # Track best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(classifier.state_dict())
            best_preds = val_preds.copy()
            
        if print_ret and epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.4f} "
                  f"Train Acc={train_acc:.4f} Val Acc={val_acc:.4f}")
    
    # Get final predictions
    _, final_acc, final_preds, final_labels = validate_epoch(test_loader, classifier)
    
    if print_ret:
        print(f"Training completed. Best Val Acc: {best_acc:.4f}, Final Acc: {final_acc:.4f}")
    
    return (best_acc, final_acc, 
            (classifier, best_state, best_preds, final_preds, final_labels),
            train_loss)
