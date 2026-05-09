import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ChainedScheduler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from tqdm import tqdm


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """
    Create a learning rate scheduler with linear warmup followed by cosine annealing.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of steps to warm up learning rate
        total_steps: Total number of training steps
    
    Returns:
        scheduler: ChainedScheduler with warmup + cosine annealing
    """
    # Warmup phase: linearly increase LR from 0 to initial_lr
    def warmup_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
    
    # Cosine annealing phase: decrease LR following cosine curve
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=0.0,
    )
    
    # Chain schedulers: warmup then cosine
    scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])
    
    return scheduler


def create_cosine_scheduler_with_warmup(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """
    Create CosineAnnealingLR scheduler with linear warmup.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup phase
        total_epochs: Total number of training epochs
        steps_per_epoch: Number of batches per epoch
    
    Returns:
        scheduler: ChainedScheduler stepping per batch
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    scheduler = create_warmup_scheduler(optimizer, warmup_steps, total_steps)
    
    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, scheduler=None, gradient_accumulation_steps=1):
    """
    Train for one epoch with optional learning rate scheduling.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)
        scaler: GradScaler for mixed precision (optional)
        scheduler: Learning rate scheduler (optional, step per batch)
        gradient_accumulation_steps: Steps to accumulate gradients (for batch size simulation)
    
    Returns:
        avg_loss: Average loss over epoch
    """
    model.train()
    total_loss = 0.0
    
    # Determine if we can use mixed precision (CUDA only)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    for i, batch in enumerate(tqdm(train_loader, desc='Training')):
        batch = batch.to(device)
        
        # Forward pass with mixed precision if available
        if scaler is not None and device_type == 'cuda':
            with torch.amp.autocast(device_type=device_type):
                out = model(batch)
                loss = criterion(out, batch.y)
        else:
            out = model(batch)
            loss = criterion(out, batch.y)
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (i + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            if scaler is not None and device_type == 'cuda':
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            
            # Step scheduler if provided
            if scheduler is not None:
                scheduler.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, eval_loader, criterion, device, find_best_threshold=False):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Neural network model
        eval_loader: Evaluation data loader
        criterion: Loss function
        device: Device (cuda/cpu)
        find_best_threshold: If True, search for the threshold that maximizes F1.
                             Should be True during Validation, False during Testing.
    
    Returns:
        dict with metrics and the best threshold found (or default 0.5).
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in eval_loader: # Loại bỏ tqdm ở đây để log train gọn gàng hơn
            batch = batch.to(device)
            
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            
            # Tính xác suất (Probability)
            probs = F.softmax(out, dim=1)
            
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Lấy xác suất của class 1 (Fake)
    
    avg_loss = total_loss / len(eval_loader)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    auc = roc_auc_score(all_labels, all_probs)
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    # Logic tìm ngưỡng tối ưu (Chỉ chạy trên tập Validation)
    if find_best_threshold:
        # Thử nghiệm 100 ngưỡng từ 0.01 đến 0.99
        thresholds = np.linspace(0.01, 0.99, 100)
        for t in thresholds:
            preds = (all_probs >= t).astype(int)
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
    
    # Tính toán kết quả cuối cùng với ngưỡng (best_threshold hoặc 0.5)
    final_preds = (all_probs >= best_threshold).astype(int)
    
    accuracy = accuracy_score(all_labels, final_preds)
    precision = precision_score(all_labels, final_preds, zero_division=0)
    recall = recall_score(all_labels, final_preds, zero_division=0)
    f1 = f1_score(all_labels, final_preds, zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'threshold': best_threshold
    }


def train_full_pipeline(model, train_loader, val_loader, test_loader, criterion, 
                       optimizer, device, epochs=80, patience=20, use_mixed_precision=True, scheduler=None, output_dir=None):
    """
    Complete training pipeline with early stopping, best model selection, and learning rate scheduling.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)
        epochs: Maximum epochs to train
        patience: Early stopping patience (# of epochs without improvement)
        use_mixed_precision: Use torch.amp for mixed precision (CUDA only)
        scheduler: Learning rate scheduler (optional, steps per batch)
    
    Returns:
        (best_model, test_metrics)
    """
    # Create GradScaler only for CUDA
    if use_mixed_precision and device.type == 'cuda':
        scaler = torch.amp.GradScaler(device='cuda')
    else:
        scaler = None
    
    best_val_f1 = -1
    best_model_state = None
    patience_count = 0
    
    print(f"\nTraining for up to {epochs} epochs (patience: {patience})...")
    print(f"Device: {device}")
    print(f"Mixed precision: {use_mixed_precision and device.type == 'cuda'}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-" * 80)
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, scheduler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_f1 = val_metrics['f1']
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Test Acc: {test_metrics['accuracy']:.4f}")
        
        # LOGIC LƯU FILE TỰ ĐỘNG
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_count = 0
            print(f"Best model updated (F1: {val_f1:.4f})")
            
            # Nếu có truyền output_dir, lưu file .pth ngay lập tức
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, 'best_model.pth')
                torch.save(best_model_state, save_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\nEarly stopping triggered.")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, evaluate(model, test_loader, criterion, device)
    print("-" * 80)
    print(f"Final Test Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    
    return model, test_metrics


def build_optimizer(model, lr=1e-3, weight_decay=1e-4):
    """
    Build AdamW optimizer with improved hyperparameters.
    
    Args:
        model: Neural network model
        lr: Learning rate (increased to 1e-3 for better convergence)
        weight_decay: L2 regularization weight
    
    Returns:
        optimizer
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    return optimizer


def get_confusion_matrix(model, eval_loader, device):
    """
    Get confusion matrix for analysis.
    
    Args:
        model: Neural network model
        eval_loader: Evaluation data loader
        device: Device (cuda/cpu)
    
    Returns:
        confusion matrix
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            out = model(batch)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    return confusion_matrix(all_labels, all_preds)
