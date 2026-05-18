import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for handling class imbalance in fake news detection.
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Formula:
        L_focal = -α * (1 - p_t)^γ * log(p_t)
    
    where:
        p_t = model probability of true class
        α = class weight for minority class (default 0.25)
        γ = focusing parameter (default 2.0)
    
    This loss down-weights easy examples and focuses on hard negatives.
    For fake news detection: helps balance precision/recall tradeoff.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, labels):
        """
        Compute focal loss.
        
        Args:
            logits: Model output (B, num_classes)
            labels: Ground truth class indices (B,)
        
        Returns:
            loss: Scalar loss value
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)  # (B, num_classes)
        
        # Get probability of true class
        p_t = probs.gather(1, labels.view(-1, 1)).squeeze(1)  # (B,)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma  # (B,)
        
        # Compute cross entropy loss (per sample)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # (B,)
        
        # Apply focal weighting and alpha scaling
        focal_loss = self.alpha * focal_weight * ce_loss  # (B,)
        
        # Aggregate
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(torch.nn.Module):
    """
    Weighted Focal Loss with per-class weight support.
    
    Extends FocalLoss to support different alpha/gamma for different classes.
    Per-sample class weights allow dataset-specific fine-tuning.
    
    For PolitiFact: Balanced class_weights=[1.0, 1.0] with α=0.32, γ=2.3
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
    
    def forward(self, logits, labels):
        """
        Compute weighted focal loss with per-class weights.
        
        Args:
            logits: Model output (B, num_classes)
            labels: Ground truth class indices (B,)
        
        Returns:
            loss: Scalar loss value
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)  # (B, num_classes)
        
        # Get probability of true class
        p_t = probs.gather(1, labels.view(-1, 1)).squeeze(1)  # (B,)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma  # (B,)
        
        # Compute cross entropy loss (per sample)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # (B,)
        
        # Apply focal weighting and alpha scaling
        focal_loss = self.alpha * focal_weight * ce_loss  # (B,)
        
        # Apply per-class weights if provided
        if self.class_weights is not None:
            device = logits.device
            class_weights = self.class_weights.to(device)
            per_sample_weights = class_weights[labels]  # (B,)
            focal_loss = focal_loss * per_sample_weights
        
        # Aggregate
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
