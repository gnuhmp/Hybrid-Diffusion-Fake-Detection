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
