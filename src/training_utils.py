import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            should_stop = self.counter >= self.patience
            if should_stop and self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return should_stop

def get_scheduler(optimizer, scheduler_type="reduce_lr", **kwargs):
    """Get learning rate scheduler"""
    if scheduler_type == "reduce_lr":
        return ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=kwargs.get('factor', 0.5), 
            patience=kwargs.get('patience', 5),
            min_lr=kwargs.get('min_lr', 1e-6),
            verbose=True
        )
    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.7)
        )
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
