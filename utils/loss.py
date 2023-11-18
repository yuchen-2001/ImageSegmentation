import torch.nn as nn
import torch.nn.functional as F
import torch 

# =========================== Define your custom loss function ===========================================
class CustomCombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2, size_average=True, ignore_index=255):
        super(CustomCombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        # CrossEntropyLoss for the main prediction
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        focal_loss = self.alpha * (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss
        ce_loss = self.beta * focal_loss

        # Combine losses
        combined_loss = focal_loss.sum()

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss
