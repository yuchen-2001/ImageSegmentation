import torch.nn as nn
import torch.nn.functional as F
import torch 

# =========================== Define your custom loss function ===========================================
class CustomCombinedLoss(nn.Module):
    def __init__(self):
        super(CustomCombinedLoss, self).__init__()
        self.alpha = 1
        self.gamma = 1
        self.ignore_index = 255
        self.size_average = True

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
