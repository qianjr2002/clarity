import torch.nn as nn
import torch.nn.functional as F

class BinauralLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, est_wav, clean_wav):
        left_loss = F.mse_loss(est_wav[:,0,:], clean_wav[:,0,:])
        right_loss = F.mse_loss(est_wav[:,1,:], clean_wav[:,1,:])
        loss = (left_loss + right_loss) / 2
        return loss