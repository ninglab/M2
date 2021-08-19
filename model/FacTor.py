import torch
import torch.nn as nn
import torch.nn.functional as F

class FacTor(nn.Module):
    def __init__(self, numItems, dim, device):
        super(FacTor, self).__init__()

        self.outEmbs = nn.Embedding(numItems, dim).to(device)
    
    def forward(self, rowIndex, colIndex):
        rows = self.outEmbs(rowIndex)
        cols = self.outEmbs(colIndex)

        res = (rows * cols).sum(-1)

        return res
