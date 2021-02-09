import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class FREQ(nn.Module):
    def __init__(self, numItems, device):
        super(FREQ, self).__init__()
        #self.global_freq is the vector contains the learned global frequency for each item
        #self.alf is a scaler to combine the personalized frequency and learned global frequency

        self.global_freq = nn.Embedding(1, numItems).to(device)
        self.input = torch.LongTensor([0]).to(device)
        self.alf = nn.Parameter(torch.rand(1)).to(device)
        self.soft = nn.Softmax(dim=1)

    def forward(self, his):
        alf = torch.sigmoid(self.alf)
        global_freq = self.global_freq(self.input)
        global_freq = self.soft(global_freq).reshape(-1)

        res = alf * his + (1.0-alf) * global_freq
        return res

