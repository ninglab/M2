import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class FREQP(nn.Module):
    def __init__(self, numItems, device):
        super(FREQP, self).__init__()
        #self.global_freq is the vector contains the learned global frequency for each item
        #self.alf is a scaler to combine the personalized frequency and learned global frequency

        #here dim represents the embedding for the hidden layer in gates
        self.dim = config.dim

        self.global_freq = nn.Embedding(1, numItems).to(device)
        self.input = torch.LongTensor([0]).to(device)
        self.soft = nn.Softmax(dim=1)

        self.his_embds    = nn.Linear(numItems, self.dim)
        self.global_embds = nn.Linear(numItems, self.dim)
        self.gate_his     = nn.Linear(self.dim, 1)
        self.gate_global  = nn.Linear(self.dim, 1)


    def forward(self, his):
        global_freq = self.global_freq(self.input)
        global_freq_soft = self.soft(global_freq).reshape(-1)

        embds_his = self.his_embds(his)
        embds_global = self.global_embds(global_freq)

        pdb.set_trace()
        gate = torch.sigmoid(self.gate_his(embds_his) + self.gate_global(embds_global))

        res = gate * his + (1.0-gate) * global_freq_soft
        return res

