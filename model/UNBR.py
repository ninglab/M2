import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class UNBR(nn.Module):
    def __init__(self, config, numItems, device, weights):
        super(UNBR, self).__init__()

        self.dim     = config.dim
        self.itemEmb = nn.Embedding(numItems+1, self.dim, padding_idx=config.padIdx).to(device)

        self.gru = nn.GRU(config.dim, self.dim, 1)

        self.out     = nn.Linear(2*self.dim, numItems)
        self.his_embds = nn.Linear(numItems, self.dim)
        self.gate_his  = nn.Linear(self.dim, 1)
        self.gate_trans= nn.Linear(2*self.dim, 1)

        if config.isPreTrain:
            with torch.no_grad():
                self.out.weight.copy_(weights)

    def forward(self, x, decay, offset, lenX, his, isEval):
        embs = self.itemEmb(x)

        if not isEval:
            embs = F.dropout(embs)
        
        #latent rep from GRU
        embsRNN = embs.mean(2)
        embsRNN = embsRNN * offset

        embsRNN = embsRNN.permute(1,0,2)
        hRNN, _ = self.gru(embsRNN)
        hRNN = hRNN.permute(1,0,2)

        hRNN = torch.bmm(lenX.unsqueeze(1), hRNN).squeeze(1)

        #latent rep from AE
        embsAE = decay * embs.sum(2)
        hAE = F.tanh(embsAE.sum(1))

        union  = torch.cat((hAE, hRNN), -1)
        scores_trans = self.out(union)
        scores_trans = F.softmax(scores_trans, dim=-1)

        embs_his = self.his_embds(his)
        gate = torch.sigmoid(self.gate_his(embs_his) + self.gate_trans(union))

        scores = gate * scores_trans + (1-gate) * his

        return scores
