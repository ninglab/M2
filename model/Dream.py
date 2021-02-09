import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Dream(nn.Module):
    def __init__(self, config, numItems, device):
        super(Dream, self).__init__()

        self.dim = config.dim

        self.itemEmb = nn.Embedding(numItems+1, self.dim, padding_idx=config.padIdx).to(device)
        self.grus    = nn.GRU(config.dim, self.dim, 1)
        self.outEmb  = nn.Embedding(numItems+1, self.dim, padding_idx=config.padIdx).to(device)
        self.outB    = nn.Embedding(numItems+1, 1, padding_idx=config.padIdx).to(device)

        self.testOut = nn.Linear(self.dim, numItems)

        with torch.no_grad():
            #for the purpose of max pooling
            self.itemEmb.weight[numItems] = -100 * torch.ones(self.dim).to(device)

    def forward(self, x, lenX, tar, neg, isEval):
        #tarAndNeg: matrix batchsize * maxTargetBasLen with padding
        embs = self.itemEmb(x)

        if not isEval:
            embs = F.dropout(embs)

        embs3d = torch.max(embs, 2)[0]

        embs3d = embs3d.permute(1,0,2)
        h, _ = self.grus(embs3d)
        h = h.permute(1,0,2)

        hActure = torch.bmm(lenX.unsqueeze(1), h)

        #calculate BPR loss
        tarEmbs = self.outEmb(tar)

        #calculate the scores
        if not isEval:
            tarB    = self.outB(tar).squeeze(2)
            scoresTar = torch.bmm(hActure, tarEmbs.permute(0,2,1)).squeeze(1)
        else:
            tarB    = self.outB(tar).squeeze(1)
            scoresTar = hActure.squeeze(1).mm(tarEmbs.permute(1,0))

        scoresTar += tarB
        scoresNeg = 0

        if not isEval:
            negEmbs = self.outEmb(neg)
            negB    = self.outB(neg).squeeze(2)
            scoresNeg = torch.bmm(hActure, negEmbs.permute(0,2,1)).squeeze(1)
            scoresNeg += negB

        return scoresTar, scoresNeg
        
