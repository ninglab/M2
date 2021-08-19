import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class RNBR(nn.Module):
    def __init__(self, config, numItems, device, weights):
        super(RNBR,self).__init__()

        self.dim = config.dim
        self.itemEmb = nn.Embedding(numItems+1, config.dim, padding_idx=config.padIdx).to(device)

        #gru with 1 layer
        self.gru = nn.GRU(config.dim, self.dim, 1)
        self.out = nn.Linear(self.dim, numItems)
        
        self.his_embds = nn.Linear(numItems, self.dim)
        self.gate_his  = nn.Linear(self.dim, 1)
        self.gate_trans= nn.Linear(self.dim, 1)

        if config.isPreTrain:
            with torch.no_grad():
                self.out.weight.copy_(weights)

    def forward(self, x, offset, lenX, his, isEval):
        embs = self.itemEmb(x)

        if not isEval:
            embs = F.dropout(embs)
        
        #need to count for the padded items
        embs3d = embs.mean(2)
        embs3d = embs3d * offset

        #grus require input of timeStamp * batch * dim
        embs3d = embs3d.permute(1,0,2)

        #hn is not valid after pad
        h, _ = self.gru(embs3d)
        h = h.permute(1,0,2)

        hActure = torch.bmm(lenX.unsqueeze(1), h).squeeze(1)
        scores_trans = self.out(hActure)
        scores_trans = F.softmax(scores_trans, -1)

        scores = scores_trans

        ##embs_his = self.his_embds(his)
        ##gate = torch.sigmoid(self.gate_his(embs_his) + self.gate_trans(hActure))

        ##scores = gate * scores_trans + (1-gate) * his

        return scores
