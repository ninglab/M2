i
port torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class SNBR(nn.Module):
    def __init__(self, config, numItems, device, weights):
        super(SNBR, self).__init__()

        self.dim     = config.dim
        self.itemEmb = nn.Embedding(numItems+1, self.dim, padding_idx=config.padIdx).to(device)
        self.out     = nn.Linear(self.dim, numItems)

        self.his_embds = nn.Linear(numItems, self.dim)
        self.gate_his  = nn.Linear(self.dim, 1)
        self.gate_trans= nn.Linear(self.dim, 1)

        if config.isPreTrain:
            with torch.no_grad():
                self.out.weight.copy_(weights)

    def forward(self, x, decay, offset, his, isEval):
        #x:    3d: batch_size * max_num_seq * max_num_bas
        #embs: 4d: batch_size * max_num_seq * max_num_bas * dim
        #decay:3d: batch_size * max_num_bas * 1
        #his:  2d: batch_size * numItems
 
        embs   = self.itemEmb(x)

        if not isEval:
            embs = F.dropout(embs)

        embs3d = decay * embs.sum(2)
        embs2d = torch.tanh(embs3d.sum(1))

        scores_trans = self.out(embs2d)
        scores_trans = F.softmax(scores_trans, dim=-1)

        scores = scores_trans

        embs_his = self.his_embds(his)
        gate = torch.sigmoid(self.gate_his(embs_his) + self.gate_trans(embs2d))

        scores = gate * scores_trans + (1-gate) * his
        return scores, gate
