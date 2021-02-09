import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class FPMC(nn.Module):

    def __init__(self, config, numItems, numUsers, device):
        super(FPMC, self).__init__()

        self.dim = config.dim
        self.EUI = nn.Embedding(numUsers, self.dim).to(device)
        self.EIU = nn.Embedding(numItems+1, self.dim, padding_idx=config.padIdx).to(device)

        self.EIL = nn.Embedding(numItems+1, self.dim, padding_idx=config.padIdx).to(device)
        self.ELI = nn.Embedding(numItems+1, self.dim, padding_idx=config.padIdx).to(device)

    def forward(self, u, x, tar, neg, offset, isEval):
        #for fpmc, the sequences is broken to several pairs (prev, next)
        #we use the prev to predict the next

        eui  = self.EUI(u)

        eTarU = self.EIU(tar)
        if not isEval:
            eNegU = self.EIU(neg)

        eTarL = self.EIL(tar)
        if not isEval:
            eNegL = self.EIL(neg)

        eli = self.ELI(x)

        #MF

        #FMC
        eli = eli.mean(1) * offset
        if not isEval:
            xmfTar = torch.bmm(eui.unsqueeze(1), eTarU.permute(0,2,1)).squeeze()
            xfpTar = torch.bmm(eTarL, eli.unsqueeze(1).permute(0,2,1)).squeeze()
        else:
            xmfTar = eui.mm(eTarU.permute(1,0))
            xfpTar = eTarL.mm(eli.permute(1,0)).permute(1,0)

        #dot = mul and sum over the last dim
        scoreTar = xmfTar + xfpTar

        scoreNeg = 0

        if not isEval:
            xmfNeg = torch.bmm(eui.unsqueeze(1), eNegU.permute(0,2,1)).squeeze()
            xfpNeg = torch.bmm(eNegL, eli.unsqueeze(1).permute(0,2,1)).squeeze()
            scoreNeg = xmfNeg + xfpNeg

        return scoreTar, scoreNeg

      
