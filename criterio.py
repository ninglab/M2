import torch

def BPR(target, neg):
   
    diff = target-neg
    diff[diff==0.0] = float('Inf') 

    loss = -torch.log(torch.sigmoid(diff)+ 1e-8)
    #the average loss for each basket
    loss = loss.sum(-1).mean(-1)

    return loss

def Bernoulli(groud, scores):
    neg_ll  = -(torch.log(scores) * groud).sum(-1).mean()

    return neg_ll
