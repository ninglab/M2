import math
import numpy as np

def calRecall(target, pred, k):
    assert len(target) == len(pred)
    sumRecall = 0
    for i in range(len(target)):
        gt = set(target[i])
        rec= set(pred[i][:k])

        if len(gt) == 0:
           print('Error')

        sumRecall += len(gt & rec) / float(len(gt))

    return sumRecall / float(len(target))

def calNDCG(target, pred, k):
    assert len(target) == len(pred)
    sumNDCG = 0
    for i in range(len(target)):
        valK = min(k, len(target[i]))
        gt = set(target[i])
        idcg = calIDCG(valK)
        dcg  = sum([int(pred[i][j] in gt) / math.log(j+2, 2) for j in range(k)])
        sumNDCG += dcg / idcg

    return sumNDCG / float(len(target))

#the gain is 1 for every hit, and 0 otherwise
def calIDCG(k):
    return sum([1.0 / math.log(i+2, 2) for i in range(k)])
