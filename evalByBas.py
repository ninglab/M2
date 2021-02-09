import torch
import numpy as np
import math
from evalMetric import calRecall, calNDCG
from utils import *

def evalByBas(model, trans, config, device):
    
    numBas   = len(trans.test.training)
    numBatch = numBas // config.batchSize + 1
    idxList  = [i for i in range(numBas)]
    Recall = []
    NDCG = []

    targetBatch = np.asarray([i for i in range(trans.numItems)])
    targetBatch= torch.from_numpy(targetBatch).type(torch.LongTensor).to(device)

    for batch in range(numBatch):
        start = config.batchSize * batch
        end   = min(numBas, start+config.batchSize)
 
        batchList  = idxList[start:end]
        trainBatch, groundTruth, userBatch = generateBatchBas(trans, batchList, isEval=1)

        trainBatch, offset = padBasWise(trainBatch, config)
        userBatch  = np.asarray(userBatch)

        trainBatch = torch.from_numpy(trainBatch).type(torch.LongTensor).to(device)
        offset     = torch.from_numpy(offset).type(torch.FloatTensor).to(device)
        userBatch  = torch.from_numpy(userBatch).type(torch.LongTensor).to(device)

        scoresTar, _ = model.forward(userBatch, trainBatch, targetBatch, None, offset, isEval=1)
        
        #get the index of top 40 items
        predIdx = torch.topk(scoresTar, 40, largest=True)[1]
        predIdx = predIdx.cpu().data.numpy().copy()

        if batch == 0:
            predIdxArray = predIdx
            targetList  = groundTruth
        else:
            predIdxArray = np.append(predIdxArray, predIdx, axis=0)
            targetList  += groundTruth

    for k in [5, 10, 20, 40]:
        Recall.append(calRecall(targetList, predIdxArray, k))
        NDCG.append(calNDCG(targetList, predIdxArray, k))

    return Recall, NDCG
