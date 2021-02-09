import torch
import numpy as np
import math
import pdb
from utils import generateBatchSamples
from evalMetric import calRecall, calNDCG

def evalByUser(model, dataLoader, config, device, isTrain):
    evalBatchSize = config.batchSize
    
    if isTrain:
        numUser  = dataLoader.numValid
        numItems = dataLoader.numItemsTrain
    else:
        numUser = dataLoader.numTest
        numItems = dataLoader.numItemsTest

    numBatch = numUser // evalBatchSize + 1
    idxList  = [i for i in range(numUser)]

    Recall = []
    NDCG = []

    for batch in range(numBatch):
        start = batch * evalBatchSize
        end   = min(batch * evalBatchSize + evalBatchSize, numUser)

        batchList = idxList[start:end]

        #target is the same with targetList in evaluation
        samples, decays, his, target, offset, lenX, _ = generateBatchSamples(dataLoader, batchList, config, isEval=1)

        samples = torch.from_numpy(samples).type(torch.LongTensor).to(device)
        decays  = torch.from_numpy(decays).type(torch.FloatTensor).to(device)
        his     = torch.from_numpy(his).type(torch.FloatTensor).to(device)
        offset  = torch.from_numpy(offset).type(torch.FloatTensor).to(device)
        lenX    = lenX.to(device)

        if config.model == 'Dream' or config.model == 'FPMC':
            allItemIdx = np.asarray([i for i in range(numItems)])
            allItemIdx = torch.from_numpy(allItemIdx).type(torch.LongTensor).to(device)
            scores, _  = model.forward(samples, lenX, allItemIdx, neg=None, isEval=1)

        with torch.no_grad():
            if config.model == 'SNBR':        
                scores, gate = model.forward(samples, decays, offset, his, isEval=1)
            elif config.model == 'RNBR':
                scores = model.forward(samples, offset, lenX, his, isEval=1)
            elif config.model == 'UNBR':
                scores = model.forward(samples, decays, offset, lenX, his, isEval=1)
            elif config.model == 'FREQ' or config.model == 'FREQP':
                scores = model.forward(his)

        #get the index of top 40 items
        predIdx = torch.topk(scores, 40, largest=True)[1]
        predIdx = predIdx.cpu().data.numpy().copy()

        if config.model == 'SNBR' and config.abla=="None":
            gate = gate.cpu().data.numpy().copy()

        if batch == 0:
            predIdxArray = predIdx
            targetList  = target
            if config.model == 'SNBR' and config.abla=="None":
                gateArray = gate
        else:
            predIdxArray = np.append(predIdxArray, predIdx, axis=0)
            if config.model == 'SNBR' and config.abla=="None":
                gateArray = np.append(gateArray, gate, axis=0)
            targetList  += target
            

    for k in [5, 10, 20, 40]:
        Recall.append(calRecall(targetList, predIdxArray, k))
        NDCG.append(calNDCG(targetList, predIdxArray, k))

    ##if config.model == 'SNBR' and config.abla=="None" and not config.isTrain and config.retrival:
    ##    np.savetxt('./gates/gateBest/gate_'+config.dataset+'_'+str(config.testOrder)+'.txt', gateArray.reshape(-1,1), fmt="%.5f")

    return Recall, NDCG
