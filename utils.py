import numpy as np
import pickle
import torch
import random
import time
import pdb

def generateBatchSamples(dataLoader, batchIdx, config, isEval):
    samples, sampleLen, his, target = dataLoader.batchLoader(batchIdx, config.isTrain, isEval)
    
    #max length of sequence
    maxLenSeq = max([len(userLen) for userLen in sampleLen])
    #max length of basket
    maxLenBas = max([max(userLen) for userLen in sampleLen])

    #pad users
    #last pad
    paddedSamples = []
    paddedDecays  = []
    paddedOffset  = []
    targetList    = []
    lenList       = []
    for user in samples:
        #the last one is to calculate errors
        trainU  = user[:-1]
        testU   = user[-1]
        targetList.append(testU)

        paddedU = []
        decayU  = []
        offsetU = []
        lenList.append([len(trainU) - 1])
        
        decayNum = len(trainU)-1
        for eachBas in trainU:
            paddedBas = eachBas + [config.padIdx] * (maxLenBas - len(eachBas))
            paddedU.append(paddedBas)
            decayU.append(config.decay ** decayNum)
            decayNum -= 1
            #offset = maxLenBas/actLenBas
            offsetU.append(maxLenBas/float(len(eachBas)))

        paddedU = paddedU + [[config.padIdx] * maxLenBas] * (maxLenSeq - len(paddedU))
        decayU  = decayU + [0] * (maxLenSeq - len(decayU))
        offsetU = offsetU + [0] * (maxLenSeq-len(offsetU))
        #add a sample
        paddedSamples.append(paddedU)
        paddedDecays.append(decayU)
        paddedOffset.append(offsetU)
 
    paddedOffset = np.asarray(paddedOffset).reshape(len(samples), -1, 1)

    #1-hot vectors
    lenTen = torch.tensor(lenList, dtype=torch.long)
    lenX   = torch.FloatTensor(len(samples), maxLenSeq)
    lenX.zero_()
    lenX.scatter_(1,lenTen,1)

    return np.asarray(paddedSamples), np.asarray(paddedDecays).reshape(len(samples), -1, 1), his, target, paddedOffset, lenX, targetList

def preTrain(dataLoader, config, model, logger, device):

    rows, cols = dataLoader.SPMI.nonzero()
    valus      = np.asarray(dataLoader.SPMI[rows, cols]).reshape(-1)

    numNonZero = len(rows)
    numBatch   = numNonZero // config.preTrainBatchSize + 1
    epochs     = config.preTrainEpochs
    idxList    = [i for i in range(numNonZero)]

    optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr, weight_decay=config.l2)

    logger.info('start preTrain')

    #in case the matrix is too big
    #we factorize it batch by batch
    for epoch in range(epochs):
        sEpoch = time.time()
        random.shuffle(idxList)
        epochErr = 0

        for batch in range(numBatch):
            start = batch * config.preTrainBatchSize
            end   = min(numNonZero, start+config.preTrainBatchSize)
            batchValues = valus[start:end]
            batchRows   = rows[start:end]
            batchCols   = cols[start:end]

            batchRows  = torch.from_numpy(batchRows).type(torch.LongTensor).to(device)
            batchCols  = torch.from_numpy(batchCols).type(torch.LongTensor).to(device) 
            batchValues = torch.from_numpy(batchValues).type(torch.FloatTensor).to(device)

            res = model.forward(batchRows, batchCols)
            err = torch.pow(res-batchValues, 2).mean()

            epochErr += err.item()

            optimizer.zero_grad()
            err.backward()
            optimizer.step()

        epochErr = epochErr / float(numBatch)
        eEpoch   = time.time()
        
        logger.info("num_epoch: %d, elaspe: %.1f, loss: %.3f" % (epoch, eEpoch - sEpoch, epochErr))

    logger.info('end preTrain')
    logger.info('\n')

    return model.outEmbs.weight

def padBasWise(trainBatch, config):

    maxBasLen = max([len(bas) for bas in trainBatch])
    paddedtrainBatch = []
    offset = []

    for bas in trainBatch:
        bas += [config.padIdx] * (maxBasLen-len(bas))
        paddedtrainBatch.append(bas)
        offset.append(maxBasLen/float(len(bas)))

    return np.asarray(paddedtrainBatch), np.asarray(offset).reshape(-1,1)

def generateBatchBas(trans, idxList, isEval):
    if not isEval:
        trainBatch = [trans.train.training[idx] for idx in idxList]
        targetBatch= [trans.train.target[idx] for idx in idxList]
        userBatch  = [trans.train.UserList[idx] for idx in idxList]
    else:
        trainBatch = [trans.test.training[idx] for idx in idxList]
        targetBatch= [trans.test.target[idx] for idx in idxList]
        userBatch  = [trans.test.UserList[idx] for idx in idxList]

    return trainBatch, targetBatch, userBatch

#generate negatives in fly
def negSamp(posInds, nItems, nSamp):
    """ Pre-verified with binary search
    `pos_inds` is assumed to be ordered
    reference: https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html
    """
    #pos_inds is an array
    posInds.sort(kind='heapsort')
    rawSamp = np.random.randint(0, nItems - len(posInds), size=nSamp)
    posIndsAdj = posInds - np.arange(len(posInds))
    negInds = rawSamp + np.searchsorted(posIndsAdj, rawSamp, side='right')
    return negInds

def generateNegatives(targetList, nItems, config):

    maxBasLen = max([len(Bas) for Bas in targetList])

    negListPad = []
    tarListPad = []

    for Bas in targetList:
        #eliminate the duplicate
        BasSet = set(Bas)
        BasArr = np.asarray(list(BasSet))
        negIdx = negSamp(BasArr, nItems, len(Bas))
        
        Bas = Bas + [config.padIdx] * (maxBasLen - len(Bas))
        negIdx = negIdx.tolist() + [config.padIdx] * (maxBasLen - len(negIdx))

        tarListPad.append(Bas)
        negListPad.append(negIdx)

    return np.asarray(tarListPad), np.asarray(negListPad)
