import torch
import numpy as np
import random
import time
from utils import *
from model.Dream import Dream
from model.FPMC import FPMC
from model.SNBR import SNBR
from model.FREQ import FREQ
from criterio import Bernoulli, BPR
import sys
import pdb
from evalByUser import evalByUser

def trainByUser(dataLoader, config, logger, device):

    if config.isTrain:
        numUsers = dataLoader.numTrain
        numItems = dataLoader.numItemsTrain
    else:
        numUsers = dataLoader.numTrainVal
        numItems = dataLoader.numItemsTest

    #preTrain
    outEmbsWeights = None
    numBatch = numUsers // config.batchSize + 1
    idxList  = [i for i in range(numUsers)]

    if config.model == 'Dream':
        model = Dream(config, numItems, device).to(device)
    elif config.model == 'SNBR':
        model = SNBR(config, numItems, device, outEmbsWeights).to(device)
    elif config.model == 'FREQ' or config.model == 'FREQP':
        model = FREQ(numItems, device).to(device)

    #open to more
    if config.opt == 'Adam':
        #condidate lr: 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    elif config.opt == 'Adagrad':
        #condidate lr: 1e-2
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr, weight_decay=config.l2)

    for epoch in range(config.numIter):
        #permutate user list
        random.shuffle(idxList)
        timeEpStr = time.time()
        epochLoss = 0

        for batch in range(numBatch):
            start = config.batchSize * batch
            end   = min(numUsers, start + config.batchSize)

            batchList = idxList[start:end]
            samples, decays, his, target, offset, lenX, targetList = generateBatchSamples(dataLoader, batchList, config, isEval=0)

            samples = torch.from_numpy(samples).type(torch.LongTensor).to(device)
            decays  = torch.from_numpy(decays).type(torch.FloatTensor).to(device)
            his     = torch.from_numpy(his).type(torch.FloatTensor).to(device)
            target  = torch.from_numpy(target).type(torch.FloatTensor).to(device)
            offset  = torch.from_numpy(offset).type(torch.FloatTensor).to(device)
            lenX    = lenX.to(device)

            if config.model == 'Dream':
                tarArr, negArr = generateNegatives(targetList, numItems, config)
                tarLongTensor = torch.from_numpy(tarArr).type(torch.LongTensor).to(device)
                negLongTensor = torch.from_numpy(negArr).type(torch.LongTensor).to(device)
                scoresTar, scoresNeg = model.forward(samples, lenX, tarLongTensor, negLongTensor, isEval=0)
            elif config.model == 'SNBR':
                scores, _  = model.forward(samples, decays, offset, his, isEval=0)
            elif config.model == 'RNBR':
                scores  = model.forward(samples, offset, lenX, his, isEval=0)
            elif config.model == 'UNBR':
                scores  = model.forward(samples, decays, offset, lenX, his, isEval=0)
            elif config.model == 'FREQ' or config.model == 'FREQP':
                scores  = model.forward(his)

            #Dream and FPMC are trained using BPR
            if config.model == 'Dream' or config.model == 'FPMC':
                loss = BPR(scoresTar, scoresNeg)
            else:
                loss = Bernoulli(target, scores)

            epochLoss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config.model != 'FREQ' and config.model != 'FREQP':
                with torch.no_grad():
                    zeros = torch.zeros(config.dim).to(device)
                    model.itemEmb.weight[config.padIdx].copy_(zeros)

        epochLoss = epochLoss / float(numBatch)
        timeEpEnd   = time.time()
        logger.info("num_epoch: %d, elaspe: %.1f, loss: %.3f" % (epoch, timeEpEnd - timeEpStr, epochLoss))
        

        if (epoch + 1)% config.evalEpoch == 0:
            timeEvalStar = time.time()
            logger.info("start evaluation")

            #recall 5, 10, 20, 40...
            recall, ndcg = evalByUser(model, dataLoader, config, device, config.isTrain)
            logger.info(', '.join(str(e) for e in recall))
            logger.info(', '.join(str(e) for e in ndcg))

            timeEvalEnd = time.time()
            logger.info("Evaluation time:{}".format(timeEvalEnd - timeEvalStar))

            if not config.isTrain:
                torch.save(model, config.saveRoot+'_'+str(epoch))

    logger.info("\n")
    logger.info("\n")



            
            

            

        
