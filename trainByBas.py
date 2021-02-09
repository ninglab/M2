import torch
import numpy as np
from model.FPMC import FPMC
from utils import *
from evalByBas import evalByBas
import random
from criterio import BPR

def trainByBas(trans, config, logger, device):
   
    numBas   = len(trans.train.training)
    numBatch = numBas // config.batchSize + 1 
    idxList  = [i for i in range(numBas)]

    if config.model == 'FPMC':
        model = FPMC(config, trans.numItems, trans.numUsers, device)

    #open to more
    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    elif config.opt == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr, weight_decay=config.l2)

    for epoch in range(config.numIter):

        random.shuffle(idxList)
        timeEpStr = time.time()
        epochLoss = 0

        for batch in range(numBatch):
            start = config.batchSize * batch
            end   = min(numBas, start+config.batchSize)

            batchList  = idxList[start:end]
            trainBatch, targetBatch, userBatch = generateBatchBas(trans, batchList, isEval=0)
            trainBatch, offset = padBasWise(trainBatch, config)
            targetBatch, negBatch = generateNegatives(targetBatch, trans.numItems, config)
            userBatch = np.asarray(userBatch)

            trainBatch = torch.from_numpy(trainBatch).type(torch.LongTensor).to(device)
            offset     = torch.from_numpy(offset).type(torch.FloatTensor).to(device)
            targetBatch= torch.from_numpy(targetBatch).type(torch.LongTensor).to(device)
            negBatch   = torch.from_numpy(negBatch).type(torch.LongTensor).to(device)
            userBatch  = torch.from_numpy(userBatch).type(torch.LongTensor).to(device)

            scoresTar, scoresNeg = model.forward(userBatch, trainBatch, targetBatch, negBatch, offset, isEval=0)

            if config.model == 'FPMC':
                loss = BPR(scoresTar, scoresNeg)
           
            epochLoss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                zeros = torch.zeros(config.dim).to(device)
                model.EIU.weight[config.padIdx].copy_(zeros)
                model.EIL.weight[config.padIdx].copy_(zeros)
                model.ELI.weight[config.padIdx].copy_(zeros)

        epochLoss = epochLoss / float(numBatch)
        timeEpEnd   = time.time()
        logger.info("num_epoch: %d, elaspe: %.1f, loss: %.3f" % (epoch, timeEpEnd - timeEpStr, epochLoss))

        if (epoch + 1)% config.evalEpoch == 0:
            timeEvalStar = time.time()
            logger.info("start evaluation")

            #recall 5, 10, 20, 40...
            recall, ndcg = evalByBas(model, trans, config, device)
            logger.info(', '.join(str(e) for e in recall))
            logger.info(', '.join(str(e) for e in ndcg))

            timeEvalEnd = time.time()
            logger.info("Evaluation time:{}".format(timeEvalEnd - timeEvalStar))

            if not config.isTrain:
                torch.save(model, config.saveRoot+'_'+str(epoch))

    logger.info("\n")
    logger.info("\n")
