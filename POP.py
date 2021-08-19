import numpy as np
from data import dataLoader
import argparse
from evalMetric import calRecall, calNDCG
import pdb

def countFre (userList, numItems):
    #calculate the global freq of items
    gloFreqVec = np.zeros(numItems)
    for userBas in userList:
        #the last one is for testing
        trainBas = userBas[:-1]
        for eachBas in trainBas:
            for item in eachBas:
                gloFreqVec[item] += 1

    return gloFreqVec

def countPerFre(uid, dataset):
    #always in testing mode
    return np.asarray(dataset.hisMatTest[uid].todense()).squeeze(0)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Gowalla')
    parser.add_argument('--testOrder', type=int, default=1)
    parser.add_argument('--isTrain', type=int, default=0)
    parser.add_argument('--isPreTrain', type=int, default=0)
    parser.add_argument('--model', type=str, default='POP')
    parser.add_argument('--mode', type=str, default='time_split')

    config = parser.parse_args()

    if config.mode == 'time_split':
        from data import dataLoader
        dataset = dataLoader(config.dataset, config)
    elif config.mode == 'seq_split':
        from data_leave_one import dataLoader
        dataset = dataLoader(config.dataset, config)

    #default it runs in testing mode
    #no need to turn parameters
    if config.model == 'POP':
        if config.mode == 'time_split':
            weight_file = 'weights/'+config.dataset+'_POP.txt'
        elif config.mode == 'seq_split':
            weight_file = 'weights_leave_one/'+config.dataset+'_POP.txt'

        gloFreqVec = countFre(dataset.testList, dataset.numItemsTest)
        with open(weight_file, 'w') as f:
            for i in range(len(gloFreqVec)):
                f.write(str(gloFreqVec[i])+'\n')
            
            

    testSet = dataset.testList
    Recall = []
    NDCG = []
    maxK = 40

    targetList = dataset.tarMatTest
    topkIdxArr = []
    for uid in range(len(testSet)):
        if config.model == 'POP':
            scores = gloFreqVec
        if config.model == 'POEP':
            scores = countPerFre(uid, dataset)
        idxArr = scores.argsort()[-maxK:][::-1].reshape(1,-1)

        if not len(topkIdxArr):
            topkIdxArr = idxArr
        else:
            topkIdxArr = np.append(topkIdxArr, idxArr, axis=0)

    for k in [5, 10, 20, 40]:
        Recall.append(calRecall(targetList, topkIdxArr, k))
        NDCG.append(calNDCG(targetList, topkIdxArr, k))

    print(config.dataset)
    print(config.testOrder)
    print(config.model)
    print("Recall: %f, %f, %f, %f" % (Recall[0], Recall[1], Recall[2], Recall[3]))
    print("NDCG: %f, %f, %f, %f" % (NDCG[0], NDCG[1], NDCG[2], NDCG[3]))
