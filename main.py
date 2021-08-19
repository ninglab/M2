from trainByUser import *
from trainByBas  import *
from transactions import transactions
from evalByUser import evalByUser
from evalByBas import evalByBas

import argparse
import logging
import pdb
import pickle

import os
import sys
##os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='time_split')
    parser.add_argument('--dataset', type=str, default='TaFeng')
    
    parser.add_argument('--batchSize', type=int, default=100)
    parser.add_argument('--opt', type=str, default='Adagrad')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--decay', type=float, default=0.6)
    parser.add_argument('--testOrder', type=int, default=1)
    parser.add_argument('--numIter', type=int, default=200)
    parser.add_argument('--evalEpoch', type=int, default=5)
    parser.add_argument('--isTrain', type=int, default=1)
    parser.add_argument('--k', type=float, default=0.0)
    parser.add_argument('--preTrainBatchSize', type=int, default=4096)
    parser.add_argument('--preTrainEpochs', type=int, default=100)
    parser.add_argument('--isPreTrain', type=int, default=0)
    parser.add_argument('--abla', type=str, default="None")
    parser.add_argument('--retrival', type=int, default=0)
    parser.add_argument('--testEpoch', type=int, default=0)
    parser.add_argument('--store', type=int, default=0, help='1 for storing training data only')

    parser.add_argument('--model', type=str, default='SNBR')

    config = parser.parse_args()

    if config.mode == 'time_split':
        if config.isTrain:
            resultName = 'all_results_valid'
        else:
            resultName = 'all_results_test'
    elif config.mode == 'seq_split':
        if config.isTrain:
            resultName = 'all_results_valid_leave_one'
        else:
            resultName = 'all_results_test_leave_one'

    if config.abla != "None":
        logName = resultName+'/'+config.model+'_'+config.abla+'/'+config.dataset+'/'+config.model+'_'+config.opt+'_'+str(config.batchSize)+'_'+str(config.decay)+'_'+str(config.dim)+'_'+str(config.l2)+'_'+str(config.isPreTrain)+'_'+str(config.k)+'_'+str(config.testOrder)
    else:
        logName = resultName+'/'+config.model+'/'+config.dataset+'/'+config.model+'_'+config.opt+'_'+str(config.batchSize)+'_'+str(config.decay)+'_'+str(config.dim)+'_'+str(config.l2)+'_'+str(config.isPreTrain)+'_'+str(config.k)+'_'+str(config.testOrder)

    if config.store == 0:
        #avoid to stack results from different runs
        if os.path.exists(logName):
            os.remove(logName)

    logging.basicConfig(filename=logName, level=logging.DEBUG)
    ##logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    if config.mode == 'time_split':
        if not config.isTrain:
            config.saveRoot = 'models_test/'+config.model+'/'+config.dataset+'/'+config.model+'_'+config.opt+'_'+str(config.batchSize)+'_'+str(config.decay)+'_'+str(config.dim)+'_'+str(config.l2)+'_'+str(config.isPreTrain)+'_'+str(config.k)+'_'+str(config.testOrder)
    elif config.mode == 'seq_split':
        if not config.isTrain:
            config.saveRoot = 'models_test_leave_one/'+config.model+'/'+config.dataset+'/'+config.model+'_'+config.opt+'_'+str(config.batchSize)+'_'+str(config.decay)+'_'+str(config.dim)+'_'+str(config.l2)+'_'+str(config.isPreTrain)+'_'+str(config.k)+'_'+str(config.testOrder)

    if config.mode == 'time_split':
        from data import dataLoader
        dataset = dataLoader(config.dataset, config)
    elif config.mode == 'seq_split':
        from data_leave_one import dataLoader
        dataset = dataLoader(config.dataset, config)

    if config.store == 1:
        with open('SetsData_leave_one/train_user_list_'+config.dataset+'.pkl', 'wb') as f:
            pickle.dump(dataset.trainList, f)

        with open('SetsData_leave_one/valid_user_list_'+config.dataset+'_'+str(config.testOrder)+'.pkl', 'wb') as f:
            pickle.dump(dataset.validList, f)

        with open('SetsData_leave_one/train_valid_user_list_'+config.dataset+'.pkl', 'wb') as f:
            pickle.dump(dataset.trainValList, f)

        with open('SetsData_leave_one/test_user_list_'+config.dataset+'_'+str(config.testOrder)+'.pkl', 'wb') as f:
            pickle.dump(dataset.testList, f)
        print('store done')
        sys.exit()
        

    if config.isTrain:
        config.padIdx = dataset.numItemsTrain
    else:
        config.padIdx = dataset.numItemsTest

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger.info('start training')

    if config.retrival:
        model = torch.load(config.saveRoot+'_'+str(config.testEpoch))
        if config.model == 'FPMC':
            recall, ndcg = evalByBas(model, trans, config, device)
            userEmbeddings = model.EUI.cpu().weight.data.numpy().copy()
            np.save('embeddings/userEmb_'+config.dataset+'_FPMC', userEmbeddings)
            trans = transactions(dataset, config)
        else:
            recall, ndcg = evalByUser(model, dataset, config, device, config.isTrain)
            W = model.itemEmb.cpu().weight.data.numpy().copy()
            bias = model.out.cpu().bias.data.numpy().copy()
            np.savetxt('weights/'+config.model+'_'+config.dataset+'_'+str(config.testOrder)+'_bias.txt',bias,fmt='%.4f',delimiter=',')
            np.savetxt('weights/'+config.model+'_'+config.dataset+'_'+str(config.testOrder)+'_W.txt',W,fmt='%.4f',delimiter=',')

        logger.info(', '.join(str(e) for e in recall))
        logger.info(', '.join(str(e) for e in ndcg))

    else:
        if config.model == 'FPMC':
            trans = transactions(dataset, config)
            trainByBas(trans, config, logger, device)
        else:
            trainByUser(dataset, config, logger, device)

    logger.info('end')
