import numpy as np
import time
import pickle
from data import dataLoader as generalLoader
from scipy import sparse
import scipy
import pdb

class dataLoader(generalLoader):
    def __init__(self, dataset, config):

        root  = dataset+'/'+dataset
        trainRoot = root+'_'+'train.pkl'
        validRoot = root+'_'+'valid.pkl'
        testRoot  = root+'_'+'test.pkl'

        with open(trainRoot, 'rb') as f:
            trainDict = pickle.load(f)

        with open(validRoot, 'rb') as f:
            validDict = pickle.load(f)

        with open(testRoot, 'rb') as f:
            testDict  = pickle.load(f)

        user2item = self.generate_user_list(trainDict, validDict, testDict)
        numRe, user2idx = self.generate_sets(user2item)
        self.numItems = self.get_num_items() + 1
        print("num_users_removed   %d" % numRe)
        print("num_valid_users   %d" % len(self.testList))
        print("num_items   %d" % self.numItems)

        #to generate his
        #Different sets share the same set of users
        self.numTrain, self.numValid, self.numTrainVal, self.numTest = len(self.testList), len(self.testList), len(self.testList), len(self.testList)
        #same number of items for training and testing
        self.numItemsTrain, self.numItemsTest = self.numItems, self.numItems
        #same id in training, validation and testing
        self.valid2train = {}
        self.test2trainVal = {}
        for i in range(len(self.trainList)):
            self.valid2train[i]   = i
            self.test2trainVal[i] = i

        if config.isTrain:
            self.lenTrain    = self.generateLens(self.trainList)
            self.lenVal      = self.generateLens(self.validList)
        else:
            self.lenTrainVal = self.generateLens(self.trainValList)
            self.lenTest     = self.generateLens(self.testList)

        start = time.time()
        if config.isTrain:
            #generate and store the matrices in the first run
            #and just load these matrices in the following runs
            ##self.hisMatTra, self.tarMatTra        = self.generateHis(self.trainList, isTrain=1, isEval=0)
            ##self.hisMatVal, self.tarMatVal        = self.generateHis(self.validList, isTrain=1, isEval=1)

            ##scipy.sparse.save_npz('his_leava_one/'+config.dataset+'_hisMatTra_'+str(config.testOrder)+'.npz', self.hisMatTra)
            ##scipy.sparse.save_npz('his_leava_one/'+config.dataset+'_hisMatVal_'+str(config.testOrder)+'.npz', self.hisMatVal)
            ##scipy.sparse.save_npz('his_leava_one/'+config.dataset+'_tarMatTra_'+str(config.testOrder)+'.npz', self.tarMatTra)
            ##with open('his_leava_one/'+config.dataset+'_tarMatVal_'+str(config.testOrder)+'.pkl', 'wb') as f:
            ##    pickle.dump(self.tarMatVal, f)

            self.hisMatTra = scipy.sparse.load_npz('his_leava_one/'+config.dataset+'_hisMatTra_'+str(config.testOrder)+'.npz')
            self.hisMatVal = scipy.sparse.load_npz('his_leava_one/'+config.dataset+'_hisMatVal_'+str(config.testOrder)+'.npz')
            self.tarMatTra = scipy.sparse.load_npz('his_leava_one/'+config.dataset+'_tarMatTra_'+str(config.testOrder)+'.npz')
            with open('his_leava_one/'+config.dataset+'_tarMatVal_'+str(config.testOrder)+'.pkl', 'rb') as f:
                self.tarMatVal = pickle.load(f)

        else:
            #generate and store the matrices in the first run
            #and just load these matrices in the following runs
            ##self.hisMatTraVal, self.tarMatTraVal  = self.generateHis(self.trainValList, isTrain=0, isEval=0)
            ##self.hisMatTest, self.tarMatTest      = self.generateHis(self.testList, isTrain=0, isEval=1)

            ##scipy.sparse.save_npz('his_test_leave_one/'+config.dataset+'_hisMatTraVal_'+str(config.testOrder)+'.npz', self.hisMatTraVal)
            ##scipy.sparse.save_npz('his_test_leave_one/'+config.dataset+'_hisMatTest_'+str(config.testOrder)+'.npz', self.hisMatTest)
            ##scipy.sparse.save_npz('his_test_leave_one/'+config.dataset+'_tarMatTraVal_'+str(config.testOrder)+'.npz', self.tarMatTraVal)
            ##with open('his_test_leave_one/'+config.dataset+'_tarMatTest_'+str(config.testOrder)+'.pkl', 'wb') as f:
            ##    pickle.dump(self.tarMatTest, f)

            self.hisMatTraVal = scipy.sparse.load_npz('his_test_leave_one/'+config.dataset+'_hisMatTraVal_'+str(config.testOrder)+'.npz')
            self.hisMatTest = scipy.sparse.load_npz('his_test_leave_one/'+config.dataset+'_hisMatTest_'+str(config.testOrder)+'.npz')
            self.tarMatTraVal = scipy.sparse.load_npz('his_test_leave_one/'+config.dataset+'_tarMatTraVal_'+str(config.testOrder)+'.npz')
            with open('his_test_leave_one/'+config.dataset+'_tarMatTest_'+str(config.testOrder)+'.pkl', 'rb') as f:
                self.tarMatTest = pickle.load(f)

        print("finish generating his matrix, elaspe: %.3f" % (time.time()-start))

    def generate_user_list(self, trainDict, validDict, testDict):
        all_users = list(trainDict.keys()) + list(validDict.keys()) + list(testDict.keys())
        user2item = {}
        for user in all_users:
            user2item[user] = trainDict.get(user, []) + validDict.get(user, []) + testDict.get(user, [])

        return user2item

    def generate_sets(self, user2item):
        self.trainList    = []
        self.validList    = []
        self.trainValList = []
        self.testList     = []
        count = 0
        count_remove = 0
        user2idx = {}

        for user in user2item:
            #only keep the users with validation and testing baskets
            if len(user2item[user]) < 4:
                count_remove += 1
                continue
            user2idx[user] = count
            count += 1
            self.trainList.append(user2item[user][:-2])
            self.trainValList.append(user2item[user][:-1])
            self.validList.append(user2item[user][:-1])
            self.testList.append(user2item[user])
        return count_remove, user2idx

    def get_num_items(self):
        numItem = 0
        for baskets in self.testList:
            #all the baskets of users
            for basket in baskets:
                for item in basket:
                    numItem = max(item, numItem)

        return numItem 

if __name__ == '__main__':
    dataset = 'TaFeng'
    data    = dataLoader('TaFeng')
    print('testing end')        
        
