import numpy as np
from collections import Counter
import pickle
import pdb
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from scipy import sparse
import scipy
from sklearn.preprocessing import normalize
import time

class dataLoader():
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

        self.trainList    = []
        self.validList    = []
        self.trainValList = []
        self.testList     = []
        numItemsTrain     = 0

        self.testOrder = config.testOrder

        self.key2idxTrain    = {}
        self.key2idxTrainVal = {}

        self.valid2train     = {}
        self.test2trainVal   = {}

        #get the number of items and train list
        #only the users have at least two items in the training set are used 
        #for validation and testing
        count = 0
        numRe = 0
        for key in trainDict:
            #keep users with as least 2 training set for the training purpose
            if len(trainDict[key]) >= 2:
                self.trainList.append(trainDict[key])
                self.key2idxTrain[key] = count
                count += 1
                for eachlist in trainDict[key]:
                    #update the number of items in training
                    for item in eachlist:
                        numItemsTrain = max(numItemsTrain, item)
            else:
                numRe += 1

        #get the valid list
        count = 0
        for key in validDict:

            #do it only in the train mode
            if not config.isTrain:
                break

            #using users in training only
            if key not in self.key2idxTrain:
                continue

            #we test on the next textOrder basket

            if len(validDict[key]) < self.testOrder:
                continue

            marker = 0
            for i in range(self.testOrder):

                while len(validDict[key][i]) and max(validDict[key][i]) > numItemsTrain:
                    validDict[key][i].remove(max(validDict[key][i]))

                if len(validDict[key][i]) == 0:
                    marker = 1
                    break

                if max(validDict[key][i]) > numItemsTrain:
                    print("error")

            if marker == 1:
                continue
            
            self.validList.append(trainDict[key] + validDict[key][:self.testOrder])
            trainIdx = self.key2idxTrain[key]
            self.valid2train[count] = trainIdx
            count += 1

        #get the test user
        count = 0
        for key in self.key2idxTrain:
            if key in validDict:
                self.trainValList.append(trainDict[key] + validDict[key])
            else:
                self.trainValList.append(trainDict[key])

            self.key2idxTrainVal[key] = count
            count += 1

        numItemsTest = 0
        for userList in self.trainValList:
            for eachlist in userList:
                for item in eachlist:
                    numItemsTest = max(numItemsTest, item)

        count = 0
        for key in testDict:
            if key not in self.key2idxTrain:
                continue

            #all the items and users in testing must appeared in training.
            if len(testDict[key]) < (self.testOrder):
                continue

            marker = 0
            for i in range(self.testOrder):

                while len(testDict[key][i]) and max(testDict[key][i]) > numItemsTest:
                    testDict[key][i].remove(max(testDict[key][i]))

                if len(testDict[key][i]) == 0:
                    marker = 1
                    break

            if marker == 1:
                continue

            if key in validDict and key in trainDict:
                self.testList.append(trainDict[key] + validDict[key] + testDict[key][:self.testOrder])
            else:
                self.testList.append(trainDict[key] + testDict[key][:self.testOrder])

            #some test users may only have one set in the history. For these very few users. We remove them.
            if key in self.key2idxTrainVal:
                trainValidIdx = self.key2idxTrainVal[key]
                self.test2trainVal[count] = trainValidIdx
                count += 1

        self.numTrain    = len(self.trainList)
        self.numValid    = len(self.validList)
        self.numTest     = len(self.testList)
        self.numTrainVal = len(self.trainValList)

        #start from 0
        self.numItemsTrain = numItemsTrain + 1
        self.numItemsTest  = numItemsTest + 1

        self.numTransTrain = sum([len(eachlist) for eachlist in [user for user in self.trainList]])

        print("num_users_removed   %d" % numRe)
        print("num_trans_train:    %d" % self.numTransTrain)
        print("max_index_items_train:    %d" % self.numItemsTrain)
        print("max_index_items_test:     %d" % self.numItemsTest)
        print("num_train_users:    %d" % self.numTrain)
        print("num_valid_users:    %d" % self.numValid)
        print("num_test_users:     %d" % self.numTest)
        print("num_trainVal_users: %d" % self.numTrainVal)

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
            self.hisMatTra, self.tarMatTra        = self.generateHis(self.trainList, isTrain=1, isEval=0)
            self.hisMatVal, self.tarMatVal        = self.generateHis(self.validList, isTrain=1, isEval=1)

            ##scipy.sparse.save_npz('his/'+config.dataset+'_hisMatTra_'+str(config.testOrder)+'.npz', self.hisMatTra)
            ##scipy.sparse.save_npz('his/'+config.dataset+'_hisMatVal_'+str(config.testOrder)+'.npz', self.hisMatVal)
            ##scipy.sparse.save_npz('his/'+config.dataset+'_tarMatTra_'+str(config.testOrder)+'.npz', self.tarMatTra)
            ##with open('his/'+config.dataset+'_tarMatVal_'+str(config.testOrder)+'.pkl', 'wb') as f:
            ##    pickle.dump(self.tarMatVal, f)

            ##self.hisMatTra = scipy.sparse.load_npz('his/'+config.dataset+'_hisMatTra_'+str(config.testOrder)+'.npz')
            ##self.hisMatVal = scipy.sparse.load_npz('his/'+config.dataset+'_hisMatVal_'+str(config.testOrder)+'.npz')
            ##self.tarMatTra = scipy.sparse.load_npz('his/'+config.dataset+'_tarMatTra_'+str(config.testOrder)+'.npz')
            ##with open('his/'+config.dataset+'_tarMatVal_'+str(config.testOrder)+'.pkl', 'rb') as f:
            ##    self.tarMatVal = pickle.load(f)

        else:
            #generate and store the matrices in the first run
            #and just load these matrices in the following runs
            self.hisMatTraVal, self.tarMatTraVal  = self.generateHis(self.trainValList, isTrain=0, isEval=0)
            self.hisMatTest, self.tarMatTest      = self.generateHis(self.testList, isTrain=0, isEval=1)

            ##scipy.sparse.save_npz('his_test/'+config.dataset+'_hisMatTraVal_'+str(config.testOrder)+'.npz', self.hisMatTraVal)
            ##scipy.sparse.save_npz('his_test/'+config.dataset+'_hisMatTest_'+str(config.testOrder)+'.npz', self.hisMatTest)
            ##scipy.sparse.save_npz('his_test/'+config.dataset+'_tarMatTraVal_'+str(config.testOrder)+'.npz', self.tarMatTraVal)
            ##with open('his_test/'+config.dataset+'_tarMatTest_'+str(config.testOrder)+'.pkl', 'wb') as f:
            ##    pickle.dump(self.tarMatTest, f)

            ##self.hisMatTraVal = scipy.sparse.load_npz('his_test/'+config.dataset+'_hisMatTraVal_'+str(config.testOrder)+'.npz')
            ##self.hisMatTest = scipy.sparse.load_npz('his_test/'+config.dataset+'_hisMatTest_'+str(config.testOrder)+'.npz')
            ##self.tarMatTraVal = scipy.sparse.load_npz('his_test/'+config.dataset+'_tarMatTraVal_'+str(config.testOrder)+'.npz')
            ##with open('his_test/'+config.dataset+'_tarMatTest_'+str(config.testOrder)+'.pkl', 'rb') as f:
            ##    self.tarMatTest = pickle.load(f)

            Trans = self.generateTransMat(self.trainValList, self.numItemsTest)
            scipy.sparse.save_npz('weights/transition_matrix_'+config.dataset+'_'+str(config.testOrder)+'.npz', Trans)

            
        print("finish generating his matrix, elaspe: %.3f" % (time.time()-start))


        if config.isPreTrain:

            if config.isTrain:
                self.SPMI = self.generateSPMI(self.trainList, config.k, self.numItemsTrain)
            else:
                self.SPMI = self.generateSPMI(self.trainValList, config.k, self.numItemsTest)


            if config.isTrain:
                self.SPMI = self.generateSPMI(self.trainList, config.k, self.numItemsTrain)
            else:
                self.SPMI = self.generateSPMI(self.trainValList, config.k, self.numItemsTest)

        ##if config.isTrain:
        ##    with open('./SetsData/train_user_list_'+dataset+'.pkl', 'wb') as f:
        ##        pickle.dump(self.trainList, f)
        ## 
        ##    with open('./SetsData/valid_user_list_'+dataset+'_'+str(config.testOrder)+'.pkl', 'wb') as f:
        ##        pickle.dump(self.validList, f) 
        ##else: 
        ##    with open('./SetsData/train_valid_user_list_'+dataset+'.pkl', 'wb') as f:
        ##        pickle.dump(self.trainValList, f)
        ## 
        ##    with open('./SetsData/test_user_list_'+dataset+'_'+str(config.testOrder)+'.pkl', 'wb') as f:
        ##        pickle.dump(self.testList, f)

    def batchLoader(self, batchIdx, isTrain, isEval):
        if isTrain and not isEval:
            train    = [self.trainList[idx] for idx in batchIdx] 
            trainLen = [self.lenTrain[idx] for idx in batchIdx]
            his      = self.hisMatTra[batchIdx,:].todense()
            target   = self.tarMatTra[batchIdx,:].todense()
        elif not isTrain and not isEval:
            train    = [self.trainValList[idx] for idx in batchIdx] 
            trainLen = [self.lenTrainVal[idx] for idx in batchIdx]
            his      = self.hisMatTraVal[batchIdx,:].todense()
            target   = self.tarMatTraVal[batchIdx,:].todense()
        elif isTrain and isEval:
            train    = [self.validList[idx] for idx in batchIdx]
            trainLen = [self.lenVal[idx] for idx in batchIdx]
            his      = self.hisMatVal[batchIdx,:].todense()
            target   = [self.tarMatVal[idx] for idx in batchIdx]
        else:
            train    = [self.testList[idx] for idx in batchIdx]
            trainLen = [self.lenTest[idx] for idx in batchIdx]
            his      = self.hisMatTest[batchIdx,:].todense() 
            target   = [self.tarMatTest[idx] for idx in batchIdx]

        return train, trainLen, his, target

    def generateLens(self, userList):
        #list of list of lens of baskets

        lens    = []
        #pre-calculate the len of each sequence and basket
        for user in userList:
            lenUser    = []
            #the last bas is the traget to calculate errors
            trainEUser = user[:-1] 
            for bas in trainEUser:
                lenUser.append(len(bas))
            lens.append(lenUser)

        return lens

    def generateHis(self, userList, isTrain, isEval):
        
        #pre generate the history vector and target vector for each user
        if isTrain and not isEval:
            hisMat = np.zeros((self.numTrain, self.numItemsTrain))
        elif isTrain and isEval:
            hisMat = np.zeros((self.numValid, self.numItemsTrain))
        elif not isTrain and not isEval:
            hisMat = np.zeros((self.numTrainVal, self.numItemsTest))
        else:
            hisMat = np.zeros((self.numTest, self.numItemsTest))

        if not isEval and isTrain:
            tarMat = np.zeros((self.numTrain, self.numItemsTrain))
        elif not isEval and not isTrain:
            tarMat = np.zeros((self.numTrain, self.numItemsTest))
        else:
            tarMat = []

        for i in range(len(userList)):
            trainUser  = userList[i][:-1]
            targetUser = userList[i][-1]

            for Bas in trainUser:
                for item in Bas:
                    hisMat[i,item] += 1

            if not isEval:
                for item in targetUser:
                    tarMat[i,item] = 1
            else:
                tarMat.append(targetUser)

        #convert numpy to csr matrix
        hisMat = csr_matrix(hisMat)
        if not isEval:
            tarMat = csr_matrix(tarMat)

        #normalize history vectors to have unit norm rows in place
        normalize(hisMat, norm='l1', axis=1, copy=False)

        return hisMat, tarMat

    def generateSPMI(self, userList, k, numItems):
        #using full matrix for small dataset and sparse for large one
        C = np.zeros((numItems, numItems))
        
        for user in userList:
            for bas in user:
                #only count for once
                bas = list(set(bas))
                for i in range(len(bas)):
                    for j in range(i+1,len(bas)):
                        C[bas[i],bas[j]] += 1
                        C[bas[j],bas[i]] += 1

        numNoEmpty = np.count_nonzero(C)
        C = csr_matrix(C)
        rowNorm = sparse.diags(1/(C.sum(axis=1).A.ravel()+1e-6))
        colNorm = sparse.diags(1/(C.sum(axis=0).A.ravel()+1e-6))

        rowNormed  = rowNorm @ (C * numNoEmpty)
        SPMI       = rowNormed @ colNorm
        SPMI.data = np.log(SPMI.data)
        #filter out < log(k)
        SPMI.data[SPMI.data<np.log(k)] = 0.0

        ##SPMI = csr_matrix(SPMI)

        return SPMI

    def generateTransMat(self, userList, numItems):
        numUsers = len(userList)
        #Transition matrix: Trans[i][j] is the times that i transits to j
        Trans = np.zeros((numItems, numItems))

        for user in userList:
            last = user[-1]
            flatten_lsit = [item for bas in user[:-1] for item in bas]
            cnt = Counter(flatten_lsit)

            #a has freq 2 in cnt and b is in the last
            #it means a has transited to b twice

            for item in last:
                for key in cnt:
                    Trans[key][item] += cnt[key]

        Trans = csr_matrix(Trans)

        return Trans
            
            
            


 


if __name__ == '__main__':
    dataset = 'TaFeng'
    data    = dataLoader('TaFeng')
    #for testing
    print('testing end')
