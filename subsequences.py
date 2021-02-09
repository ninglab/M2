import numpy as np

class subsequences:

    #windowSize 2 for FPMC. Could extend it to n
    #n-1 to predict the next 1
    def __init__(self, basSeqs, mapping=None, windowSize=2):

        self.training = []
        self.target   = []

        #the userId is the same with the training one
        self.UserList = []

        self.generateSeqs(basSeqs, mapping, windowSize)

    def generateSeqs(self, basSeqs, mapping, windowSize):

        countUser = 0
        for basSeq in basSeqs:
            if not mapping:
                for i in range(len(basSeq)-windowSize):
                    #in each subsequence of len windowSize, the last one is for predicting the rest is for training
                    subBasSeq = basSeq[i:i+windowSize]
                    subBasSeqTrain = subBasSeq[:-1]
                    subBasSeqTest  = subBasSeq[-1]

                    if len(subBasSeqTrain) == 1:
                        subBasSeqTrain = subBasSeqTrain[0]

                    self.training.append(subBasSeqTrain)
                    self.target.append(subBasSeqTest)
                    self.UserList.append(countUser)

            else:
                subBasSeqTrain = basSeq[-2]
                subBasSeqTest  = basSeq[-1]

                self.training.append(subBasSeqTrain)
                self.target.append(subBasSeqTest)
                self.UserList.append(mapping[countUser])
         
            countUser+=1

        return
