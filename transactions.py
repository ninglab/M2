import numpy as np
from subsequences import subsequences

class transactions:

    def __init__(self, dataLoader, config):
       
        if config.isTrain:
            self.numUsers = dataLoader.numTrain
            self.numItems = dataLoader.numItemsTrain
            self.train = subsequences(dataLoader.trainList)
            self.test  = subsequences(dataLoader.validList, dataLoader.valid2train)

        else:
            self.numUsers = dataLoader.numTrainVal
            self.numItems = dataLoader.numItemsTest 
            self.train = subsequences(dataLoader.trainValList)
            self.test  = subsequences(dataLoader.testList, dataLoader.test2trainVal)

