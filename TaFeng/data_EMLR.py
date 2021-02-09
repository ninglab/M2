import numpy
import pickle
import pdb

class dataloader():
    def __init__(self, trainPkl, validPkl, testPkl):

        with open(trainPkl, 'rb') as f:
            train_dict = pickle.load(f)

        with open(validPkl, 'rb') as f:
            valid_dict = pickle.load(f)

        with open(testPkl, 'rb') as f:
            test_dict = pickle.load(f)

        self.train_user_list = []
        self.valid_user_list = []
        self.train_valid_user_list = []
        self.test_user_list = []
        self.test_user_list_FPMC = []
        items_train = 0

        self.key2idx_train = {}
        self.key2idx_train_valid = {}

        self.valid2train = {}
        self.test2train_valid = {}

        #get the number of items and train list
        count = 0
        for key in train_dict:
            self.train_user_list.append(train_dict[key])
            self.key2idx_train[key] = count
            count += 1
            for eachlist in train_dict[key]:
                for item in eachlist:
                    items_train = max(items_train, item)

        #get the valid list
        count = 0
        for key in valid_dict:
            marker = 0
            if key not in train_dict:
                continue
            for item in valid_dict[key][0]:
                if item > items_train:
                    marker = 1
            if marker == 1:
                continue
            self.valid_user_list.append(train_dict[key] + [valid_dict[key][0]])
            train_idx = self.key2idx_train[key]
            self.valid2train[count] = train_idx
            count += 1

        #get the test user
        count = 0
        for key in train_dict:
            if key in valid_dict:
                self.train_valid_user_list.append(train_dict[key] + valid_dict[key])
            else:
                self.train_valid_user_list.append(train_dict[key])

            self.key2idx_train_valid[key] = count
            count += 1

        for key in valid_dict:
            if len(valid_dict[key]) < 2:
                continue
            if key not in train_dict:
                self.train_valid_user_list.append(valid_dict[key])
                self.key2idx_train_valid[key] = count
                count += 1

        items_test = 0
        for user_list in self.train_valid_user_list:
            for eachlist in user_list:
                for item in eachlist:
                    items_test = max(items_test, item)

        count = 0
        for key in test_dict:
            marker = 0
            if key not in train_dict and key not in valid_dict:
                continue
            for item in test_dict[key][0]:
                if item > items_test:
                    marker = 1

            if marker == 1:
                continue

            if key in valid_dict and key in train_dict:
                self.test_user_list.append(train_dict[key] + valid_dict[key] + [test_dict[key][0]])
                
            elif key in valid_dict:
                self.test_user_list.append(valid_dict[key] + [test_dict[key][0]])
            else:
                self.test_user_list.append(train_dict[key] + [test_dict[key][0]])

            #soem test users may only have one set in the history. For these very few users. We can not test them on FPMC.
            if key in self.key2idx_train_valid:
                train_valid_idx = self.key2idx_train_valid[key]
                self.test2train_valid[count] = train_valid_idx
                count += 1

                if key in valid_dict and key in train_dict:
                    self.test_user_list_FPMC.append(train_dict[key] + valid_dict[key] + [test_dict[key][0]])

                elif key in valid_dict:
                    self.test_user_list_FPMC.append(valid_dict[key] + [test_dict[key][0]])
                else:
                    self.test_user_list_FPMC.append(train_dict[key] + [test_dict[key][0]])
                
            else:
                if len(valid_dict[key]) != 1:
                    print ('error')

        self.num_train = len(self.train_user_list)
        self.num_valid = len(self.valid_user_list)
        self.num_test = len(self.test_user_list)
        self.num_train_valid = len(self.train_valid_user_list)

        #start from 0
        self.items_train = items_train + 1
        self.items_test = items_test + 1

        total_transactions_train = sum([len(eachlist) for eachlist in [user for user in self.train_user_list]])

        print (total_transactions_train)
        print("num_items_train: %d" % self.items_train)
        print("num_items_test: %d" % self.items_test)
        print("num_train: %d" % self.num_train)
        print("num_valid: %d" % self.num_valid)
        print("num_test: %d" % self.num_test)
        print("num_train_valid: %d" % self.num_train_valid)
        print("num_valid_test_FPMC: %d" % len(self.test2train_valid))
        print("num_valid_test_FPMC: %d" % len(self.test_user_list_FPMC))

        ##with open('train_user_list_Gowalla.pkl', 'wb') as f:
        ##    pickle.dump(self.train_user_list, f)
        ## 
        ##with open('valid_user_list_Gowalla.pkl', 'wb') as f:
        ##    pickle.dump(self.valid_user_list, f) 
        ## 
        ##with open('train_valid_user_list_Gowalla.pkl', 'wb') as f:
        ##    pickle.dump(self.train_valid_user_list, f)
        ## 
        ##with open('test_user_list_Gowalla.pkl', 'wb') as f:
        ##    pickle.dump(self.test_user_list, f)

    def batch_loader(self, batch_user_list):
        return [self.train_user_list[idx] for idx in batch_user_list]

    def batch_loader_test(self, batch_user_list):
        return [self.train_valid_user_list[idx] for idx in batch_user_list]
