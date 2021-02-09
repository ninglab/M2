import pandas as pd
import numpy as np
import argparse
import datetime
import pdb
import pickle

import collections

parser = argparse.ArgumentParser()
parser.add_argument('--inputFile', type=str, default='TaFeng_raw.csv')
parser.add_argument('--outputFile', type=str, default='')
parser.add_argument('--n', type=int, default=10)

args = parser.parse_args()

df = pd.read_csv(args.inputFile)

df = df[["TRANSACTION_DT", "CUSTOMER_ID", "PRODUCT_ID"]]

counts_user = df['CUSTOMER_ID'].value_counts()
df = df[df["CUSTOMER_ID"].isin(counts_user[counts_user >= args.n].index)]

counts_item = df['PRODUCT_ID'].value_counts()
df = df[df["PRODUCT_ID"].isin(counts_item[counts_item >= args.n].index)]

ts_unique = df.groupby("CUSTOMER_ID")["TRANSACTION_DT"].nunique()
df = df[df["CUSTOMER_ID"].isin(ts_unique[ts_unique >= 2].index)]

counts_item = df['PRODUCT_ID'].nunique()
counts_user = df['CUSTOMER_ID'].nunique()
print('num_users: %d' % counts_user)
print('num_items: %d' % counts_item)

df["TRANSACTION_DT"] = pd.to_datetime(df["TRANSACTION_DT"], format="%m/%d/%Y")
df_ordered = df.sort_values(["TRANSACTION_DT"], ascending=True)

seq_data = df_ordered.groupby(["CUSTOMER_ID", "TRANSACTION_DT"])["PRODUCT_ID"].apply(list)

train_dict = collections.defaultdict(list)
valid_dict = collections.defaultdict(list)
test_dict = collections.defaultdict(list)

valid_cutoff_str = "2/1/2001"
valid_cutoff = datetime.datetime.strptime(valid_cutoff_str, "%m/%d/%Y")

test_cutoff_str = "2/14/2001"
test_cutoff = datetime.datetime.strptime(test_cutoff_str, "%m/%d/%Y")

count_sets = 0
count_trans = 0

for (user_id, data), item_list in seq_data.iteritems():
    if data < valid_cutoff:
        train_dict[user_id].append(item_list)

    elif data < test_cutoff:
        valid_dict[user_id].append(item_list)

    else:
        test_dict[user_id].append(item_list)

    count_sets += 1
    count_trans += len(item_list)

print("num_sets: %d" % count_sets)
print("num_trans: %d" % count_trans)


item_dict = {}
count = 0

train_filtered = {}

for key in train_dict:

    if len(train_dict[key]) < 2:
        continue

    for eachlist in train_dict[key]:
        for i in range(len(eachlist)):
            if eachlist[i] in item_dict:
                eachlist[i] = item_dict[eachlist[i]]

            else:
                item_dict[eachlist[i]] = count
                count += 1
                eachlist[i] = item_dict[eachlist[i]]

    train_filtered[key] = train_dict[key]

print(count)

for key in valid_dict:
    for eachlist in valid_dict[key]:

        for i in range(len(eachlist)):
            if eachlist[i] in item_dict:
                eachlist[i] = item_dict[eachlist[i]]

            else:
                item_dict[eachlist[i]] = count
                count += 1
                eachlist[i] = item_dict[eachlist[i]]

print(count)

for key in test_dict:
    for eachlist in test_dict[key]:

        for i in range(len(eachlist)):
            if eachlist[i] in item_dict:
                eachlist[i] = item_dict[eachlist[i]]

            else:
                item_dict[eachlist[i]] = count
                count += 1
                eachlist[i] = item_dict[eachlist[i]]

print(count)    

with open('./TaFeng_train.pkl', 'wb') as f:
    pickle.dump(train_filtered, f)

with open('./TaFeng_valid.pkl', 'wb') as f:
    pickle.dump(valid_dict, f)

with open('./TaFeng_test.pkl', 'wb') as f:
    pickle.dump(test_dict, f)



print('END')

