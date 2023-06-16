import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize


def _onehot(categories):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(categories)
    n=len(np.unique(labels))
    identity_matrix = np.eye(n)
    one_hot_encoding = identity_matrix[labels]

    return one_hot_encoding

def one_hot_encode(dataframe,lab): return _onehot(dataframe.loc[:,lab].to_numpy())

def encode_cabin(dataframe, lab):
    arr = dataframe.loc[:,lab].to_numpy()
    list1, list2, list3 = [], [], []
    for s in arr:
        try:
            split_s = s.split('/')
            if len(split_s) == 3:
                list1.append(split_s[0])
                list2.append(int(split_s[1]))
                list3.append(split_s[2])
        except:
            list1.append("asdf")
            list2.append(-1)
            list3.append("dasf")
    print(len(list2), len(arr))
    n= len(list2)
    a1=_onehot(list1)
    a2=np.array(list2,dtype="float64").reshape((n,1))
    a3=_onehot(list3)
    return np.concatenate((a1,a2,a3),axis=1)

def encode_id(dataframe, lab):
    arr = dataframe.loc[:,lab].to_numpy()
    list1, list2, listids = [], [], []
    for s in arr:
        try:
            split_s = s.split('_')
            if len(split_s) == 2:
                list1.append(int(split_s[0]))
                list2.append(int(split_s[1]))
                listids.append(s)
        except:
            list1.append(10000)
            list2.append(1)
    print(len(list1), len(arr))
    n = len(arr)
    a1=np.array(list1,dtype="float64").reshape((n,1))
    a2=np.array(list2,dtype="float64").reshape((n,1))
    return np.concatenate((a1,a2),axis=1), listids

def quantities_preprocess(dataframe, indicies):
    arr = dataframe.iloc[:,indicies].to_numpy(dtype="float64",na_value=0)
    arr = normalize(arr)
    return arr

def preprocess(dataframe,includeLabel=True):
    data=[]
    for col in ["Destination","HomePlanet","CryoSleep","VIP"]:
        data.append(one_hot_encode(dataframe,col))
    data = np.concatenate(data, axis=1)
    print(data.shape)
    quants = quantities_preprocess(dataframe,[5,7,8,9,10,11])
    ids, listids = encode_id(dataframe,"PassengerId")
    cabins = encode_cabin(dataframe,"Cabin")
    for thing in (data,quants,ids,cabins):
        print(thing.shape)
    data=np.concatenate((data,quants,ids,cabins),axis=1)

    if includeLabel:
        labels = _onehot(dataframe.loc[:,"Transported"].to_numpy())
        x=labels[:,0].reshape((8693,1))
        print(x)
        print(x.shape)

        return data, labels, listids
    else: return data, listids