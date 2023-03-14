import os
import time

import numpy as np
from random import seed # 用于固定每次生成的随机数都是确定的（伪随机数）
from random import randrange
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.autograd import Variable

path='datasets'
allpath=os.listdir(path)
print(allpath)


def datapre(path):
    alldata=[]
    maxdf,maxrssi,mindf=0,0,0
    maxdf,maxrssi,mindf=0,0,0
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            x = line.split()
            a = float(x[3])
            b=float(x[4])
            #print(b)
            if a>maxdf:
                maxdf=a
            if a<mindf:
                mindf=a
            if b<maxrssi:
                maxrssi=b
    #print(path)
    with open(path, "r") as f:
        for line in f.readlines():
            data = []
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            x = line.split()
            data.append(int(x[1]))
            mmm = float(x[3])
            if mmm > 0:
                mmm = mmm / maxdf
            if mmm < 0:
                mmm = mmm / mindf
            data.append(mmm)
            data.append(float(x[4]) / maxrssi)
            data.append(float(x[5]) / 6.280117345603815)
            alldata.append(data)

    newdata=[]
    for k in range(16):
        newdata.append([])
    for j in alldata:
        a = j[0]
        newdata[a].append(j[1])
        newdata[a].append(j[2])
        #newdata[a].append(j[3])
    for q in range(len(newdata)):
        if len(newdata[q]) < 100:
            x = 100 - len(newdata[q])
            buchong = [0] * x
            newdata[q] = newdata[q] + buchong
        if len(newdata[q]) > 100:
            newdata[q] = newdata[q][:100]
    newdata=list(np.array(newdata).flatten())
    #print(newdata)
    return newdata



#datapre()

def data_lodar(path):
    datas=[]
    labels=[]
    path1 = os.listdir(path)
    #print(path1)
    for i in path1:
        path2=os.path.join(path,str(i))
       #print(path2)
        path3=os.listdir(path2)
        #print(path3)
        for j in path3:
            label=j.split('_')[1][:-4]
            labels.append(label)
            path4=os.path.join(path2,j)
            #print(path4)
            data=datapre(path4)
            datas.append(data)

    print('label',len(labels),type(labels))
    print('data',len(datas),type(datas))
    return datas,labels

datas,labels=data_lodar(path)


import random
def split_train_test(data,label,test_ratio):
    #设置随机数种子，保证每次生成的结果都是一样的

    '''random.seed(42)
    random.shuffle(data)
    random.seed(42)
    random.shuffle(label)
    test_set_size = int(len(data) * test_ratio)
    print(test_set_size)
    test_data = torch.Tensor(data[:test_set_size])
    test_label = torch.Tensor(label[:test_set_size])
    train_data = torch.Tensor(data[test_set_size:])
    train_label = torch.Tensor(label[test_set_size:])'''

    random.seed(42)
    random.shuffle(data)
    random.seed(42)
    random.shuffle(label)
    test_set_size = int(len(data) * test_ratio)
    newdata,newlabel=[],[]
    for k in range(21):
        newdata.append([]),newlabel.append([])
    for i in range(len(label)):
        newdata[int(label[i])].append(data[i])
    train_data,test_data,train_label,test_label=[],[],[],[]
    #print('ssr',len(newdata[0]),len(label[0]),len(label),len(newdata))
    for i in range(len(newdata)):
        for j in range(len(newdata[i])):
            if j <int(len(newdata[i])*test_ratio):
                test_data.append(newdata[i][j])
                test_label.append(i)
            else:
                train_data.append(newdata[i][j])
                train_label.append(i)
    random.seed(42)
    random.shuffle(train_data)
    random.seed(42)
    random.shuffle(train_label)
    '''test_data = np.array(test_data)
    test_label=np.array(test_label)
    train_data = np.array(train_data)
    train_label=np.array(train_label)'''
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    #iloc选择参数序列中所对应的行
    return train_data,train_label,test_data,test_label

traindata,trainlabel,testdata,testlabel=split_train_test(datas,labels,0.2)
from torch.utils.data import Dataset, DataLoader, TensorDataset

EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.001

def get_confusion_matrix(preds, labels, num_classes, normalize="true"):
    """
    Calculate confusion matrix on the provided preds and labels.
    Args:
        preds (tensor or lists of tensors): predictions. Each tensor is in
            in the shape of (n_batch, num_classes). Tensor(s) must be on CPU.
        labels (tensor or lists of tensors): corresponding labels. Each tensor is
            in the shape of either (n_batch,) or (n_batch, num_classes).
        num_classes (int): number of classes. Tensor(s) must be on CPU.
        normalize (Optional[str]) : {‘true’, ‘pred’, ‘all’}, default="true"
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix
            will not be normalized.
    Returns:
        cmtx (ndarray): confusion matrix of size (num_classes x num_classes)
    """

    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)))#, normalize=normalize) 部分版本无该参数
    return cmtx



from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
if os.path.isfile("svm_model3.pkl"):
    svm = pickle.load(open("svm_model3.pkl", "rb"))
else:
    svm = SVC(kernel='linear', gamma=0.001)
    svm.fit(traindata, trainlabel)
    pickle.dump(svm, open("svm_model3.pkl", "wb"))
print("Testing...\n")
preds = []
labels = []
right = 0
total = 0
for x, y in zip(testdata, testlabel):
    x = x.reshape(1, -1)
    prediction = svm.predict(x)[0]
    if y == prediction:
        right += 1
    total += 1
    preds.append(prediction)
    labels.append(y)
preds=torch.Tensor(preds)
labels=torch.Tensor(labels)
cmtx = get_confusion_matrix(preds, labels, 21)
print(cmtx.tolist())
accuracy = float(right) / float(total)
print(str(accuracy) + "% accuracy")
print("Manual Testing\n")
