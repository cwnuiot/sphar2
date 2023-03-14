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
    with open(path, "r") as f:
        for line in f.readlines():
            data = []
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            x = line.split()
            data.append(int(x[1]))
            mmm = float(x[4])
            if mmm > 0:
                mmm = mmm / 1089.625
            if mmm < 0:
                mmm = mmm / 1402.125
            data.append(mmm)
            data.append(float(x[5]) / (-81.5))
            data.append(float(x[6]) / 6.280117345603815)
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
    test_data = torch.Tensor(test_data)
    test_label = torch.Tensor(test_label)
    train_data = torch.Tensor(train_data)
    train_label = torch.Tensor(train_label)
    #iloc选择参数序列中所对应的行
    return train_data,train_label,test_data,test_label

traindata,trainlabel,testdata,testlabel=split_train_test(datas,labels,0.2)
from torch.utils.data import Dataset, DataLoader, TensorDataset

traindata = TensorDataset(traindata, trainlabel)
testdata = TensorDataset(testdata, testlabel)
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.001
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(     # LSTM 效果要比 nn.RNN() 好多了
            input_size=1600,      # 图片每行的数据像素点
            hidden_size=300,     # rnn hidden unit
            num_layers=1,       # 有几层 RNN layers
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
            bidirectional=True
        )
        #self.con1 = nn.Conv1d(1, 1, kernel_size=16, stride=2, padding=7)
        self.fc = nn.Linear(600, 200)
        self.out = nn.Linear(200, 21)    # 输出层
    def forward(self, x):
        #print(x.size())
        #x = self.con1(x)
        #print(x.size())
        #r_out, (h_n, h_c) = self.rnn(x, None)   # None 表示 hidden state 会用全0的 state
        r_out, h_n = self.rnn(x, None)
        #r_out, h_n = self.rnn1(r_out, None)
        #print(r_out.size())
        out = self.fc(r_out[:, -1, :])
        out = self.out(out)
        return out
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
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    # If labels are one-hot encoded, get their indices.
    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    # Get the predicted class indices for examples.
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)
    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)))#, normalize=normalize) 部分版本无该参数
    return cmtx


device = torch.device('cuda')
rnn=RNN().to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all parameters
loss_func = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(dataset=traindata, batch_size=BATCH_SIZE, shuffle=True)
def train(EPOCH):
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
            #print(b_x)
            b_y=torch.tensor(b_y,dtype=torch.int64)
            b_x, b_y = b_x.to(device), b_y.to(device)
            b_x = b_x.view(-1, 1, 1600)
            #print(b_y)
            output = rnn(b_x)  # rnn output
            #print(output.dtype)
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            '''if step % 50 == 0:
                test_output = rnn(b_x)  # (samples, time_step, input_size)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)'''
best_acc =[]
def test(ep):
    test_loss = 0
    correct = 0
    preds = []
    labels = []
    # data, label = data0,label0
    test_data = DataLoader(testdata, batch_size=32, shuffle=True)
    if ep == 1:
        torch.save(rnn.state_dict(), 'RNN500.mdl')
    with torch.no_grad():
        for data, target in test_data:
            target = torch.tensor(target, dtype=torch.int64)
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 1, 1600)
            data, target = data.to(device), target.to(device)
            data, target = Variable(data, volatile=True), Variable(target)
            output = rnn(data).to(device)
            preds.append(output.cpu())
            labels.append(target.cpu())
            test_loss += loss_func(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_data.dataset)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        cmtx = get_confusion_matrix(preds, labels, 21)
        print(cmtx)
        # print('============================')
        # print(len(test_data.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_data.dataset),
            100. * correct / len(test_data.dataset)))
        acc = correct / len(test_data.dataset)
        best_acc.append(float(acc))
        print(best_acc)
        print(max(best_acc))
        return test_loss
if __name__ == '__main__':
    begintime=time.time()
    for epoch in range(0, 25):
        print(epoch)
        train(epoch)
        test(epoch)
        if epoch %10==0:
            LR /= 10
    endtime=time.time()
    print('time',endtime-begintime)
