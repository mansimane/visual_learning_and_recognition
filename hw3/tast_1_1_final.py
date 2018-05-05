import pickle
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy
from torch import max as torch_max
from logger import *
import math
#with open('annotated_train_set.p', "rb",) as input_file:
from torch import mean as torch_mean

'''
Loading data
Data is list of dictionaries (1 dict per video)
each dict has keys: dict_keys([b'class_num', b'class_name', b'features'])
data[0][b'features'].shape: (10, 512)

Ref: https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
'''
with open('annotated_train_set.p', "rb",) as input_file:
    data = pickle.load(input_file, encoding='bytes')
no_of_classes = 51
data = data[b'data']
no_of_frames, feature_len = data[0][b'features'].shape

no_of_videos = len(data)
x = np.zeros((no_of_videos, no_of_frames, feature_len))
y = np.zeros((no_of_videos, no_of_classes))
class_map_dict = {}
for i, video in enumerate(data):
    x[i] = video[b'features']
    y[i, video[b'class_num']] = 1
    class_map_dict[video[b'class_num']] = video[b'class_name']

'''
Load Test data
'''
with open('randomized_annotated_test_set_no_name_no_num.p', "rb",) as input_file:
    test_data = pickle.load(input_file, encoding='bytes')
test_data = test_data[b'data']

x_test = np.zeros((len(test_data), no_of_frames, feature_len))
for i, video in enumerate(test_data):
    x_test[i] = video[b'features']


'''
Random shuffling as keras val_split option takes last few samples from train 
data sequentially
'''
P = [i for i in range(no_of_videos)]
split_ratio = 0.8
no_of_train = int(no_of_videos * split_ratio)
no_of_valid = no_of_videos-no_of_train

shuffle(P)
x = x[P]
y = y[P]
x_train = x[P[0:no_of_train]]
x_val = x[P[no_of_train:]]

y_train = y[P[0:no_of_train]]
y_val = y[P[no_of_train:]]


'''
Pytorch model definition
Avgpool: inp (N,C,Lin)
         out (N,C,Lout)
'''
no_of_epochs = 80
lr = 0.0001
weight_decay = 0.00000
logdir = './tboard_'+ 'batch_' '+ ''epochs_'+ str(no_of_epochs)+ '_lr_' + str(lr)+ '_wd_' + str(weight_decay)+ '_bn1d'
print(logdir)
logger_t = Logger(logdir)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features= feature_len, out_features = 1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features= 1024, out_features = 256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features= 256, out_features = no_of_classes),
        )

        self.p1 = nn.Sequential(
            nn.AvgPool1d(no_of_frames, stride=None, padding=False, ceil_mode=False, count_include_pad=False),
            #nn.MaxPool1d(no_of_frames, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False),
        )

    def forward(self, x):

        # size batchx 10x512
        x = self.fc1(x)
        # size bx10x51
        #x = x.view(x.size()[0], x.size()[2], x.size()[1])
        # size 1x51x10
        #x = self.p1(x)
        #size 1x51x1
        #x = x.view(x.size()[0], x.size()[1])
        #reshaping again as cross entropy needs NxC inp
        #size 1 x 51
        return x

net = Net()
for f in net.fc1:
            if isinstance(f, nn.Linear):
                #from IPython.core.debugger import Tracer; Tracer()()
                sum_io = f.weight.size()[0] + f.weight.size()[1]
                f.weight.data.normal_(0, math.sqrt(2.0 / sum_io))

'''
Training pytorch model
'''
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay= weight_decay)
optimizer = optim.Adam(net.parameters(), lr=lr)
P = [i for i in range(len(y_train))]
total = no_of_train
batch_size = 64
no_of_batches = math.floor(no_of_train/batch_size)

for epoch in range(no_of_epochs):  # loop over the dataset multiple times
    #Shuffle
    shuffle(P)
    x_train = x_train[P]
    y_train = y_train[P]

    running_loss = 0.0
    correct = 0.0
    total = no_of_train
    for i in range(no_of_batches):
        start = i*batch_size
        end = start + batch_size
        x_single, y_single = x_train[start:end], y_train[start:end]
        # wrap them in Variable
        inputs, labels = Variable(from_numpy(x_single)).float(), Variable(from_numpy(y_single)).long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        outputs = torch_mean(outputs, 1)

        loss = criterion(outputs, torch_max(labels, 1)[1])
        pred = torch_max(outputs, 1)[1]
        pred_np = np.array(pred.data)
        correct += np.sum(pred_np == [np.argmax(y_single)])
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i %3 == 2:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / i))
    train_acc = correct/total
    print(' train_acc: %.3f' %
          ( train_acc))

    logger_t.scalar_summary(tag='train_acc', value=train_acc, step=epoch)
    # get validation acc at end of each epoch
    total, correct = 0.0, 0.0
    for i in range(no_of_valid):

        x_single, y_single = x_val[i], y_val[i]
        x_single = x_single[np.newaxis,:,:]
        y_single = y_single[np.newaxis, :]
        # wrap them in Variable
        inputs, labels = Variable(from_numpy(x_single)).float(), Variable(from_numpy(y_single)).long()
        outputs = net(inputs)
        outputs = torch_mean(outputs, 1)

        _, predicted = torch_max(outputs.data, 1)
        total += labels.size(0)
        predicted = predicted.numpy()
        true = np.argmax(y_single, axis=1)
        correct += (predicted == true).sum()
        #correct += (predicted == torch_max(labels, 1)[1]).sum()
    val_acc = correct/total

    logger_t.scalar_summary(tag= 'val_acc', value= val_acc, step= epoch)
    logger_t.scalar_summary(tag= 'train_loss', value= running_loss/i, step= epoch)

    print('[%d] val_acc: %.3f' %
          (epoch + 1, correct / total))

to_write = []
for i in range(len(test_data)):
    x_single = x_test[i]
    x_single = x_single[np.newaxis,:,:]
    inputs = Variable(from_numpy(x_single)).float()
    outputs = net(inputs)
    outputs = torch_mean(outputs, 1)
    _, predicted = torch_max(outputs.data, 1)
    predicted = np.array(predicted)
    for p in predicted:
        to_write.append(p)

file_name = 'part1.1.txt'
file = open(file_name, 'w')

for i in to_write:
    file.write("%s\n" % i)
file.close()
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#



