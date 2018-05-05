import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.optimizers import Adam, SGD
from random import shuffle
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import numpy as np
#with open('annotated_train_set.p', "rb",) as input_file:
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
y = np.zeros((no_of_videos,no_of_classes))
class_map_dict = {}
for i, video in enumerate(data):
    x[i] = video[b'features']
    y[i, video[b'class_num']] = 1
    class_map_dict[video[b'class_num']] = video[b'class_name']

'''
Random shuffling as keras val_split option takes last few samples from train 
data sequentially
'''
P = [i for i in range(no_of_videos)]
x_shuf = np.zeros((no_of_videos, no_of_frames, feature_len))
y_shuf = np.zeros((no_of_videos,no_of_classes))

shuffle(P)
x = x[P]
y = y[P]

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
Keras model definition
'''
tbCallBack =TensorBoard(log_dir='./1_2_Graph', histogram_freq=0,
          write_graph=True, write_images=False)
model = Sequential()
model.add(LSTM(256, input_shape=(10, 512)))
#model.add(Dropout(0.2))
model.add(Dense(no_of_classes, activation='softmax'))   #should be no of classes

sgd = SGD(lr=0.0001, momentum=0.9, decay=1e-4, nesterov=False)
adam = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


#history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=128, class_weight=class_weight)
history = model.fit(x, y, validation_split=0.2, epochs=50, batch_size=64, callbacks=[tbCallBack])
#best epochs = 50, sgd

'''
Writing test data predictions
'''
to_write = []
outputs = model.predict(x_test)
pred = np.argmax(outputs, axis=1)

file_name = 'part1.2.txt'
file = open(file_name, 'w')

for i in pred:
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
# plt.close()
# ## loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()