#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
#grab datafile using pandas
dataFrame = pd.read_csv("yeast.data", delim_whitespace=1, names=["mcg","gvh","alm","mit","erl","pox","vac","nuc","class"])

#outlier detection using Isolation Forest:
df = dataFrame.drop(columns="class")
dfOnlyClasses = dataFrame.drop(columns=["mcg","gvh","alm","mit","erl","pox","vac","nuc"])
IF = IsolationForest(n_estimators=8)
IF.fit(df)
#negative values are outliers
#print(len(outliersWithIF))
#find outliers and remove them
count = 0
IFlen = []
for x in IF.predict(df):
    if x < 0 and count < len(df):
        df = df.drop(df.index[count])
        dfOnlyClasses = dfOnlyClasses.drop(dfOnlyClasses.index[count])
        IFlen.append(x)
    count = count + 1
#print(df)
#print(dfOnlyClasses)

#outlier dectection using one-class SVM
OCSVM = OneClassSVM(gamma='auto').fit(dataFrame.drop(columns="class"))
#negative values are outliers
outliersWithOCSVM = [x for x in OCSVM.predict(dataFrame.drop(columns="class")) if x < 0]
print('amount of outliers using Isolation Forest:')
print(len(IFlen))
print('amount of outliers using one-class SVM:')
print(len(outliersWithOCSVM))


# In[2]:


#import what's needed to build ann
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

#encode classes using one hot encoder:
enc = OneHotEncoder()
enc.fit(dfOnlyClasses)
OneHotEncoder(categorical_features=None, categories=None, drop=None, handle_unknown='ignore', n_values=None)
y = enc.transform(dfOnlyClasses).toarray()

import numpy as np
np.set_printoptions(threshold=np.inf)
y = np.array(y)
#split data into training and testing:
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.34, random_state=42)
#print(X_train) #test
#build ann model using keras:
model = Sequential()
model.add(Dense(3, activation='sigmoid', input_dim=8))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
#comile the model:
from keras.optimizers import SGD
opt = SGD(lr=10)
model.compile(
  optimizer=opt,
  loss='mean_squared_error',
  metrics=['accuracy'],
)
weights_history = []
_biases= []
# A custom callback to get weights for plotting
class MyCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs):
        weights = model.get_weights()[0]
        biases = model.get_weights()[1]
        w1,w2,w3,w4,w5,w6,w7,w8 = weights
        w01,w02,w03 = biases
        biases = [w01]
        weights = [w1[0], w2[0], w3[0]]
        weights_history.append(weights)
        _biases.append(biases)

#create checkpoint for plot, to grab val_loss
callback = MyCallback()
filepath="weights-improvement-{epoch:04d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

#fit the model with training set:
history = model.fit(
  X_train,
  y_train,
  epochs=150,
  batch_size=100, 
  verbose=1,
  validation_split=0.1,
  callbacks=[callback]
)

#test model using training set:
loss, accuracy = model.evaluate(
  X_test,
  y_test
)
print('Loss: %.2f' % loss)
print("Accuracy %.2f" %(accuracy))
model.save_weights('model.h5')
#get weights and biases from output layer for problem #4:
from tensorflow.contrib.keras import layers
output_layer_weights_lastLayer = model.layers[2].get_weights()[0]
output_layer_weights_secondtolastlayer  = model.layers[1].get_weights()[0]
output_layer_biases_lastLayer = model.layers[2].get_weights()[1]
output_layer_biases_secondtolastlayer  = model.layers[1].get_weights()[1]
print("output_layer_weights_lastLayer: ")
print(output_layer_weights_lastLayer) #test
print("output_layer_weights_secondtolastLayer: ")
print(output_layer_weights_secondtolastlayer) #test
print("output_layer_biases_lastLayer: ")
print(output_layer_biases_lastLayer) #test
print("output_layer_biases_secondtolastLayer: ")
print(output_layer_biases_secondtolastlayer) #test

import matplotlib.pyplot as plt
#plot the training and testing error:
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('train/test error per iteration')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plot the weights for CYT class:
plt.axis([0, 150, -1, 1])
plt.plot(weights_history)
plt.plot(_biases)
plt.title('weights per iteration for CYT class')
plt.ylabel('weights')
plt.xlabel('epoch')
plt.legend(['w0','w1', 'w2','w3'], loc='upper left')
plt.show()










# In[3]:



#create df including all data as stated in directions
dataFrameWithAllData = pd.read_csv("yeast.data", delim_whitespace=1, names=["mcg","gvh","alm","mit","erl","pox","vac","nuc","class"])
dfe = dataFrameWithAllData.drop(columns="class")
#grab classes for model:
cdfe = dataFrameWithAllData.drop(columns=["mcg","gvh","alm","mit","erl","pox","vac","nuc"])

#use new one hot encoder for new y column vector
from sklearn.preprocessing import LabelEncoder
#label encode for decoding later:
l = LabelEncoder()
e = l.fit_transform(cdfe)
#use one hot encoder to encode classes, this time without removing data as instructed

newenc = OneHotEncoder()
newenc.fit(cdfe)
OneHotEncoder(categorical_features=None, categories=None, drop=None, handle_unknown='ignore', n_values=None)
y = enc.transform(cdfe).toarray()
#turn into numpy array
y = np.array(y)
#create counter 
it=1
#create empty list to append argmax's to, basically what index 1 lands on from ohe
f = []
#decodes on hot encoder: 
#create a list of indexes to decode
for i in y:
    f.append(np.argmax(i))
#reset counter:
it = 1
#decode and print, this will show you my encoded classes
#what row you are on, and the decoded class
for j in l.inverse_transform(f):
    print(it, end =" ")
    print(y[it-1])
    print(j)
    it = it + 1
#Train model with all data
#it will show less in model because I am using 0.1 for validation
history = model.fit(
  dfe,
  y,
  epochs=150,
  batch_size=100, 
  verbose=1,
  validation_split=0.1
)
#now get training error, to do this I run model with training data, in this case it is all the data
loss, accuracy = model.evaluate(dfe,y)
print("acc: ")
print(accuracy)


# In[4]:



X_train= np.array(X_train)
def createModel(i,j):
    modelSearch = Sequential()
    for nodes in range(0,j+1):
        modelSearch.add(Dense(i, activation='sigmoid'))
    modelSearch.add(Dense(10, activation='sigmoid'))
    modelSearch.compile(
      optimizer=opt,
      loss='mean_squared_error',
      metrics=['accuracy'],
    )
    modelSearch.fit(
      X_train,
      y_train,
      epochs=150,
      batch_size=100
    )
    loss, accuracy = modelSearch.evaluate(
      X_test,
      y_test
    )
    print('Loss: %.2f' % loss)
    print("Accuracy %.2f" %(accuracy))
for i in range(1,4):
    for j in range(3,13):
        if(j%3 == 0):
            createModel(i,j)


# In[5]:



#create np array for prediction
x = np.array([[0.52], [0.47], [0.52], [0.23], [0.55], [0.03], [0.52], [0.39]])
#using model for prediction, change model name for two seperate models!
prediction = model.predict(x.T, verbose=1)
print(prediction)
#highest number is class, can find class from decoding.


# In[6]:



newModel = Sequential()
newModel.add(Dense(3, activation='relu', input_dim=8))
newModel.add(Dense(3, activation='relu'))
newModel.add(Dense(10, activation='softmax'))
#comile the model:
newModel.compile(
  optimizer=opt,
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)
history = newModel.fit(
  X_train,
  y_train,
  epochs=150,
  batch_size=100, 
  verbose=1,
  validation_split=0.1
)
loss, accuracy = newModel.evaluate(
  X_test,
  y_test
)
print(accuracy)
#now using grid search:
def createNewModel(i,j):
    modelSearch = Sequential()
    for nodes in range(0,j+1):
        modelSearch.add(Dense(i, activation='relu'))
    modelSearch.add(Dense(10, activation='softmax'))
    modelSearch.compile(
      optimizer=opt,
      loss='categorical_crossentropy',
      metrics=['accuracy'],
    )
    modelSearch.fit(
      X_train,
      y_train,
      epochs=150,
      batch_size=100,
      verbose=1,
      validation_split=0.1,
      callbacks=[callback]
    )
    loss, accuracy = modelSearch.evaluate(
      X_test,
      y_test
    )
    print('Loss: %.2f' % loss)
    print("Accuracy %.2f" %(accuracy))
for i in range(1,4):
    for j in range(3,13):
        if(j%3 == 0):
            createNewModel(i,j)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('train error per iteration')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['training error', 'testing error'], loc='upper left')
plt.show()


# In[ ]:




