import numpy
import matplotlib.pyplot as plt
import pandas
import math
from time import time, sleep
from keras.models import Sequential,Input
from keras.layers import Dense
from keras.layers import LSTM,SimpleRNN,GRU
from keras.layers import Flatten,BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.losses import quantile
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import ReLU
import keras.backend as K
import keras.regularizers as reg
from keras.callbacks import TensorBoard
import tensorflow as tf
import plotly.plotly as py
import plotly.graph_objs as go
#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from IPython.display import clear_output
#import lime
plt.rcParams.update({'font.size': 22})

numpy.random.seed(3)

def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()
def nrm(data):
    cf = numpy.empty([data.shape[1],2])
    for i in range(0,data.shape[1]):
        dt = data[:,i]
        mn = numpy.amin(dt)
        mx = numpy.amax(dt)
        #print(mn," ",mx)
        data[:,i] = (dt - mn) / (mx - mn)
        cf[i] = [mn,mx]
    return data,cf

def denrm(data,values):
    mn = values[0,0]
    mx = values[0,1]
    data = numpy.multiply(data,(mx-mn)) + mn
    return data

#to replace cols with 0, except analyzing one
def clearData(data,skiprow):
    for i in range(0,data.shape[1]):
        if(i!= skiprow):
          data[:,i] = 0
    return data


K.clear_session()

#Set of parameters

tsteps = 1
batch_size = 16

momentum =0
beta_1 = 0.75
beta_2 = 0.985
epsilon = 1e-12
decay = 0

#column number for Price
pred_colnum = 0

rawdata = pandas.read_csv('path/Bitcoin.csv', engine='python')
rawdata = rawdata.dropna()
rawdata["Price"] = rawdata["Price"].shift(-2)



BTC = rawdata.iloc[:,1:8].copy()

BTC = BTC.values
BTC = BTC.astype('float32')

train_size = (int((len(BTC)-1) * 0.80) //batch_size)*batch_size
val_size = (int(train_size * 0.25) // batch_size)*batch_size
test_size = ((len(BTC) - train_size) // batch_size)*batch_size
test_size_skip = len(BTC) - train_size - test_size - 1
train_size = train_size - val_size

print("Train size: ", train_size, " Val size: ", val_size, " Test size: ", test_size, " Test size skip: ", test_size_skip)


train, validate, test = BTC[0:train_size,:], BTC[train_size+1:(train_size + val_size)+1,:], BTC[(train_size + val_size)+1+test_size_skip:len(BTC),]

test[-1,0], test[-2,0] = test[-3,0], test[-3,0]

train, train_n = nrm(train)
test, test_n = nrm(test)
validate, validate_n = nrm(validate)

trainX = numpy.reshape(train, (train.shape[0], train.shape[1], tsteps))
trainY = trainX[:,pred_colnum,:]
trainX = numpy.delete(trainX,pred_colnum,axis=1)


testX = numpy.reshape(test, (test.shape[0], test.shape[1], tsteps))
testY = testX[:,pred_colnum,:]
testX = numpy.delete(testX,pred_colnum,axis=1)

validateX = numpy.reshape(validate, (validate.shape[0], validate.shape[1], tsteps))
validateY = validateX[:,pred_colnum,:]
validateX = numpy.delete(validateX,pred_colnum,axis=1)




early_stop= EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

def baseline_model (ep, learning_rate, unit_lstm, unit_rnn):

    keras_adam = Adam(lr = learning_rate, beta_1 = beta_1, beta_2 = beta_2,
                      epsilon = epsilon, decay = False, amsgrad = True)
    re = reg.l1_l2(l1 = 0, l2=0.01)

#model by itself
    model = Sequential()
    model.add(LSTM(unit_lstm,return_sequences = True, stateful = True,batch_size=batch_size,input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(0.05))
    model.add(SimpleRNN(units=unit_rnn,return_sequences = True, stateful = True))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss=quantile, optimizer=keras_adam)
    return model

#loop for hyperparameters search
def model_params( ep, learning_rate, unit_lstm, unit_rnn):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                  patience=2, min_lr=0.00001)
    params = list()
    for i in unit_lstm:
        for j in unit_rnn:
            for k in ep:
                for l in learning_rate:
                    mdl = baseline_model(k, l, i, j)
                    history = mdl.fit(trainX, trainY, epochs=k, batch_size=batch_size, verbose=2,
                                      validation_data = (validateX, validateY), callbacks = [plot_losses, reduce_lr,early_stop])
                    trainPredict = mdl.predict(trainX,batch_size=batch_size)
                    testPredict = mdl.predict(testX,batch_size=batch_size)
                    trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
                    testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
                    
                    params.append([str(k) + "," + str(l) + "," +str(i) + "," + str(j),trainScore,testScore])
    print('Total grid size: %d' % len(params))
    return params


start= time()
grid = []
unit_lstm= [11, 25]
unit_rnn= [5, 10]
ep = [35, 55]
learning_rate = [0.005, 0.0055]  #learning rate
grid = model_params(ep,learning_rate,unit_lstm,unit_rnn)
end = (time() - start)
print(end)

def sortsecond(val2):
    return val2[2]

grid1=grid
grid1.sort(key=sortsecond)
