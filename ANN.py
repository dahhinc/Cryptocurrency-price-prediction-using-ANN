import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential,Input
from keras.layers import Dense
from keras.layers import LSTM,SimpleRNN,GRU
from keras.layers import Flatten,BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.callbacks import Callback,TensorBoard
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import ReLU
from keras.losses import quantile
import keras.backend as K
import keras.regularizers as reg
#from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from IPython.display import clear_output
import lime
import time

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

def clearData(data,skiprow):
    for i in range(0,data.shape[1]):
        if(i!= skiprow):
          data[:,i] = 0
    return data

K.clear_session()

tsteps = 1
batch_size = 16
ep = 100
lr = 0.0055
momentum =0
beta_1 = 0.75
beta_2 = 0.985
epsilon = 1e-12
decay = 0

pred_colnum = 0

rawdata = pandas.read_csv('Bitcoin.csv', engine='python')
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

keras_adam = Adam(lr = lr, beta_1 = beta_1, beta_2 = beta_2,
                      epsilon = epsilon, decay = False, amsgrad = True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                              patience=2, min_lr=0.0001)


early_stop= EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

#quantile = 0.3
#qnt = lambda y,f: tilted_loss(quantile,y,f)

re = reg.l1_l2(l1 = 0, l2=0.001)

model = Sequential()
model.add(LSTM(12,return_sequences = True, stateful = True,batch_size=batch_size,input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.05))
model.add(SimpleRNN(units=9,return_sequences = True, stateful = False))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss=quantile, optimizer=keras_adam)
history = model.fit(trainX, trainY, epochs=ep, batch_size=batch_size, verbose=2,validation_data = (validateX, validateY),
                    callbacks = [plot_losses, reduce_lr, early_stop])


# make predictions
trainPredict = model.predict(trainX,batch_size=batch_size)
testPredict = model.predict(testX,batch_size=batch_size)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:-2,0], testPredict[:-2,0]))
print('Test Score: %.2f RMSE' % (testScore))

# invert predictions
trainPredict = denrm(trainPredict,train_n)
testPredict = denrm(testPredict,test_n)

trainY_den =denrm(trainY,train_n)
testY_den = denrm(testY,test_n)

trainScore = mean_absolute_error(trainY_den[:,0], trainPredict[:,0])
print('Train Score in $: %.2f MAE' % (trainScore))
testScore = mean_absolute_error(testY_den[:-2,0], testPredict[:-2,0])
print('Test Score in $: %.2f MAE' % (testScore))


def mapee(y_true, y_pred):
    return 100 * numpy.sum(numpy.divide(numpy.abs(numpy.add(y_true, numpy.multiply(y_pred, -1))),y_pred))/y_true.shape[0]


trainMAPE =mapee(trainY_den[:,0], trainPredict[:,0])
print('Train mean absolute error : %.2f MAPE' % (trainMAPE))
testMAPE = mapee(testY_den[:-2,0], testPredict[:-2,0])
print('Test mean absolute error : %.2f MAPE' % (testMAPE))


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(BTC)
trainPredictPlot[:, :] = 0
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(BTC)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+1:len(trainPredict)+len(testPredict)+1, :1] = testPredict
# plot baseline and predictions

Bitcoin = pandas.read_csv('path/Bitcoin.csv', engine='python')
Bitcoin = Bitcoin.dropna()
Bitcoin["Price"] = Bitcoin["Price"].shift(-2)
BTC = Bitcoin.values
#BTC = BTC.astype('float32')

expected = BTC[len(trainPredict)+1+val_size+11:len(BTC),1]


plt.clf()
firstplot = plt.figure(2,figsize=(8, 6))
expec, = plt.plot(expected,  label='Expected')
predic, = plt.plot(testPredict,  label='Predicted')
plt.legend(handles=[expec, predic])
plt.show()

plt.clf()
plt.plot()
plt.show()
