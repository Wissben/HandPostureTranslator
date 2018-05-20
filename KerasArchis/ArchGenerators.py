import os
import pathlib
from threading import Thread
import sys
from tensorflow.python.keras import optimizers
from tensorflow.python.keras._impl.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from variator import Variator


def layerGenerator(numberOfLayers=3, step=10, maxPerLayer=40):
    layers = [40, ]
    activations = ['relu']
    for num in range(numberOfLayers):
        layers.append(step)
        # default is relu
        activations.append('relu')
        numberOfSteps = int(maxPerLayer / step)
        for layerSize in range(numberOfSteps):
            yield layers, activations
            layers[-1] += step


def optimiezersGenerator():

    return ['Adadelta','SGD','Adam','RMSprop','Adadelta','Adagrad','Adamax','Nadam']


def plotCallback(variator: Variator, model: Model):
    # get last history
    pathlib.Path('Graphs').mkdir(parents=True, exist_ok=True)
    history = variator.histories[-1]
    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(variator.currentParameters['modelName'] + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig.savefig("Graphs/"+variator.currentParameters['modelName'] + '.png', dpi=fig.dpi)
    # from PIL import Image
    # Image.open(variator.currentParameters['modelName']+'.png').show()
    # plt.show()
    # plot(plt)


def updateBestCallback(variator: Variator, bestModel: Model, bestScore):
    pathlib.Path('Graphs').mkdir(parents=True, exist_ok=True)
    # get last history
    history = variator.histories[-1]
    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(variator.currentParameters['modelName'] + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig.savefig('Graphs/bestModel.png', dpi=fig.dpi)
    from PIL import Image
    # Image.open('bestModel.png').show()


def exludeUser(values,userId):
    leftUser = []
    otherUsers = []
    for val in values:
        if (val[1] == userId):
            leftUser.append(val)
        else:
            otherUsers.append(val)
    return np.array(leftUser),np.array(otherUsers)


if(len(sys.argv)<2):
	print("Too few arguments !")
	exit

# selector = Selector("allUsers.lcl.csv",6,2500)
# values = selector.getData()


#MUST SPECIFY WHERE THE DATA IS


#data = read_csv("../DataSets/dataOneOf5.csv").replace('?', 0)
#sys.argv must containt codified data.
data = read_csv(sys.argv[1]).replace('?', 0)
leftUser, values = exludeUser(data.values, 14)
values = values[1:, :]
leftUser = leftUser[1:, :]
print(values, leftUser)
np.random.seed(7)
np.random.shuffle(values)
np.random.shuffle(leftUser)

train_X = values[:, 2:-5]
train_Y = values[:, -5:]

test_X = leftUser[:, 2:-5]
test_Y = leftUser[:, -5:]

# create tesnsorboard callback
pathlib.Path('Logs').mkdir(parents=True, exist_ok=True)
os.system("rm Logs/*")
# tb = TensorBoard(log_dir='./Logs', histogram_freq=0, write_graph=True, write_images=True)
# create a csv logger callback
csv_logger = CSVLogger("Logs/modeltraining.log")
trainGenerator = {'layer': layerGenerator(), 'optimizer': optimiezersGenerator()}
paramGenerator = Variator.createTrainParamsGenerator(train_X, train_Y, trainGenerator, retrains=2,
                                                     epochs=20, batch=200, callbacks=[csv_logger])

# training the model
variator = Variator(evaluationCallbacks=[plotCallback], updateBestCallbacks=[updateBestCallback])
model, score = variator.trainCustom(train_X,train_Y,test_X,test_Y, paramGenerator)
