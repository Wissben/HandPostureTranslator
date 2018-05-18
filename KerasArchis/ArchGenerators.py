from threading import Thread

from tensorflow.python.keras._impl.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from KerasArchis.variator import Variator


def layerGenerator(numberOfLayers=1,step=20,maxPerLayer=40):
    layers = [40,]
    activations = ['relu']
    for num in range(numberOfLayers):
        layers.append(step)
        #default is relu
        activations.append('relu')
        numberOfSteps = int(maxPerLayer/step)
        for layerSize in range(numberOfSteps):
            yield layers,activations
            layers[-1] += step

def optimiezersGenerator():
    return ['sgd','adamax']

def plotCallback(variator:Variator,model:Model):
    # get last history
    history = variator.histories[-1]
    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(variator.currentParameters['modelName']+' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig.savefig(variator.currentParameters['modelName']+'.png', dpi=fig.dpi)
    # from PIL import Image
    # Image.open(variator.currentParameters['modelName']+'.png').show()
    # plt.show()
    # plot(plt)



def updateBestCallback(variator:Variator,bestModel:Model,bestScore):
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
    fig.savefig('bestModel.png', dpi=fig.dpi)
    from PIL import Image
    # Image.open('bestModel.png').show()


# selector = Selector("allUsers.lcl.csv",6,2500)
# values = selector.getData()

data = read_csv("../DataSets/newFileTrain.csv").replace('?', 0)
values = data.astype('float64').values
values = values[1:,:]
print(values.shape)
np.random.seed(7)
np.random.shuffle(values)

X = values[:, 2:-5]
Y = values[:, -5:]

data = read_csv("../DataSets/newFileTest.csv").replace('?', 0)
values = data.astype('float64').values
values = values[1:,:]
print(values.shape)
np.random.seed(7)
np.random.shuffle(values)

tX = values[:, 2:-5]
tY = values[:, -5:]

def writeCSV(variator:Variator,model:Model,counter = [0]):
    history = variator.histories[-1]
    score = model.evaluate(tX,tY,verbose=0)


    with open("logs/modelStats.csv","a") as f:
        if counter[0] == 0:
            counter[0] += 1
            f.write("Model,"+str(model.loss))
            for metric in model.metrics:
                f.write(',' + str(metric))
            f.write("\n")

        f.write(variator.currentParameters['modelName']+","+str(score[0]))
        index = 1
        for metric in model.metrics:
            f.write(','+str(score[index]))
            index += 1
        f.write("\n")



monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto')
# create tesnsorboard callback
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
# create a csv logger callback
csv_logger = CSVLogger("logs/modeltraining.log")
trainGenerator = {'layer':layerGenerator(),'optimizer':optimiezersGenerator()}
paramGenerator = Variator.createTrainParamsGenerator(X,Y,trainGenerator,retrains=2,
                                                     epochs=20,batch=200,callbacks=[monitor,tb,csv_logger])




# training the model
variator = Variator(evaluationCallbacks=[plotCallback,writeCSV],updateBestCallbacks=updateBestCallback)
model,score = variator.trainB(X,Y,tX,tY,paramGenerator)

model_json = model.to_json()
with open("Models/JSON/modelCluster_Architecure.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save("Models/Weights/bestModelCluster.hd5")