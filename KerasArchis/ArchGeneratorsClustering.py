import pathlib
from threading import Thread
import sys
from tensorflow.python.keras._impl.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from variator import Variator


###################### callbacks ###########################


def plotCallback(variator:Variator,model:Model):
    # get last history
    pathlib.Path('Graphs').mkdir(parents=True, exist_ok=True)
    history = variator.histories[-1]
    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(variator.currentParameters['modelName']+' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig.savefig("Graphs/"+variator.currentParameters['modelName']+'.png', dpi=fig.dpi)



def updateBestCallback(variator:Variator,bestModel:Model,bestScore):
    # get last history
    history = variator.histories[-1]
    pathlib.Path('Graphs').mkdir(parents=True, exist_ok=True)
    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(variator.currentParameters['modelName'] + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig.savefig("Graphs/bestModel.png", dpi=fig.dpi)
    from PIL import Image
    # Image.open('bestModel.png').show()

def writeCSV2(variator:Variator,model:Model,counter = [0,0]):
    history = variator.histories[-1]
    score = model.evaluate(tX,tY,verbose=0)

    acc = history.history['acc'][-1]
    val_acc = history.history['val_acc'][-1]
    print("counter: "+str(counter[1]))
    if score[1] > counter[1]:
        counter[1] = score[1]
        model_json = model.to_json()
        with open("Models/JSON/modelEvCluster_Architecure.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save("Models/Weights/bestEvModelCluster.hd5")
    with open("Logs/modelStats.csv","a") as f:
        if counter[0] == 0:
            counter[0] += 1
            f.write("Model,acc,val_acc,test_acc\n")
        f.write(variator.currentParameters['modelName']+","+str(acc)+","+str(val_acc)+","+str(score[1]))
        f.write("\n")



######################## generators of architectures ###############################

def layerGenerator(numberOfLayers=2, step=10, maxPerLayer=40):
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


if(len(sys.argv)<3):
	print("Too few arguments !")
	exit


################## training set #####################
pathToTrainingSet = sys.argv[1]
data = read_csv(pathToTrainingSet).replace('?', 0)
values = data.astype('float64').values
values = values[1:,:]
print(values.shape)
np.random.seed(7)
np.random.shuffle(values)

X = values[:, 2:-5]
Y = values[:, -5:]

################## one user left for testing ####################

data = read_csv(sys.argv[2]).replace('?', 0)
values = data.astype('float64').values
values = values[1:,:]
print(values.shape)
np.random.seed(7)
np.random.shuffle(values)

tX = values[:, 2:-5]
tY = values[:, -5:]




############### keras callbacks #####################
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto')
csv_logger = CSVLogger("Logs/modeltraining.log")

############## setting generators ####################
trainGenerator = {'layer':layerGenerator(),'optimizer':optimiezersGenerator()}
paramGenerator = Variator.createTrainParamsGenerator(X,Y,trainGenerator,retrains=4,
                                                     epochs=100,batch=200,callbacks=[monitor,csv_logger])


# training the model
variator = Variator(evaluationCallbacks=[plotCallback,writeCSV2],updateBestCallbacks=[updateBestCallback])
model,score = variator.trainCustom(X,Y,tX,tY,paramGenerator)

# save model
model_json = model.to_json()
with open("Models/JSON/modelCluster_Architecure.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save("Models/Weights/bestModelCluster.hd5")
