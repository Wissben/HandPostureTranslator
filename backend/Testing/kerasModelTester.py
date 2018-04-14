import numpy as np
from pandas import read_csv
from tensorflow.python.keras.models import load_model


def fileToArray(path):  # Here you may redifine how u read from your file
    fileReader = open(path, 'r')
    data = read_csv(path, header=None).replace('?', 0)
    values = data.values
    np.random.shuffle(values)

    dataTestSize = 1500
    return values[0:dataTestSize, 2:38]


def loadModel(pathToModel):
    model = load_model(pathToModel)
    # Just in case you'd want to do stuff with your loaded model
    return model


pathToTestinFile = "/home/wiss/CODES/TP-AARN/Mini-Project/DataSets/dataOneOf5.csv"
pathToModel = "/home/wiss/CODES/TP-AARN/Mini-Project/backend/training/neuralNetworkModel/Models/Weigths/modelTest.hd5"
batchSize = 100
resultPath = "./results.csv"

testingSet = fileToArray(pathToTestinFile)
model = load_model(pathToModel)
score = model.predict(testingSet, batchSize)
print(score)
