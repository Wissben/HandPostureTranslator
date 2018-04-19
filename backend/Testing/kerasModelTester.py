import numpy as np
from pandas import read_csv
from tensorflow.python.keras.models import load_model

MAX = 78071


def fileToArray(path, replaceMissing):  # Here you may redifine how u read from your file
    fileReader = open(path, 'r')
    data = read_csv(path, header=0).replace('?', replaceMissing).astype('float')

    values = data.values
    np.random.shuffle(values)

    dataTestSize = int(MAX)
    # return data.values[1500:1501,2:38]
    return values[0:dataTestSize, 2:38], values[0:dataTestSize, 38:42]


def loadModel(pathToModel):
    model = load_model(pathToModel)
    # Just in case you'd want to do stuff with your loaded model
    return model


pathToTestinFile = "/home/wiss/CODES/TP-AARN/Mini-Project/DataSets/dataOneOf5.csv"
pathToModel = "/home/wiss/CODES/TP-AARN/Mini-Project/backend/training/Models/Weigths/modelTest.hd5"
batchSize = 100
resultPath = "./results.csv"

testingSet, targetSet = fileToArray(pathToTestinFile, 0)
model = load_model(pathToModel)
Y = model.predict_proba(testingSet)
matching = []
for i, j in zip(Y, targetSet):
    matching.append([np.unravel_index(i.argmax(), i.shape) == np.unravel_index(j.argmax(), j.shape)])
unique_elements, counts_elements = np.unique(matching, return_counts=True)
print("ACCURACY:")
print(np.asarray((unique_elements, counts_elements))[1][1] * 100 / MAX)
