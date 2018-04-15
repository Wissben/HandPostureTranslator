import numpy as np
from pandas import read_csv
from tensorflow.python.keras.models import load_model


def fileToArray(path):  # Here you may redifine how u read from your file
    fileReader = open(path, 'r')
    data = read_csv(path, header=None).replace('?', 0).astype('float')
    values = data.values
    # np.random.shuffle(values)

    dataTestSize = 10000
    # return data.values[1500:1501,2:38]
    return values[0:dataTestSize, 2:38], values[0:dataTestSize, 37:42]


def loadModel(pathToModel):
    model = load_model(pathToModel)
    # Just in case you'd want to do stuff with your loaded model
    return model


pathToTestinFile = "/home/wiss/CODES/TP-AARN/Mini-Project/DataSets/dataOneOf5.csv"
pathToModel = "/home/wiss/CODES/TP-AARN/Mini-Project/backend/training/Models/Weigths/modelTest.hd5"
batchSize = 100
resultPath = "./results.csv"

testingSet, targetSet = fileToArray(pathToTestinFile)
model = load_model(pathToModel)
Y = model.predict_proba(testingSet)
matching = []
for i, j in zip(Y, targetSet):
    matching.append([np.unravel_index(i.argmax(), i.shape)[0] == np.unravel_index(j.argmax(), j.shape)[0]])
unique_elements, counts_elements = np.unique(matching, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))
