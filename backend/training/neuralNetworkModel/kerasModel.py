import numpy as np
from pandas import read_csv
from sklearn.preprocessing import Imputer
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

stats = open("Models/stats.csv", 'w')
stats.write("layers,tranFnc,opt,epochs,batch,score,loss\n")
file = "/home/wiss/CODES/TP-AARN/Mini-Project/DataSets/dataOneOf5.csv"
# SIZE OF INPUT DATASET TO TRAIN ON (MAX IS 78071)
MAX = 1000
trainingRows = int(MAX * 75 / 100)
testingRows = MAX - trainingRows


def replaceMissingData(X, values):
    # fill missing values with mean column values
    imputer = Imputer()
    transformed_X = imputer.fit_transform(X)
    transformed_values = imputer.fit_transform(values)
    # count the number of NaN values in each column
    print(np.isnan(transformed_values).sum())
    return transformed_X


def evaluateModel(model, X, Y, batch):
    score = model.evaluate(X, Y, batch_size=batch, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # sleep(25)
    return score


def trainModel(modelName, train_X, train_Y, test_X, test_Y, layers, inputDime, outputDim, transfnc, optimizer, epochs,
               batch):
    # create tesnsorboard callback
    tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    ## create model
    model = Sequential()
    model.add(Dense(layers[0], input_dim=inputDime, activation=transfnc))

    # setup model's architecure
    for i in range(1, len(layers)):
        model.add(Dense(layers[i], activation=transfnc))
    model.add(Dense(outputDim, activation='softmax'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Fit the model
    model.fit(train_X, train_Y, epochs=epochs, batch_size=batch, callbacks=[tb], verbose=1, validation_split=0.15,
              shuffle=True)

    # evaluate the model
    scores = evaluateModel(model, test_X, test_Y, batch)

    # Saving the model
    architecture = ""
    for layer in layers:
        architecture = architecture + str(layer) + "-"
    architecture = architecture[0:len(architecture) - 1]
    stats.write(architecture + "," +
                transfnc + "," +
                optimizer + "," +
                str(epochs) + "," +
                str(batch) + "," +
                str(scores[1] * 100) +
                "," + str(scores[0]) + "\n")

    # serialize model to JSON
    model_json = model.to_json()
    with open("Models/JSON/" + modelName + "_Architecure.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save("Models/Weigths/" + modelName + ".hd5")
    print("SAVING " + modelName + " TO DISK")
    print("FINISHED TRAINING MODEL : " + modelName)
    # sleep(30)
    return model


##Biggest bullshit ever made :
# data = read_csv(file,header=None).replace('?',np.NaN)
##
data = read_csv(file, header=None).replace('?', 0)
values = data.values
np.random.shuffle(values)

training_X = values[0:trainingRows, 2:38]
training_Y = values[0:trainingRows, 37:42]
transformedTraining_X = replaceMissingData(training_X, data.values)
print(training_Y)

# print(values[trainingRows+1])
testing_X = values[trainingRows + 1:MAX, 2:38]
testing_Y = values[trainingRows + 1:MAX, 37:42]
transformedTesting_X = replaceMissingData(testing_X, data.values)
# print(transformed_X[2])


#training the model
model = trainModel("modelTest", transformedTraining_X, training_Y, transformedTesting_X, testing_Y, [50, 25, 15], 36, 5,
                   'relu', 'sgd', 150, 100)

# #
# Uncomment to launch multiple models training
# transFncs = ['relu', 'tanh', 'sigmoid']
# optimizers = ['sgd', 'adagrad', 'adam']
# inDim = 36
# outDim = 5
# batch = 125
# epochs = 150
# modelBaseNam= "model_"
# for tranFnc in transFncs:
#     for opt in optimizers:
#         # layers =[]
#         for i in range(40, 51):
#             for j in range(20, 26):
#                 for k in range(10, 15):
#                     modeName = modelBaseName + \
#                                str(i) + "_" + \
#                                str(j) + "_" + \
#                                str(k) + "_" + \
#                                tranFnc + "_" + \
#                                opt + "_" + str(batch)
#                     layers = [i, j, k]
#                     print("TRAINING MODEL  :" + modeName)
#                     trainModel(modeName, transformedTraining_X, training_Y, transformedTesting_X, testing_Y, layers,
#                                inDim, outDim, tranFnc, opt, epochs, batch)


stats.close()

