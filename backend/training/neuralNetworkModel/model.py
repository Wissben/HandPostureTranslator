import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from time import sleep

stats = open("Models/stats.csv", 'w')
stats.write("layers,tranFnc,opt,epochs,batch,score\n")


def trainModel(modelName, X, Y, layers, inputDime, outputDim, transfnc, optimizer, epochs, batch):
    # # create model
    print("TRAINING MODEL : " + modelName)
    # sleep(0.5)

    model = Sequential()
    model.add(Dense(layers[0], input_dim=inputDime, activation=transfnc))
    model.add(Dense(layers[1], activation=transfnc))
    model.add(Dense(layers[2], activation=transfnc))
    model.add(Dense(outputDim, activation='softmax'))
    #
    # # Compile model
    # optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # # Fit the model
    model.fit(X, Y, epochs=epochs, batch_size=batch)
    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n/%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    stats.write(str(layers[0])+"-"+str(layers[1])+"-"+str(layers[2])+","+
                transfnc+","+optimizer+","+str(epochs)+","+str(batch)+","+str(scores[1]*100)+"\n")
    # serialize model to JSON
    model_json = model.to_json()
    with open("Models/JSON/" + modelName + "_Architecure.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Models/Weigths/" + modelName + "_WEIGHTS.h5")
    print("SAVING " + modelName + " TO DISK")


file = "/home/wiss/CODES/TP-AARN/Mini-Project/DataSets/dataOneOf5.csv"
# MAX IS 78071
rows = 78071

# load pima indians dataset
# data = np.genfromtxt("/home/wiss/CODES/TP-AARN/Mini-Project/DataSets/dataSetHandPosture.csv",
#                      dtype=np.float64, delimiter=',', names=True)

data = np.genfromtxt(file, dtype=float, delimiter=',', missing_values='?', filling_values=0.0)
X = data[0:rows, 2:38]
Y = data[0:rows, 37:42]
# print(X)


transFncs = ['relu', 'tanh', 'sigmoid']
optimizers = ['sgd', 'adagrad', 'adam', 'adamax']
inDim = 36
outDim = 5
batch = 125
epochs = 150
modelBaseName = "model_"
for tranFnc in transFncs:
    for opt in optimizers:
        # layers =[]
        for i in range(5, 15):
            for j in range(12, 25):
                for k in range(40, 50):
                    modeName = modelBaseName + \
                               str(k) + "_" + \
                               str(i) + "_" + \
                               str(j) + "_" + \
                               tranFnc + "_" + \
                               opt + "_"+str(batch)
                    layers = [i, j, k]
                    print(layers)
                    trainModel(modeName, X, Y, layers, inDim, outDim, tranFnc, opt, epochs, batch)

stats.close()
















# later...
#
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
#
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
