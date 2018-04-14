import numpy as np
from pandas import read_csv
from sklearn.preprocessing import Imputer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

stats = open("Models/stats.csv", 'w')
stats.write("layers,tranFnc,opt,epochs,batch,score\n")
file = "/home/wiss/CODES/TP-AARN/Mini-Project/DataSets/dataOneOf5.csv"
# SIZE OF INPUT DATASET TO TRAIN ON (MAX IS 78071)
dataRows = 1000



def replaceMissingData(X,values) :
    # fill missing values with mean column values
    imputer = Imputer()
    transformed_X = imputer.fit_transform(X)
    transformed_values = imputer.fit_transform(values)
    # count the number of NaN values in each column
    print(np.isnan(transformed_values).sum())
    return transformed_X




def trainModel(modelName, X, Y, layers, inputDime, outputDim, transfnc, optimizer, epochs, batch):
    ## create model
    model = Sequential()
    model.add(Dense(layers[0], input_dim=inputDime, activation=transfnc))

    for i in range(1,len(layers)) :
        model.add(Dense(layers[i],activation=transfnc))
    model.add(Dense(outputDim, activation='softmax'))
    # # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # # Fit the model
    model.fit(X, Y, epochs=epochs, batch_size=batch)
    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n/%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    architecture = ""
    for layer in layers :
        architecture=architecture+str(layer)+"-"
    architecture=architecture[0:len(architecture)-1]
    stats.write(architecture+","+
                transfnc+","+optimizer+","+str(epochs)+","+str(batch)+","+str(scores[1]*100)+"\n")
    # serialize model to JSON
    model_json = model.to_json()
    with open("Models/JSON/" + modelName + "_Architecure.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Models/Weigths/" + modelName + "_WEIGHTS.h5")
    print("SAVING " + modelName + " TO DISK")
    print("FINISHED TRAINING MODEL : " + modelName)


##Biggest bullshit ever made :
# data = read_csv(file,header=None).replace('?',np.NaN)
##
data = read_csv(file,header=None).replace('?',0)
values=data.values

X = values[0:dataRows, 2:38]
Y = values[0:dataRows, 37:42]
transformed_X=replaceMissingData(X,data.values)
# print(transformed_X[2])


#training the model
trainModel("modelTest",transformed_X,Y,[50,25,15,10,5],36,5,'relu','sgd',150,15)
#
#Uncomment to launch multiple models training
# transFncs = ['relu', 'tanh', 'sigmoid']
# optimizers = ['sgd', 'adagrad', 'adam']
# inDim = 36
# outDim = 5
# batch = 125
# epochs = 150
# modelBaseName = "model_"
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
#                                opt + "_"+str(batch)
#                     layers = [i, j, k]
#                     print("TRAINING MODEL  :"+modeName)
#                     trainModel(modeName, X, Y, layers, inDim, outDim, tranFnc, opt, epochs, batch)
#
stats.close()



# Uncomment to load model from JSON and its weights from hd5 files
#
# load json and create model
# jsonModelName=""
# hd5WeightsName=""
# json_file = open(jsonModelName, 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights(hd5WeightsName)
# print("LOADED MODEL : " +jsonModelName+" FROM DISK")
#
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
