import time
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import Imputer
from tensorflow.python.keras._impl.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
import sys

if(len(sys.argv)<2):
	print("Too few arguments")
	exit


statsFile = "Models/trainingStats.csv"
stats = open(statsFile, 'w')
stats.write("layers,tranFnc,opt,epochs,batch,accuracy,mse,trainingTime\n")
stats.close()
file = str(sys.argv[1])
# SIZE OF INPUT DATASET TO TRAIN ON (MAX IS 78071)
MAX = int(78071)
trainingRows = int(MAX * 75 / 100)
testingRows = MAX - trainingRows


def replaceMissingData(X, strategy):
    mean_imputer = Imputer(missing_values='NaN', strategy=strategy, axis=0)
    mean_imputer = mean_imputer.fit(X)
    imputed_df = mean_imputer.transform(X.values)
    return imputed_df


def evaluateModel(model, X, Y, batch):
    score = model.evaluate(X, Y, batch_size=batch, verbose=1)
    print('Test LOSS:', score[0])
    print('Test ACCURACY:', score[1])
    print('Test MSE :', score[2])
    # sleep(25)
    return score


def trainModel(modelName, train_X, train_Y, test_X, test_Y, layers, inputDime, outputDim, transfnc, optimizer, epochs,
               batch):
    # create tesnsorboard callback
    tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    # create a csv logger callback
    csv_logger = CSVLogger("logs/" + modelName + 'training.log')

    ## create model
    model = Sequential()
    model.add(Dense(layers[0], input_dim=inputDime, activation=transfnc))

    # setup model's architecure
    for i in range(1, len(layers)):
        model.add(Dense(layers[i], activation=transfnc))
    model.add(Dense(outputDim, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', 'mse'])
    start_time = time.time()
    # Fit the model
    model.fit(train_X, train_Y, epochs=epochs, shuffle=True, batch_size=batch, callbacks=[tb, csv_logger], verbose=1,
              validation_split=0.15,
              )

    # evaluate the model
    scores = evaluateModel(model, test_X, test_Y, batch)
    elapsed_time = time.time() - start_time

    # Saving the model
    architecture = ""
    for layer in layers:
        architecture = architecture + str(layer) + "-"
    architecture = architecture[0:len(architecture) - 1]

    stats = open(statsFile, 'a')
    stats.write(architecture + "," +
                transfnc + "," +
                optimizer + "," +
                str(epochs) + "," +
                str(batch) + "," +
                str(scores[1] * 100) + "," +
                str(scores[2]) + "," +
                (time.strftime("%H:%M:%S.", time.gmtime(elapsed_time))) +
                "\n")
    stats.close()
    # serialize model to JSON
    model_json = model.to_json()
    with open("Models/JSON/" + modelName + "_Architecure.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save("Models/Weights/" + modelName + ".hd5")
    print("SAVING " + modelName + " TO DISK")
    print("FINISHED TRAINING MODEL : " + modelName)

    # sleep(30)
    return model

def splitData(values,ratio):
    assert 1 > ratio > 0
    size = list(values.shape)[0]
    thresh = int(size*ratio)
    training_X = values[0:thresh, 0:-5]
    training_Y = values[0:thresh, -5:]

    testing_X = values[thresh+1:, 0:-5]
    testing_Y = values[thresh+1:, -5:]
    return training_X,training_Y,testing_X,testing_Y

data = read_csv(file).replace('?', 0)
values = data.astype('float64').values
values = values[1:,:]
np.random.seed(7)
np.random.shuffle(values)

training_X,training_Y,testing_X,testing_Y = splitData(values,0.75)
print(training_X.shape)
print(testing_X.shape)

# training the model
model = trainModel("modelTest", training_X, training_Y, testing_X, testing_Y, [40,40,20],testing_X.shape[1], testing_Y.shape[1],
                   'relu', 'sgd', 1000, 100)

y = model.predict(testing_X)

