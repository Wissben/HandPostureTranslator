from collections import Iterator
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import backend as K

# from backend.training.neuralNetworkModel.treat import Selector
import numpy as np


class Variator(object):

    evaluationCallbacks = None
    updateBestCallbacks = None
    evaluateScoreFunction = None


    bestModel = Model
    bestScore = None
    histories = []
    currentParameters = {}

    def __init__(self,evaluationCallbacks=None,updateBestCallbacks=None,
                 evaluateScoreFunction=None) -> None:
        super().__init__()
        if evaluationCallbacks:
            if isinstance(evaluationCallbacks, Iterator):
                self.evaluationCallbacks = evaluationCallbacks
            else:
                self.evaluationCallbacks = [evaluationCallbacks]

        if not updateBestCallbacks:
            updateBestCallbacks = defaultUpdateBestCallback
        if isinstance(updateBestCallbacks, Iterator):
            self.updateBestCallbacks = updateBestCallbacks
        else:
            self.updateBestCallbacks = [updateBestCallbacks]

        if not evaluateScoreFunction:
            evaluateScoreFunction = defaultEvaluateScoreFunction
        self.evaluateScoreFunction = evaluateScoreFunction

    def train(self,inputs,targets,splitRatio,trainParamsGenerator=None):
        train_X, train_Y, test_X, test_Y = splitData(inputs,targets,splitRatio)
        if not trainParamsGenerator:
            trainParamsGenerator = self.createTrainParamsGenerator(inputs,targets,
                                                                   defaultTrainGenerators())
        return self._trainModel(trainParamsGenerator,train_X, train_Y, test_X, test_Y)


    def _trainModel(self,trainParamsGenerator: Iterator,train_X,train_Y,test_X,test_Y):
        for model,parameters in trainParamsGenerator:
            epochs = parameters['epochs']
            batch = parameters['batch']
            callbacks = parameters['callbacks']
            verbose = parameters['verbose']
            validation_split = parameters['validation_split']
            self.currentParameters = parameters
            self.histories.append(model.fit(train_X, train_Y, epochs=epochs, shuffle=True, batch_size=batch, callbacks=callbacks, verbose=verbose,
                validation_split=validation_split))
            self.evaluateModel(model,test_X,test_Y)

        return self.bestModel,self.bestScore

    def evaluateModel(self,model: Model,test_X,test_Y) -> Model:
        if self.evaluationCallbacks:
            for callback in self.evaluationCallbacks:
                callback(self,model)

        score = defaultEvaluation(model,test_X,test_Y)
        return self.updateBestModel(model,score)

    def updateBestModel(self,model: Model,score) -> Model:
        if not self.bestScore:
            self.bestScore = score
            self.bestModel = model

            if self.updateBestCallbacks is not None:
                for callback in self.updateBestCallbacks:
                    callback(self, self.bestModel, self.bestScore)

        elif self.evaluateScoreFunction(self.bestScore,score) < 0: #new score is better than best
            self.bestScore = score
            self.bestModel = model

            if self.updateBestCallbacks is not None:
                for callback in self.updateBestCallbacks:
                    callback(self, self.bestModel, self.bestScore)


        return self.bestModel

    @staticmethod
    def reset_weights(model):
        session = K.get_session()
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

    @staticmethod
    def createTrainParamsGenerator(inputs, targets, trainGenerators, callbacks=None,
                                   epochs=40, batch=100, validation_split=0.15, verbose=0,
                                   outputActivationFunc='softmax', loss='categorical_crossentropy',
                                   metrics=['accuracy', 'mse'],retrains=1):
        inputDime = list(inputs.shape)[-1]
        outputDim = list(targets.shape)[-1]

        generators = trainGenerators
        archs = generators['layer']
        optimizers = generators['optimizer']
        for layers, activations in archs:
            for optimizer in optimizers:
                for retrain in range(retrains):
                    archi = ": layers: "+str(layers)+" "+str(activations)+" optimizer: "+str(optimizer)
                    modelName = "model "+archi
                    print("train "+archi)
                    ## create model
                    model = Sequential()
                    model.add(Dense(layers[0], input_dim=inputDime, activation=activations[0]))
                    # setup model's architecure
                    for i in range(1, len(layers)):
                        model.add(Dense(layers[i], activation=activations[i]))
                    model.add(Dense(outputDim, activation=outputActivationFunc))

                    # Compile model
                    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

                    parameters = {'epochs': epochs, 'batch': batch, 'callbacks': callbacks, 'verbose': verbose,
                                  'validation_split': validation_split,'modelName':modelName}

                    yield model, parameters



# print loss and metrics values
def defaultEvaluation(model: Model,test_X,test_Y):
    score = model.evaluate(test_X, test_Y)

    print('Test ' + str(model.loss) + ':', score[0])
    index = 1
    for metric in model.metrics:
        print('Test ' + str(metric) + ':', score[index])
        index += 1
    return score

# evaluate according to first metric of model
def defaultEvaluateScoreFunction(bestScore,score) -> int:
    if bestScore[1] > score[1]:
        return 1
    elif bestScore[1] < score[1]:
        return -1
    return 0

def defaultUpdateBestCallback(variator:Variator,model: Model,score):
    print('best is '+str(model.metrics[0]) + ': '+str(score[1]))

def splitData(inputs,targets,ratio):
    assert 1 > ratio > 0
    assert list(inputs.shape)[0] == list(targets.shape)[0]
    size = list(inputs.shape)[0]
    thresh = int(size*ratio)
    training_X = inputs[0:thresh,:]
    training_Y = targets[0:thresh,:]

    testing_X = inputs[thresh+1:,:]
    testing_Y = targets[thresh+1:,:]
    return training_X,training_Y,testing_X,testing_Y

def defaultTrainGenerators():
    import KerasArchis.ArchGenerators as ag
    return {'layer':ag.layerGenerator(),'optimizer':ag.optimiezersGenerator()}


