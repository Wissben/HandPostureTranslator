import tensorflow as tf
import numpy as np
from pandas import read_csv

pathToTestinFile = "/home/wiss/CODES/TP-AARN/Mini-Project/DataSets/dataOneOf5.csv"


def fileToArray(path, replaceMissing):  # Here you may redifine how u read from your file
    dataTestSize = 15000
    fileReader = open(path, 'r')
    data = read_csv(path, header=None).replace('?', replaceMissing)
    values = data.values[1:dataTestSize]
    header = data.values[0:1, 2:38]
    # np.random.shuffle(values)
    # return data.values[1500:1501,2:38]
    return header, values[0:dataTestSize, 2:38], values[0:dataTestSize, 37:42]


def input_evaluation_set():
    feats = {}
    header, att, target = fileToArray(pathToTestinFile, 0)
    print()
    for i in range(36):
        # print(header[:, 2], att[0:, 2])
        feats[str(header[:, i])] = att[0:, i]
        # print(feats[str
    labels = np.array(target)
    print(target)
    return feats, labels


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(7).repeat().batch(batch_size)


# Feature columns describe how to use the input.
my_feature_columns = []
feats, labels = input_evaluation_set()
for key in feats.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[50, 25, 15],
    # The model must choose between 3 classes.
    n_classes=5)
# Train the Model.
header, att, target = fileToArray(pathToTestinFile, 0)
print(train_input_fn(feats,target,100))
# classifier.train(
#     input_fn=lambda:train_input_fn(att, target, 100),
#     steps=150)

# import tensorflow as tf
# from pandas import read_csv
# import numpy as np
#
# layers = [50, 50, 50]
# nb_attributes = 36
# nb_classe = 5
# batch_size = 1000
# nb_data = 78071
#
# x = tf.placeholder('float', [None, nb_attributes])
# y = tf.placeholder('float', [None, nb_classe])
#
#
# def fileToArray(path, replaceMissing):  # Here you may redifine how u read from your file
#     fileReader = open(path, 'r')
#     data = read_csv(path, header=None).replace('?', replaceMissing).astype('float')
#     values = data.values
#     np.random.shuffle(values)
#
#     dataTestSize = 15000
#     # return data.values[1500:1501,2:38]
#     return values[0:dataTestSize, 2:38], values[0:dataTestSize, 37:42]
#
#
# def defineModel(data):
#     hidden_layer1 = {'weights': tf.Variable(tf.random_normal([nb_attributes, layers[0]])),
#                      'biases': tf.Variable(tf.random_normal([layers[0]]))}
#     hidden_layer2 = {'weights': tf.Variable(tf.random_normal([layers[0], layers[1]])),
#                      'biases': tf.Variable(tf.random_normal([layers[1]]))}
#     hidden_layer3 = {'weights': tf.Variable(tf.random_normal([layers[1], layers[2]])),
#                      'biases': tf.Variable(tf.random_normal([layers[2]]))}
#
#     output_layer = {'weights': tf.Variable(tf.random_normal([nb_classe, nb_classe])),
#                     'biases': tf.Variable(tf.random_normal([nb_classe]))}
#
#     layer1 = tf.add(tf.matmul(data, hidden_layer1['weights']), hidden_layer1['biases'])
#     layer1 = tf.nn.relu(layer1)
#
#     layer2 = tf.add(tf.matmul(layer1, hidden_layer2['weights']), hidden_layer2['biases'])
#     layer2 = tf.nn.relu(layer2)
#
#     layer3 = tf.add(tf.matmul(layer2, hidden_layer3['weights']),hidden_layer3['biases'])
#     layer3 = tf.nn.relu(layer3)
#
#     output = tf.matmul(layer3, output_layer['weights']) + output_layer['biases']
#     # output = tf.nn.softmax(output)
#     return output
#
#
# def trainModel(X):
#     prediction = defineModel(X)
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
#     optimize = tf.train.AdamOptimizer().minimize(cost)
#
#     epochs = 150
#     with tf.Session() as sess:
#         sess.run(tf.initialize_all_variables())
#         for epoch in range(epochs):
#             epoch_loss = 0
#             for _ in range(int(nb_data / batch_size)):
#
#
# attributes, target = fileToArray("/home/wiss/CODES/TP-AARN/Mini-Project/DataSets/dataOneOf5.csv", 0)
# print(attributes, target)
