from sklearn.cluster import  KMeans
import pandas as pd
import numpy as np
from backend.training.neuralNetworkModel.treat import Selector

def splitMarkersInNP(dataF,markers):
    array = []
    for i in range(markers):
        array.append(np.array(
            [np.array(x) for x in
             zip(dataF['X' + str(i)].values, dataF['Y' + str(i)].values, dataF['Z' + str(i)].values)]))

    array = np.concatenate(array)
    return array


# read all data
data = pd.read_csv("allUsers.lcl.csv").replace('?',0).astype('float64')
# select class and user
data = data[(data['Class'] == 1) & (data['User'] == 2)]
print(data)
################ prepare data from pandas dataframe to np array ready for kmeans ####################
array = splitMarkersInNP(data,12)

numberOfMarkers = 11
############# learning ###############
kmeans = KMeans(numberOfMarkers,max_iter=1000).fit(array)


# data = selector.selectGesture(1,5,9000)
# array2 = splitMarkersInNP(data,5)
# array = np.concatenate([array,array2])
# print(array)

# found = [0 for i in range(12)]
# for i in kmeans.labels_:
#     found[i] = found[i]+1






######################### data to be rewritten #############################
allData = pd.read_csv("allUsers.lcl.csv").replace('?',0).astype('float64')
allData = allData[(allData['Class'] == 1) & (allData['User'] == 2)]
print(allData.shape)

########### loop to rewrite data according to Kmeans Predictions ###############
for index, row in allData.iterrows():
    a1 = np.zeros(shape=(numberOfMarkers,3))
    temp = np.array([np.array([row['X'+str(i)],row['Y'+str(i)],row['Z'+str(i)]]) for i in range(numberOfMarkers)])
    done = [False for i in range(numberOfMarkers)]
    prediction = kmeans.predict(temp)
    ind = 0
    for t in temp:
        if t[0] != 0:
            if done[prediction[ind]]:
                print(str(t) + " " + str(prediction[ind]))
            a1[prediction[ind]] = t
            done[prediction[ind]] = True
        ind += 1
    a1 = np.concatenate(a1)
    # for i in range(12):
    #     allData.loc[index:index,'X'+str(i):'Z'+str(i)] = a1[i]
    allData.loc[index:index, 'X0':'Z'+str(numberOfMarkers-1)] = a1
    print(index)
for i in range(numberOfMarkers,12):
    del allData['X' + str(i)]
    del allData['Y'+str(i)]
    del allData['Z'+str(i)]
############################ saving data to csv ##########################
allData.to_csv("newFile.csv")