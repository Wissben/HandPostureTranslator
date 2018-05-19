from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from Blender.vectors import Vector, coSystem
from sklearn.neighbors import NearestNeighbors
from Gestures import getGesture
def splitMarkersInNP(dataF,markers):
    array = []
    for i in range(markers):
        array.append(np.array(
            [np.array(x) for x in
             zip(dataF['X' + str(i)].values, dataF['Y' + str(i)].values, dataF['Z' + str(i)].values)]))

    array = np.concatenate(array)
    return array

def allDone(l:np.array):
    index = 0
    for a in l:
        if not a:
            return index
        index+=1
    return index


origin = Vector(-0.734604, -0.167507, 1.454207)
x = Vector(-0.736820, -0.150441, 1.461019)
y = Vector(-0.760898, -0.168561, 1.457838)

s = coSystem(origin, x, y)
gestures = ['nailMiddle', 'little', 'lowBig', 'ring', 'nailIndex', 'middle', 'midBig', 'nailRing', 'nailBig', 'index', 'nailLittle']



# read all data
# data = pd.read_csv("allUsers.lcl.csv").replace('?',0).astype('float64')
# # select class and user
# data = data[(data['Class'] == 1) & (data['User'] == 2)]

################ prepare data from pandas dataframe to np array ready for kmeans ####################
# array = splitMarkersInNP(X,11)
# print(array)
dataFrames = []
dropped = 0
allRows = 0
for j in range(1,6):
    gest = getGesture(j)
    X = [s.newit(np.array(gest[v])) * 1400 for v in gestures]
    print(X)
    numberOfMarkers = 11
    ############# learning ###############
    # kmeans = KMeans(numberOfMarkers, max_iter=1000).fit(X)
    neigh = NearestNeighbors(n_neighbors=3)
    neigh.fit(X)

    # data = selector.selectGesture(1,5,9000)
    # array2 = splitMarkersInNP(data,5)
    # array = np.concatenate([array,array2])
    # print(array)

    # found = [0 for i in range(12)]
    # for i in kmeans.labels_:
    #     found[i] = found[i]+1






    ######################### data to be rewritten #############################
    names = "Class,User,X0,Y0,Z0,X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,X4,Y4,Z4,X5,Y5,Z5,X6,Y6,Z6,X7,Y7,Z7,X8,Y8,Z8,X9,Y9,Z9,X10,Y10,Z10,X11,Y11,Z11,a,w,i,s,e"
    names = names.split(',')
    allData = pd.read_csv("Partitionnement-one-user-left/usersLeft/user5/Training.csv",names=names).replace('?', 0).astype('float64')
    allData = allData[(allData['Class'] == j)]
    print(allData.shape)

    ########### loop to rewrite data according to Kmeans Predictions ###############
    for index, row in allData.iterrows():
        a1 = np.zeros(shape=(numberOfMarkers, 3))
        temp = np.array(
            [np.array([row['X' + str(i)], row['Y' + str(i)], row['Z' + str(i)]]) for i in range(numberOfMarkers)])
        done = [False for i in range(numberOfMarkers)]
        prediction = np.array(neigh.kneighbors(temp,return_distance=False))
        # print(temp)
        # print(prediction)
        ind = 0
        countDiff = 0
        for t in temp:
            if t[0] != 0:
                myInd = allDone([done[int(k)] for k in prediction[ind]])
                if myInd >= 3:
                    print(str(t) + " " + str(prediction[ind]))
                else:
                    countDiff += 1
                    a1[prediction[ind][myInd]] = t
                    done[prediction[ind][myInd]] = True
            ind += 1
        print(str(index) +" countDiff + " +str(countDiff))
        if countDiff > 3:
            a1 = np.concatenate(a1)
            # for i in range(12):
            #     allData.loc[index:index,'X'+str(i):'Z'+str(i)] = a1[i]
            allData.loc[index:index, 'X0':'Z' + str(numberOfMarkers - 1)] = a1
            print("added")
        else:
            allData.drop([index])
            dropped += 1
        allRows += 1

    for i in range(numberOfMarkers, 12):
        del allData['X' + str(i)]
        del allData['Y' + str(i)]
        del allData['Z' + str(i)]
        ############################ saving data to csv ##########################
    dataFrames.append(allData)

print("dropped "+str(dropped)+"/"+str(allRows))
allData = pd.concat(dataFrames)
allData.to_csv("newFileTrain.csv")
