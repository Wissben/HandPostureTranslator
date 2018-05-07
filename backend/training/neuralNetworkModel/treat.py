import pandas as pd
import numpy as np

# path = "../../DataSets/dataOneOf5.csv"
def transformClassToOneHot(data):
    data['Class'] = pd.get_dummies(data.Class).astype("float64").values.tolist()
    return data

class Selector(object):
    allData = pd.DataFrame
    gestures = pd.DataFrame

    def __init__(self,path ,minMarkers = None,numberOfInstances = None) -> None:
        super().__init__()
        self.allData = pd.read_csv(path)
        if minMarkers is not None and numberOfInstances is not None:
            self._select(minMarkers,numberOfInstances)

    def selectGesture(self,gesture,minMarkers,numberOfInstances) -> pd.DataFrame:
        allData = self.allData
        if 11 >= minMarkers >= 0:
            selected = allData[(allData['X' + str(minMarkers)] != '?') & (allData['Class'] == gesture)]
        else:
            selected = allData
        del selected['User']
        for i in range(minMarkers+1,12):
            del selected['X' + str(i)]
            del selected['Y' + str(i)]
            del selected['Z' + str(i)]
        selected = selected.replace(['?'], 0).astype('float64')
        return selected.head(numberOfInstances)

    def _select(self,minMarkers,numberOfInstances) -> pd.DataFrame:
        gests = self.selectGesture(1,minMarkers,numberOfInstances)
        for i in range(2,6):
            s = self.selectGesture(i,minMarkers,numberOfInstances)
            gests = gests.append(s,ignore_index=True)
        gests = transformClassToOneHot(gests)
        self.gestures = gests

        return gests

    def getInputOutput(self):
        return self.gestures.loc[:, 'X0':].values , np.array([a for a in self.gestures['Class']])

    def getData(self):
        return np.concatenate(self.getInputOutput(),axis=1)

    def getGesturesDataFrame(self):
        return self.gestures


# selector = Selector("allUsers.lcl.csv",6,10000)
# gests = []
# inp, out = selector.getInputOutput()
# print(inp.shape)
# print(out)
# print(selector.getData().shape)
# for i in range(1,6):
#     gests.append(selector.selectGesture(i,6,2600))
#     print(gests[i-1].size / len(list(gests[i-1])))

# result = pd.concat(gests, ignore_index=True)
# print(result.size / len(list(result)))
# target = result['Class'].values
# features = result.loc[:,'X0':].values
# print(len(target))
# print(len(features[0]))

