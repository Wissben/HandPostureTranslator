import numpy as np
from tensorflow.python.keras.models import load_model

class Vector(object):
    x = 0.0
    y = 0.0
    z = 0.0

    def __init__(self,x,y,z) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.z = z


def createV(co) -> np.array:
    return np.array([co.x,co.y,co.z])

class coSystem(object):
    origin = None
    Vx = None
    Vy = None
    Vz = None

    def __init__(self,oCo, xCo, yCo) -> None:
        super().__init__()
        self.origin, self.Vx, self.Vy, self.Vz = self.createCoordinateSystem(oCo, xCo, yCo)

    def createCoordinateSystem(self,oCo, xCo, yCo):
        origin = createV(oCo)
        x = createV(xCo)
        y = createV(yCo)
        Vx = x - origin
        Vy = y - origin
        Vz = np.cross(Vx, Vy)
        Vx = Vx / np.linalg.norm(Vx)
        Vy = Vy / np.linalg.norm(Vy)
        Vz = Vz / np.linalg.norm(Vz)
        return origin, Vx, Vy, Vz

    def newit(self,vec):
        # translation:
        vec = vec - self.origin

        # rotation:
        nnx,nny,nnz = self.getAxis()
        # old axes:
        nox, noy, noz = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float).reshape(3, -1)

        # ulgiest rotation matrix you can imagine
        top = [np.dot(nnx, n) for n in [nox, noy, noz]]
        mid = [np.dot(nny, n) for n in [nox, noy, noz]]
        bot = [np.dot(nnz, n) for n in [nox, noy, noz]]

        xn = sum([p * q for p, q in zip(top, vec)])
        yn = sum([p * q for p, q in zip(mid, vec)])
        zn = sum([p * q for p, q in zip(bot, vec)])
        return np.array([xn, yn, zn])

    def getAxis(self):
        return self.Vx,self.Vy,self.Vz

# origin = Vector(-0.734604,-0.167507,1.454207)
# x = Vector(-0.736820,-0.150441,1.461019)
# y = Vector(-0.760898,-0.168561,1.457838)
#
#
# s = coSystem(origin,x,y)
# tab = []
#
# markers = {'nailIndex':131, 'nailMiddle':170, 'nailRing':43, 'nailLittle': 60,
#            'index': 372, 'middle': 144,'ring': 32, 'little': 45,
#            'nailBig': 306, 'midBig': 64, 'lowBig': 322}
# tab.append( [-0.821039080619812, -0.11133310198783875, 1.4654878377914429] )
# tab.append( [-0.810570240020752, -0.16549016535282135, 1.4615062475204468] )
# tab.append( [-0.7860807180404663, -0.07898025214672089, 1.4308576583862305] )
# tab.append( [-0.8262975811958313, -0.13195575773715973, 1.4653825759887695] )
# tab.append( [-0.8010895848274231, -0.09128524363040924, 1.42415189743042] )
# tab.append( [-0.7645509839057922, -0.07662639021873474, 1.436549186706543] )
# tab.append( [-0.8184627294540405, -0.149211585521698, 1.4653873443603516] )
# dic = {}
# # dic['nailIndex'] = [-0.8627850413322449, -0.11155961453914642, 1.4638097286224365]
# # dic['nailMiddle'] = [-0.8721364140510559, -0.1308656632900238, 1.4647668600082397]
# # dic['nailRing'] = [-0.8609100580215454, -0.14880047738552094, 1.4603713750839233]
# # dic['nailLittle'] = [-0.8373788595199585, -0.16685912013053894, 1.4586029052734375]
# # dic['index'] = [-0.8299517035484314, -0.11130033433437347, 1.4631918668746948]
# # dic['middle'] = [-0.8354308605194092, -0.13193881511688232, 1.4640510082244873]
# # dic['ring'] = [-0.827552080154419, -0.149312362074852, 1.4614354372024536]
# # dic['little'] = [-0.8119481801986694, -0.16592566668987274, 1.4578845500946045]
# # dic['nailBig'] = [-0.8021453022956848, -0.08142299950122833, 1.4349708557128906]
# # dic['midBig'] = [-0.7839306592941284, -0.07367655634880066, 1.437751054763794]
# # dic['lowBig'] = [-0.7630971074104309, -0.07598352432250977, 1.4393389225006104]
#
# dic['nailIndex'] = [-0.863071858882904, -0.11165919899940491, 1.4615262746810913]
# dic['nailMiddle'] = [-0.8724150061607361, -0.13091284036636353, 1.4621859788894653]
# dic['nailRing'] = [-0.8610506057739258, -0.14889322221279144, 1.4589096307754517]
# dic['nailLittle'] = [-0.8373264670372009, -0.16683652997016907, 1.4589587450027466]
# dic['index'] = [-0.8302159905433655, -0.11135396361351013, 1.4621939659118652]
# dic['middle'] = [-0.8357042670249939, -0.13195903599262238, 1.4627721309661865]
# dic['ring'] = [-0.8277260661125183, -0.14934709668159485, 1.4607553482055664]
# dic['little'] = [-0.8119130730628967, -0.16592073440551758, 1.4580062627792358]
# dic['nailBig'] = [-0.802054226398468, -0.08091585338115692, 1.4351868629455566]
# dic['midBig'] = [-0.7837497591972351, -0.0734078586101532, 1.4378550052642822]
# dic['lowBig'] = [-0.7630001902580261, -0.07594172656536102, 1.4393526315689087]
#
#
#
#
# mars = ['little','ring','middle','nailMiddle','index','nailBig','lowBig']
# tab = [dic[k] for k in mars]
# tab2 = np.array([s.newit(np.array(v))*1200 for v in tab])
# # for i in range(5):
# #     tab2.append([0,0,0])
# a = np.array([np.hstack(tab2)])
# print(a)
# print(a.T.shape)
# # for key,v in dic.items():
# #     print(key + ": " + str(s.newit(np.array(v))*1200))
#
# pathToModel = "../backend/training/Models/Weights/modelTest.hd5"
# model = load_model(pathToModel)
# cat = model.predict_classes(a)
#
# print(cat)
# import scipy.io
# scipy.io.savemat('/home/ressay/data2.mat', mdict={'sample': (a.tolist())})
# print(s.newit(np.array([-0.8218, -0.1054, 1.4619])))