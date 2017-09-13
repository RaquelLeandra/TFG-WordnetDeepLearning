import numpy as np
from nltk.corpus import wordnet as wn

"""
vgg...npz es el FNE mapeado con los labels
Consiste en un diccionario con:

claves de data: ['labels', 'data_matrix']
labels: La clasificación resultante para cada imagen
data_matrix: el embedding correspondiente

synset.txt es la lista con los synsets de imagenet
"""

data = np.load("/media/raquel/Datos/Programación-git/tfg/Data/vgg16_ImageNet_ALLlayers_C1avg_imagenet_train.npz")
discretData = np.load(
    "/media/raquel/Datos/Programación-git/tfg/Data/vgg16_ImageNet_imagenet_C1avg_E_FN_KSBsp0.15n0.25_Gall_val_.npy")
imagenet_id = np.genfromtxt("/media/raquel/Datos/Programación-git/tfg/Data/synset.txt", dtype=np.str)

labels = data['labels']
matrix = data['data_matrix']

del data

print(labels.shape)
"""
print(imagenet_id[labels[0:50]])
['n03937543' 'n02085620' 'n03146219' 'n02641379' 'n03871628' 'n03871628'
 'n02655020' 'n04147183' 'n02110627' 'n01688243' 'n04067472' 'n01807496'
 'n01644373' 'n02092339' 'n01491361' 'n03791053' 'n03271574' 'n02894605'
 'n04493381' 'n02492660' 'n02894605' 'n01774750' 'n01807496' 'n02871525'
 'n03874293' 'n02389026' 'n07836838' 'n04154565' 'n03109150' 'n02105855'
 'n03791053' 'n02092339' 'n02110627' 'n03937543' 'n07693725' 'n02168699'
 'n02102973' 'n02102973' 'n03891332' 'n02086079' 'n02486261' 'n04606251'
 'n02138441' 'n02112706' 'n04147183' 'n01770393' 'n07583066' 'n03874293'
 'n02951585' 'n02871525']

print(labels[0:50])
[720 151 524 395 692 692 397 780 252  43 758  86  31 178   3 670 545 460
 876 379 460  76  86 454 694 339 960 784 512 230 670 178 252 720 931 303
 221 221 704 154 371 913 299 262 780  71 924 694 473 454]
"""


def get_wn_ss(imagenet_id):
    return wn.of2ss(imagenet_id[1:] + '-' + imagenet_id[0])


def get_in_id(wordnet_ss):
    wn_id = wn.ss2of(wordnet_ss)
    print(wn_id)
    return wn_id[-1] + wn_id[:8]


def get_wn_id(imagenet_id):
    return imagenet_id[1:] + '-' + imagenet_id[0]


def get_ss_from_label(label):
    print(imagenet_id[labels[label]])
    return get_wn_ss(imagenet_id[labels[label]])

print(labels[0])
print(imagenet_id[labels[0]])
print(get_ss_from_label(0))
