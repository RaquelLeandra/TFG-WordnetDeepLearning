import numpy as np
from nltk.corpus import wordnet as wn

"""
vgg...npz es el FNE mapeado con los labels
Consiste en un diccionario con:

claves de data: ['labels', 'data_matrix']
labels: La clasificación resultante para cada imagen
data_matrix: el embedding correspondiente

synsets_in_imagenet.txt es la lista con los synsets de imagenet
"""

data = np.load("/media/raquel/Datos/Programación-git/tfg/Data/vgg16_ImageNet_ALLlayers_C1avg_imagenet_train.npz")
#discretData = np.load(
#    "/media/raquel/Datos/Programación-git/tfg/Data/vgg16_ImageNet_imagenet_C1avg_E_FN_KSBsp0.15n0.25_Gall_val_.npy")
imagenet_id = np.genfromtxt("/media/raquel/Datos/Programación-git/tfg/Data/synsets_in_imagenet.txt", dtype=np.str)


labels = data['labels']
matrix = data['data_matrix']

del data


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
    return wn_id[-1] + wn_id[:8]


def get_wn_id(imagenet_id):
    return imagenet_id[1:] + '-' + imagenet_id[0]


def get_ss_from_label(label):
    print(imagenet_id[labels[label]])
    return get_wn_ss(imagenet_id[labels[label]])


def get_index_from_ss(synset):
    hypo = lambda s: s.hyponyms()
    path = '/media/raquel/Datos/Programación-git/tfg/Data/' + str(synset) + '.txt'
    hyponim_file = open(path, "w")
    synset_list = []
    for thing in list(synset.closure(hypo)):
        hyponim_file.write(get_in_id(thing) + '\n')
        synset_list.append(get_in_id(thing))

    hyponim_file.close()
    path2 = '/media/raquel/Datos/Programación-git/tfg/Data/' + str(synset) + '_' + 'index' + '.txt'
    index_file = open(path2, 'w')
    i = 0
    for lab in labels:
        if imagenet_id[lab] in synset_list:
            index_file.write(str(i) + '\n')
        i += 1

    index_file.close()

dog = wn.synsets('dog')[0]
mammal = wn.synsets('mammal')[0]
living_things = wn.synset('living_thing.n.01')
hunting_dogs = wn.synsets('hunting_dog')[0]

synsets = [living_things, mammal, dog, hunting_dogs]
for synset in synsets:
    get_index_from_ss(synset)

def stats_from_synset(synset):
    path = '/media/raquel/Datos/Programación-git/tfg/Data/' + str(synset) + '.txt'
    index_path = '/media/raquel/Datos/Programación-git/tfg/Data/' + str(synset) + '_index' + '.txt'
    res_path = path + '_stats' + '.txt'
    #TODO: añadir exception si no tenemos los datos generados
    #data = np.genfromtxt(path, dtype=np.str)
    index = np.genfromtxt(index_path, dtype=np.int)

    labels_size = labels.shape[0]
    labels_clases = labels.ptp() + 1
    synset_range = index.ptp()+1

    print('Tenemos ' + str(labels_size) + 'imagenes, de las cuales el '
          + str(float(index.shape[0]) /labels_size * 100) + ' son ' + str(synset))

#suponiendo que estan ordenados de más general a menos


def instra_synset_stats(synsets):
    j = 0
    for synset in synsets:
        index_path = '/media/raquel/Datos/Programación-git/tfg/Data/' + str(synset) + '_index' + '.txt'
        syn_index = np.genfromtxt(index_path, dtype=np.int)
        #np.sum(np.in1d(b, a))
        syn_size = syn_index.shape[0]
        for i in range(j, len(synsets)):
            child_path = '/media/raquel/Datos/Programación-git/tfg/Data/' + str(synsets[i]) + '_index' + '.txt'
            child_index = np.genfromtxt(child_path, dtype=np.int)
            child_in_synset = np.sum(np.in1d(child_index, syn_index))
            print('Tenemos ' + str(syn_size) + ' ' + str(synset) + ' de los cuales ' + str(child_in_synset)
                  + ' son ' + str(synsets[i]))
        j = j+1


instra_synset_stats(synsets)

for synset in synsets:
    print(str(synset))
    stats_from_synset(synset)
    ind = np.where(synsets == synset)
