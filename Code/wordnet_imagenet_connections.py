import numpy as np
from nltk.corpus import wordnet as wn


class Data:
    """
    Esta clase consiste en los datos que voy a necesitar para hacer las estadísticas.
    """
    def __init__(self):
        # todo: generalizar el tema de los paths
        self.embedding_path = "../Data/vgg16_ImageNet_ALLlayers_C1avg_imagenet_train.npz"
        self.imagenet_id_path = "../Data/synset.txt"
        self.discretized_embedding_path = '../Data/vgg16_ImageNet_imagenet_C1avg_E_FN_KSBsp0.15n0.25_Gall_val_.npy'
        self.embedding = np.load(self.embedding_path)
        self.labels = self.embedding['labels']
        #self.matrix = self.embedding['data_matrix']
        self.dmatrix = np.load(self.discretized_embedding_path)
        self.imagenet_all_ids = np.genfromtxt(self.imagenet_id_path, dtype=np.str)
        self.layers = {
                'conv1_1': [0,64],        # 1
                'conv1_2': [64,128],      # 2
                'conv2_1': [128,256],     # 3
                'conv2_2': [256,384],     # 4
                'conv3_1': [384,640],     # 5
                'conv3_2': [640,896],     # 6
                'conv3_3': [896,1152],    # 7
                'conv4_1': [1152,1664],   # 8
                'conv4_2': [1664,2176],   # 9
                'conv4_3': [2176,2688],   # 10
                'conv5_1': [2688,3200],   # 11
                'conv5_2': [3200,3712],   # 12
                'conv5_3': [3712,4224],   # 13
                'fc6':[4224,8320],       # 14
                'fc7':[8320,12416],      # 15
                'conv1':[0,128],         # 16
                'conv2':[128,384],       # 17
                'conv3':[384,1152],      # 18
                'conv4':[1152,2688],     # 19
                'conv5':[2688,4224],     # 20
                'conv':[0,4224],         # 21
                '2_5conv':[128,4224],    # 22
                'fc6tofc7':[4224,12416], # 23
                #'all':[0,12416]          # 24
                    }

    def get_in_id(wordnet_ss):
        # Esta funcion genera la id de imagenet a partir del synset de wordnet
        wn_id = wn.ss2of(wordnet_ss)
        return wn_id[-1] + wn_id[:8]

    def get_index_from_ss(self, synset):
        """Esta función genera un archivo con los índices de la aparición de un synset y sus hiponimos"""
        hypo = lambda s: s.hyponyms()
        path = '../Data/' + str(synset) + '.txt'
        hyponim_file = open(path, "w")
        synset_list = []
        for thing in list(synset.closure(hypo)):
            hyponim_file.write(self.get_in_id(thing) + '\n')
            synset_list.append(self.get_in_id(thing))

        hyponim_file.close()
        index_path = '../Data/' + str(synset) + '_' + 'index' + '.txt'
        index_file = open(index_path, 'w')
        i = 0
        for lab in self.labels:
            if self.imagenet_all_ids[lab] in synset_list:
                index_file.write(str(i) + '\n')
            i += 1

        index_file.close()


class Statistics:
    def __init__(self, synsets):
        self.data = Data()
        self.synsets = synsets
        self.stats_path = '../Data/' + str(self.synsets[0:3]) + '_stats.txt'

    def data_stats(self):
        stats_file = open(self.stats_path, 'a')
        labels_size = self.data.labels.shape[0]
        for synset in self.synsets:
            synset_path = '../Data/' + str(synset) + '.txt'
            index_path = '../Data/' + str(synset) + '_index' + '.txt'
            index = np.genfromtxt(index_path, dtype=np.int)
            if len(index) == 0:
                self.data.get_index_from_ss(synset)
                index = np.genfromtxt(index_path, dtype=np.int)

            text = 'Tenemos ' + str(labels_size) + ' imagenes, de las cuales el ' + str(float(index.shape[0]) / labels_size * 100) + ' son ' + str(synset) + '\n'
            stats_file.write(text)
            print(text)
        stats_file.close()

    def inter_synset_stats(self):
        stats_file = open(self.stats_path, 'a')
        labels_size = self.data.labels.shape[0]
        for synset in self.synsets:
            synset_path = '../Data/' + str(synset) + '.txt'
            index_path = '../Data/' + str(synset) + '_index' + '.txt'
            index = np.genfromtxt(index_path, dtype=np.int)
            if len(index) == 0:
                self.data.get_index_from_ss(synset)
                index = np.genfromtxt(index_path, dtype=np.int)

    def compare_intra_embedding(self,synset):
        index_path = '../Data/' + str(synset) + '_index' + '.txt'
        syn_index = np.genfromtxt(index_path, dtype=np.int)
        total = 0
        for i in syn_index:
            for j in range(i,len(syn_index)):
                if j != i:
                    total += sum(np.equal(self.data.dmatrix[i, :], self.data.dmatrix[j, :]))
        return total

    def intra_synset_stats(self, synsets):
        j = 0
        stats_file = open(self.stats_path, 'a')
        total_embeddings_communes = []
        for synset in synsets:
            index_path = '../Data/' + str(synset) + '_index' + '.txt'
            syn_index = np.genfromtxt(index_path, dtype=np.int)
            # np.sum(np.in1d(b, a))
            syn_size = syn_index.shape[0]
            for i in range(j, len(synsets)):
                child_path = '../Data/' + str(synsets[i]) + '_index' + '.txt'
                child_index = np.genfromtxt(child_path, dtype=np.int)
                child_in_synset = np.sum(np.in1d(child_index, syn_index))
                text = 'Tenemos ' + str(syn_size) + ' ' + str(synset) + ' de los cuales ' + str(child_in_synset) \
                       + ' son ' + str(synsets[i]) + ' el ' + str(child_in_synset/syn_size * 100) + ' % \n'
                print(text)
                stats_file.write(text)
            j = j + 1
            print('embedding común')
            total_embeddings_communes.append(self.compare_intra_embedding(synset))
            text2 = 'Para el synset ' + str(synset) + ' hay un total de ' + str(total_embeddings_communes) + 'coincidencias respecto a un total de ' + self.data.dmatrix.shape[1]) + '\n'
            stats_file.write(text2)








