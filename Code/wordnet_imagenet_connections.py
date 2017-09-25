import numpy as np
from nltk.corpus import wordnet as wn
from itertools import combinations
import _pickle as pickle
from os import path
from os import makedirs


class Data:
    """
    Esta clase consiste en los datos que voy a necesitar para hacer las estadísticas.
    """
    def __init__(self):
        #TODO cambiar la inicialización de Data para que dependa de la matriz de embeddings
        self.embedding_path = "../Data/vgg16_ImageNet_ALLlayers_C1avg_imagenet_train.npz"
        self.imagenet_id_path = "../Data/synset.txt"
        self.discretized_embedding_path = '../Data/vgg16_ImageNet_imagenet_C1avg_E_FN_KSBsp0.15n0.25_Gall_val_.npy'
        self.embedding = np.load(self.embedding_path)
        self.labels = self.embedding['labels']
        #self.matrix = self.embedding['data_matrix']
        self.dmatrix = np.array(np.load(self.discretized_embedding_path))
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


class Statistics:
    def __init__(self, synsets):
        self.data = Data()
        self.synsets = synsets
        self.dir_path = '../Data/' + str(self.synsets) + '/'
        if not path.exists(self.dir_path):
            makedirs(self.dir_path)
        self.stats_path = self.dir_path + str(self.synsets) + '_stats.txt'
        self.matrix_size = self.data.dmatrix.shape
        self.total_features = self.matrix_size[0]*self.matrix_size[1]
        self.all_features = self.count_features(self.data.dmatrix)
        self.synsets_features = {}
        self.features_path = self.dir_path + 'features' + str(synsets) + '.pkl'
        self.features = {}
        self.features_per_layer_path = self.dir_path + 'features_per_layer' + str(synsets) + '.pkl'
        self.features_per_layer = {}

    def get_in_id(self, wordnet_ss):
        # Esta funcion genera la id de imagenet a partir del synset de wordnet
        wn_id = wn.ss2of(wordnet_ss)
        return wn_id[-1] + wn_id[:8]

    def get_index_from_ss(self, synset):
        """
        Esta función genera un archivo con los índices(0:999) de la aparición de un synset y sus hiponimos
        y otro con los códigos imagenet de todos los hipónimos
        """
        hypo = lambda s: s.hyponyms()
        path = self.dir_path + str(synset) + '_index_hyponim' + '.txt'
        hyponim_file = open(path, "w")
        synset_list = []
        for thing in list(synset.closure(hypo)):
            hyponim_file.write(self.get_in_id(thing) + '\n')
            synset_list.append(self.get_in_id(thing))

        hyponim_file.close()
        index_path = self.dir_path + str(synset) + '_' + 'index' + '.txt'
        index_file = open(index_path, 'w')
        i = 0
        for lab in self.data.labels:
            if self.data.imagenet_all_ids[lab] in synset_list:
                index_file.write(str(i) + '\n')
            i += 1

        index_file.close()

    def data_stats(self):
        stats_file = open(self.stats_path, 'a')
        labels_size = self.data.labels.shape[0]
        for synset in self.synsets:
            synset_path = self.dir_path + str(synset) + '.txt'
            index_path = self.dir_path + str(synset) + '_index' + '.txt'
            index = np.genfromtxt(index_path, dtype=np.int)
            if len(index) == 0:
                self.get_index_from_ss(synset)
                index = np.genfromtxt(index_path, dtype=np.int)

            text = 'Tenemos ' + str(labels_size) + ' imagenes, de las cuales ' + str(float(index.shape[0])) + \
                   ', el ' + str(float(index.shape[0]) / labels_size * 100) + ' son ' + str(synset) + '\n'
            stats_file.write(text)
            print(text)
        stats_file.close()

    def count_features(self, matrix):
        """
        Devuelve un diccionario con la cantidad de features de cada tipo de la matriz matrix
        """
        features = {1: 0, -1: 0, 0: 0}
        features[1] += np.sum(np.equal(matrix, 1))
        features[-1] += np.sum(np.equal(matrix, -1))
        features[0] += np.sum(np.equal(matrix, 0))
        return features

    def inter_synset_stats(self):
        stats_file = open(self.stats_path, 'a')
        labels_size = self.data.labels.shape[0]
        #todo: arreglar esto para no tener que hardcodearlo
        text = 'Dentro de las 50k imágenes tenemos: \n un total de ' \
               + str(self.total_features) + 'de la matriz de tamaño ' + str(self.matrix_size) \
               + '\n -Features de tipo -1: ' + str(self.all_features[-1]) + ' el ' + str(self.all_features[-1]/self.total_features * 100) + ' %' \
               + '\n -Features de tipo 0: ' + str(self.all_features[0]) + ' el ' + str(self.all_features[0]/self.total_features * 100) + ' %' \
               + '\n -Features de tipo 1: ' + str(self.all_features[1]) + ' el ' + str(self.all_features[1]/self.total_features * 100) + ' %'
        stats_file.write(text)
        for synset in self.synsets:
            synset_path = self.dir_path+ str(synset) + '.txt'
            index_path = self.dir_path+ str(synset) + '_index' + '.txt'
            index = np.genfromtxt(index_path, dtype=np.int)
            if len(index) == 0:
                self.get_index_from_ss(synset)
                index = np.genfromtxt(index_path, dtype=np.int)

            self.synsets_features[synset] = self.count_features(self.data.dmatrix[index, :])
            synset_features = self.count_features(self.data.dmatrix[index, :])
            synset_total_features = len(index)*self.matrix_size[1]
            text = '\nEn el ' + str(synset) + ' tenemos ' + str(synset_total_features) + 'features en total : ' \
                   + '\n -Features de tipo -1: ' + str(self.synsets_features[synset][-1]) + ' el ' + str(self.synsets_features[synset][-1] / synset_total_features * 100) + ' % respecto todas las features -1' \
                   + '\n -Features de tipo 0: ' + str(self.synsets_features[synset][0]) + ' el ' + str(self.synsets_features[synset][0] / synset_total_features * 100) + ' % respecto todas las features 0' \
                   + '\n -Features de tipo 1: ' + str(self.synsets_features[synset][1]) + ' el ' + str(self.synsets_features[synset][1] / synset_total_features * 100) + ' % respecto todas las features 1'
            stats_file.write(text)

        stats_file.close()

    def compare_intra_embedding(self, synset):
        index_path = self.dir_path + str(synset) + '_index' + '.txt'
        syn_index = np.genfromtxt(index_path, dtype=np.int)
        total = 0
        for i,j in combinations(syn_index,2):
            total += np.sum(np.equal(self.data.dmatrix[i, :], self.data.dmatrix[j, :]))
        return total

    def intra_synset_stats(self):
        j = 0
        stats_file = open(self.stats_path, 'a')
        total_embeddings_communes = []
        trol = 0
        for synset in self.synsets:
            index_path = self.dir_path+ str(synset) + '_index' + '.txt'
            syn_index = np.genfromtxt(index_path, dtype=np.int)
            # np.sum(np.in1d(b, a))
            syn_size = syn_index.shape[0]
            for i in range(j, len(self.synsets)):
                child_path = self.dir_path+ str(self.synsets[i]) + '_index' + '.txt'
                child_index = np.genfromtxt(child_path, dtype=np.int)
                child_in_synset = np.sum(np.in1d(child_index, syn_index))
                text = 'Tenemos ' + str(syn_size) + ' ' + str(synset) + ' de los cuales ' + str(child_in_synset) \
                       + ' son ' + str(self.synsets[i]) + ' el ' + str(child_in_synset/syn_size * 100) + ' % \n'
                print(text)
                stats_file.write(text)
            j = j + 1
            print('embedding común')
            #TODO: esto estaba mal, comprobar que pasa
            #total_embeddings_communes.append(self.compare_intra_embedding(synset))
            #stats_file.write('Para el synset ' + str(synset) + ' hay un total de ' + str(total_embeddings_communes)
            #                 + 'coincidencias respecto a un total de ' + str(self.data.dmatrix.shape[1]*len(syn_index)) + '\n')

        stats_file.close()

    def feature_generate(self):
        """Genera un archivo con el diccionario siguiente:
            Para cada feature(0,...,12k):
                Para cada tipo(-1,0,1)
                    Para cada synset:
                        - cantidad de imágenes del synset que tienen ese tipo en la feature en cuestión

        """
        self.features = {}

        for feature in range(0, self.data.dmatrix.shape[1]):
            self.features[feature] = {}
            feature_column = self.data.dmatrix[:, feature]
            for i in [-1, 0, 1]:
                self.features[feature][i] = {}
                feature_index = np.where(np.equal(feature_column, i))
                for synset in self.synsets:
                    index_path = self.dir_path+ str(synset) + '_index' + '.txt'
                    synset_index = np.genfromtxt(index_path, dtype=np.int)
                    self.features[feature][i][str(synset)] = np.sum(np.in1d(synset_index, feature_index))
        with open(self.features_path, 'wb') as handle:
            pickle.dump(self.features, handle)

    def features_stats(self):
        """"
        Aquí debería sacar las estadísticas de las features y guardarlas en features_stats

        """
        if self.features == {}:
            self.features = pickle.load(open(self.features_path, 'rb'))
        feature_stats_path = self.features_path + '_stats'
        feature_stats_file = open(feature_stats_path,'a')
        for feature in self.features:
            feature_stats_file.write(str(feature) + '\n')
            for i in [-1,0,1]:
                #feature_stats_file.write(str(i) + '\n')
                feature_stats_file.write(str(i) + ': ' + str(self.features[feature][i]) + '\n')
        feature_stats_file.close()

    def get_features_per_layer(self):
        """
        Crea un archivo de texto con la información de features
        :return:
        """
        #TODO: falta que calcule las estadísticas
        for layer in self.data.layers:
            section = self.data.dmatrix[:, range(self.data.layers[layer][0], self.data.layers[layer][1])]
            self.features_per_layer[layer] = self.count_features(section)
        with open(self.features_per_layer_path, 'wb') as handle:
            pickle.dump(self.features_per_layer, handle)

    def features_per_layer_stats(self, synsets):
        """"Aquí debería sacar las estadísticas de las features"""
        pass


