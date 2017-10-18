import numpy as np
from nltk.corpus import wordnet as wn
from itertools import combinations
import _pickle as pickle
from os import path
from os import makedirs
import matplotlib.pyplot as plt

class Data:
    """
    Esta clase consiste en los datos que voy a necesitar para hacer las estadísticas.
    Que no dependen de los synsets elegidos.


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
        self.features_category = [-1,0,1]
        self.colors = ['#3643D2', 'c', '#722672','#BF3FBF']
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
        """
            Esta clase genera todas las estadísticas para un conjunto de synsets
        :param synsets:
        """
        self.data = Data()
        self.synsets = synsets
        self.textsynsets = [str(s)[8:-7] for s in synsets]
        self.dir_path = '../Data/' + str(self.synsets) + '/'
        if not path.exists(self.dir_path):
            makedirs(self.dir_path)
        self.stats_path = self.dir_path + str(self.synsets) + '_stats.txt'
        self.matrix_size = self.data.dmatrix.shape
        self.total_features = self.matrix_size[0]*self.matrix_size[1]
        self.all_features = self.count_features(self.data.dmatrix)
        self.synset_in_data = {}
        self.synsets_features = {}
        self.features_path = self.dir_path + 'features' + str(synsets) + '.pkl'
        self.images_per_feature_path = self.dir_path + 'images_per_feature' + '.pkl'
        self.images_per_feature = {}
        self.features_per_layer_path = self.dir_path + 'features_per_layer' + str(synsets) + '.pkl'
        self.features_per_image_path = self.dir_path + 'features_per_image' + str(synsets) + '.pkl'
        self.synset_in_data_path = self.dir_path + 'synset_in_data_path' + str()
        #Todo: preguntar si es mejor tener esto cargado en la memoria o pillarlo de un archivo cuando lo necesite.
        self.features_per_layer = {}
        self.features_per_image = {}
        self.basic_stats = {}

    def get_in_id(self, wordnet_ss):
        """
        Input: Synset
        :param wordnet_ss:
        :return: imagenet id
        """
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
        """
        This function generates a dictionary with the basic stats
        devuelve synset in data donde:
        synset_in_data[str(synset)] = cantidad de elementos del synset en los datos
        synset_in_data['total'] =  cantidad total de elementos

        """

        stats_file = open(self.stats_path, 'a')
        labels_size = self.data.labels.shape[0]
        self.synset_in_data['total'] = labels_size
        for synset in self.synsets:
            synset_path = self.dir_path + str(synset) + '.txt'
            index_path = self.dir_path + str(synset) + '_index' + '.txt'
            index = np.genfromtxt(index_path, dtype=np.int)
            if len(index) == 0:
                self.get_index_from_ss(synset)
                index = np.genfromtxt(index_path, dtype=np.int)
            self.synset_in_data[str(synset)] = index.shape[0]
            text = 'Tenemos ' + str(labels_size) + ' imagenes, de las cuales ' + str(float(index.shape[0])) + \
                   ', el ' + str(float(index.shape[0]) / labels_size * 100) + ' son ' + str(synset) + '\n'
            stats_file.write(text)
            print(text)
        with open(self.synset_in_data_path, 'wb') as handle:
            pickle.dump(self.images_per_feature_per_synset, handle)

        stats_file.close()

    def count_features(self, matrix):
        """
        Devuelve un diccionario con la cantidad de features de cada tipo de la matriz matrix
        """
        features = {-1: 0, 0: 0, 1: 0}
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

    def images_per_feature_per_synset_gen(self):
        """Genera un archivo con el diccionario siguiente:
            Para cada feature(0,...,12k):
                Para cada tipo(-1,0,1)
                    Para cada synset:
                        - cantidad de imágenes del synset que tienen ese tipo en la feature en cuestión

        """

        for feature in range(0, self.data.dmatrix.shape[1]):
            self.images_per_feature_per_synset[feature] = {}
            feature_column = self.data.dmatrix[:, feature]
            for i in self.data.features_category:
                self.images_per_feature_per_synset[feature][i] = {}
                feature_index = np.where(np.equal(feature_column, i))
                for synset in self.synsets:
                    index_path = self.dir_path+ str(synset) + '_index' + '.txt'
                    synset_index = np.genfromtxt(index_path, dtype=np.int)
                    self.images_per_feature_per_synset[feature][i][str(synset)] = np.sum(np.in1d(synset_index, feature_index))
        with open(self.features_per_synset_path, 'wb') as handle:
            pickle.dump(self.images_per_feature_per_synset, handle)

    def images_per_feature_gen(self):
        """Genera un archivo con el diccionario siguiente:
            Para cada feature(0,...,12k):
                Para cada tipo(-1,0,1)
                    cantidad de imágenes del synset que tienen ese categoria en la feature en cuestión

        """
        for feature in range(0, self.data.dmatrix.shape[1]):
            self.images_per_feature[feature] = {}
            feature_column = self.data.dmatrix[:, feature]
            for i in self.data.features_category:
                self.images_per_feature[feature][i] = np.sum(np.equal(feature_column,i))
        with open(self.images_per_feature_path, 'wb') as handle:
            pickle.dump(self.images_per_feature, handle)

    def images_per_feature_stats(self):
        """"
        Aquí debería sacar las estadísticas de las features y guardarlas en features_stats

        """
        if self.images_per_feature == {}:
            self.images_per_feature = pickle.load(open(self.images_per_feature_path, 'rb'))
        feature_stats_path = self.features_path + '_stats'
        feature_stats_file = open(feature_stats_path,'a')
        for feature in self.images_per_feature:
            feature_stats_file.write(str(feature) + '\n')
            for i in self.data.features_category:
                feature_stats_file.write(str(i) + ': ' + str(self.images_per_feature[feature][i]) + '\n')
        feature_stats_file.close()

    def plot_images_per_feature(self):
        """
        Here I want to plot the images per feature in an histogram per category
        :return:
        """
        if self.images_per_feature == {}:
            self.images_per_feature = pickle.load(open(self.images_per_feature_path, 'rb'))
        for category in self.data.features_category:
            values = {}
            for key in self.images_per_feature.keys():
                values[key] = self.images_per_feature[key][category]
            plt.hist(list(values.values()))
            plt.title('Images per feature of ' + str(category) + ' category')
            plt.savefig('Images per feature of ' + str(category) + ' category' + '.png')
            plt.show()

    def plot_images_per_feature_of_synset(self, synset):
        """
        Here I want to plot the images per feature in an histogram per category
        :return:
        """
        if self.images_per_feature == {}:
            self.images_per_feature = pickle.load(open(self.features_path, 'rb'))
        for category in self.data.features_category:
            values = {}
            for key in self.images_per_feature.keys():
                values[key] = self.images_per_feature[key][category][str(synset)]
            plt.hist(list(values.values()))
            plt.title('Images per feature of ' + str(category) + ' category')
            plt.savefig('Images per feature of ' + str(category) + ' category' + '.png')
            plt.xlabel('Quantity of ' + str(category))
            plt.ylabel('Quantity of images')
            plt.savefig(self.dir_path + 'images_per_feature' + str(category))
            plt.show()

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

    def features_per_image_gen(self):
        """
        Esta función debería calcular para cada imagen cuantas features de cada tipo se activan
        Output:
        Un diccionario tal que:
        dic[imagen][tipo]=cantidad de features de este tipo que se activan
        """
        for image in range(0, len(self.data.labels)):
            self.features_per_image[image] = self.count_features(self.data.dmatrix[image,:])
        with open(self.features_per_image_path, 'wb') as handle:
            pickle.dump(self.features_per_image, handle)
        return self.features_per_image

    def plot_features_per_image(self):
        """
        It does a plot of the features per image for each category.
        la cantidad de imagenes que tienen tantas features -1
        :return:
        """
        if self.features_per_image == {}:
            self.features_per_image = pickle.load(open(self.features_path, 'rb'))
        if self.features_per_image == {}:
            self.features_per_image_gen()
        for category in self.data.features_category:
            values = {}
            for key in self.features_per_image.keys():
                values[key] = self.features_per_image[key][category]
            plt.hist(list(values.values()))
            plt.title('Features per image for ' + str(category) + ' category')
            plt.ylabel('Quantity of ' + str(category))
            plt.xlabel('Quantity of images')
            plt.show()
            plt.savefig(self.dir_path + 'features_per_image' + str(category))