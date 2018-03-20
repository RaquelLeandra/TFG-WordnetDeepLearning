import numpy as np
from nltk.corpus import wordnet as wn
from itertools import combinations
import _pickle as pickle
from os import path
from os import makedirs
import matplotlib.pyplot as plt
import gc
from scipy import stats as st
import json


class Data:
	"""
	Esta clase consiste en los datos que voy a necesitar para hacer las estadísticas.
	Que no dependen de los synsets elegidos.

	Attributes:
		version (int): versión del embedding que utilizo puede ser 19, 25 o 31
		embedding_path (str): path
		layers (dict): Un diccionario tal que
			layers[string correspondiente al layer] = [inicio del layer, final del layer]

		 labels ()

		 :parameter version = Version del embedding que utilizo
	"""

	def __init__(self, my_path):
		"""

		:param version: Es la versión del embedding que queremos cargar (25,31,19)
		"""
		self.version = version
		_embedding_path = "../Data/Embeddings/vgg16_ImageNet_ALLlayers_C1avg_imagenet_train.npz"
		self.imagenet_id_path = "../Data/Distances/Common_Data/synsets_in_imagenet.txt"
		_embedding = 'vgg16_ImageNet_imagenet_C1avg_E_FN_KSBsp0.15n0.25_Gall_train_.npy'
		self.discretized_embedding_path = '../Data/Embeddings/' + _embedding
		print('Estamos usando ' + _embedding[-20:-16])
		# embedding = np.load(_embedding_path)
		self.labels = np.load('../Data/Embeddings/labels.npy')
		# self.matrix = self.embedding['data_matrix']
		self.dmatrix = np.array(np.load(self.discretized_embedding_path))
		self.imagenet_all_ids = np.genfromtxt(self.imagenet_id_path, dtype=np.str)
		self.features_category = [-1, 0, 1]
		self.colors = ['#3643D2', 'c', '#722672', '#BF3FBF']
		self.layers = {
			'conv1_1': [0, 64],  # 1
			'conv1_2': [64, 128],  # 2
			'conv2_1': [128, 256],  # 3
			'conv2_2': [256, 384],  # 4
			'conv3_1': [384, 640],  # 5
			'conv3_2': [640, 896],  # 6
			'conv3_3': [896, 1152],  # 7
			'conv4_1': [1152, 1664],  # 8
			'conv4_2': [1664, 2176],  # 9
			'conv4_3': [2176, 2688],  # 10
			'conv5_1': [2688, 3200],  # 11
			'conv5_2': [3200, 3712],  # 12
			'conv5_3': [3712, 4224],  # 13
			'fc6': [4224, 8320],  # 14
			'fc7': [8320, 12416],  # 15
			'conv1': [0, 128],  # 16
			'conv2': [128, 384],  # 17
			'conv3': [384, 1152],  # 18
			'conv4': [1152, 2688],  # 19
			'conv5': [2688, 4224],  # 20
			'conv': [0, 4224],  # 21
			'fc6tofc7': [4224, 12416],  # 23
			# 'all':[0,12416]          # 24
		}
		self.reduced_layers = {
			'conv1': [0, 128],
			'conv2': [128, 384],
			'conv3': [384, 1152],
			'conv4': [1152, 2688],
			'conv5': [2688, 4224],
			'fc6': [4224, 8320],
			'fc7': [8320, 12416]
		}
		_all_synsets_and_sons_path = '../Data/Distances/Common_Data/all_synsets_and_sons.pkl'
		self.all_synsets_and_sons = []
		if path.isfile(_all_synsets_and_sons_path):
			self.all_synsets_and_sons = pickle.load(open(_all_synsets_and_sons_path, 'rb'))
			print(len(self.all_synsets_and_sons), self.all_synsets_and_sons)
		else:
			self.all_synsets_and_sons = self.all_synsets_and_sons_gen()

	def get_wn_ss(self, imagenet_id):
		return wn.of2ss(imagenet_id[1:] + '-' + imagenet_id[0])

	def get_in_id(self, wordnet_ss):
		wn_id = wn.ss2of(wordnet_ss)
		return wn_id[-1] + wn_id[:8]

	def get_wn_id(self, imagenet_id):
		return imagenet_id[1:] + '-' + imagenet_id[0]

	def ss_to_text(self, synset):
		""" returns the string of the name of the input synset"""
		return str(synset)[8:-7]

	def wn_id_to_label(self):
		_path = '../Data/Distances/Common_Data/'
		if path.isfile(_path + 'wordnet_to_label.pkl'):
			wordnet_to_label = pickle.load(open(_path + 'wordnet_to_label.pkl', 'rb'))

		else:
			label_to_wordnet = pickle.load(open(_path + 'imagenet_label_to_wordnet_synset.pkl', 'rb'))
			wordnet_to_label = {}
			for i in range(0, 1000):
				wordnet_to_label[label_to_wordnet[i]['id']] = i
			with open(_path + 'wordnet_to_label.pkl', 'wb') as handle:
				pickle.dump(wordnet_to_label, handle)

	def all_synsets_and_sons_gen(self):
		"""
		This function calculates all the synsets and their hyponims presents in imagenet and saves it in a pickle.
		:return: np array with the synsets and their hyponims
		"""
		with open("../Data/Distances/Common_Data/synsets_in_imagenet.txt") as f:
			content = f.readlines()

		imagenet_ids = [x.strip() for x in content]

		synsets = []
		for i in imagenet_ids:
			synset = self.get_wn_ss(i)
			synsets.append(self.get_in_id(synset))
			hypo = lambda s: s.hyponyms()
			for thing in list(synset.closure(hypo)):
				synsets.append(self.get_in_id(thing))
		with open('../Data/Distances/Common_Data/all_synsets_and_sons.pkl', 'wb') as handle:
			pickle.dump(synsets, handle)
		return synsets

	def __del__(self):
		self.embedding = None
		self.dmatrix = None
		self.version = None
		self.embedding_path = None
		self.layers = None
		self.labels = None
		self.features_category = None
		self.colors = None
		gc.collect()

class Distances:
    """
    This class is formed for several distances and auxiliar functions
    """
	def __init__(self, data):
		self.data = data
		self.dir_path = '../Data/' + 'Distances' + '/'
		self.plot_path = self.dir_path + 'plots/'
		if not path.exists(self.dir_path):
			makedirs(self.dir_path)
		if not path.exists(self.plot_path):
			makedirs(self.plot_path)

	def get_in_id(self, wordnet_ss):
		"""
		Input: Synset
		:param wordnet_ss:
		:return: imagenet id (string)
		"""
		wn_id = wn.ss2of(wordnet_ss)
		return wn_id[-1] + wn_id[:8]

	def ss_to_text(self, synset):
		""" devuelve el string del nombre del synset en cuestion"""
		return str(synset)[8:-7]

	def ss_to_label(self):
		_path = '../Data/Distances/Common_Data/'
		label_to_wordnet = json.load(_path + 'imagenet_label_to_wordnet_synset.txt')
		wordnet_to_label = {}
		for i in range(0, 999):
			wordnet_to_label[label_to_wordnet[i]] = i
		print(wordnet_to_label)

	def get_index_from_ss(self, synset):
		"""
		Esta función genera un archivo con los índices(0:999) de la aparición de un synset y sus hiponimos
		y otro con los códigos imagenet de todos los hipónimos
		"""
		ss_path = self.dir_path + self.ss_to_text(synset) + '_index_hyponim' + '.npy'
		if path.isfile(ss_path):
			index = np.load(ss_path)
			return index
		else:
			hypo = lambda s: s.hyponyms()
			hyponim_file = open(ss_path, "w")
			synset_list = []
			for thing in list(synset.closure(hypo)):
				hyponim_file.write(self.get_in_id(thing) + '\n')
				synset_list.append(self.get_in_id(thing))

			index = []
			hyponim_file.close()
			i = 0
			for lab in self.data.labels:
				if self.data.imagenet_all_ids[lab] in synset_list:
					index.append(i)
				i += 1
			np.save(ss_path, index)
			return index

	def count_features(self, matrix):
		"""
		Devuelve un diccionario con la cantidad de features de cada tipo de la matriz matrix
		features[category] = cantidad de category de la matriz
		"""
		features = {-1: 0, 0: 0, 1: 0}
		features[1] += np.sum(np.equal(matrix, 1))
		features[-1] += np.sum(np.equal(matrix, -1))
		features[0] += np.sum(np.equal(matrix, 0))
		return features

	def get_represention_fast(self, synset):
		"""
		Quiero que me devuelva un vector tal que el valor i sea el que tiene mayor proporción dentro del synset.
		rep[feature] = 1, -1 o 0 según el valor que se repite más veces.
		:param synset:
		:return: rep
		"""
		index = self.get_index_from_ss(synset)
		if index != []:
			sub_matrix = self.data.dmatrix[index, :]
			rep = st.mode(sub_matrix, axis=0)[0]
			# print(self.ss_to_text(synset), sub_matrix.shape)
			return rep
		return []

	def distance_between_synsets_reps(self, synset1, synset2):
		"""
		Quiero que esta función me calcule la distancia entre dos synsets adyacentes de wordnet.
		Con la distancia definida como:
		abs(proporcion1(representante de s1) - proporcion1(representante de s2))
		siendo proporcion1 la proporción de 1 del representante.
		:return: distance (float)
		"""
		r1 = self.get_represention_fast(synset1)
		r2 = self.get_represention_fast(synset2)
		cf1 = self.count_features(r1)
		cf2 = self.count_features(r2)
		if (cf1[-1] + cf1[0] + cf1[1]) == 0 or (cf2[-1] + cf2[0] + cf2[1]) == 0:
			return 9999
		prop1 = cf1[1] / (cf1[-1] + cf1[0] + cf1[1])
		prop2 = cf2[1] / (cf2[-1] + cf2[0] + cf2[1])
		# print('prop1: ', prop1)
		# print(synset1)
		# print(r1)
		# print('prop2: ', prop2)
		# print(synset2)
		# print(r2)
		distance = np.abs(prop1 - prop2) * 100
		return distance

	def NEW_distance_between_synsets_reps(self, synset1, synset2):
		# print(self.ss_to_text(synset1), self.ss_to_text(synset2))
		r1 = self.get_represention_fast(synset1)
		r2 = self.get_represention_fast(synset2)
		cf1 = self.count_features(r1)
		cf2 = self.count_features(r2)
		if (cf1[-1] + cf1[0] + cf1[1]) == 0 or (cf2[-1] + cf2[0] + cf2[1]) == 0:
			return 9999
		sharedones = np.sum(np.equal(r1, 1) & np.equal(r1, r2))
		totalones = cf1[1] + cf2[1]
		d = 1 - (sharedones / (totalones - sharedones))
		# print(self.ss_to_text(synset1), self.ss_to_text(synset2), 'distance', d)
		return d

	def plot_changes_between_synset_reps(self, synsets):
		"""
		Quiero que printe una gráfica tal que en el valor de las x tenga los elementos de synsets y en el de las ordenadas
		un acumulative plot con la  cantidad de 1, 0 y -1 de los representantes del synset en cuestión.
		changes[synset][-1]
		:return: void
		"""
		plt.rcParams['figure.figsize'] = [40.0, 8.0]
		changes_in_synset = {}
		ones = []
		zeros = []
		negones = []
		l = 0
		textsynsets = []
		for synset in synsets:
			rep = self.get_represention_fast(synset)
			if rep == []:
				continue
			l += 1
			textsynsets.append(str(synset)[8:-7])
			changes_in_synset[self.ss_to_text(synset)] = self.count_features(rep)
			negones.append(changes_in_synset[self.ss_to_text(synset)][-1])
			zeros.append(changes_in_synset[self.ss_to_text(synset)][0])
			ones.append(changes_in_synset[self.ss_to_text(synset)][1])

		plot_index = np.arange(l)
		p_negones = plt.bar(plot_index, negones, color='#4C194C')
		p_zeros = plt.bar(plot_index, zeros, color='#7F3FBF', bottom=negones)
		p_ones = plt.bar(plot_index, ones, color='#3F7FBF', bottom=[sum(x) for x in zip(negones, zeros)])

		plt.ylabel('Cantidad')
		plt.title('Comparativa entre las categorias por synset')

		plt.xticks(plot_index, textsynsets)
		plt.legend((p_negones[0], p_zeros[0], p_ones[0]), ('-1', '0', '1'))
		plt.grid()
		name = 'Comparative_of_synsets.png'
		plt.savefig(self.plot_path + name)
		plt.cla()
		plt.clf()
		plt.rcParams['figure.figsize'] = [8.0, 8.0]
