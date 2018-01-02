import numpy as np
from nltk.corpus import wordnet as wn
import time
from datetime import timedelta
import queue
from Code.wordnet_imagenet_connections import Data
from Code.wordnet_imagenet_connections import Distances as dis
import _pickle as pickle
import pygraphviz as PG

import json


def get_wn_ss(imagenet_id):
	return wn.of2ss(imagenet_id[1:] + '-' + imagenet_id[0])


def get_in_id(wordnet_ss):
	wn_id = wn.ss2of(wordnet_ss)
	return wn_id[-1] + wn_id[:8]


def get_wn_id(imagenet_id):
	return imagenet_id[1:] + '-' + imagenet_id[0]


def ss_to_text(synset):
	""" devuelve el string del nombre del synset en cuestion"""
	return str(synset)[8:-7]


def in_imagenet(synset, imagenet):
	hypo = lambda s: s.hyponyms()

	for thing in list(synset.closure(hypo)):
		if thing in imagenet:
			return True
	return False

def breadth_first_search(synset, imagenet):
	dat = Data('', 25)
	mydis = dis(dat)
	open_set = queue.Queue()
	graph = PG.AGraph()
	graph.node_attr.update(color='#3F7FBF', style="filled")
	graph.edge_attr.update(color="blue", len="4.0", width="2.0")
	# initialize
	start = synset
	open_set.put(start)
	tree = {}
	str_tree = {}
	i = 0
	distances = {}
	hypo = lambda s: s.hyponyms()
	for thing in list(synset.closure(hypo)):
		distances[thing] = 9999
	distances[synset] = 0
	tree[i] = {}
	str_tree[str(i)] = {}
	tree[i][None] = [synset]
	str_tree[str(i)][None] = [ss_to_text(synset)]
	graph.add_node(ss_to_text(synset))
	i = 0
	while not open_set.empty():
		parent_state = open_set.get()
		for child_synset in parent_state.hyponyms():
			if distances[child_synset] == 9999:
				open_set.put(child_synset)
				distances[child_synset] = distances[parent_state] + 1
				if not distances[child_synset] in list(tree.keys()):
					tree[distances[child_synset]] = {}
					str_tree[str(distances[child_synset])] ={}
				if not parent_state in list(tree[distances[child_synset]].keys()):
					tree[distances[child_synset]][parent_state] = []
					str_tree[str(distances[child_synset])][ss_to_text(parent_state)] = []
				if in_imagenet(child_synset, imagenet):
					distance = mydis.NEW_distance_between_synsets_reps(parent_state, child_synset)
					print(ss_to_text(parent_state), ss_to_text(child_synset), distance)
					if distance < 9999:
						if distance < 1:
							distance += 1
						graph.add_edge(ss_to_text(parent_state), ss_to_text(child_synset), len=distance)
					graph.draw('../Data/Distances/plots/mammals/' + ss_to_text(synset) + str(i) + '.png',  format='png', prog='neato')
					i = i + 1
					tree[distances[child_synset]][parent_state].append(child_synset)
					str_tree[str(distances[child_synset])][ss_to_text(parent_state)].append(ss_to_text(child_synset))
	_filename = '../Data/Distances/plots/' + ss_to_text(synset) + '.png'
	graph.draw(_filename, format='png', prog='neato')
	return tree, str_tree


def graph_from_dict(tree):
	graph = PG.AGraph()
	graph.node_attr.update(color='#3F7FBF', style="filled")
	graph.edge_attr.update(color="blue", len="2.0", width="2.0")

	for deph in list(tree.keys()):
		if deph != 0:
			father =list(tree[deph].keys())[0]

	_filename = '../Data/Distances/plots/' + 'tree' + '.png'
	graph.draw(_filename, format='png', prog='neato')


def get_hyponims(synset):
	imagenet_clases = load_imagenet_synsets()
	print(len(imagenet_clases))
	synset_tree, synset_str_tree = breadth_first_search(synset, imagenet_clases)
	return synset_tree, synset_str_tree


def get_all_hypernims_from_classes(synsets):
	clases = set()
	hyper = lambda s: s.hypernyms()
	for synset in synsets:
		hypernims = list(synset.closure(hyper))
		for h in hypernims:
			clases.add(h)
	return clases


def get_subtree_from_synset(synset, depht=4):
	"""
	Falta testear
	:param synset:
	:param depht:
	:return:
	"""
	aux = [synset]
	print(aux)
	subtree = {}
	subtree[4] = [synset]
	for i in range(1, depht):
		aux = aux[0].hyponyms()[0]
		subtree[4 - i] = [aux]
		hypo = lambda s: s.hyponyms()
		hyponims = list(aux[0].closure(hypo))
		if 4 - i + 1 not in list(subtree.keys()):
			subtree[4 - i + 1] = []
		for h in hyponims:
			if h not in subtree[4 - i + 1]:
				subtree[4 - i + 1].append()
	print(subtree)
	return subtree


def load_imagenet_synsets():
	"""
	:return:
	"""
	with open("../Data/Distances/Common_Data/synsets_in_imagenet.txt") as f:
		content = f.readlines()

	imagenet_ids = [x.strip() for x in content]
	synsets = []
	for i in imagenet_ids:
		synset = get_wn_ss(i)
		synsets.append(synset)
		hypo = lambda s: s.hyponyms()
		for thing in list(synset.closure(hypo)):
			synsets.append(thing)
	return synsets


def testin_dog():
	dog = wn.synsets('dog')[0]
	ini_time = time.time()
	ss_list = []
	hypo = lambda s: s.hyponyms()
	for thing in list(dog.closure(hypo)):
		ss_list.append(thing)
	print(ss_list)
	data = Data('', 25)
	distance = dis(data)

	distance.plot_changes_between_synset_reps(ss_list)
	print('total time', timedelta(seconds=(time.time() - ini_time)))


def test_graph():
	living_things = wn.synsets('mammal')[0]
	syn_mammals, syn_str_mammals = get_hyponims(living_things)
	print(syn_str_mammals)
	print(json.dumps(syn_str_mammals, indent=4, sort_keys=True))



def main():
	ini_time = time.time()
	test_graph()
	print('total time', timedelta(seconds=(time.time() - ini_time)))

if __name__ == "__main__":
	main()
