import numpy as np
from nltk.corpus import wordnet as wn
import time
from datetime import timedelta
import queue
from Code.wordnet_imagenet_connections import Data, Distances

import json


def get_wn_ss(imagenet_id):
    return wn.of2ss(imagenet_id[1:] + '-' + imagenet_id[0])


def get_in_id(wordnet_ss):
    wn_id = wn.ss2of(wordnet_ss)
    return wn_id[-1] + wn_id[:8]


def get_wn_id(imagenet_id):
    return imagenet_id[1:] + '-' + imagenet_id[0]


def breadth_first_search(synset):
    # a FIFO open_set
    open_set = queue.Queue()
    # an empty set to maintain visited nodes
    closed_set = set()
    # a dictionary to maintain meta information (used for path formation)
    meta = {}  # key -> (parent state, action to reach child)

    # initialize
    start = synset
    open_set.put(start)
    tree = {}
    i = 0
    distances = {}
    hypo = lambda s: s.hyponyms()
    for thing in list(synset.closure(hypo)):
        distances[thing] = 9999
    distances[synset] = 0
    tree[i] = {}
    tree[i][None] = [synset]
    while not open_set.empty():
        parent_state = open_set.get()
        for child_synset in parent_state.hyponyms():
            if distances[child_synset] ==9999:
                open_set.put(child_synset)
                distances[child_synset] = distances[parent_state] + 1
                if not distances[child_synset] in list(tree.keys()):
                    tree[distances[child_synset]] = {}
                if not parent_state in list(tree[distances[child_synset]].keys()):
                    tree[distances[child_synset]][parent_state] = []
                tree[distances[child_synset]][parent_state].append(child_synset)

    return tree


def get_hyponims(synset):
    hypo = lambda s: s.hyponyms()
    synset_list = []
    synset_tree = breadth_first_search(synset)
    # for key in list(synset_tree.keys()):
    #     print(key, '\n', synset_tree[key])
    # synset_tree[0] = synset.hyponyms()
    # for thing in list(synset.closure(hypo)):
    #     print(thing)
    #     synset_list.append(thing)
    return synset_tree


def get_all_hypernims_from_classes(synsets):
    clases = set()
    hyper = lambda s: s.hypernyms()
    for synset in synsets:
        hypernims = list(synset.closure(hyper))
        for h in hypernims:
            clases.add(h)
    return clases

def get_subtree_from_synset(synset):
    depht = 100
    aux = synset
    for i in range(depht):
        aux = aux.hyponym()

def load_clases():
    """
    Falta testear que lo lea bien
    :return:
    """
    imagenet_ids = open("../Data/synset.txt", 'r').read()
    print(imagenet_ids)
    synsets = []
    for i in imagenet_ids:
        synsets.append(get_wn_ss(i))
    print(synsets)
    get_tree_from_classes(synsets)


def main():
    dog = wn.synsets('dog')[0]
    ini_time = time.time()
    ss_list = []
    hypo = lambda s: s.hyponyms()
    for thing in list(dog.closure(hypo)):
        ss_list.append(thing)
    print(ss_list)
    data = Data('', 25)
    distance = Distances(data)
    distance.plot_changes_between_synset_reps(ss_list)
    print('total time', timedelta(seconds=(time.time() - ini_time)))

if __name__ == "__main__":
    main()