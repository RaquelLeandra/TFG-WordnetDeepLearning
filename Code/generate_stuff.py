from Code.wordnet_imagenet_connections import Statistics, Data
from nltk.corpus import wordnet as wn
import numpy as np
import sys
import os
import time
import datetime

dog = wn.synsets('dog')[0]
mammal = wn.synsets('mammal')[0]
living_things = wn.synset('living_thing.n.01')
hunting_dogs = wn.synsets('hunting_dog')[0]

synsets_living = [living_things, mammal, dog, hunting_dogs]
minisyn = [hunting_dogs]

artifact = wn.synsets('artifact')[0]
instrum = wn.synset('instrumentality.n.03')
conv = wn.synset('conveyance.n.03')
wheeled_vehicle = wn.synsets('wheeled_vehicle')[0]

synsets_non_living = [artifact, instrum, conv, wheeled_vehicle]

SUPERCOMBO = [living_things, mammal, dog, hunting_dogs, artifact, instrum, conv, wheeled_vehicle]


def generate_stuff():
    #embeddings_version = [19, 25, 31]
    embeddings_version = [25]
    for version in embeddings_version:
        print('Loading data...')
        ini_time = time.time()
        data = Data('', version)
        data_time = time.time() - ini_time

        #stats_living = Statistics(synsets_living, data)
        #stats_non_living = Statistics(synsets_non_living, data)
        stats_SUPERCOMBO = Statistics(SUPERCOMBO, data)
        print('Loaded in ', datetime.timedelta(seconds=(time.time() - ini_time)), 'seconds')
        # for synset in stats_living.synsets:
        #     stats_living.generate_restricted_labels(synset)
        # for synset in stats_non_living.synsets:
        #     stats_non_living.generate_restricted_labels(synset)
        ini_time = time.time()
        stats_SUPERCOMBO.plot_changes_between_synset()
        print('plot all time: ', datetime.timedelta(seconds=(time.time() - ini_time)))
        # print('Calculating distance1...')
        # ini_time = time.time()
        # print('distance1 living mamal: ', stats_SUPERCOMBO.distance_between_synsets_reps(SUPERCOMBO[0],SUPERCOMBO[1]))
        # print('distance time: ', datetime.timedelta(seconds=(time.time() - ini_time)))
        # print('Loading data...')
        # ini_time = time.time()
        # print('distance2 living mamal: ', stats_SUPERCOMBO.distance_between_synsets_global(SUPERCOMBO[0], SUPERCOMBO[1]))
        # print('distance time: ',datetime.timedelta(seconds=(time.time() - ini_time)))

        data.__del__()

        sys.stdout.write("\n")


def generate_for_one_synset():
    one_synset_stuff = [living_things, artifact, instrum, conv, hunting_dogs, dog]
    dir_path = '../Data/Embeddings/new_labels'
    files_list = os.listdir(dir_path)
    for i in range(len(files_list)):
        data = Data(dir_path + files_list[i])
        one_synsets = [one_synset_stuff[i]]
        stats_one = Statistics(one_synsets, data)
        print(stats_one.textsynsets)
        stats_one.plot_all()
        data.__del__()

toymatrix = np.array([[1, 0, -1, 0], [0, 1, 1, 1]])
toydict = {0: {-1: 1, 0: 2, 1: 3}, 1: {-1: 4, 0: 5, 1: 6}}


def main():
    ini_time = time.time()
    generate_stuff()
    print('total time', datetime.timedelta(seconds=(time.time() - ini_time)))

if __name__ == "__main__":
    main()
