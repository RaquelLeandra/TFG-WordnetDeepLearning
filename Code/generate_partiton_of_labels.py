from wordnet_imagenet_connections import Data, Statistics
from nltk.corpus import wordnet as wn
import numpy as np
import matplotlib.pyplot as plt
dog = wn.synsets('dog')[0]
mammal = wn.synsets('mammal')[0]
living_things = wn.synset('living_thing.n.01')
hunting_dogs = wn.synsets('hunting_dog')[0]

synsets_living = [living_things, mammal, dog, hunting_dogs]
minisyn = [hunting_dogs]

object = wn.synsets('artifact')[0]
instrum = wn.synset('instrumentality.n.03')
conv = wn.synset('conveyance.n.03')
wheeled_vehicle = wn.synsets('wheeled_vehicle')[0]

synsets_non_living = [object, instrum, conv, wheeled_vehicle]

version = [25]
print('Loading data of version', str(version[0]), '...')
data = Data(version[0])


synsets = input('Living things(L) or Non Living(N)? (print exit to exit)')
while synsets != 'L' and synsets != 'N':
    synsets = input('Please, put a valid option: Living things(L) or Non Living(N)(print exit to exit)?')
while synsets != 'exit':

    if synsets == 'L':
        stats_living = Statistics(synsets_living, data)
        print('The synsets are: ', str(stats_living.textsynsets))
        print('What synset do you want to use?')
        i = 0
        for synset in stats_living.textsynsets:
            print(synset, '=', i)
            i += 1
        index_synset = input('Pick one of the numerical values: ')
        synset = stats_living.synsets[int(index_synset)]
        rl = stats_living.generate_restricted_labels(synset)
        print(rl[0:50], len(rl))

    elif synsets == 'N':
        stats_non_living = Statistics(synsets_non_living, data)
        print('The synsets are: ', str(stats_non_living.textsynsets))
        print('What synset do you want to use?')
        i = 0
        for synset in stats_non_living.textsynsets:
            print(synset, '=', i)
            i += 1
        index_synset = input('Pick one of the numerical values: ')
        synset = stats_non_living.synsets[int(index_synset)]
        rl = stats_non_living.generate_restricted_labels(synset)
        print(rl[0:50], len(rl))