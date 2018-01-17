"""
In this code I make the different experiments in the FNE ussing the class stats and Data.
"""

from Code.wordnet_imagenet_connections import Statistics, Data
from nltk.corpus import wordnet as wn
import sys
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

all = [living_things, mammal, dog, hunting_dogs, artifact, instrum, conv, wheeled_vehicle]


def generate_stuff():
    embeddings_version = [19, 25, 31]
    for version in embeddings_version:
        print('Loading data...')
        ini_time = time.time()
        data = Data('', version)
        stats_living = Statistics(synsets_living, data)
        stats_non_living = Statistics(synsets_non_living, data)
        stats_all = Statistics(all, data)

        print('Loaded in ', datetime.timedelta(seconds=(time.time() - ini_time)), 'seconds')
        ini_time = time.time()
        stats_all.plot_all()
        stats_living.plot_all()
        stats_non_living.plot_all()
        print('plot all time: ', datetime.timedelta(seconds=(time.time() - ini_time)))

        data.__del__()

        sys.stdout.write("\n")


def main():
    ini_time = time.time()
    generate_stuff()
    print('total time', datetime.timedelta(seconds=(time.time() - ini_time)))

if __name__ == "__main__":
    main()
