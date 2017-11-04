from Code.wordnet_imagenet_connections import Statistics, Data
from nltk.corpus import wordnet as wn
import numpy as np
import sys

toolbar_width = 40

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

embeddings_version = [19, 25, 31]
for version in embeddings_version:

    data = Data(version)

    stats_living = Statistics(synsets_living, data)
    stats_non_living = Statistics(synsets_non_living, data)

    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

    stats_living.plot_all()
    stats_non_living.plot_all()

    for i in range(toolbar_width):
        sys.stdout.write("-")
        sys.stdout.flush()
    data.__del__()

    sys.stdout.write("\n")


toymatrix = np.array([[1, 0, -1, 0], [0, 1, 1, 1]])
toydict = {0: {-1: 1, 0: 2, 1: 3}, 1: {-1: 4, 0: 5, 1: 6}}

