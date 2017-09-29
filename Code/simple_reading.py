import sys
sys.modules[__name__].__dict__.clear()


import numpy as np
import matplotlib.pyplot as plt
from wordnet_imagenet_connections import Statistics
from nltk.corpus import wordnet as wn


dog = wn.synsets('dog')[0]
mammal = wn.synsets('mammal')[0]
living_things = wn.synset('living_thing.n.01')
hunting_dogs = wn.synsets('hunting_dog')[0]

synsets = [living_things, mammal, dog, hunting_dogs]

stats = Statistics(synsets)