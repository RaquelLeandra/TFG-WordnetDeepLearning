"""
El objetivo de este programa es tener código vacío para testear funciones a parte
"""
from Code.wordnet_imagenet_connections import Data, Statistics
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
data = Data(version)
stats_living = Statistics(synsets_living, data)
stats_non_living = Statistics(synsets_non_living, data)
