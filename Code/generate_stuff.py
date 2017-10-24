from Code.wordnet_imagenet_connections import Statistics
from nltk.corpus import wordnet as wn
import numpy as np
import matplotlib.pyplot as plt

import time
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


stats_living = Statistics(synsets_living)
stats_non_living = Statistics(synsets_non_living)

# setup toolbar
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
for i in range(toolbar_width):
    #stats.plot_all()
    stats_living.images_per_feature_per_synset_gen()
    stats_non_living.images_per_feature_per_synset_gen()
    sys.stdout.write("-")
    sys.stdout.flush()


sys.stdout.write("\n")

#for syn in synsets:
#    stats.get_index_from_ss(syn)

toymatrix = np.array([[1,0,-1,0],[0,1,1,1]])
toydict = {0:{-1:1,0:2,1:3},1:{-1:4,0:5,1:6}}

#stats.data_stats()
#stats.intra_synset_stats()
#print('inter')
#stats.inter_synset_stats()
#print('features_per_layer')
# #stats.get_features_per_layer()
# print('features_stats')
# #stats.features_stats()
# m = stats.features_per_image_gen()
#
# negone = {}
# for key in m.keys():
#     negone[key] = m[key][-1]
# print(str(negone))
#
# zeros = {}
# for key in m.keys():
#     zeros[key] = m[key][0]
#
# one = {}
# for key in m.keys():
#     one[key] = m[key][1]
#
#
# plt.hist( negone.values(), color='g')
# plt.show()