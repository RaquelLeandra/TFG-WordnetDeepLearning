from Code.wordnet_imagenet_connections import Statistics
from nltk.corpus import wordnet as wn
import numpy as np
import matplotlib.pyplot as plt
dog = wn.synsets('dog')[0]
mammal = wn.synsets('mammal')[0]
living_things = wn.synset('living_thing.n.01')
hunting_dogs = wn.synsets('hunting_dog')[0]

synsets = [living_things, mammal, dog, hunting_dogs]
minisyn = [hunting_dogs]

stats = Statistics(synsets)

stats.plot_synsets_on_data()
textsynset = [str(s)[8:-7] for s in synsets]

print(str(textsynset))
print(str(synsets))

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