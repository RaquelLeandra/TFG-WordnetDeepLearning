from wordnet_imagenet_connections import Statistics
from nltk.corpus import wordnet as wn

dog = wn.synsets('dog')[0]
mammal = wn.synsets('mammal')[0]
living_things = wn.synset('living_thing.n.01')
hunting_dogs = wn.synsets('hunting_dog')[0]

synsets = [living_things, mammal, dog, hunting_dogs]
minisyn = [hunting_dogs]

stats = Statistics(synsets)

for syn in synsets:
    stats.get_index_from_ss(syn)


#stats.data_stats()
#stats.intra_synset_stats()
#print('inter')
#stats.inter_synset_stats()
#print('features_per_layer')
#stats.get_features_per_layer()
print('features_stats')
stats.features_stats()