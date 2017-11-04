"""
El objetivo de este programa es tener código vacío para testear funciones a parte
"""
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

embeddings_version = [25]
# for version in embeddings_version:
#
#     data = Data(version)
#     stats_living = Statistics(synsets_living, data)
#     stats_non_living = Statistics(synsets_non_living, data)
#
#     for synset in synsets_living:
#         stats_living.plot_images_per_feature_of_synset_per_layer(synset)
#
#     for synset in synsets_non_living:
#         stats_non_living.plot_images_per_feature_of_synset_per_layer(synset)

rep1 = [1,-1,1]
rep2 = [0,0,1]
changes = np.zeros([3,3])
for feature in range(0,len(rep1)):
    r1 = rep1[feature]
    r2 = rep2[feature]
    if r1 == r2 == -1:
        changes[0][0] +=1
    elif r1 == r2 == 0:
        changes[1][1] += 1
    elif r1 == r2 == 1:
        changes[2][2] += 1

    elif r1 == -1 and r2 == 0:
        changes[1][0] += 1
    elif r1 == -1 and r2 == 1:
        changes[2][0] += 1

    elif r1 == 0 and r2 == -1:
        changes[0][1] += 1
    elif r1 == 0 and r2 == 1:
        changes[2][1] += 1

    elif r1 == 1 and r2 == -1:
        changes[0][2] += 1
    elif r1 == 1 and r2 == 0:
        changes[1][2] += 1
print(rep1)
print(rep2)
print(changes)

fig, ax = plt.subplots(figsize=(5, 5))
diag = np.zeros([3,3])
diag[0,0] += 1
diag[1,1] += 1
diag[2,2] += 1
ax.matshow(diag, cmap=plt.cm.Blues, alpha=0.3)
for i in range(changes.shape[0]):
    for j in range(changes.shape[1]):
        ax.text(x=j, y=i, s=changes[i, j], va='center', ha='center', fontsize=20)
plt.xticks([0,1,2],[-1,0,1])
plt.yticks([0,1,2],[-1,0,1])

plt.xlabel('Original values')
plt.ylabel('New values')
plt.title('Changes from a to b')
plt.tight_layout()

plt.show()