import numpy as np
from nltk.corpus import wordnet as wn
from time import time
import pickle

start = time()
discretized_embedding_path = '../Data/vgg16_ImageNet_imagenet_C1avg_E_FN_KSBsp0.15n0.25_Gall_val_.npy'
dmatrix = np.array(np.load(discretized_embedding_path))
embedding_path = "../Data/vgg16_ImageNet_ALLlayers_C1avg_imagenet_train.npz"
embedding = np.load(embedding_path)
labels = embedding['labels']
del embedding

dog = wn.synsets('dog')[0]
mammal = wn.synsets('mammal')[0]
living_things = wn.synset('living_thing.n.01')
hunting_dogs = wn.synsets('hunting_dog')[0]
synsets = [living_things, mammal, dog, hunting_dogs]

total_images = dmatrix.shape[0]

features = {}
features_path = '../Data/features' + str(synsets) + '.pkl'
print('entering loop')
beginloop = time()
print(time())
for feature in range(0, dmatrix.shape[1]):
    features[feature] = {}
    feature_column = dmatrix[:, feature]
    if feature % 1000 == 0:
        print(feature)
        print(time()-beginloop)
    for i in [-1, 0, 1]:
        features[feature][i] = {}
        feature_index = np.where(np.equal(feature_column, i))
        for synset in synsets:
            index_path = '../Data/' + str(synset) + '_index' + '.txt'
            synset_index = np.genfromtxt(index_path, dtype=np.int)
            features[feature][i][str(synset)] = np.sum(np.in1d(synset_index, feature_index))

endloop = time()
with open(features_path, 'wb') as handle:
    pickle.dump(features, handle)


print(features[0][-1][str(dog)])

end = time()

print('Total time = ' + str(end - start) + '\n' 'Loop time = ' + str(endloop - beginloop))