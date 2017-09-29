from Code.wordnet_imagenet_connections import *

dog = wn.synsets('dog')[0]
mammal = wn.synsets('mammal')[0]
living_things = wn.synset('living_thing.n.01')
hunting_dogs = wn.synsets('hunting_dog')[0]
synsets = [living_things, mammal, dog, hunting_dogs]


stats = Statistics()