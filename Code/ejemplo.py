from nltk.corpus import wordnet as wn


dog = wn.synsets('dog')[0]
print(dog)
print(dog.offset())


def get_wn_ss(imagenet_id):
    return wn.of2ss(imagenet_id[1:] + '-' + imagenet_id[0])


def get_in_id(wordnet_ss):
    wn_id = wn.ss2of(wordnet_ss)
    print(wn_id)
    return wn_id[-1] + wn_id[:8]


def get_wn_id(imagenet_id):
    return imagenet_id[1:] + '-' + imagenet_id[0]


in_id = get_in_id(dog)
print(get_wn_id(in_id))
print(in_id)
wn_ss = get_wn_ss(in_id)
print(wn_ss)


hypo = lambda s: s.hyponyms()
living_things = wn.synset('living_thing.n.01')
mammals = wn.synset('mammal.n.01')
dogs = wn.synsets('dog')[0]
hunting_dogs = wn.synsets('hunting_dog')[0]
"""
text_file = open("/media/raquel/Datos/Programación-git/tfg/Data/mammals", "w")
for mammal in list(mammals.closure(hypo)):
    text_file.write(get_in_id(mammal)+'\n')
text_file.close()
"""
text_file = open("/media/raquel/Datos/Programación-git/tfg/Data/hunting_dogs.txt", "w")
for thing in list(hunting_dogs.closure(hypo)):
    text_file.write(get_in_id(thing)+'\n')
text_file.close()

