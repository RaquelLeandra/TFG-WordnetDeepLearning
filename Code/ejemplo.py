from nltk.corpus import wordnet as wn
dog = wn.synsets('dog')[0]
print(dog)
print(dog.offset())


def get_wn_ss(imagenet_id):
    return wn.of2ss(imagenet_id[1:] + '-' + imagenet_id[0])


def get_in_id(wordnet_ss):
    wn_id = wn.ss2of(wordnet_ss)
    return wn_id[-1] + wn_id[:8]


in_id = get_in_id(dog)
print(in_id)
wn_ss = get_wn_ss(in_id)
print(wn_ss)