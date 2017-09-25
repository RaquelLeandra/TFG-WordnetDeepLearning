import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
dog = wn.synsets('dog')[0]
mammal = wn.synsets('mammal')[0]
living_things = wn.synset('living_thing.n.01')
hunting_dogs = wn.synsets('hunting_dog')[0]

synsets = [living_things, mammal, dog, hunting_dogs]

strsynsets = [str(living_things)[8:-7], str(mammal)[8:-7], str(dog)[8:-7], str(hunting_dogs)[8:-7]]

totalimagenes = 50000

cantidad_synsets = {strsynsets[0]: 20500, strsynsets[1]:10900,strsynsets[2]:5900,strsynsets[3]:3150}
porcentajes_synsets = {strsynsets[0]: 20500/totalimagenes * 100, strsynsets[1]:10900/totalimagenes * 100,strsynsets[2]:5900/totalimagenes * 100,strsynsets[3]:3150/totalimagenes * 100}
D = porcentajes_synsets

plt.bar(range(len(D)), D.values(), align='center', color=['#3643D2', 'c', '#722672','#BF3FBF'])
plt.xticks(range(len(D)), D.keys())
plt.title('Porcentajes de los synsets respecto las 50k imágenes')

plt.show()

features_total = {-1: 67.534703125, 0: 9.16827835052, 1: 23.2970185245}

plt.pie([float(v) for v in features_total.values()], labels=[float(k) for k in features_total.keys()], autopct=None, colors=['#3643D2', '#722672','#1B8C4A'])
plt.title('Distribución de las features sobre el total')
plt.show()

""" -Features de tipo 1: 144627891 el 23.2970185245 %
 -Features de tipo -1: 419255437 el 67.534703125 %
 -Features de tipo 0: 56916672 el 9.16827835052 %
 """