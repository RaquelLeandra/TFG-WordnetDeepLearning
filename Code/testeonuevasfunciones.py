"""
El objetivo de este fichero es testear que funcionan las nuevas funciones que le meto.
Comprobado, parece que funciona. Mantengo la función por si tengo que testear más funciones nuevas
"""
import numpy as np
import matplotlib.pyplot as plt

toymatrix = np.array([[1,0,-1,0],[0,1,1,1],[1,-1,0,1]])
toydict = {0:{-1:1,0:2,1:3},1:{-1:4,0:5,1:6}}

toylayers = {
        'a': [0, 2],
        'b': [2, 4],
               }
synsets = ['a','b']

ones = [1,3]
zeros = [1,2]
negones = [1,3]

plot_index = np.arange(len(synsets))
p_negones = plt.bar(plot_index, negones, color='#4C194C')
p_zeros = plt.bar(plot_index, zeros, color='#7F3FBF',bottom=negones)
p_ones = plt.bar(plot_index, ones, color='#3F7FBF',bottom=[sum(x) for x in zip(negones,zeros)])

plt.ylabel('Cantidad')
plt.title('Comparativa entre las categorias por synset')
plt.xticks(plot_index, synsets)
plt.legend((p_negones[0], p_zeros[0], p_ones[0]), ('-1', '0', '1'))
plt.show()