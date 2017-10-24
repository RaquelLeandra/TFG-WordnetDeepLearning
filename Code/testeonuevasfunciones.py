"""
El objetivo de este fichero es testear que funcionan las nuevas funciones que le meto.
Comprobado, parece que funciona. Mantengo la función por si tengo que testear más funciones nuevas
"""
import numpy as np

toymatrix = np.array([[1,0,-1,0],[0,1,1,1],[1,-1,0,1]])
toydict = {0:{-1:1,0:2,1:3},1:{-1:4,0:5,1:6}}

toylayers = {
        'a': [0, 2],
        'b': [2, 4],
               }


def count_features(matrix):
    """
    Devuelve un diccionario con la cantidad de features de cada tipo de la matriz matrix
    features[category] = cantidad de category de la matriz
    """
    features = {-1: 0, 0: 0, 1: 0}
    features[1] += np.sum(np.equal(matrix, 1))
    features[-1] += np.sum(np.equal(matrix, -1))
    features[0] += np.sum(np.equal(matrix, 0))
    return features
toyresult = {}
def images_per_feature_per_layer_gen():
    """
    Quiero generar un diccionario tal que

    :return:dict[layer][category] = cantidad de imagenes que tienen la feature con valor category en este layer
    """
    print(toymatrix)
    for layer in toylayers:
        section =toymatrix[:, range(toylayers[layer][0], toylayers[layer][1])]
        print(section)
        toyresult[layer] = count_features(section)
    print(toyresult)

images_per_feature_per_layer_gen()