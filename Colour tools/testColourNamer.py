import random
import matplotlib.pyplot as plt
import numpy as np
from colourNamer import *
from colourCodes import *

"""
Simple test file for random colours
"""

def randomRGB():
    r = random.randint(1, 255)
    g = random.randint(1, 255)
    b = random.randint(1, 255)
    return [r,g,b]

def showColour(rgbValues):
    hsl = RGBToHSL(rgbValues)
    name = humanColour(hsl)
    rgb_array = np.zeros((100, 100, 3), dtype=np.uint8)  # Créer un tableau RGB de 100x100 pixels
    rgb_array[:,:] = rgbValues  # Remplir le tableau avec la couleur spécifiée

    plt.imshow(rgb_array)
    plt.axis('off')  # Désactiver les axes
    plt.title(f'\n name : {name} \n  rgb : {rgbValues} \n  hsl : {hsl}')
    plt.show()

def rdTest():
    c = randomRGB()
    print(f'RGB : {c}')
    showColour(c)

rdTest()