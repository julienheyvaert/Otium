import numpy as np
import math
import cv2

"""
Pourquoi valeur négative dans calcul des gradients pas erreur ?

-->objectif :  detection d'image similaire si nb de points d'inflexions et intesection verticales et horizontales sur grille
"""

def getImage(path):
    path = path.replace("\\", "/")
    return cv2.imread(path)

def greyMatrix(image_matrix):
    return cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)

def smooth_matrix(matrix):
    return cv2.blur(matrix, (5, 5))

def xyGradients(imageMatrix):
    """
    Le but est de calculer l'intensité de la variation de gris par pixel (horizontalement et verticalement)
    --> on calcule la dérivée en ce pixel en calculant la variation de la valeur de gris du pixel précédent et suivant
    --> f'(x) = (f(x+1) - f(x-1))/2 
    -- --> Effectivement, la fonction n'est pas vraiment continue (suite d'entiers) on est obligé de prendre 
    une pseudo-dérivée, approximation de la dérivée, assez large (précédent, suivant)

    --> NOYEAU DE SOBEL, matrice 3*3, servant de facteurs de calculs autour du point dans une direction donnée.
    -- --> Matrice 3*3 pour prendre aussi en compte, les pixels en diagonal
    -- --> poids plus important pour les pixels adjacants que pour les diagonaux (ex : +-2 adjacants, +-1 diagonaux)
    -- --> dim SOBEL = 3 car, la prise en compte des diagonaux, limites les effets du bruits (un bord de 1pixel n'est pas un bord)
    -- -- --> plus dim SOBEL est grande, moins il y a de bruit, mais aussi moins de précision de détection de bord réel


    input : -- matrice de l'image en grayscale --> np.array
            -- (optionnel) noiseCancelTreshold (dimension de SOBEL) -- réel
    output : -- matrices de gradients x, y et matrices d'angles des bords -- liste de 3 np.array                                
    """

    #Image dimensions
    lines = imageMatrix.shape[0]
    columns = imageMatrix.shape[1]
    
    # SOBEL matrix
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    # gradients matrixes initialisation
    grad_x = np.zeros((lines, columns))
    grad_y = np.zeros((lines, columns))

    # angles matrix initialisation
    angles = np.zeros((lines, columns))


    # Convolution
    for line in range(1, lines-1):
        for col in range(1, columns-1):

            # sub matrix 3*3
            region = imageMatrix[line-1:line+2, col-1:col+2]

            # convolution on current 3*3 sub matrix
            gx = np.sum(region * sobel_x)
            gy = np.sum(region * sobel_y)
            
            # Update Gradient matrixes
            grad_x[line, col] = gx
            grad_y[line, col] = gy

            # Calculate angle and Update angle Matrix
            angles[line, col] = math.atan(gy/gx)

    return grad_x, grad_y, angles


def magnitude(gradientsMatrix_x, gradientsMatrix_y):
    """
    Combine les contours détectés en direction x et y, en une valeur d'intensité de changement.
    """
    lines = gradientsMatrix_x.shape[0]
    columns = gradientsMatrix_x.shape[1]
    magnitudes = np.zeros_like(gradientsMatrix_x, dtype=np.float64)

    for line in range(lines):
        for column in range(columns):
            magnitude = (gradientsMatrix_x[line, column]**2 + gradientsMatrix_y[line, column]**2)**(1/2)
            magnitudes[line, column] = magnitude
    
    return magnitudes

def onlyMaxima(magnitudes, angles):
    lines = magnitudes.shape[0]
    columns = magnitudes.shape[1]
    maxima_only = np.zeros_like(magnitudes, dtype=np.float64)

    return maxima_only

image = getImage(r"C:\Users\julienn\Downloads\testImage.png")
cv2.imwrite('created images/defaultImage.jpg', image)

image = greyMatrix(image)
cv2.imwrite('created images/greyImage.jpg', image)

image = smooth_matrix(image)
cv2.imwrite('created images/smoothImage.jpg', image)

grad_x, grad_y, angles = xyGradients(image)[0], xyGradients(image)[1], xyGradients[2]
cv2.imwrite('created images/gradients_x.jpg', grad_x)
cv2.imwrite('created images/gradients_y.jpg', grad_y)

mag = magnitude(grad_x, grad_y)
cv2.imwrite('created images/magnitudes.jpg', mag)

print(angles)
