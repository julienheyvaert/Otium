import numpy as np
from scipy.ndimage import convolve
import cv2
import time

"""
-- Ajouter des bordures avant calcul pour only_maxima
"""

def getImage(path):
    path = path.replace("\\", "/")
    image = cv2.imread(path)
    if image is None:
        print(f"Error: Could not load image from {path}")
        return None
    return image

def grayscale_converter(image):
    """
    Input : image np.array, (Blue, green, red)
    Output : image_grayscale np.array (0.11 * blue + 0.59 * green + 0.3 * red)
    -- colour weight for the human eye.
    """
    # Input validity verification
    if len(image.shape) != 3 or image.shape[2] != 3:
        print('Invalid image matrix.')
        return None
    
    image_grayscale = np.dot(image, [0.11, 0.59, 0.3])

    return image_grayscale

def gen_gaussian_kernel(dim, sd):
    """
    Generate a Gaussian kernel
    Input : dimension, real number
            sd (standart deviation), real number
    Output : Kernel, np.array
    -- a square matrix with weights that become weaker the further you are from its centre.
    -- weight distributed according to a Gaussian distribution.
    """
    if dim % 2 == 0:
        dim += 1
    if dim < 3:
        dim = 3

    kernel = np.zeros((dim, dim), dtype=np.float32)
    center = dim // 2

    for line in range(dim):
        for col in range(dim):
            x = line - center
            y = col - center
            kernel[line, col] = np.exp(-(x**2 + y**2) / (2 * sd**2))

    kernel /= np.sum(kernel)
    return kernel

def gaussian_blur(image_matrix, kernel_dim = 5, sd=1):
    """
    Input : image_matrix, np.array
            kernel_dim, real number
            sd (standart deviation), real number

    Output : blurred_matrix, np.array

    -- Convolve a Gaussian kernel with the image, 
    -- Applies a 'summary' of neighbouring pixels to each pixel.
    """
    if kernel_dim >= image_matrix.shape[0] or kernel_dim >= image_matrix.shape[1]:
        print('Invalid kernel.')
        return None
    
    gaussian_kernel = gen_gaussian_kernel(kernel_dim, sd)
    
    if len(image_matrix.shape) == 2:
        # image is in grayscale
        blurred_matrix = convolve(image_matrix, gaussian_kernel, mode='reflect')

    elif len(image_matrix.shape) == 3:  
        # Image is in BGR
        blurred_matrix = np.zeros_like(image_matrix)
        for channel in range(image_matrix.shape[2]):
            blurred_matrix[:, :, channel] = convolve(image_matrix[:, :, channel], gaussian_kernel, mode='reflect')
    else:
        return None

    return blurred_matrix

def xy_gradients(image_matrix):
    """
    Input : image_matrix, a grayscale image matrix, np.array (dim 2)
    Output: grad_x, grad_y, the gradients in each directions, np.array (dim 2)
            magnitudes, the matrix of the borders, np.array (dim 2)
            angles, the matrix of the borders, np.array (dim 2) 


    -- The goal is to calculate a rate of change around each pixel in the x and y directions.
    We look at the delta of the next pixel with the previous one, giving more importance to the direct neighbouring pixels.
    --> Convolution with Sobel matrix.
    The rates of change are grouped into Magnitudes (gives positive values and groups the directions).
    We calculate their orientation in angles.
    """

    # SOBEL matrix
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
    # Convolution
    grad_x = convolve(image_matrix, sobel_x, mode='reflect', output=np.float32)
    grad_y = convolve(image_matrix, sobel_y, mode='reflect', output=np.float32)

    # Calculate Angles
    angles = np.arctan2(grad_y, grad_x)

    # Calculate Magnitudes
    magnitudes = np.sqrt(np.square(grad_x) + np.square(grad_y))

    return grad_x, grad_y, magnitudes, angles

def only_maxima(magnitudes_matrix, angles_matrix):
    """
    Input:  magnitudes_matrix, np.array (dim 2)
            angles_matrix, np.array (dim 2)
    
    Output: outline_matrix, the matrix with the weak and isolated borders removed, np.array (dim 2)

    -- Analyses the weakness of the borders, keeping only does who are connected with strong ones 
    in each 4 directions.
    """
    rows, cols = angles_matrix.shape
    outline_matrix = np.zeros_like(magnitudes_matrix)
    
    angle = angles_matrix % np.pi  # Normalisation des angles entre -pi et pi
    
    # Définition des directions (0: horizontal, 1: diag montante, 2: verticale, 3: diag descendante)
    direction = np.zeros_like(angle, dtype=int)
    direction[np.where((angle >= -np.pi/8) & (angle < np.pi/8))] = 0
    direction[np.where((angle >= np.pi/8) & (angle < 3*np.pi/8))] = 1
    direction[np.where((angle >= 3*np.pi/8) & (angle < 5*np.pi/8))] = 2
    direction[np.where((angle >= 5*np.pi/8) & (angle < 7*np.pi/8))] = 3
    direction[np.where((angle >= 7*np.pi/8) & (angle <= np.pi))] = 0
    direction[np.where((angle >= -7*np.pi/8) & (angle < -5*np.pi/8))] = 0
    direction[np.where((angle >= -5*np.pi/8) & (angle < -3*np.pi/8))] = 1
    direction[np.where((angle >= -3*np.pi/8) & (angle < -np.pi/8))] = 3

    # Définir les offsets pour les directions
    offset = [(-1, 0), (1, 1), (1, 0), (1, -1)]
    
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            magnitude = magnitudes_matrix[row, col]
            dir_idx = direction[row, col]
            offset1 = offset[dir_idx]
            offset2 = (-offset1[0], -offset1[1])
            
            prev_mag = magnitudes_matrix[row + offset1[0], col + offset1[1]]
            next_mag = magnitudes_matrix[row + offset2[0], col + offset2[1]]

            if magnitude >= prev_mag and magnitude >= next_mag:
                outline_matrix[row, col] = magnitude

    return outline_matrix

def hysteresis(outlined_matrix,threshold_high = 120, threshold_low = 80):
    """
    Input:  outline_matrix, matrix of the borders, np.array (dim 2)
            threshold_high, treshold_low, the gaps where the borders are described as relevant or not, real number (0 - 255)

    Output: jungle_matrix, the matrix where only the relevant borders are keeped, np.array (dim 2)
    """

    if(threshold_high < 0 or threshold_high > 255 or threshold_low < 0 or threshold_low > 255 or threshold_low > threshold_high ):
        raise ValueError(f"Invalid thresholds: threshold_high must be between 0 and 255, and threshold_low must be between 0 and 255 and less than or equal to threshold_high. treshold_low : {threshold_low}, treshold_high : {threshold_high}")
    rows, cols = outlined_matrix.shape
    jungle_matrix = np.full((rows, cols), 0, dtype=np.uint8)

    strong_r, strong_c = np.where(outlined_matrix >= threshold_high)
    weak_r, weak_c = np.where((outlined_matrix >= threshold_low) & (outlined_matrix < threshold_high))

    jungle_matrix[strong_r, strong_c] = 255

    for i, j in zip(weak_r, weak_c):
        if (jungle_matrix[i-1:i+2, j-1:j+2] == 255).any():
            jungle_matrix[i, j] = 255

    return jungle_matrix

def canny(image, threshold_high = 100, threshold_low = 30, smoothness = (5, 1)):
    if(threshold_high < 0 or threshold_high > 255 or threshold_low < 0 or threshold_low > 255 or threshold_low > threshold_high ):
        raise ValueError(f"Invalid thresholds: threshold_high must be between 0 and 255, and threshold_low must be between 0 and 255 and less than or equal to threshold_high. treshold_low : {threshold_low}, treshold_high : {threshold_high}")
    image_g = grayscale_converter(image)
    image_b = gaussian_blur(image_g, smoothness[0], smoothness[1])
    grad_x, grad_y, image_m, image_a = xy_gradients(image_b)
    image_o = only_maxima(image_m, image_a)
    return hysteresis(image_o, threshold_high, threshold_low)

def test(image):
    cv2.imwrite('rendered/1_initial.jpg', image)

    edges = canny(image)
    cv2.imwrite('rendered/2_canny.jpg', edges)

    cv2.imwrite('rendered/3_realCanny.jpg', cv2.Canny(image, 50, 150))

image = getImage(r"C:\Users\julienn\Pictures\animals\leopard.jpg")
#test(image)