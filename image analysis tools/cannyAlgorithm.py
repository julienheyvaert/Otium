import numpy as np
from scipy.ndimage import convolve
import cv2
import time

"""
-- Ajouter des bordures avant calcul pour only_maxima
-- Optimiser only_maxima
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
    -- colour weight for the human eye
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
    -- a square matrix with weights that become weaker the further you are from its centre
    -- weight distributed according to a Gaussian distribution
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

def gaussian_blur(image_matrix, kernel_dim = 10, sd=10):
    """
    Input : image_matrix, np.array
            kernel_dim, real number
            sd (standart deviation), real number

    Output : blurred_matrix, np.array

    -- Convolve a Gaussian kernel and the image, 
    -- Applies a 'summary' of neighbouring pixels to each pixel
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
    Le but est de calculer un taux de variation autour de chaque pixel dans les directions x et y.
    On regarde le delta du pixel suivant avec le précédent, en accordant plus d'importance aux pixels voisins directs.
    --> Convolution avec matrice de Sobel.
    On regroupe les taux de variations en Magnitudes
    On calcul leurs orientation
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
    rows, cols = angles_matrix.shape
    outline_matrix = np.zeros_like(magnitudes_matrix)
    
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            angle = angles_matrix[row, col]
            magnitude = magnitudes_matrix[row, col]

            # Normalisation de l'angle entre -pi et pi
            angle = angle % np.pi
            
            if (-np.pi / 8 <= angle < np.pi / 8) or (7 * np.pi / 8 <= angle <= np.pi) or (-np.pi <= angle < -7 * np.pi / 8):
                # Horizontal
                prev_mag = magnitudes_matrix[row, col - 1]
                next_mag = magnitudes_matrix[row, col + 1]
            elif (np.pi / 8 <= angle < 3 * np.pi / 8) or (-7 * np.pi / 8 <= angle < -5 * np.pi / 8):
                # Up diag
                prev_mag = magnitudes_matrix[row + 1, col + 1]
                next_mag = magnitudes_matrix[row - 1, col - 1]
            elif (3 * np.pi / 8 <= angle < 5 * np.pi / 8) or (-5 * np.pi / 8 <= angle < -3 * np.pi / 8):
                # Vertical
                prev_mag = magnitudes_matrix[row - 1, col]
                next_mag = magnitudes_matrix[row + 1, col]
            else:
                # Down diag
                prev_mag = magnitudes_matrix[row - 1, col + 1]
                next_mag = magnitudes_matrix[row + 1, col - 1]

            if (magnitude >= prev_mag and magnitude >= next_mag):
                outline_matrix[row, col] = magnitude

    return outline_matrix

def hysteresis(outlined_matrix,threshold_high = 120, threshold_low = 80):
    """
    L'hystérésis permet de suivre les contours en connectant les contours faibles aux contours forts, 
    mais seulement s'ils sont adjacents à un contour fort. Cela garantit que seuls les contours réels, 
    qui sont connectés et significatifs, sont retenus, réduisant ainsi les faux positifs dus au bruit.
    """
    rows, cols = outlined_matrix.shape
    jungle_matrix = np.full((rows, cols), 0, dtype=np.uint8)

    strong_r, strong_c = np.where(outlined_matrix >= threshold_high)
    weak_r, weak_c = np.where((outlined_matrix >= threshold_low) & (outlined_matrix < threshold_high))

    jungle_matrix[strong_r, strong_c] = 255

    for i, j in zip(weak_r, weak_c):
        if (jungle_matrix[i-1:i+2, j-1:j+2] == 255).any():
            jungle_matrix[i, j] = 255

    return jungle_matrix

def canny(image, treshold_high = 120, treshold_low = 80, smoothness = (10,10)):
    image_g = grayscale_converter(image)
    image_b = gaussian_blur(image_g, smoothness[0], smoothness[1])
    image_m = xy_gradients(image_b)[2]
    image_a = xy_gradients(image_b)[3]
    image_o = only_maxima(image_m, image_a)

    return hysteresis(image_o, treshold_high, treshold_low)

def test():
    start = time.time()
    
    image = getImage(r"C:\Users\julienn\Pictures\animals\girafe.jpg")
    
    cv2.imwrite('rendered/1_initial.jpg', image)
    
    start_step = time.time()
    image_gray = grayscale_converter(image)
    cv2.imwrite('rendered/2_grayscale.jpg', image_gray)
    end = time.time()
    print(f"-- Grayed in {end - start_step} seconds.")

    start_step = time.time()
    image_blurred = gaussian_blur(image_gray, 10, 10)
    cv2.imwrite('rendered/3_blur.jpg', image_blurred)
    end = time.time()
    print(f"-- Blured in {end - start_step} seconds.")

    start_step = time.time()
    grad_x, grad_y, mag, angles = xy_gradients(image_gray)
    cv2.imwrite('rendered/4_grad_x.jpg', grad_x)
    cv2.imwrite('rendered/4_grad_y.jpg', grad_y)
    cv2.imwrite('rendered/5_magnitudes.jpg', mag)
    end = time.time()
    print(f"-- Gradients computed in {end - start_step} seconds.")

    start_step = time.time()
    image_outlined = only_maxima(mag, angles)
    cv2.imwrite('rendered/6_outlined.jpg', image_outlined)
    end = time.time()
    print(f"-- Outlined computed in {end - start_step} seconds.")

    start_step = time.time()
    image_hysteresis = hysteresis(image_outlined)
    cv2.imwrite('rendered/7_hysteresis.jpg', image_hysteresis)
    end = time.time()
    print(f"-- Hysteresis computed in {end - start_step} seconds.")

    start_step = time.time()
    myCanny = canny(image, 50, 80)
    cv2.imwrite('rendered/8_myCanny.jpg', myCanny)
    end = time.time()
    print(f"-- Canny computed in {end - start_step} seconds.")

    start_step = time.time()
    realCanny = cv2.Canny(image, 50, 150)
    cv2.imwrite('rendered/9_realCanny.jpg', realCanny)
    end = time.time()
    print(f"-- Real Canny computed in {end - start_step} seconds.")

    end = time.time()
    print(f"\n ==== Done in {end - start} seconds. ====\n")

test()
