import numpy as np
import cv2
"""
flouter une image en grayscale va 3 fois plus vite.
"""
def getImage(path):
    path = path.replace("\\", "/")
    image = cv2.imread(path)
    if image is None:
        print(f"Error: Could not load image from {path}")
        return None
    return image

def gen_gaussian_kernel(dim, sd):
    """
    Génère un noyau de Gauss 
    (matrice carrée de poids de moins en moins fort au plus on s'éloigne de son centre)
    poids distribué selon une distribution de Gauss
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

def gaussian_blur(image_matrix, kernel_dim, sd=10):
    if(kernel_dim >= image_matrix.shape[0] or kernel_dim >= image_matrix.shape[1]):
        print('Invalid kernel.')
        return None
    
    gaussian_kernel = gen_gaussian_kernel(kernel_dim, sd)
    
    # grayscale ou couleur

    if len(image_matrix.shape) == 2:
        # image en grayscale
        lines, columns = image_matrix.shape
        blurred_matrix = np.zeros((lines, columns), dtype=np.uint8)
        padded_image = cv2.copyMakeBorder(image_matrix, kernel_dim//2, kernel_dim//2, kernel_dim//2, kernel_dim//2, cv2.BORDER_REPLICATE)

        for line in range(lines):
            for col in range(columns):
                region = padded_image[line:line+kernel_dim, col:col+kernel_dim]
                blurred_matrix[line, col] = np.sum(gaussian_kernel * region)

    elif len(image_matrix.shape) == 3:  
        # Image en couleur
        lines, columns, channels = image_matrix.shape
        blurred_matrix = np.zeros((lines, columns, channels), dtype=np.uint8)
        padded_image = cv2.copyMakeBorder(image_matrix, kernel_dim//2, kernel_dim//2, kernel_dim//2, kernel_dim//2, cv2.BORDER_REPLICATE)

        for line in range(lines):
            for col in range(columns):
                for channel in range(3):
                    region = padded_image[line:line+kernel_dim, col:col+kernel_dim, channel]
                    blurred_matrix[line, col, channel] = np.sum(gaussian_kernel * region)

    else:
        return None

    return blurred_matrix