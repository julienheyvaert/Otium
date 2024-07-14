import numpy as np
from scipy.ndimage import convolve

def grayscale_converter(image):
    """
    Input : image np.array, (Blue, green, red)
    Output : image_grayscale np.array (0.11 * blue + 0.59 * green + 0.3 * red)
    -- colour weight for the human eye.
    """
    # Input validity verification
    if len(image.shape) == 3:      
        image_grayscale = np.dot(image, [0.11, 0.59, 0.3])

    elif len(image.shape) == 2:
        return image
    else:
        raise ValueError('Error in grayscale_convertor. Invalid matrix shape.')

    return image_grayscale

def binary_reverse(edges_matrix):
    return 255 - edges_matrix

def black_white_converter(image_matrix):
    if image_matrix.ndim == 3:
        image_matrix = grayscale_converter(image_matrix)
    else:
        if(image_matrix.ndim != 2):
            raise ValueError('black_white_converter error : image_matrix must have 3 dimensions')
    return np.where(image_matrix > 127, 255, 0).astype(int)

def gen_gaussian_kernel_1d(kernel_dim= 10, sd = 5):
    """
    Generate a 1D Gaussian kernel.
    """
    if kernel_dim % 2 == 0:
        kernel_dim += 1
    if kernel_dim < 3:
        kernel_dim = 3

    center = kernel_dim // 2
    x = np.arange(-center, center + 1)
    kernel_1d = np.exp(-0.5 * (x / sd) ** 2)
    kernel_1d /= np.sum(kernel_1d)

    return kernel_1d

def gaussian_blur(image_matrix, kernel_dim=5, sd=1):
    """
    Input : image_matrix, np.array
            kernel_dim, int
            sd (standard deviation), real number

    Output : blurred_matrix, np.array

    -- Convolve a Gaussian kernel with the image,
    -- Applies a 'summary' of neighbouring pixels to each pixel.
    """
    if kernel_dim >= image_matrix.shape[0] or kernel_dim >= image_matrix.shape[1]:
        raise ValueError('Invalid kernel dimension.')

    gaussian_kernel_1d = gen_gaussian_kernel_1d(kernel_dim, sd)

    if image_matrix.ndim == 2:
        # Image is in grayscale
        blurred_matrix = convolve(image_matrix, gaussian_kernel_1d[:, None], mode='reflect')
        blurred_matrix = convolve(blurred_matrix, gaussian_kernel_1d[None, :], mode='reflect')

    elif image_matrix.ndim == 3:
        # Image is in BGR
        blurred_matrix = np.zeros_like(image_matrix)
        for channel in range(image_matrix.shape[2]):
            blurred_channel = convolve(image_matrix[:, :, channel], gaussian_kernel_1d[:, None], mode='reflect')
            blurred_matrix[:, :, channel] = convolve(blurred_channel, gaussian_kernel_1d[None, :], mode='reflect')
    else:
        raise ValueError('Invalid image_matrix shape.')

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
    magnitudes = np.hypot(grad_x, grad_y)

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
    angle = angles_matrix % np.pi
    padded_magnitudes = np.pad(magnitudes_matrix, 1, mode='constant')
    max_by_dir_matrix = np.zeros_like(magnitudes_matrix)
    
    # Directions classification
    direction = np.zeros_like(angle, dtype=int)
    direction[np.where((angle >= -np.pi/8) & (angle < np.pi/8))] = 0
    direction[np.where((angle >= np.pi/8) & (angle < 3*np.pi/8))] = 1
    direction[np.where((angle >= 3*np.pi/8) & (angle < 5*np.pi/8))] = 2
    direction[np.where((angle >= 5*np.pi/8) & (angle < 7*np.pi/8))] = 3
    direction[np.where((angle >= 7*np.pi/8) & (angle <= np.pi))] = 0
    direction[np.where((angle >= -7*np.pi/8) & (angle < -5*np.pi/8))] = 0
    direction[np.where((angle >= -5*np.pi/8) & (angle < -3*np.pi/8))] = 1
    direction[np.where((angle >= -3*np.pi/8) & (angle < -np.pi/8))] = 3

    # List for directions assignement
    direction_vector = [(-1, 0), (1, 1), (1, 0), (1, -1)]
    
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            magnitude = padded_magnitudes[row, col]
            dir_idx = direction[row, col]
            dir_prev = direction_vector[dir_idx]
            dir_next = (-dir_prev[0], -dir_prev[1])
            
            prev_mag = padded_magnitudes[row + dir_prev[0], col + dir_prev[1]]
            next_mag = padded_magnitudes[row + dir_next[0], col + dir_next[1]]

            if magnitude >= prev_mag and magnitude >= next_mag:
                max_by_dir_matrix[row, col] = magnitude

    return max_by_dir_matrix


def hysteresis(outlined_matrix,threshold_high = 120, threshold_low = 80):
    """
    Input:  outline_matrix, matrix of the borders, np.array (dim 2)
            threshold_high, treshold_low, the gaps where the borders are described as relevant or not, real number (0 - 255)

    Output: jungle_matrix, the matrix where only the relevant borders are keeped, np.array (dim 2)
    """

    if not (0 <= threshold_low <= threshold_high <= 255):
        raise ValueError(f"Invalid thresholds ([0,255], treshold_high <= treshold_low)")
    
    rows, cols = outlined_matrix.shape
    jungle_matrix = np.zeros_like(outlined_matrix, dtype=np.uint8)

    strong = np.where(outlined_matrix >= threshold_high)
    weak = np.where((outlined_matrix >= threshold_low) & (outlined_matrix < threshold_high))

    jungle_matrix[strong[0], strong[1]] = 255

    for weak_pixel_index in range(len(weak)):
        weak_pixel = weak[weak_pixel_index]
        row = weak_pixel[0]
        col = weak_pixel[1]

        if (jungle_matrix[row-1:row+2, col-1:col+2] == 255).any():
            jungle_matrix[row, col] = 255

    return jungle_matrix

def canny(image, threshold_high = 100, threshold_low = 30, gaussian_kernel_dim = (5, 1)):
    if not (0 <= threshold_low <= threshold_high <= 255):
        raise ValueError(f"Invalid thresholds.")
    
    image_g = grayscale_converter(image)
    image_b = gaussian_blur(image_g, gaussian_kernel_dim[0], gaussian_kernel_dim[1])
    _, _, image_m, image_a = xy_gradients(image_b)
    image_o = only_maxima(image_m, image_a)

    return hysteresis(image_o, threshold_high, threshold_low)

def outliner(edges_matrix, threshold = 15, kernel_dim= 31):
    """
    treshold = nombre de pixels admis par quadrant de kernel
    Verifier si membre d'une chaine avant suppression
    A retenir : 
    pixel isolé peut etre très important.
    """
    if kernel_dim < 11 or kernel_dim % 2 == 0:
        raise ValueError('Function error : kernel dim must be an odd int greater than 11.')
    
    sub_m_radius = kernel_dim // 2
    rows, cols = edges_matrix.shape
    outlined_matrix = np.copy(edges_matrix)

    border_locations = np.argwhere(edges_matrix == 255)

    for row, col in border_locations:
        row_min = max(0, row - sub_m_radius)
        row_max = min(rows, row + sub_m_radius + 1)
        col_min = max(0, col - sub_m_radius)
        col_max = min(cols, col + sub_m_radius + 1)

        working_window = edges_matrix[row_min:row_max, col_min:col_max]

        kernel_radius = working_window.shape[0] // 2

        quarter_top_left = working_window[:kernel_radius, :kernel_radius]
        quarter_top_right = working_window[:kernel_radius, kernel_radius:]
        quarter_bottom_left = working_window[kernel_radius:, :kernel_radius]
        quarter_bottom_right = working_window[kernel_radius:, kernel_radius:]

        # Count the pixels by quarter
        count_tl = np.sum(quarter_top_left == 255)
        count_tr = np.sum(quarter_top_right == 255)
        count_bl = np.sum(quarter_bottom_left == 255)
        count_br = np.sum(quarter_bottom_right == 255)

        # If all quarters are exceed the treshold, delete the pixel
        if (count_tl > threshold and count_tr > threshold 
            and count_bl > threshold and count_br > threshold):
            
            outlined_matrix[row, col] = 0
    
    return outlined_matrix

def contour(image_matrix, edges_matrix = None):
    if not edges_matrix:
        edges_matrix = canny(image_matrix)

    contour_matrix = np.copy(image_matrix)

    if image_matrix.ndim == 3:
        contour_matrix[edges_matrix > 127] = [0, 255, 0]
    elif image_matrix.ndim == 2:
        contour_matrix[edges_matrix > 127] = 0
    else:
        raise ValueError('Invalid image_matrix shape.')

    return contour_matrix

def edges_extrimity_finder(edges_matrix):
    padded_matrix = np.pad(edges_matrix, 1, mode='constant')
    rows, cols = padded_matrix.shape
    edges_extrimity_matrix = np.zeros_like(padded_matrix)
    
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            sub_matrix = edges_matrix[row-1:row+2, col-1:col+2]
            if(500 <= np.sum(sub_matrix) <=755):
                edges_extrimity_matrix[row, col] = 255
    return edges_extrimity_matrix