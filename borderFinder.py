import cv2
import numpy as np

"""
Obsolete
"""

def border_finder(image_matrix, border_detection_treshold, border_detection_treshold_vertical):
    # Dimensions de l'image
    lines, columns = image_matrix.shape[0], image_matrix.shape[1]
    
    # Créer une matrice de bordure initialement remplie de 255 (blanc)
    border_matrix = np.ones((lines, columns), dtype=np.uint8) * 255

    # Point initial dans le coin supérieur gauche de l'image (0, 0)
    current_point_location = (0, 0)
    
    shape_found = False
    different_pixel_count_vertical = 0
    while not shape_found and current_point_location[0] < lines:
        # Valeur du pixel à current_point_location
        current_point_value = image_matrix[current_point_location[0], current_point_location[1]]
        if(current_point_value < 127):
            current_point_value = 0
        else:
            current_point_value = 255
        
        
        if current_point_value != 0:
            different_pixel_count_vertical += 1
            if(different_pixel_count_vertical >= border_detection_treshold_vertical):
                shape_found = True
        
        # Afficher la valeur du pixel initial
        print(f"Current point value: {current_point_value}")

        # Vérifier horizontalement vers la droite
        x, y = current_point_location
        different_pixel_count_horizontal = 0
        
        while y < columns:
            if image_matrix[x, y] == current_point_value:
                different_pixel_count_horizontal = 0  # Reset counter si pixel est le même
            else:
                different_pixel_count_horizontal += 1
                if different_pixel_count_horizontal >= border_detection_treshold:  # Vérifier 10 pixels consécutifs différents
                    border_matrix[x, y] = 0
                    print("Border detected")
                    current_point_location = (current_point_location[0] + 1, 0)
                    break
            y += 1
        else:
            print("Current line cleared")
            current_point_location = (current_point_location[0] + 1, 0)

    cv2.imshow('clearing image', border_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return border_matrix

# Charger l'image en niveaux de gris
def replace_backslashes_with_slashes(path):
    return path.replace("\\", "/")

image_path = replace_backslashes_with_slashes(r"C:\Users\julienn\Downloads\Group 28.jpg")
print(image_path)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Appeler la fonction border_finder
border_matrix = border_finder(image, 50, 50)

# Afficher l'image en niveaux de gris
cv2.imshow('greyed image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()