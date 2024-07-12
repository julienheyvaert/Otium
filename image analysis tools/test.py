from imageAnalysisTools import *
import time

def outliner(extremity_edge_matrix, track_width, threshold = 50):
    """
    Prendre une matrice des extrémités des bords d'une image.
    Faire passer des couloirs droits, relier les extrémités voisins (treshold max de distance).
    """
    start = time.time()

    outline_matrix = np.zeros_like(extremity_edge_matrix)
    rows, cols = extremity_edge_matrix.shape

    for col in range(0, cols, track_width):
        if col + track_width < cols:
            track = extremity_edge_matrix[:, col:col + track_width]
            if(col >= cols/2):
                end = time.time()
                print(f"Done in {end - start} seconds.")
                return track
        else:
            track = extremity_edge_matrix[:, col:cols]
    
    return outline_matrix
    


# Initial image
image = cv2.imread("animals/eiffel.jpg")
cv2.imwrite('rendered/0_initial.jpg', image)

# Canny
edge_canny = canny(image)
cv2.imwrite('rendered/1_canny.jpg', edge_canny)

# Continuous Canny
print('Start.')
continuous_canny = outliner(edge_canny, 100)
cv2.imwrite('rendered/2_Continuous_Canny.jpg', continuous_canny)
print('Done.')