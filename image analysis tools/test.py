from imageAnalysisTools import *
import math
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

def delta_rate(x, y):
    return (math.log10(x) - math.log10(y))**2

