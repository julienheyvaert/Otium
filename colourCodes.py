import numpy as np

def RGBToHSL(rgb):
    """
    input : rgb code (list --> [R,G,B])
    -- R,G,B [0,255]
    output : hsl code (list -> [H,S,L])
    -- H [0,360] (degrees), S [0, 100] (intensity), L [0,100] (intensity)
    """
    if not rgb or len(rgb) != 3:
        return None
    
    # Normalize the RGB values
    NRGB = [value / 255 for value in rgb]

    # Find extrema
    maxRGB = max(NRGB)
    minRGB = min(NRGB)

    # Lightness "L" (color's brightness)
    L = (maxRGB + minRGB) / 2

    # Saturation "S" (color's purity)
    if maxRGB == minRGB:
        S = 0
    else:
        if L < 0.5:
            S = (maxRGB - minRGB) / (maxRGB + minRGB)
        else:
            S = (maxRGB - minRGB) / (2.0 - maxRGB - minRGB)
    
    # Hue "H" (color itself)
    if maxRGB == minRGB:
        H = 0
    else:
        if maxRGB == NRGB[0]:
            H = (NRGB[1] - NRGB[2]) / (maxRGB - minRGB)
        elif maxRGB == NRGB[1]:
            H = 2.0 + (NRGB[2] - NRGB[0]) / (maxRGB - minRGB)
        else:
            H = 4.0 + (NRGB[0] - NRGB[1]) / (maxRGB - minRGB)
        
        H = H * 60
        if H < 0:
            H += 360

    # Convert S and L to percentages
    S *= 100
    L *= 100

    # Rounding values
    H = round(H, 2)
    S = round(S, 2) 
    L = round(L, 2)

    return [H, S, L]


def npRGBtoHSL(rgb):
    """
    input : rgb code (np.array -> [R G B])
    -- R,G,B [0,255]
    output : hsl code (np.array -> [H S L])
    -- H [0,360] (degrees), S [0, 100] (intensity), L [0,100] (intensity)
    """
    if rgb.size != 3:
        return None
    
    # Normalize the RGB values
    NRGB = rgb / 255.0

    # Find extrema
    maxRGB = np.amax(NRGB)
    minRGB = np.amin(NRGB)

    # Lightness "L" (color's brightness)
    L = (maxRGB + minRGB) / 2

    # Saturation "S" (color's purity)
    if maxRGB == minRGB:
        S = 0.001
    else:
        if L < 0.5:
            S = (maxRGB - minRGB) / (maxRGB + minRGB)
        else:
            S = (maxRGB - minRGB) / (2.0 - maxRGB - minRGB)
    
    # Hue "H" (color itself)
    if maxRGB == minRGB:
        H = 0
    else:
        if maxRGB == NRGB[0]:
            H = (NRGB[1] - NRGB[2]) / (maxRGB - minRGB)
        elif maxRGB == NRGB[1]:
            H = 2.0 + (NRGB[2] - NRGB[0]) / (maxRGB - minRGB)
        else:
            H = 4.0 + (NRGB[0] - NRGB[1]) / (maxRGB - minRGB)
        
        H = H * 60
        if H < 0:
            H += 360

    # Convert S and L to percentages
    S *= 100
    L *= 100

    # Rounding values
    H = round(H, 2)
    S = round(S, 2) 
    L = round(L, 2)

    return [H, S, L]
    