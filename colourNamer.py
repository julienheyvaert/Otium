import random
import matplotlib.pyplot as plt
import numpy as np

def randomRGB():
    r = random.randint(1, 255)
    g = random.randint(1, 255)
    b = random.randint(1, 255)
    return [r,g,b]


def RGBToHSL(rgb):
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


def showColour(rgbValues):
    hsl = RGBToHSL(rgbValues)
    name = humanColour(hsl)
    rgb_array = np.zeros((100, 100, 3), dtype=np.uint8)  # Créer un tableau RGB de 100x100 pixels
    rgb_array[:,:] = rgbValues  # Remplir le tableau avec la couleur spécifiée

    plt.imshow(rgb_array)
    plt.axis('off')  # Désactiver les axes
    plt.title(f'\n name : {name} \n  rgb : {rgbValues} \n  hsl : {hsl}')
    plt.show()

def humanColour(hslCode):
    hue = hslCode[0]
    saturation = hslCode[1]
    lightness = hslCode[2]

    # Determine the main color based on the hue
    if 0 <= hue < 15 or 350 <= hue <= 360:
        colour = 'red'
        if 0 <= saturation < 33:
            if 0 <= lightness < 33:
                colour = 'brown'
                if(lightness < 9):
                    colour = 'black'
            elif 33 <= lightness < 66:
                colour = 'brown'
            else:
                colour = 'beige'
        elif 33 <= saturation < 66:
            if 0 <= lightness < 33:
                colour = 'dark brown'
            elif 33 <= lightness < 66:
                colour = 'brown red'
            else:
                colour = 'beige red'
        else:
            if 0 <= lightness < 33:
                colour = 'dark red'
                if(lightness <= 6):
                    colour = 'black'
            elif 33 <= lightness < 66:
                colour = 'red'
            else:
                colour = 'magenta red'
                if lightness > 87:
                    colour = 'magenta'
    elif 15 <= hue < 40:
        colour = 'orange'
        
        if 0 <= saturation < 33:
            if 0 <= lightness < 33:
                colour = 'brown'
                if(lightness <= 11):
                    colour = 'dark brown'
            elif 33 <= lightness < 66:
                colour = 'brown beige'
            else:
                colour = 'light brown beige'

        elif 33 <= saturation < 66:
            if 0 <= lightness < 33:
                colour = 'brown'
                if lightness <= 5:
                    colour = 'black'
            elif 33 <= lightness < 66:
                colour = 'brown'
            else:
                colour = 'light brown beige'
        else:
            if 0 <= lightness < 33:
                colour = 'brown'
            elif 33 <= lightness < 66:
                colour = 'orange'
            else:
                colour = 'light orange beige'
        if lightness <=5:
            colour = 'black'
    elif 40 <= hue < 65:
        colour = 'yellow'
        if 0 <= saturation < 33:
            if 0 <= lightness < 33:
                colour = 'brown yellow (kaki)'
                if (saturation <5):
                    colour = 'grey (from kaki)'
            elif 33 <= lightness < 66:
                colour = 'brown yellow (kaki)'
            else:
                colour = 'beige'
                if lightness > 94:
                    colour = 'light beige'
        elif 33 <= saturation < 66:
            if 0 <= lightness < 33:
                colour = 'brown kaki'
            elif 33 <= lightness < 66:
                colour = 'yellow kaki'
            else:
                colour = 'light yellow'
        else:
            if 0 <= lightness < 33:
                colour = 'brown kaki'
                if (lightness < 6):
                    colour = 'black'
            elif 33 <= lightness < 66:
                colour = 'yellow'
            else:
                colour = 'light yellow'
    elif 65 <= hue < 160:
        colour = 'green'
        if 0 <= saturation < 33:
            if 0 <= lightness < 33:
                colour = 'dark green'
            elif 33 <= lightness < 66:
                colour = 'green'
            else:
                colour = 'light green'
        elif 33 <= saturation < 66:
            if 0 <= lightness < 33:
                colour = 'green'
            elif 33 <= lightness < 66:
                colour = 'green'
            else:
                colour = 'light green'
        else:
            if 0 <= lightness < 33:
                colour = 'green'
                if lightness < 10:
                    colour = 'dark green'
            elif 33 <= lightness < 66:
                colour = 'green'
            else:
                colour = 'light green'

        if lightness < 7:
                    colour = 'black'
    elif 160 <= hue < 200:
        colour = 'cyan'
        if 0 <= saturation < 33:
            if 0 <= lightness < 33:
                colour = 'green cyan'
            elif 33 <= lightness < 66:
                colour = 'green cyan'
            else:
                colour = 'light grey cyan'
        elif 33 <= saturation < 66:
            if 0 <= lightness < 33:
                colour = 'green cyan'
            elif 33 <= lightness < 66:
                colour = 'cyan'
            else:
                colour = 'light cyan'
        else:
            if 0 <= lightness < 33:
                colour = 'green cyan'
            elif 33 <= lightness < 66:
                colour = 'cyan'
            else:
                colour = 'light cyan'
    elif 200 <= hue < 240:
        colour = 'blue'
        if 0 <= saturation < 33:
            if 0 <= lightness < 33:
                colour = 'blue'
            elif 33 <= lightness < 66:
                colour = 'grey blue'
            else:
                colour = 'light grey blue'
        elif 33 <= saturation < 66:
            if 0 <= lightness < 33:
                colour = 'dark blue'
            elif 33 <= lightness < 66:
                colour = 'blue'
            else:
                colour = 'light blue'
        else:
            if 0 <= lightness < 33:
                colour = 'dark blue'
            elif 33 <= lightness < 66:
                colour = 'blue'
            else:
                colour = 'light blue'
    elif 240 <= hue < 300:
        colour = 'violet'
        if 0 <= saturation < 33:
            if 0 <= lightness < 33:
                colour = 'dark grey violet'
            elif 33 <= lightness < 66:
                colour = 'grey violet'
            else:
                colour = 'light grey violet'
        elif 33 <= saturation < 66:
            if 0 <= lightness < 33:
                colour = 'dark violet'
            elif 33 <= lightness < 66:
                colour = 'violet'
            else:
                colour = 'light violet'
        else:
            if 0 <= lightness < 33:
                colour = 'dark violet blue'
            elif 33 <= lightness < 66:
                colour = 'violet'
            else:
                colour = 'light violet'
    elif 300 <= hue < 350:
        colour = 'magenta'
        if 0 <= saturation < 33:
            if 0 <= lightness < 33:
                colour = 'dark grey magenta'
            elif 33 <= lightness < 66:
                colour = 'grey magenta'
            else:
                colour = 'light grey magenta'
        elif 33 <= saturation < 66:
            if 0 <= lightness < 33:
                colour = 'dark magenta'
            elif 33 <= lightness < 66:
                colour = 'magenta'
            else:
                colour = 'light magenta'
        else:
            if 0 <= lightness < 33:
                colour = 'dark magenta'
            elif 33 <= lightness < 66:
                colour = 'magenta'
            else:
                colour = 'light magenta'
    else:
        colour = 'unknown colour'

    return colour

def rdTest():
    c = randomRGB()
    print(f'RGB : {c}')
    showColour(c)

rdTest()