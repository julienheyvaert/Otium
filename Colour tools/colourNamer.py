def humanColour(hslCode):
    """
    input : hsl code (list -> [H,S,L])
    -- H [0,360] (degrees), S [0, 100] (intensity), L [0,100] (intensity)
    output : A human name corresponding to the colour
    """
    hue = hslCode[0]
    saturation = hslCode[1]
    lightness = hslCode[2]

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