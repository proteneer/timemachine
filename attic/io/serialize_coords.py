import numpy as np

def write(xyz, masses, recenter=True):
    if recenter:
        xyz = xyz - np.mean(xyz, axis=0, keepdims=True)
    buf = str(len(masses)) + "\n"
    buf += "timemachine\n"
    for m, (x, y, z) in zip(masses, xyz):
        if int(round(m)) == 12:
            symbol = "C"
        elif int(round(m)) == 14:
            symbol = "N"
        elif int(round(m)) == 16:
            symbol = "O"
        elif int(round(m)) == 32:
            symbol = "S"
        elif int(round(m)) == 35:
            symbol = "Cl"
        elif int(round(m)) == 1:
            symbol = "H"
        elif int(round(m)) == 31:
            symbol = "P"
        elif int(round(m)) == 19:
            symbol = "F"
        elif int(round(m)) == 80:
            symbol = "Br"
        elif int(round(m)) == 127:
            symbol = "I"
        else:
            raise Exception("Unknown mass:" + str(m))

        buf += symbol + " " + str(round(x, 5)) + " " + str(round(y, 5)) + " " + str(round(z, 5)) + "\n"
    return buf
