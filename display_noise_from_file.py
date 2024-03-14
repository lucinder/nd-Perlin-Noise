import numpy as np
import matplotlib.pyplot as plt 
m = 0
n = 0
grid = []
with open("./PerlinNoise/perlin_out.txt") as noiseFile:
    lines = noiseFile.readlines()
    linecount = 0
    for line in lines:
        if (linecount == 0):
            dim = line.split(",")
            m = int(dim[0])
            n = int(dim[1])
        else:
            grid.append(float(line))
        linecount += 1
    noiseFile.close()
print("m: " + str(m) +", n " + str(n))
print("Matrix size: " + str(linecount-1))

pixels = np.array([np.array([grid[i*m+j] for j in range(m)]) for i in range(m)])
pixel_plot = plt.figure()
plt.title("Noise for Matrix Size " + str(m) + "^"+str(n)) 
pixel_plot = plt.imshow(pixels, interpolation='nearest') 
plt.colorbar(pixel_plot)
plt.show()