import random
import math
import numpy as np
import matplotlib.pyplot as plt 
n = 2
idx = 0
hash = [ 151,160,137,91,90,15,                 
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,   
    190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
    129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
    49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
]
hash = hash+hash
# assuming our data is n dimensional, our n dimensional grid length can only be a maximum of rootn(256000/n)
# limit to 256 KB!
X_MAX_DEFAULT = int(256000**(1/n))
def getEdges(n):
    edges = [[0 for x in range(n)] for y in range(pow(2,n))]
    for i in range(0,pow(2,n)):
        for j in range (0,n):
            edges[i][j] = (i // (2**j))%2
    return edges
def getGradientsRecursive(tail, len):
    # print("Tail: " + str(tail))
    if len == 0:
        return tail # base case
    gradients = []
    if not 0 in tail:
        gradients.append([grad for grad in getGradientsRecursive([0] + tail, len-1)])
    gradients.append([grad for grad in getGradientsRecursive([1] + tail, len-1)])
    gradients.append([grad for grad in getGradientsRecursive([-1] + tail, len-1)])
    return gradients
def getGradients():
    gradients = []
    gradients.append([grad for grad in getGradientsRecursive([0], n-1)])
    gradients.append([grad for grad in getGradientsRecursive([1], n-1)])
    gradients.append([grad for grad in getGradientsRecursive([-1], n-1)])
    for i in range(0, n-1):
        gradients = sum(gradients, []) # reduce
    return gradients
def getDistances(point, edges):
    # global idx
    # print("Grid index: " + str(idx))
    n = len(point)
    vectors = [[0 for x in range(n)] for y in range(pow(2,n))]
    for i in range(0,pow(2,n)):
        for j in range (0,n):
            vectors[i][j] =  edges[i][j]- point[j]
        # print("Distance " + str(i) + " = " + str(vectors[i]))
    # idx += 1
    return vectors
def getInfluence(i, distance, gradients):
    grad = [gradients[int((hash[(i+x)%len(hash)]/len(hash))*len(gradients))] for x in range(pow(2,n))] # get gradient for index i
    # print("Gradient of length " + str(len(grad)) + ": " + str(grad))
    # print("Distance of length " + str(len(distance)) + ": " + str(distance))
    influence = [sum([(grad[j][k]) * distance[j][k] for k in range(n)]) for j in range(pow(2,n))]
    return influence
def interpRecursive(influence, point, iter):
    if(len(influence) == 1):
        return influence[0]
    result = [0 for i in range(int(len(influence)/2))]
    i = 0
    # print("Point: " + str(point))
    # print("Influence: " + str(influence))
    # print("Iteration: " + str(iter))
    while(i < len(influence)):
        target = int(i / 2)
        result[target] = influence[i] + point[iter]*(influence[i+1]-influence[i])
        i += 2
    return interpRecursive(result, point, iter+1)
def interp(influences, grid):
    interpolations = [0 for i in range(len(influences))]
    for i in range(len(influences)):
        interpolations[i] = interpRecursive(influences[i], grid[i], 0)
    return interpolations
def fade(x):
    return x*x*x*(x*(x*6-15)+10)

m = 100
grid = [[float((i % pow(m,n-x))/float(m)) for x in range(n)] for i in range(pow(m,n))]
# grid = [[(1.0/X_MAX_DEFAULT)*((y*n+x)%(pow(2,x))) for x in range(n)] for y in range(pow(X_MAX_DEFAULT,n))]
print("Grid size: " + str(len((grid))) + " ("+str(m)+"^"+str(n)+")")
# point = grid[0]
# print("Point: " + str(point))
edges = getEdges(n)
gradients = getGradients()
# print("Edges: " + str(edges))
# print("Gradient (size "+str(len(gradient))+"): " + str(gradient))
distances = [getDistances(point, edges) for point in grid]
print("Distance calculation complete!")
influences = [getInfluence(i, distances[i], gradients) for i in range(len(distances))]
print("Influence calculation completed!")
# print(str(influences))
interpolations = interp(influences, grid)
print("Interpolation done!")
# interpolations = [fade(x) for x in interpolations]
# print("Fade function done!")
# print(str(interpolations))
dim = 100
pixels = np.array([np.array([interpolations[i*m+j] for j in range(dim)]) for i in range(dim)])

pixel_plot = plt.figure()
plt.title("pixel_plot") 
pixel_plot = plt.imshow(pixels, cmap='twilight', interpolation='nearest') 
plt.colorbar(pixel_plot)
plt.show()