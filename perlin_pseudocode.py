import random
import math
import numpy as np
import matplotlib.pyplot as plt 
n = 2
idx = 0
hash = [96,131,210,231,40,179,184,56,133,209,188,207,176,245,218,230,185,70,76,105,214,182,174,72,146,159,162,14,227,160,82,212,192,191,172,74,157,236,39,26,226,201,250,211,81,254,244,219,107,161,24,53,5,154,253,34,145,197,112,233,23,68,87,49,8,156,196,248,27,149,111,19,4,62,203,190,25,132,140,202,65,16,141,118,104,153,113,144,67,175,37,216,114,60,243,189,101,150,220,217,12,252,206,124,71,103,69,171,38,126,152,167,121,178,78,106,86,15,194,66,10,237,99,55,77,13,57,205,30,44,89,138,88,41,187,73,241,221,92,215,125,168,1,46,29,239,193,52,143,251,128,61,129,45,242,64,63,213,0,123,238,43,35,208,22,33,169,222,50,51,59,32,83,20,180,183,17,108,198,177,18,80,199,94,3,9,2,170,130,186,95,165,247,204,142,28,229,102,195,116,224,163,164,97,47,36,31,223,151,225,100,122,135,136,109,84,166,249,119,7,246,155,120,235,200,181,255,127,147,21,137,58,42,75,228,54,90,232,85,93,6,79,117,98,134,173,91,148,234,240,158,11,110,48,139,115]

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
    grad = [gradients[hash[(i+x)%256]%len(gradients)] for x in range(pow(2,n))] # get gradient for index i
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
# print(str(interpolations))
dim = 100
pixels = np.array([np.array([interpolations[i*m+j] for j in range(dim)]) for i in range(dim)])

pixel_plot = plt.figure()
plt.title("pixel_plot") 
pixel_plot = plt.imshow(pixels, cmap='twilight', interpolation='nearest') 
plt.colorbar(pixel_plot)
plt.show()