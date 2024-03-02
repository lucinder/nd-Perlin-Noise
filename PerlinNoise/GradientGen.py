import sys, csv
n = 1 if len(sys.argv) == 1 else int(sys.argv[1])

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
with open("gradients/gradient-"+str(n)+".csv", 'w') as outputFile:
    output = csv.writer(outputFile)
    gradients = getGradients()
    for gradient in gradients:
        # print(gradient)
        output.writerow(gradient)