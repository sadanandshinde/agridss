from scipy.stats import norm
import numpy as np


# Bin center , Bins of 5 input variabls, including two tails
def defineBeans(mean, sd, intvl):
    intvl -= 1
    return np.arange(mean - 2 * sd - 2 * sd / intvl, (mean + 2 * sd + 2 * sd / intvl) + 4 * sd / intvl, 4 * sd / intvl)
    # (np.linspace(mean - 2 * sd - 2 * sd / intvl, mean + 2 * sd + 2 * sd / intvl, intvl+1 ))


# normal probability for 2 sd deviation (-2 sd plus 2 sd) use this probability for all other
def normIntervals(mean, sd, intvl):
    return (np.linspace(mean - 2 * sd - 2 * sd / intvl, mean + 2 * sd + 2 * sd / intvl, intvl))

def findProbabilityNDensity(values,mean, sd):

    pdfValues = norm.pdf(values, mean, sd)
    print('====pdf of values=== ' + str(pdfValues))
    print('====sum of pdfPrice ' + str(sum(pdfValues)))
    probV = pdfValues / sum(pdfValues)
    print('=====probcY == ' + str(probV))
    print('Sum probcY ' + str(sum(probV)) + ' len ' + str(len(probV)))

    vectorV = []
    # vectorizing the input scalar variables with their associated probabilities
    for i in range(len(values)):
        vectorV.append([values[i], probV[i]])

    print("vectorcY : " + str(vectorV))

    return  vectorV
