#from django.shortcuts import render
from django.shortcuts import render, redirect,get_object_or_404,render_to_response

#from statistics import mean, stdev,variance
import csv
import numpy as np
import pandas as pd
#from django.contrib.staticfiles.templatetags.staticfiles import static
from django.conf import settings as djangoSettings
from ..models import PDataset1
#from django.db.models import Q
#from bokeh.models import Range1d,LinearAxis
#import math
import logging, gc
from scipy.stats import norm
from decimal import *
#from django_pandas.io import read_frame
from ..scripts import sensitivitysimulation,BinsNDistributionCreator, BokehPlotter
from datetime import datetime
import itertools
#getcontext().prec = 6

app_name = 'passapp'
# Get an instance of a logger
logger = logging.getLogger(__name__)
logger.info("inside result")


'''
Numeric Simulation logic for 5 independent variables(Price , cost, Y0, Ymax,Nymax etc)
assumed everything normal distribution for now, read ECPA 2017 paper
'''
def simulation(request):

    startTime = datetime.now()
    '''
    #reading user input prvided earlier
    awdr = request.POST['AWDR']
    soiltype = request.POST['soilType']
    tilType = request.POST['tillType']
    cprice = request.POST['cprice']
    ncost = request.POST['ncost']

    if awdr == 'weak':
        lbound = 0
        ubound = 50
    elif awdr == 'average':
        lbound = 50
        ubound = 100
    elif awdr == 'heavy':
        lbound = 100
        ubound = 200
    '''

    soilTexture = request.POST['soilTexture']
    # fetch matching records from database
    #logger.debug("calling filter for matching records")
    #resultset = PDataset1.objects.filter(AWDR__gte=lbound, AWDR__lte=ubound, SoilType=soiltype, TillType=tilType)
    resultset = PDataset1.objects.filter(SoilTexture_cls=soilTexture)

    # converting the queryset into pandas dataframe
    #dfResult = read_frame(resultset)


    MeanPrice = 175
    StdPrice = 20
    # trying new approach for Price and Cost Prob from PDF function
    normPrice = np.linspace(130,220,7 ) #BinsNDistributionCreator.normIntervals(MeanPrice, StdPrice, 7)  # [130,145,160,175,190,205,220]#
    print("=====price===== " + str(normPrice))
    cY = normPrice.tolist()
    vectorcY=BinsNDistributionCreator.findProbabilityNDensity(normPrice,MeanPrice,StdPrice)

    MeanCost = 1
    StdCost = 0.25
    cost = np.linspace(0.4,1.6,7) #BinsNDistributionCreator.normIntervals(MeanCost, StdCost, 7)
    print("=====normCost===== " + str(cost))
    cN = cost.tolist()
    vectorcN=BinsNDistributionCreator.findProbabilityNDensity(cost,MeanCost,StdCost)

    dltYld = 0.5

    Ymin = np.arange(0, 20 + dltYld, dltYld)
    Ymax = np.arange(0, 20 + dltYld, dltYld)
    Nymax = np.arange(10, 260, 10)

    # saving the production and profit input parameters to dataframe, local csv file
    dfInput = pd.DataFrame(index=np.arange(0, len(Ymax)))
    dfInput['Y0'] = pd.Series(Ymin, index=np.arange(0, len(Ymin)))
    dfInput['Ymax'] = pd.Series(Ymax, index=np.arange(0, len(Ymax)))
    dfInput['Nymax'] = pd.Series(Nymax, index=np.arange(0, len(Nymax)))
    dfInput['cY'] = pd.Series(cY, index=np.arange(0, len(cY)))
    dfInput['cN'] = pd.Series(cN, index=np.arange(0, len(cN)))
    dfInput.index.name = 'index'
    print(dfInput.head())

    dfInput.to_csv('C:/Users/Student.Student11.000/Desktop/AgriDSS/Data/Sadanand/dfInputs'+soilTexture+'.csv')
    #dfInput.to_csv(djangoSettings.STATIC_ROOT + '/datafiles/dfInputs.csv', encoding='utf-8')

    # define the production paramters for combinations
    iterables = [Ymin, Ymax, Nymax]

    # products of the production paramters , numpy array
    products = [np.array(p) for p in itertools.product(*iterables)]
    print('all combinations count ' + str(len(products)))

    # create dataframe of the production paramters with respective columns
    dfAllProducts = pd.DataFrame(products, columns=['Y0', 'Ymax', 'Nymax'])
    # print(dfAllProducts.head())
    print("dfAllProducts count" + str(dfAllProducts.count()))

    # delete rows where Y0>Ymax from the dataframe
    dfProducts = dfAllProducts.drop(dfAllProducts[dfAllProducts.Y0 > dfAllProducts.Ymax].index)
    print("Combinations count after ymin check " + str(dfProducts.count(axis=0)))

    # aggregate all input variables in a dataframe
    #dfInputs = pd.DataFrame({'cY': cY, 'cN': cN, 'Ymin': Ymin, 'Ymax': Ymax, 'NYmax': Nymax})
    # print(dfInputs.head())
    #dfInputs.to_csv('C:/Users/Student.Student11.000/Desktop/AgriDSS/Data/dfInputs.csv')

    print('===Calling errorToProbabilityCalculation () =====')
    dfProducts=sensitivitysimulation.errorToProbabilityCalculation(dfProducts,resultset)
    #dfProducts.to_csv('C:/Users/Student.Student11.000/Desktop/AgriDSS/Data/Sadanand/dfProducts+soilTexture+'.csv')
    dfProducts.to_csv(djangoSettings.STATIC_ROOT + '/datafiles/dfProducts.csv', encoding='utf-8')
    print('===Calling profitSensitivityAnalysis()===')
    # calling to script module for numeric simulation or sensitivity analysis
    df= sensitivitysimulation.profitSensitivityAnalysis(dfProducts, vectorcY, vectorcN)
    print(df.head())

    print("Total End Time Diffrenence")
    print(datetime.now() - startTime)

    print('plotting Bokeh ')
    pfModel = BokehPlotter.plotProfitFunction(df)
    #call to proft classes functuion
    profClasses= BokehPlotter.plotProfitClasses(df)

    #converting the dataframe results to html table format to show on result page
    dfhtml = df.to_html()

    del df
    gc.collect()

    print('rendering  result page  ')
    context = {'resultset': resultset,  'pfModel': pfModel,'profClasses':profClasses,'dfresult':dfhtml}
    return render(request, 'passapp/result.html', context)

def uploadcsv(request):

    if request.POST and request.FILES:

        print('Start,uploadcsv() ')
        startTime = datetime.now()
        dfInputData = pd.read_csv(request.FILES['csv_file'],sep='\t')
        print(dfInputData.head())

        pY0, pYmax, pNymax,pCN,pCY = [], [], [],[],[]
        Y0, Ymax, Nymax, CY,CN = [], [], [],[],[]

        for row in dfInputData.itertuples():

            if row.param == 'Y0':
                Y0_min = row.min
                Y0_max = row.max
                Y0_delta = row.delta

            elif row.param == 'Ymax':

                Ymax_min = row.min
                Ymax_max = row.max
                Ymax_delta = row.delta

            elif row.param == 'Nymax':

                Nymax_min = row.min
                Nymax_max = row.max
                Nymax_delta = row.delta

        MeanPrice = 175
        StdPrice = 20
        # trying new approach for Price and Cost Prob from PDF function
        normPrice = np.linspace(130, 220,7)  # BinsNDistributionCreator.normIntervals(MeanPrice, StdPrice, 7)  # [130,145,160,175,190,205,220]#
        print("=====price===== " + str(normPrice))
        cY = normPrice.tolist()
        vectorcY = BinsNDistributionCreator.findProbabilityNDensity(normPrice, MeanPrice, StdPrice)

        MeanCost = 1
        StdCost = 0.25
        cost = np.linspace(0.4, 1.6, 7)  # BinsNDistributionCreator.normIntervals(MeanCost, StdCost, 7)
        print("=====normCost===== " + str(cost))
        cN = cost.tolist()
        vectorcN = BinsNDistributionCreator.findProbabilityNDensity(cost, MeanCost, StdCost)

        #Initilizing production parameters according the configuration file specifications
        Ymin = np.arange(Y0_min, Y0_max + Y0_delta, Y0_delta)
        Ymax = np.arange(Ymax_min, Ymax_max + Ymax_delta, Ymax_delta)
        Nymax = np.arange(Nymax_min, Nymax_max+Nymax_delta, Nymax_delta)

        # define the production paramters for combinations
        iterables = [Ymin, Ymax, Nymax]

        # products of the production paramters , numpy array
        products = [np.array(p) for p in itertools.product(*iterables)]
        print('all combinations count ' + str(len(products)))

        # create dataframe of the production paramters with respective columns
        dfAllProducts = pd.DataFrame(products, columns=['Y0', 'Ymax', 'Nymax'])
        # print(dfAllProducts.head())
        print("dfAllProducts count" + str(dfAllProducts.count()))

        # delete rows where Y0>Ymax from the dataframe
        dfProducts = dfAllProducts.drop(dfAllProducts[dfAllProducts.Y0 > dfAllProducts.Ymax].index)
        print("Combinations count after ymin check " + str(dfProducts.count(axis=0)))


        # print('Price' + str(CY))
        # print('Cost' + str(CN))
        # print('Y0' + str(Y0))
        # print('Ymax' + str(Ymax))
        # print('Nymax' + str(Nymax))
        # print('pY0' + str(pY0))
        resultset = PDataset1.objects.filter(SoilTexture_cls='Fine')

        print('===Calling errorToProbabilityCalculation () =====')
        dfProducts = sensitivitysimulation.errorToProbabilityCalculation(dfProducts, resultset)
        # dfProducts.to_csv('C:/Users/Student.Student11.000/Desktop/AgriDSS/Data/Sadanand/dfProducts+soilTexture+'.csv')


        print('===Calling profitSensitivityAnalysis()===')
        # calling to script module for numeric simulation or sensitivity analysis
        df = sensitivitysimulation.profitSensitivityAnalysis(dfProducts, vectorcY, vectorcN)
        print(df.head())

        print("Total End Time Diffrenence")
        print(datetime.now() - startTime)

        print('plotting Bokeh ')
        pfModel = BokehPlotter.plotProfitFunction(df)
        # call to proft classes functuion
        profClasses = BokehPlotter.plotProfitClasses(df)

        # converting the dataframe results to html table format to show on result pag
        dfhtml = df.to_html()

        del df
        gc.collect()

        context = { 'pfModel': pfModel,'profClasses':profClasses,'dfresult':dfhtml}

    else:
        print('file not found')

    print('rendering  result page  ')
    return render(request, 'passapp/result.html',context)

