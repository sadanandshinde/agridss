import numpy as np
import pandas as pd
from datetime import datetime
import  gc
from scipy.stats import norm
import os, psutil
from statistics import mean, stdev,variance
from django.conf import settings as djangoSettings
#from functools import lru_cache as cache

# def usage():
#     process = psutil.Process(os.getpid())
#     return process.memory_full_info()

'''
This function takes input of independent variables array/combiations with respective associated joint probability, calculate
or perform sensitivity of production paramters on profit function at each N rate and return expected profit, profit classes probability and NRCNF in a dataframe.
'''
def profitSensitivityAnalysis(dfProducts, vectorcY, vectorcN):

    startTime = datetime.now()

    '''
       vectorPrice, vectorCost = [],[]
       for i in range(len(pCY)):
           vectorPrice.append([cY[i], pCY[i]])

       for i in range(len(pCN)):
           vectorCost.append([cN[i], pCY[i]])
       '''
    JProb = []  # joint prob is combination of i *j *k *l *m 5 indepenet varaible prob

    Nrate, allProfit, expProfit, JProbSum, ExpProfSum = [], [], [], [], []

    Nstep = 10  # this will be later taken form database as a constant input
    vectProfClass = []
    print(" Time Diffrenence before main iteration ")
    print(datetime.now() - startTime)

    dfProfitCls = pd.DataFrame(columns=np.arange(100, 2600, 100), index=np.arange(0, 200 + Nstep, Nstep))
    # print(dfProfitCls.head())

    # strating sensitivity analysis/ uncertainty analyasis for each of the possible values of the input variables
    # on the predicted production and/or profit functions
    for N in range(0, 200 + Nstep, Nstep):
        # iterate N rate from 0 to 200 by Interval of Nstep
        for cost in vectorcN:

            for price in vectorcY:

                for prd in dfProducts.itertuples():
                    # ECPA 2017 paper, Adamchuk Nitrogen paper, derived co-efficents for QP model
                    a0 = prd.Y0
                    a1 = 2 * (prd.Ymax - prd.Y0) / prd.Nymax
                    a2 = (prd.Y0 - prd.Ymax) / (prd.Nymax ** 2)

                    # QP model taken from ECPA 2017 paper, yld=Y t per ha
                    yld = a0 + (a1 * N) + (a2 * N ** 2) if N <= prd.Nymax else prd.Ymax

                    # profit for each of the input combincations yld price over nitrogen fertilization cost
                    profit = (yld * price[0]) - (N * cost[0])  # Net retun over cost of fertilization

                    allProfit.append(profit)
                    # joint prob is combination of i *j *k *l *m  n=bins of the 5 indepenet varaible
                    jointProb = (price[1] * cost[1] * prd.Jprob)
                    JProb.append(jointProb)

                    # NRCNF Net return over cost of fertilization
                    expProfit.append(profit * jointProb)  # use JProb as joint Probability for all the 5 variables

                    Nrate.append(N)
                    # End paroduction params loop
                    # End Price loop
        # End Cost loop


        JProbSum.append(sum(JProb))
        ExpProfSum.append(sum(expProfit))
        print("Sum of EXp Profit(NRCNF) " + str(sum(expProfit)))

        # profit classes of ranging 100 to 2500 at each N rate
        dfAllValues = pd.DataFrame()
        dfAllValues['JProb'] = pd.Series(JProb)
        dfAllValues['expProfit'] = pd.Series(expProfit)
        dfAllValues['Profit'] = pd.Series(allProfit)
        dfAllValues['Nrate'] = pd.Series(Nrate)

        # slicing probability of acheving 100 to 2500 profits classes for given N rate
        for i in range(100, 2600, 100):
            temp = list(dfAllValues['JProb'][dfAllValues['Profit'] >= i][dfAllValues['Nrate'] == N])
            dfProfitCls.at[N, i] = sum(temp)
            temp.clear()

        allProfit.clear()
        Nrate.clear()
        expProfit.clear()
        JProb.clear()

        del dfAllValues
        gc.collect()

        print("End Time Diffrenence for N App rate " + str(N))
        print(datetime.now() - startTime)
        # End of N rate main loop

    # this arrray cointains returns or additon to profit function over Ymin result
    Nreturns = [ExpProfSum[i] - ExpProfSum[0] for i in range(0, len(ExpProfSum))]  # format(x, '.6')

    # dfProfitCls.to_csv('djangoSettings.STATIC_ROOT + '/datafiles/dfProfitCls.csv', encoding='utf-8')
    dfProfitCls.to_csv(djangoSettings.STATIC_ROOT + '/datafiles/dfProfitCls.csv', encoding='utf-8')

    Napp = np.arange(0, 200 + Nstep, Nstep)

    # Dataframe with profit levels threshhold and their respective probabiliteis
    df = pd.DataFrame(index=Napp)

    df['N Rate'] = pd.Series(Napp, index=dfProfitCls.index)
    df.loc[:, 'prob_1000'] = pd.Series(dfProfitCls.loc[:, 1000], index=dfProfitCls.index)
    df.loc[:, 'prob_1500'] = pd.Series(dfProfitCls.loc[:, 1500], index=dfProfitCls.index)  # pd.Series(Nprob1500)
    df['prob_2000'] = pd.Series(dfProfitCls[2000], index=dfProfitCls.index)  # pd.Series(Nprob2000)
    df.loc[:, 'prob_2500'] = pd.Series(dfProfitCls[2500], index=dfProfitCls.index)
    df.loc[:, 'EProf'] = pd.Series(ExpProfSum, index=dfProfitCls.index)
    # df.loc[:, 'JProb'] = pd.Series(JProbSum)
    df.loc[:, 'NRCNF'] = pd.Series(Nreturns, index=dfProfitCls.index)
    # df.dropna()
    #  exporting the results from a a dataframe to csv format
    df.to_csv(djangoSettings.STATIC_ROOT + '/datafiles/dfProductonProfit.csv')

    del dfProfitCls
    gc.collect()

    return df

'''
this function finds the errors/difference between estimated yield by QP model and observed yeild in fertility trails
and find the density of avg with respect to 0 in normal density pdf function and normalize the density to probability
'''
def errorToProbabilityCalculation(dfProducts,resultset):

    startTime= datetime.now()
    print('start: errorToProbabilityCalculation: ')

    # dfFtrails = pd.read_csv('C:/Users/Student.Student11.000/Desktop/AgriDSS/Data/Sadanand/mDataCSV_Fine.csv',
    #                         names=['SoilTexture_cls', 'Yld0', 'Yld50', 'Yld100', 'Yld150', 'Yld200'], skiprows=1)
    # # print(dfFtrails.head)

    dfFPlots = pd.DataFrame(columns=['N', 'Y', 'L'])
    i = 0

    for rs in resultset:
        # dfFPlots.set_value(4, 'A', 10)  # row, column, value
        # This will make an index labeled `2` and add the new values
        # print(plt)
        i += 1
        dfFPlots.loc[i] = [0, rs.Yld0, 1]
        i += 1
        dfFPlots.loc[i] = [50, rs.Yld50, 1]
        i += 1
        dfFPlots.loc[i] = [100, rs.Yld100, 1]
        i += 1
        dfFPlots.loc[i] = [150, rs.Yld150, 1]
        i += 1
        dfFPlots.loc[i] = [200, rs.Yld200, 1]

    print(" ==== plots =======" + str(len(dfFPlots)))
    print(dfFPlots.head)

    yld, Eerror, normDist = [], [], []

    i = 0
    for row in dfProducts.itertuples():

        i += 1
        # ECPA 2017 paper, Adamchuk Nitrogen paper, derived co-efficents for QP model

        b = 2 * (row.Ymax - row.Y0) / row.Nymax
        c = (row.Y0 - row.Ymax) / (row.Nymax ** 2)

        for ft in dfFPlots.itertuples():
            # QP model taken from ECPA 2017 paper, yld=Y t per ha

            N = ft.N
            yld = (row.Y0 + (b * N) + (c * N ** 2) if N <= row.Nymax else row.Ymax)
            Eerror.append((yld - ft.Y) * ft.L)


            # End ftrails loop

        avg = (mean(Eerror))
        sd = stdev(Eerror)
        normDist.append(norm.pdf(0, avg, sd))
        if i <= 1:
            print('avg ' + str(avg))
            print('sd ' + str(sd))
            print('normDist ' + str(normDist))
            print('yld ' + str(yld))
            print('Eerror ' + str(Eerror))
        Eerror.clear()

    print('len n dist' + str(len(normDist)))
    print('total count ,i ' + str(i))

    sumNDist = sum(normDist)
    Jprob = normDist / sumNDist
    # prob= list(map(lambda  x: x/sumNDist,normDist))
    print('prob len ' + str(len(Jprob)))
    print('prob  sum ' + str(sum(Jprob)))
    dfProducts.loc[:, 'Jprob'] = Jprob
    print('end: errorToProbabilityCalculation, end time :' + str(datetime.now() - startTime))

    del dfFPlots
    del resultset
    normDist.clear()

    gc.collect()

    return dfProducts