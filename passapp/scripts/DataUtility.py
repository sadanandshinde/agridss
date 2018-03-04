
import pandas as pd
import numpy as np
#access n culumn
#n[:,1] -- first column
#n[1:,] -- first row
def imputeBinomFun(x):
    print('Start imputeBinomFun')
    print(x)
    x1 =  x[np.where( x == 1 )]
    countall =x.size - np.count_nonzero(np.isnan(x))
    count1=x1.size
    print('count1 ', count1)
    print('countall', countall)
    xprob= count1/countall
    print('xprob ',xprob)
    tempImp = [np.random.binomial(1, xprob, size=None) if np.isnan(xval) else xval for xval in x]
    #print('bionomial Imputation ', tempImp)
    print('End imputeBinomFun')
    return  tempImp

def imputeContFun(x):

    print('Start imputeContFun')
    xmean= np.nanmean(x)#x.mean(skipna=True)
    xstd=np.nanstd (x)#x.std(skipna=True)
    print('mean ' , xmean)
    print('std ', xstd)
    #norm.rvs(loc=0, scale=1, size=1, random_state=None)
    #tempImp=[]

    tempImp=[np.random.normal(xmean, xstd, size=None) if np.isnan(xval) else xval for xval in x]

    tempImp =[ 0 if val < 0 else val for val in tempImp]
    print('End imputeContFun')
    #print('continious Imputation ',tempImp)
    return  tempImp

# x=np.array([1,2,np.nan,8,np.nan,4,5])
#
# b= np.array ([0,1,1,1,0,1,np.nan])
# print('Calling imputation function ')
# imputeContFun(x)
#
# imputeBinomFun(b)