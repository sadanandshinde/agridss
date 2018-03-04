
import quandl
import math
from statistics import *
quandl.ApiConfig.api_key = 'pBBajE28yrkDVEqBwVHG'

from dateutil.relativedelta import relativedelta
import datetime

years_ago = datetime.datetime.now() - relativedelta(years=5)

print("five years ago" + str(years_ago.strftime('%Y-%m-%d')))
'''
CORN & SORGHUM (56 lb/bu)
Corn : 352 cents per bushel
1 bushel = .0254 metric ton
1 metric ton = 39.368 bushels

WHEAT & SOYBEANS (60 lb/bu)
Soybean: 996 cents per bushel

1 bushel = .0272155 metric ton
1 metric ton = 36.7437 bushels
'''

print("today "+str (datetime.datetime.now().strftime('%Y-%m-%d')))
'''
return  mean and std of Soya price in USD $/t
'''
def getSoybeanPrice():
   try:
       SoyaCMEdata = quandl.get("CHRIS/CME_S1", start_date=years_ago.strftime('%Y-%m-%d'),
                                end_date=datetime.datetime.now().strftime('%Y-%m-%d'))
       # print(SoyaCMEdata.head)
       meanSoyaPrice = mean(SoyaCMEdata['Settle'])
       stdPrice = 36.7437 * stdev(SoyaCMEdata['Settle']) / 100
       print("==============Soybean price Mean===========" + str(meanSoyaPrice))
       # 1 metric ton = 36.7437 bushels
       meanPrice = (36.7437 * meanSoyaPrice) / 100
       # 1 bushel corn(56# ) = 25.40 (25) kilograms
       # 1 bushel wheat / soybeans(60  # ) = 27.22 (27) kilograms
       # 1 kilogram(kg) = 2.205(2.2) pounds
       # 1 pound = .4536(.45) kilograms
       price = {'mean': meanPrice, 'std': stdPrice}

       return price
   except:

       return None

'''
return  mean and std of Corn price in USD $/t
'''
def getCornPrice():
    try:
        CornData = quandl.get("CHRIS/CME_C1", start_date=years_ago.strftime('%Y-%m-%d'),
                              end_date=datetime.datetime.now().strftime('%Y-%m-%d'))
        # print(CornData.head())
        # 1 bushel wheat / soybeans(60  # ) = 27.22 (27) kilograms
        meanCornPrice = mean(CornData['Settle'])
        stdPrice = 39.368 * stdev(CornData['Settle']) / 100
        print('SD price ' + str(stdPrice))
        print("==============Corn price Mean cents per bushel===========" + str(meanCornPrice))

        meanPrice = (39.368 * meanCornPrice) / 100
        price = {'mean': round(meanPrice, 2), 'std': round(stdPrice, 2)}
        return price
    except:
        print("Connection Error")


        return None


'''
return  mean and std of pure Nitrogen(nutrient) cost in USD $/kg
'''
def getNitrogenCost():
    try:
        #UreaEurop = quandl.get("WORLDBANK/WLD_UREA_EE_BULK", start_date=years_ago.strftime('%Y-%m-%d'),
         #                      end_date=datetime.datetime.now().strftime('%Y-%m-%d'))
        #changed the API call on 25 July 2017 as the previous worlbankd API was not working
        UreaEurop = quandl.get("COM/WLD_UREA_EE_BULK", start_date=years_ago.strftime('%Y-%m-%d'),
                            end_date=datetime.datetime.now().strftime('%Y-%m-%d'))
        # print(UreaEurop.head)
        meanUreaPrice = mean(UreaEurop['Column 1'])
        # convert from metric ton to kg (devide by 1000) and then urea nutirent price to pure nitrogen price divide by 0.46

        stdUrea = stdev((UreaEurop['Column 1']))
        stdNcost = (stdUrea / 0.46) / 1000

        # print("std urea price " + str(stdUrea))
        meanNCost = ((meanUreaPrice) / 1000) / .46

        print('stdNCost  ' + str(stdNcost))
        print(' meanNCost  ' + str(meanNCost))

        cost = {'mean': round(meanNCost, 2), 'std': round(stdNcost, 2)}
        # print(cost['mean'])


    except Exception as ex:
        print("Connection Error", ex)
        #default values if connection error
        cost = {'mean': 0.70, 'std': 0.15}
        return cost

    return cost
# print("calling getSoybeanPrice")
# print("getSoybeanPrice " + str(getSoybeanPrice()))
# getNitrogenCost()
# print("priceMetricTon " + str(getCornPrice()))