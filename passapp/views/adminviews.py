from django.contrib import messages
from django.shortcuts import render, redirect
import csv
from django.http import HttpResponse,HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from ..models import PlotYield,UserTransaction,UserRquestSite
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from statistics import mean, stdev
from django_pandas.io import read_frame
from ..scripts import DataUtility
import pandas as pd
@login_required()
def loadadminview(request):
    print('Start loadAdminview')
    if request.session['isstaff']:
        plot_list = PlotYield.objects.filter(Verified='N')
        trail_count = PlotYield.objects.count()
        unverfied_count = plot_list.count()
        active_users_count = User.objects.filter(is_active=True).count()
        total_request_count = UserRquestSite.objects.count()
        # processtime=UserTransaction.objects.only("request_process_time")[:30]
        # print('process time list ',processtime[0].id)
        # print('mean processtime', mean(processtime['UserTransaction'))
        processTime = []
        processtimeQuery = UserTransaction.objects.values_list('request_process_time', flat=True).order_by('-id')[:30]

        for pt in processtimeQuery:
            if pt is not None:
                processTime.append(pt)

        meanProcessTime = int(mean(processTime))
        print(' mean process time  ', meanProcessTime)
        context = {'plot_list': plot_list, 'trail_count': trail_count, 'active_users_count': active_users_count
            , 'unverfied_count': unverfied_count, 'total_request_count': total_request_count,
                   'mean_processing_time': meanProcessTime}

        return render(request, 'admin/admin.html', context)
    else:
        return HttpResponse('you do not have permissions as an admin')

@login_required()
def verifyTrail(request, id):

    print('start, verifyTrail ')
    try:
        obj = PlotYield.objects.get(pk=id)
        obj.Verified = "Y"
        obj.save()
        print('verifyTrail successfully trail id ', id)

        messages.info(request, " successfully verified the trail  id : " + str(id))
    except:
        messages.error("there is some error! could not verify the trail")


    print('End, verifyTrail ')
    return loadadminview(request)

@login_required()
def verifySelected(request):

    print('start verifySelected')
    if request.method == 'POST':
        id_list = request.POST.getlist('instance')
        # This will submit an array of the value attributes of all the
        # checkboxes that have been checked, that is an array of {{obj.id}}

        # Now all that is left is to iterate over the array fetch the
        # object with the ID and delete it.
        trailIds = " "
        for plot_id in id_list:
            #plot = PlotYield.objects.get(id=plot_id)
            obj=PlotYield.objects.get(pk=plot_id)
            obj.Verified = "Y"
            obj.save()

            print('plot id  verified', plot_id)
            trailIds += ', '+str(plot_id)

        messages.info(request, " successfully verified the selected trail ids "+ trailIds)
        print('End, verifySelected ')
        return loadadminview(request)
    else:
        return HttpResponse('Get method is not supported')

@login_required()
def deleteSelected(request):

    if request.method == 'POST':

        print('start deleteSelected')
        id_list = request.POST.getlist('instance')
        # This will submit an array of the value attributes of all the
        # checkboxes that have been checked, that is an array of {{obj.id}}

        # Now all that is left is to iterate over the array fetch the
        # object with the ID and delete it.
        trailIds = " "
        for plot_id in id_list:
            PlotYield.objects.get(id=plot_id).delete()
            trailIds += ','+ str(plot_id)

        messages.info(request, "successfully deleted the selected trail ids  "+ trailIds)
        print('successfully deleted the selected trail ids'+ trailIds)
        print('End deleteSelected')
        return loadadminview(request)
    else:
        return HttpResponse('Get method is not supported')

@login_required()
def reportview(request):
    print('start reportView')
    usersList =None
    user=None


    if request.method == 'GET':
        if request.session['isstaff'] == True:
            usersList = User.objects.all()
            print('user count', usersList.count())
        return  render(request,'admin/reports.html',{'userList':usersList})

    if request.session['isstaff'] == True:
        user = request.POST['user']
        usersList = User.objects.all()
        print('user count', usersList.count())
    else:
        user = request.session['userid']

    reqType=request.POST['reqType']

    print('request type ', reqType)
    print('user id ', user)

    reqList=None
    trailList=None
    transList=None
    if user=='None':

        if reqType=='viewRequest':
            reqList= UserRquestSite.objects.all()
            print('return all User Request')
        elif reqType=='viewTrails':
            trailList=PlotYield.objects.all()
            print('return all Trails')

        elif reqType == 'viewTrans':
            print('return all executed Transaction ')
            transList=UserTransaction.objects.all()
    else:

        if reqType == 'viewRequest':
            print('return  User Request for user id ', user)
            userObj = User.objects.get(id=user)
            reqList= UserRquestSite.objects.filter(user=userObj)
            print('reqList count' , reqList.count())

        elif reqType == 'viewTrails':
            print('return Trails only entered by user id ', user)
            userObj = User.objects.get(id=user)
            trailList=PlotYield.objects.filter(Source=userObj.username)
            print('trailList count', trailList.count())

        elif reqType == 'viewTrans':
            userObj = User.objects.get(id=user)
            print('return all executed Transaction for user', userObj.id)
            transList = UserTransaction.objects.filter(user=userObj)

    return render(request, 'admin/reports.html', {'userList': usersList,'reqList':reqList,'trailList':trailList,'transList':transList})

@login_required()
def imputeMissingValues(request):

    dbSites = PlotYield.objects.all()
    if dbSites.count() > 1:
        dfDataset = read_frame(dbSites)
        #'convert None or null values to  np NaN '
        from numpy import nan
        dfDataset.fillna(value=nan, inplace=True)
        print(dfDataset.head())
        dfDataset['AWDR'] = DataUtility.imputeContFun(dfDataset['AWDR'].values)
        dfDataset['ClayRatio'] = DataUtility.imputeContFun(dfDataset['ClayRatio'].values)
        dfDataset['PrevContribN_int'] = DataUtility.imputeContFun(dfDataset['PrevContribN_int'].values)
        dfDataset['SOM'] = DataUtility.imputeContFun(dfDataset['SOM'].values)
        dfDataset['TillType_int'] = DataUtility.imputeBinomFun(dfDataset['TillType_int'].values)
        dfDataset['CHU'] = DataUtility.imputeContFun(dfDataset['CHU'].values)
        print('Data has been imputed')

        print(dfDataset.head())
        for row in dfDataset.itertuples():
            plotObj = PlotYield.objects.get(id=row.id)
            plotObj.AWDR = row.AWDR
            plotObj.ClayRatio = row.ClayRatio
            plotObj.SOM = row.SOM
            plotObj.PrevContribN_int = row.PrevContribN_int
            plotObj.TillType_int = row.TillType_int
            plotObj.CHU=row.CHU
            plotObj.save()

        print('All plots/trials are successfully updated')
        # dfDataset.to_csv('C:/Users/Student.Student11.000/Desktop/ImputedDataset.csv')
        print(dfDataset.head())
        return HttpResponse('Data imputation is completed ')

    return HttpResponse('No missing record present in the database ')



def downloadAllDBTrails(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="NumericAgDBTrials.csv"'

    writer = csv.writer(response)
    writer.writerow(['id', 'SiteFieldId', 'Latitude','Longitude','SoilType',
                                                   'ClayRatio','SOM','TillType','PrevCrop','CHU','PPT','AWDR','Nrate','Yield','Source','Verified'])
    #trailList = PlotYield.objects.all()
    writer.writerows(PlotYield.objects.values_list('id', 'SiteFieldId', 'Latitude','Longitude','SoilType',
                                                   'ClayRatio','SOM','TillType','PrevCrop','CHU','PPT','AWDR','Nrate','Yield','Source','Verified'))

    return response

def uploadTrialsInDB(request):
    # file = request.FILES['csv_file','rb']
    # data = [row for row in csv.reader(file.read().splitlines())]

    if request.POST and request.FILES:
        uploadedfile = request.FILES['csv_file']

        #dfInputData = pd.read_csv(uploadedfile.read())
        data=[]
        for line in uploadedfile:
            data.append(line)

        labels=['id', 'SiteFieldId', 'Latitude','Longitude','SoilType',

                                                   'ClayRatio','SOM','TillType','PrevCrop','CHU','PPT','AWDR','Nrate','Yield','Source','Verified']
        print(data[0])
        dfdata =pd.DataFrame.from_records(data)
        #dfdata.to_csv('C:/Users/Student.Student11.000/Desktop/dfdata.csv')
        print(dfdata.head(5))



        messages.info(request,
                      'This functionality is under development, please try later ')
        return render(request, 'passapp/notification.html')
