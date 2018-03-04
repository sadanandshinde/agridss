#from django.shortcuts import render
from django.shortcuts import render, redirect,get_object_or_404,render_to_response

from statistics import mean, stdev,variance

from ..models import UserRquestSite,UserTransaction,PlotYield,SiteField,dsslookup
#from django.db.models import Q

import csv
from django.http import HttpResponse,HttpResponseRedirect
from django.conf import settings as djangoSettings
#from django_pandas.io import read_frame
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required

from ..scripts import UserOperations, OnlineResourcesFinder
from django.contrib.auth import authenticate, login
from django.conf import settings
from django.contrib import messages
from django.db import transaction,IntegrityError

app_name = 'passapp'


def confirmUser(request,id):

    user=UserOperations.userActivation(id)
    if user is not None:
        print('rendering user account confirmation page')
        return render(request, 'registration/confirmation.html')
    else:
        messages.error(request, 'Could not find account with the user, please register here!')
        return render(request, 'registration/registration.html')

def registerUser(request):


    firstname= request.POST.get('firstname', '')
    lastname= request.POST.get('lastname', '')
    email= request.POST.get('email', '')
    password= request.POST.get('password', '')
    passwordconfirm =  request.POST.get('passwordconfirm', '')

    if not passwordconfirm == password:
        messages.error(request,'password does not match with confirm password, please try again ')
        return render(request, 'registration/registration.html')

    print('firstname : '+ firstname)
    print('lastname : '+ lastname)
    print('email : '+ email)

    try:
        user=UserOperations.findUserbyUsername(email)
        if user is not None:
            messages.error(request, "User Email already exist, please try with another email id")
            return render(request, 'registration/registration.html')
    except:
       pass

    user=UserOperations.createUser(firstname,email,lastname,password)

    print("usercreation successfull ")

    #send verification email to cofirm user.
    link= str(request.get_host()) +'/'+ str(user.id)+'/confirmUser'
    print('link: '+ link)
    UserOperations.sendVerificationEmail(user, link)
    # print('host: ' + str(request.get_host()))
    # print('ful path ' + request.get_full_path())
    # print(' raw uri ' + request.get_raw_uri())

    messages.info (request, "Your are successfully registered,please verify your email to login")

    return render(request, 'registration/login.html')

def signup(request):

    return render(request, 'registration/registration.html')

def signin(request):

    return render(request, 'registration/login.html')

def logoutUser(request):

    logout(request)
    try:
        del request.session['username']
        del request.session['userid']
    except KeyError:
        pass
    # Redirect to a success page.
    print('Successfully Logged out')
    messages.info (request, "Your are successfully logged out, login here again !")
    return render(request, 'registration/login.html')

''' verify user credentials '''
def authenticateUser(request):

    if request.method == 'POST':
        print("Start, general Views,  authenticateUser()")
        password = request.POST.get('password','')
        username=request.POST.get('username','')
        print('user: '+username)
        user = authenticate(username=username, password=password)
        if user is not None:
            print('authenticated user : '+username)

            if UserOperations.isUserActive(username) is False:
                messages.error(request, "User email verification pending!, please verify")
                return render(request, 'registration/login.html')

            if not request.POST.get('remember_me', None):
                request.session.set_expiry(0)

            login(request, user)

            request.session['username'] = user.first_name
            request.session['userid'] = user.id
            request.session['isstaff'] = user.is_staff
            request.session['useremail'] = user.email
            print("login is successfull, redirecting to home page")
            request.session.set_expiry(1200)
            return redirect('index')

        else:

            messages.error(request, "Authentication failed, Invalid User, please try again !")
            print('Authentication failed, Invalid User')

    return render(request, 'registration/login.html')

@login_required()
def index(request):
        print('Start , generalviews, index()')
        # if not request.user.is_authenticated:
        #     print('use os not authenticated, please login')
        #     return redirect((settings.LOGIN_URL, request.path))

        if request.user.is_active == False:
            messages.error(request, "Activatioin pending, check your email and click on link given to activate!")

        ' commented  this, added the voptions manually in the home.html file as plotyield table containing the data is belong to dssservice module'
        # print('fetching basic input values from database')
        # latest_dataset_list = PDataset1.objects.all()
        # #tillTypes=latest_dataset_list.order_by('TillType').distinct('TillType')
        # tillTypes = PDataset1.objects.order_by('TillType').values('TillType').distinct() #PDataset1.objects.order_by('TillType').distinct('TillType')
        # # soilTypes = [] #PDataset1.objects.order_by('SoilType').distinct('SoilType') works only with PostgreSQL
        # soilTypes=PDataset1.objects.order_by('SoilType').values('SoilType').distinct() #MySQL + PostgreSQL
        # prevCrops = PDataset1.objects.order_by('PrevCrop').values('PrevCrop').distinct()#PDataset1.objects.order_by('PrevCrop').distinct('PrevCrop')

        print('calling online resoruce finder for price and cost')
        cornPrice=OnlineResourcesFinder.getCornPrice()
        print('corn price fetched : '+ str(cornPrice))

        Ncost= OnlineResourcesFinder.getNitrogenCost()
        print('N cost fetched :'+ str(Ncost))

        context = {'Ncost':Ncost,'cornPrice': cornPrice}

        print('End , generalviews, index()')

        return  render(request,'passapp/home.html',context)


@transaction.atomic
@login_required()
def saveUserRequest(request):

    print('start, general view, saveUserRequest() ')
    if request.user.is_active == False:
        return  HttpResponseRedirect('/')

    fertilizer=request.POST.get('fertilizer', '')
    currentcrop=request.POST.get('currentcrop', '')
    season=request.POST.get('season', '')
    soilType = request.POST.get('soilType', '')
    tillType = request.POST.get('tillType', '')
    SOM = request.POST.get('som', '')
    latitude = request.POST.get('latitude', '')
    longitude = request.POST.get('longitude', '')
    climate = request.POST.get('climate', '')
    prevCrop = request.POST.get('prevCrop', '')
    CHU= request.POST.get('chu', '')
    print((request.POST.get('meanprice', '').strip()))
    meanprice = float(request.POST.get('meanprice', '').strip())
    stdprice = float(request.POST.get('stdprice', '').strip())
    meancost = float(request.POST.get('meancost', ''))
    stdcost = float(request.POST.get('stdcost', ''))

    print("=============user specified paramters: =============")

    print("currentcrop :" + str(currentcrop))
    print("season :" + str(season))
    print("soilType :" + str(soilType))
    print("tillType :" + str(tillType))
    print("latitude :" + str(latitude))
    print("longitude :" + str(longitude))
    print("climate :" + str(climate))
    print("prevCrop :" + str(prevCrop))
    print("meanprice :" + str(meanprice))
    print("stdprice :" + str(stdprice))
    print("meancost :" + str(meancost))
    print("stdcost :" + str(stdcost))
    print("SOM :" , SOM)
    print("CHU :", CHU)

    awdr= None
    chu=None
    som=None
    #split the categorical variables to take a concrete values, if other is psecified then take from other
    # textbox for respective attribute
    if climate=='other':
        awdr=float(request.POST['awdrother'])
    else:

        lbound, ubound = climate.split('-')
        awdr = float((int(lbound) + int(ubound)) / 2)
    if CHU=='other':
        chu= float(request.POST['chuother'])
    else:

        lbound, ubound= CHU.split('-')
        chu=float((int(lbound) + int(ubound)) / 2);

    if SOM=='other':
        som = float(request.POST['somother'])
    else:
        lbound, ubound = SOM.split('-')
        som = float((int(lbound) + int(ubound)) / 2);

    print('concreate values ' )

    print("chu :", chu)
    print("awdr :", awdr)
    print("som :", som)

    #user = User.objects.get(username=request.session['username'])
    #Store user request in user trans request
    ur = UserRquestSite(user= request.user,fertilizer=fertilizer,current_crop=currentcrop,season=season,soiltype=soilType,tilltype=tillType,
                        latitude=latitude,longitude=longitude,awdr=awdr,prev_crop=prevCrop,CHU=chu,SOM=som,price_mean=meanprice,price_std=stdprice,costmean=meancost,coststd=stdcost)
    ur.save()
    print('user site request has been saved successfully, id', ur.id)

    #create entry in user transaction model, to proces request
    utrans=UserTransaction(usersite=ur,user=request.user)
    utrans.save()
    print('user trnsaction request has been saved successfully, id',utrans.id)
    messages.info(request,'Your request will be processed as soon as possible.You should receive an e-mail message acknowledging your request and another message with results.'\
                'Due to the heavy computation, this might take several minutes.')

    # send email of notificaton
    UserOperations.sendEmailNotification( request.user.username)

    print('End, general view, saveUserRequest() ')

    # redirect to notify about  processing started for request.
    return render(request,'passapp/notification.html')

def aboutusview(request):

    return render(request, 'passapp/aboutus.html')

def descriptionview (request):

    return render(request, 'passapp/description.html')




def downlaodCSV(request):
    # Create the HttpResponse object with the appropriate CSV header.
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="dbTrialsNumericAg.csv"'

    writer = csv.writer(response)
    trailList = PlotYield.objects.all()
    with open(djangoSettings.STATIC_ROOT+'/datafiles/dbTrialsNumericAg.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        for row in spamreader:
            #print(', '.join(row))
            writer.writerow(row)

    return response

@login_required()
def addtrails(request):
    print('in addtrails view')
    return render(request, 'passapp/addtrails.html')


@login_required()
def submittrails(request):

    print('in submittrails view')
    if request.method == 'POST':
        if request.is_ajax():
            import json
            # user filled values, hold in variable
            fertilizer = request.POST['fertilizer']
            currentcrop = request.POST['currentcrop']
            season = int(request.POST['season'])
            soilType = request.POST['soilType']
            tillType = request.POST['tillType']
            SOM = request.POST['som']
            latitude = request.POST['latitude']
            longitude = request.POST['longitude']
            climate = request.POST['climate']
            prevCrop = request.POST['prevCrop']
            CHU= (request.POST['chu'])
            Nrate = int(request.POST['nrate'])
            yld = float(request.POST['yield'])
            sitename = request.POST['sitename']

            latitude = None if latitude == '' else float(latitude)
            longitude = None if longitude == '' else float(longitude)
            #som = None if som == '' else float(som)
            sitename = None if sitename == '' else sitename

            # province=request.POST.get('province', '')
            # city=request.POST.get('city', '')
            # region=request.POST.get('region', '')
            # fieldnumber=int(request.POST.get('fieldnumber', ''))


            print('user submitted values are :')

            print("currentcrop :" + str(currentcrop))
            print("season :", season)
            print("soilType :" + str(soilType))
            print("tillType :" + str(tillType))
            print("som :", SOM)
            print("chu :", CHU)
            print("latitude :", latitude)
            print("longitude :", longitude)
            print("sitename :", sitename)
            print("climate :", climate)
            print("prevCrop :" + str(prevCrop))
            print("Nrate :", Nrate)
            print("yield :", yld)

            # check for outliers/rules
            if Nrate < 0 or Nrate > 350:
                # messages.error(request, "Fertilizer rate should not be negative and maximum up to 350 kg/ha, please verify")
                message = 'Error!, the fertilizer rate should not be negative and maximum up to 350 kg/ha. Please correct the input'
                return HttpResponse(message)

            if yld < 0 or yld > 30:
                # messages.error(request, "Yield should not be negative and maximum up to 30 t/ha, please verify")
                message = 'Error!, the yield value should not be negative and maximum up to 30 t/ha. Please correct the input'
                return HttpResponse(message)

            # source =useremail
            source = request.session['useremail']

            awdr, chu, som = None,None,None
            if climate == 'other':
                awdr = float(request.POST['awdrother'])
            else:

                lbound, ubound = climate.split('-')
                awdr = float((int(lbound) + int(ubound)) / 2)

            if CHU == 'other':
                chu = float(request.POST['chuother'])
            else:

                lbound, ubound = CHU.split('-')
                chu = float((int(lbound) + int(ubound)) / 2);

            if SOM == 'other':
                som = float(request.POST['somother'])
            else:
                lbound, ubound = SOM.split('-')
                som = float((int(lbound) + int(ubound)) / 2);

            print('descrete values ')
            print("chu :", chu)
            print("awdr :", awdr)
            print("som :", som)

            tillTypelookup = int(dsslookup.objects.filter(fieldname='till_type', key=tillType)[0].value)
            clayratiolookup = float(dsslookup.objects.filter(fieldname='soil_type_clay_ratio', key=soilType)[0].value)
            prevCropNcontribLookup = int(
                dsslookup.objects.filter(fieldname='prev_crop_N_contrib', key=prevCrop)[0].value)

            print("source ", source)

            print("tillTypelookup ", tillTypelookup)
            print("clayratiolookup ", clayratiolookup)
            print("prevCropNcontribLookup ", prevCropNcontribLookup)

            # if admin privilage, verified = Y, for user by default verified is = N
            # if request.session['isstaff']:
            #     verified = 'Y'
            # else:
            #     verified = 'N'
            #default verified as N
            verified = 'N'
            # check here if user has existing trail with same parameters, then it should be considered as duplicate and prevent from adding it.
            trailList = PlotYield.objects.filter(Latitude=latitude, Longitude=longitude,Year=season,
                                                 SoilType=soilType, ClayRatio=clayratiolookup, TillType=tillType,
                                                 SOM=som,
                                                 TillType_int=tillTypelookup, PrevCrop=prevCrop, CHU=chu,
                                                 PrevContribN_int=prevCropNcontribLookup,
                                                 AWDR=awdr, Nrate=Nrate, Yield=yld, Source=source)

            if (trailList.count() > 0):

                if sitename is not None and sitename is not '':

                    siteList = SiteField.objects.filter(Site_Field_Name=sitename)
                    if siteList.count() > 0:

                        filterTrail = trailList.filter(SiteFieldId=siteList.last())
                        if filterTrail.count() > 0:
                            message = 'Error!, a trail already exist with the inputed details under your account. Please change any of the input field to treat as a new trial and submit again'
                            return HttpResponse(message)
                else:

                    message = 'Error!, a trail already exist with the inputed details under your account. Please change any of the input field to treat as a new trial and submit again'
                    return HttpResponse(message)
                # if latitude is not None and longitude is not None:
                #
                #     filterTrail = trailList.filter(Latitude=latitude, Longitude=longitude)
                #     if filterTrail.count() > 0:
                #
                #         message = 'Error!, a trail already exist with the inputed details under your account. Please change any of the input field to treat as a new trial and submit again'
                #         return HttpResponse(message)



            print("verified ", verified)

            sitefield = SiteField(Site_Field_Name=sitename)
            sitefield.save()
            print('SiteField is successfully inserted id ', sitefield.id)
            try:

                print('inserting plot/trial now ')
                plot = PlotYield(SiteFieldId=sitefield, Latitude=latitude, Longitude=longitude, Year=season,
                                 SoilType=soilType, ClayRatio=clayratiolookup, TillType=tillType, SOM=som,
                                 TillType_int=tillTypelookup, PrevCrop=prevCrop,CHU=chu,
                                 PrevContribN_int=prevCropNcontribLookup,
                                 AWDR=awdr, Nrate=Nrate, Yield=yld, Source=source, Verified=verified)
                plot.save()
                print("PlotYield is successfully inserted id", plot.id)

                #messages.success(request,'your farm trail has been successfully submitted, your efforts are appreciated, Thank you!')
                message =  'The data have been successfully submitted. Thank you for your submission'
                return HttpResponse(message)
            except Exception as ex:
                print('Could not insert user trail into database, ', ex)
                messages.error(request, 'oops!, there was some exception, please try again ...')
                transaction.rollback()
                # raise Exception ('oops!, there was some exception, please try again ...')

        return render(request, 'passapp/addtrails.html')


