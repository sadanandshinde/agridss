from django.db import models
from django.utils.encoding import python_2_unicode_compatible
from django.contrib.auth.models import User
# Create your models here.

@python_2_unicode_compatible
class PDataset1(models.Model):
    BDoriginale = models.CharField(max_length=100)
    Champ = models.CharField(max_length=100)
    #REcorded_Year = models.DateTimeField(null=True)
    AWDR = models.FloatField(null=True)

    TillPractice = models.CharField(max_length=100)
    TillType = models.CharField(max_length=100)
    TillTypeChoices = (
        (0, 'No till'),
        (1, 'Conventional'),
    )
    TillType_int = models.IntegerField(null=True, choices=TillTypeChoices)
    PrevCrop = models.CharField(max_length=100)
    PrevContribN = models.IntegerField(null=True)
    PrevContribNChoices = (
        ('Weak', '0 - 25 kg/ha'),
        ('Medium', '25 - 50 kg/ha'),
        ('Strong', '50 -100 kg/ha'),
    )
    PrevContribN_cls = models.CharField(null=True, choices=PrevContribNChoices, max_length=50)
    SoilType = models.CharField(max_length=50)
    SoilTexture_cls = models.CharField(max_length=50)
    Yld0    = models.FloatField(null=True)
    Yld50   = models.FloatField(null=True)
    Yld100  = models.FloatField(null=True)
    Yld150  = models.FloatField(null=True)
    Yld200  = models.FloatField(null=True)
    Ymodel0 = models.FloatField(null=True)
    Ymax = models.FloatField(null=True)
    Nymax = models.FloatField(null=True)
    a = models.FloatField(null=True)
    b = models.FloatField(null=True)
    c = models.FloatField(null=True)
    #pub_date = models.DateTimeField('date published')
    class Meta:
          db_table = 'PaasDataset1'



class UserRquestSite(models.Model):
    user=   models.ForeignKey(User)
    fertilizer= models.CharField(max_length=100)
    current_crop= models.CharField(max_length=100)
    season =    models.IntegerField()
    soiltype=   models.CharField(max_length=100)
    tilltype=   models.CharField(max_length=100)
    latitude=   models.CharField(max_length=100)
    longitude=  models.CharField(max_length=100)
    awdr = models.FloatField(null=True)
    prev_crop=  models.CharField(max_length=100)
    price_mean= models.FloatField(null=True)
    price_std=  models.FloatField(null=True)
    costmean=   models.FloatField(null=True)
    coststd=    models.FloatField(null=True)
    request_date = models.DateField(auto_now_add=True)
    CHU = models.IntegerField(null=True)
    SOM = models.FloatField(null=True)
    class Meta:
          db_table = 'dssservice_userrquestsite'


class UserTransaction(models.Model):

    usersite = models.OneToOneField(
        UserRquestSite,
        verbose_name="user request site to user trans mapping",
    )
    user = models.ForeignKey(User)
    #'0:Pending, 1: Completed'
    status = models.IntegerField(null=True,default=0)
    creation_date=models.DateField(auto_now_add=True)
    retry_count=models.IntegerField(null=True, default=0)
    #'Y:Yes,N:No'
    isEmailSent=models.CharField(max_length=1,default='N')
    request_process_time=models.IntegerField(null=True)
    class Meta:
          db_table = 'dssservice_usertransaction'



class SiteField(models.Model):
    Site_Field_Name = models.CharField(null=True, max_length=200)
    Data_Soruce = models.CharField(null=True, max_length=100)
    Province = models.CharField(null=True, max_length=100)
    Region = models.CharField(null=True, max_length=200)
    Town = models.CharField(null=True, max_length=100)
    Site = models.CharField(null=True, max_length=100)
    Field_Number = models.IntegerField(null=True)

    class Meta:
        db_table = 'dssservice_sitefield'



class PlotYield(models.Model):
    SiteFieldId=models.ForeignKey(SiteField)
    Latitude=models.FloatField(null=True)
    Longitude=models.FloatField(null=True)
    Year=models.IntegerField(null=True)
    SoilType=models.CharField(max_length=100)

    ClayRatio=models.FloatField(null=True)
    SOM=models.FloatField(null=True)

    TillType=models.CharField(null=True,max_length=100)

    # TillTypeChoices=(
    #     (0, 'No till'),
    #     (1, 'Conventional'),
    # )
    TillType_int =models.IntegerField(null=True)

    PrevCrop=models.CharField(null=True,max_length=100)
    PrevContribN_int=models.IntegerField(null=True)

    CHU=models.IntegerField(null=True)
    PPT=models.IntegerField(null=True)
    AWDR=models.FloatField(null=True)
    Nrate=models.IntegerField(null=False)
    Yield=models.FloatField(null=False)
    Source = models.CharField(max_length=150)
    Verified = models.CharField(default='N', max_length=1)

    class Meta:
        db_table = 'dssservice_plotyield'

class dsslookup(models.Model):
    fieldname = models.CharField(null=True,max_length=100)
    key     =   models.CharField(null=True,max_length=100)
    value   =   models.FloatField(null=True)

    def __str__(self):
        return self.fieldname+self.key

    class Meta:
        db_table = 'dssservice_dsslookup'

