"""passapp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from . import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    #by default load this sign in page
    url(r'^$', views.signin, name='signin'),
    url(r'^signup$', views.signup, name='signup'),
    url(r'^logoutUser', views.logoutUser, name='logoutUser'),
    url(r'^index', views.index, name='index'),
    url(r'^authenticateUser', views.authenticateUser, name='authenticateUser'),
    url(r'^registerUser', views.registerUser, name='registerUser'),
    url(r'^saveUserRequest', views.saveUserRequest, name='saveUserRequest'),
    url(r'^(?P<id>[0-9]+)/confirmUser/$', views.confirmUser, name='confirmUser'),
    url(r'^addtrails', views.addtrails, name='addtrails'),
    url(r'^submittrails', views.submittrails, name='submittrails'),
    url(r'^loadadminview', views.loadadminview, name='loadadminview'),
    url(r'^(?P<id>[0-9]+)/verifyTrail/$', views.verifyTrail, name='verifyTrail'),
    url(r'^verifySelected', views.verifySelected, name='verifySelected'),
    url(r'^deleteSelected', views.deleteSelected, name='deleteSelected'),
    url(r'^reportview', views.reportview, name='reportview'),
    url(r'^imputeMissingValues', views.imputeMissingValues, name='imputeMissingValues'),

    url(r'^downloadAllDBTrails', views.downloadAllDBTrails, name='downloadAllDBTrails'),
    url(r'^uploadTrialsInDB', views.uploadTrialsInDB, name='uploadTrialsInDB'),

    url(r'^aboutusview', views.aboutusview, name='aboutusview'),
    url(r'^description', views.descriptionview, name='description'),

    #url(r'^(?P<id>[0-9]+)/detail/$', views.detail, name='detail'),

]