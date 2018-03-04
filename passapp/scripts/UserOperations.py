from django.contrib.auth.models import User
from django.core.mail import send_mail,EmailMessage, BadHeaderError

def createUser(firstname,email,lastname,password):
    try:

        #creating new user wth attributes and retunred to user object
        user = User.objects.create_user(username=email, first_name=firstname,last_name=lastname,email=email,password= password,is_active=False)
        print('user sucessfully created, user id ', user.id)
        # At this point, user is a User object that has already been saved
        # to the database. You can continue to change its attributes
        # if you want to change other fields.

        #user.set_password('secret')
        #user.save()
        #print(' user save is successfull')

    except Exception:
        return None
    return  user

def userActivation(userId):
    try:
        user = User.objects.get(id=userId)
        user.is_active = True
        print("user email is verified and confirmed for userid " , user.id)
        user.save()
        return user
    except Exception as ex:
        print('Exception in  activating user, useroperations,', ex)
    return None


def findUserbyUsername(username):
    user=User.objects.get(email=username)
    return  user
def isUserActive(userrname):
    user = User.objects.get(email=userrname)
    return  user.is_active

def sendEmailNotification(userName):
    print('Start, UserOperations, sendEmailNotification()')

    user=User.objects.get(username=userName)
    subject="NumericAg - Acknowledgment of your request"
    message ='Hello '+user.first_name +'  \n \n Your request has been received. You will receive an e-mail with computation results in a few minutes.'\
    '\n \n Thank you for your submission \n\n NumericAg'

    from_email= 'passagridss@gmail.com'
    try:
        send_mail(subject, message, from_email, [user.email])

        print('Email sent successfully')

    except BadHeaderError:
        print("Error in sending email, please check error "+str(BadHeaderError))
        return False

    return True

def sendVerificationEmail(user, link):
    print('start , UserOperations, sendVerificationEmail()')
    print('Link formed :'+ link)
    subject = "NumericAg, Please verify your E-Mail."
    message = ' <html> <body> <p> <span>Welcome, <b>' + user.first_name + '</b> </span><br>' \
            '<br>Thank you for registering with NumericAg. <br>.' \
            ' <br/>  <a href= '+link+' > please click here to verify and activate your profile. </a>  or copy paste below link in browser<br>'+link+'</p></body></html>'

    from_email = 'passagridss@gmail.com'
    try:
        msg = EmailMessage(subject, message, from_email, [user.email])
        msg.content_subtype = "html"  # Main content is now text/html
        msg.send()

        print('Email sent successfully')

    except BadHeaderError:
        print("Error in sending email, please check error " + str(BadHeaderError))
        return False
    print('end , UserOpeartions, sendVerificationEmail()')
    return True





def send_email(to_list, subject, message, sender="Aircourts <noreply@aircourts.com>"):
    msg = EmailMessage(subject, message, sender, to_list)
    msg.content_subtype = "html"  # Main content is now text/html
    return msg.send()