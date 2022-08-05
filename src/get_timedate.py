from datetime import datetime
from datetime import date
from datetime import time

now = datetime.now()
nowdate = date.today()
nowtime = time(now.hour, now.minute)
date_and_time = str(nowdate)+"_"+str(nowtime)
print("############## "+date_and_time+" ##############")
