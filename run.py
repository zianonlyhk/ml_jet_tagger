import os
import subprocess
from pathlib import Path
from datetime import datetime
from datetime import date
from datetime import time

cwd = os.getcwd()
results_dir = cwd+'/results'
log_dir = results_dir+'/logs/'

now = datetime.now()
nowdate = date.today()
nowtime = time(now.hour, now.minute)
date_and_time = str(nowdate)+"_"+str(nowtime)

log_file_addr = log_dir+date_and_time+".txt"

try:
    Path(results_dir).mkdir()
except FileExistsError:
    pass
try:
    Path(log_dir).mkdir()
except FileExistsError:
    pass
Path(log_file_addr).touch()

with open(log_file_addr, "w+") as output:
    subprocess.call(["python", cwd+"/src/get_timedate.py"], stdout=output)
    subprocess.call(["python", cwd+"/user_interface.py"], stdout=output)
