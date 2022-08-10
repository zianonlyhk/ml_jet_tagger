import os  # for pwd/cwd
import subprocess  # automatically log to a txt file
from pathlib import Path  # for mkdir and touch
# obtain information about the task
from datetime import datetime
from datetime import date
from datetime import time

# working with directory
cwd = os.getcwd()
results_dir = cwd+'/results'
temp_dir = results_dir+'/temp'
# working with date and time
now = datetime.now()
nowdate = date.today()
nowtime = time(now.hour, now.minute)
date_and_time = str(nowdate)+"_"+str(nowtime)
# prepare the destination of logging output
log_file_addr = temp_dir+'/'+date_and_time+".txt"
try:  # sub dir 1, for outputs of all the submitted tasks
    Path(results_dir).mkdir()
except FileExistsError:
    pass
try:  # sub dir 2, for this specific run
    Path(temp_dir).mkdir()
except FileExistsError:
    pass
Path(log_file_addr).touch()  # create the log file

# run the task here
with open(log_file_addr, "w+") as output:
    # call "user_interface.py"
    subprocess.call(["python", cwd+"/user_interface.py"], stdout=output)
    # rename the output folder using the date and time when the task was submitted
    subprocess.run(["mv", temp_dir, results_dir+"/"+date_and_time])
