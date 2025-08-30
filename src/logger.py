import logging 
import os
from datetime import datetime

# gives the current time
LOG_FILE=f"{datetime.now().strftime("%m_%d_%Y_%H_%M_%s")}.log"

# Combining logs with present current directory
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)

# creating the path
os.makedirs(logs_path,exist_ok=True)

# final log path
LOG_FILE_PATH=os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)