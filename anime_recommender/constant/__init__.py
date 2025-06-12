"""
Defining logging directory and file name
"""
from datetime import datetime 
 
LOGS_DIRECTORY:str = "Logs"
LOGS_FILENAME:str = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"
