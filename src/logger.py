import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" # Log file name with timestamp

logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE) # This line constructs the path for the log file, placing it in a 'logs' directory within the current working directory.

os.makedirs(logs_path, exist_ok=True) # This line creates the 'logs' directory if it does not already exist and ensures that the directory structure is created as needed.

LOG_FILE_PATH  = os.path.join(logs_path, LOG_FILE) # This line constructs the full path for the log file.

logging.basicConfig(
    filename = LOG_FILE_PATH, # This line sets up the logging configuration, specifying the file where logs will be stored.
    format = '[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s', # This line defines the format of the log messages, including the timestamp, line number
    level = logging.INFO # This line sets the logging level to INFO, meaning that messages at this level and above will be logged.
) # This line initializes the logging system with the specified configuration.

if __name__ == "__main__": # This line checks if the script is being run directly (not imported as a module).
    logging.info("Logging has been set up successfully.") # This line logs an informational message indicating that the logging setup is complete.