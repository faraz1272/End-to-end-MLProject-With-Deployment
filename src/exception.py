## Python's sys module provides functions and variables used to manipulate different parts of the Python runtime environment. 
# It's a powerful tool for handling system-specific parameters and functions.

import sys
from src.logger import logging  # Importing the logging module from the src package to handle logging of messages.


def error_message_detail(error, error_detail:sys):
    """ This function takes an error and an error_detail object,
        and returns a formatted error message with details about the exception.
    """
    _,_,exc_tb = error_detail.exc_info() # This line unpacks the exception information into three variables: _, _, and exc_tb.

    file_name = exc_tb.tb_frame.f_code.co_filename # This line retrieves the name of the file where the exception occurred.

    error_message = "Error occured in Python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error) # Converting the error to a string for better readability
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """ Custom exception class that inherits from the built-in Exception class.
            It initializes with an error and its details, and formats the error message.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        """ Returns the formatted error message when the exception is converted to a string. """
        return self.error_message


if __name__ == "__main__":
    try:
        a=1/0  # This line will raise a ZeroDivisionError
    except Exception as e:
        logging.info("Division by zero error occurred")
        raise CustomException(e, sys)  # This line raises the custom exception with the error details.