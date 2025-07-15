class Error(Exception):
    """
        This exception is the generic class
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class ArgumentValueError(Error):
    """
        This exception may be raised when one argument of a function has a wrong value
    """