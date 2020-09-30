class RangeProofException(Exception):
    pass

class ProvingException(RangeProofException):
    """Exception raised for errors in range proof generation.

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message):
        self.message = message

class VerificationException(RangeProofException):
    """Exception raised for errors in range proof verification.

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message):
        self.message = message