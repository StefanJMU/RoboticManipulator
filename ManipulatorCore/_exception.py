class ControlException(Exception) :

    def __init__(self, message : str) :
    
        super(ControlException,self).__init__(message)
        self.message = message
        
     