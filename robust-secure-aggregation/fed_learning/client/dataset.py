

class DataSet(object):
    
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train / 255.0
        self.y_train = y_train
        self.x_test = x_test / 255.0
        self.y_test = y_test
