class RawToOrganized:


    def __init__(self, X,y):

        self.deep=X
        self.wide=y[:,0,0].reshape(-1,1)
        self.labels=y[:,1,0]


