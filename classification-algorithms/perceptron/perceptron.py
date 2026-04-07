import numpy as np

class Perceptron:
    def __init__(self, eta= 0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        #generate random numbers using the state
        rgen = np.random.RandomState(self.random_state)

        #generate random weights in normal distribution
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.errors_ = []
        
        #loop over the number of iterations
        for _ in range(self.n_iter):
            #keep track of the error
            error = 0
            for xi, target in zip(X, y):
                #plot the update function
                update = self.eta * (target - self.predict(xi)) 
                
                #perform the update
                self.w_ += update * xi
                self.b_ += update

                #the error for this run
                error += int(update != 0)

            #add the error to the array of errors
            self.errors_.append(error)


    def net_input(self, X):
        
        return np.dot(X, self.w_) + self.b_
    
    #predicts the ouput y for that input of x
    def predict(self, X):
        #return the decisions for the preceptron
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    

        


