'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.01, n_iter=100):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.JHist = None
    

    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        n,d = X.shape
        self.JHist = []
        for i in xrange(self.n_iter):
            self.JHist.append( (self.computeCost(X, y, theta), theta) )
            print "Iteration: ", i+1, " Cost: ", self.JHist[i][0], " Theta: ", theta
            # TODO:  add update equation here
            # tempTheta = np.copy(theta)
            # for j in range(d):
            #     sum = 0
            #     for k in range(n):
            #         sum = sum + (np.dot(X[k], theta) - y[k]) * X[k,j]
            #     sum = sum * self.alpha / n
            #     tempTheta[j] = tempTheta[j] - sum
            # theta = np.copy(tempTheta)
            theta =  np.dot(np.linalg.inv(( np.dot(np.transpose(X), X) ) ), np.transpose(X)) * y
        return theta
    

    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
              ** make certain you don't return a matrix with just one value! **
        '''

        # for i in range(n):
        #     cost += ( np.dot(X[i], theta) - y[i] ) ** 2
        # cost = cost/(2*n)

        n = len(X)

        cost = (np.transpose(np.dot(X,theta) - y) * (np.dot(X,theta)- y) ) / (2*n)


        return cost.item(0)


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n = len(y)
        n,d = X.shape
        if self.theta==None:
            self.theta = np.matrix(np.zeros((d,1)))
        self.theta = self.gradientDescent(X,y,self.theta)    


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        # TODO:  add prediction function here