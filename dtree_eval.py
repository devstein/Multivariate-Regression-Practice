'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score



def evaluatePerformance(numTrials = 100):
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree

    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape

    #run 100 trial
    folds = 10

    accuracyResults = []  
    stumpAccuracyResults = []
    DT3AccuracyResults = []

    for t in range(numTrials):
        # shuffle the data at beggining of each trial 
        idx = np.arange(n)
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
        
        clf = tree.DecisionTreeClassifier()
        clfStump = tree.DecisionTreeClassifier(max_depth=1)
        clfDT3 = tree.DecisionTreeClassifier(max_depth=3)

        # split the data 
        for i in range(folds):
            #pick 10% of the data for testing   
            start =  int(round((len(X)/10.0) * i))
            stop = int(round((len(X)/10.0) * (i+1) ))

            XFirstHalf = X[0:start, :]
            XSecondHalf = X[stop:, :]

            Xtrain = np.concatenate((XFirstHalf, XSecondHalf))

            Xtest = X[start:stop, :]

            yFirstHalf = y[0:start, :]
            ySecondHalf = y[stop:, :]

            ytrain = np.concatenate((yFirstHalf, ySecondHalf))

            ytest = y[start:stop, :]
            
            #unbounded lvl
            clf = clf.fit(Xtrain,ytrain)

            y_pred = clf.predict(Xtest)

            accuracyResults.append(accuracy_score(ytest, y_pred))

            #stump
            clfStump = clfStump.fit(Xtrain, ytrain)

            y_predStump = clfStump.predict(Xtest)

            stumpAccuracyResults.append(accuracy_score(ytest, y_predStump))

            #lvl 3
            clfDT3 = clfDT3.fit(Xtrain, ytrain)

            y_predDT3 = clfDT3.predict(Xtest)

            DT3AccuracyResults.append(accuracy_score(ytest, y_predDT3))
            
    # TODO: update these statistics based on the results of your experiment
    meanDecisionTreeAccuracy = np.mean(accuracyResults)  
    stddevDecisionTreeAccuracy = np.std(accuracyResults)

    meanDecisionStumpAccuracy = np.mean(stumpAccuracyResults)
    stddevDecisionStumpAccuracy = np.std(stumpAccuracyResults)

    meanDT3Accuracy = np.mean(DT3AccuracyResults)
    stddevDT3Accuracy = np.std(DT3AccuracyResults)

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print "Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")"
    print "Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")"
    print "3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")"
# ...to HERE.
