#the dataset represents OR logic gate which is linearly separable and therefore, suitable for perceptron learning

#importing libraries
import numpy as np
import matplotlib.pyplot as plt

#creating dataset
X_or = np.array([[0,0], [0,1], [1,0] , [1,1]]) #input data points , each input has two features here x1 and x2
y_or = np.array([0,1,1,1]) #labels for each data point


#defining the perceptron class
class Perceptron: # this is the class of perceptron which has the first functiont taking parameters learning rate and epochs
    def __init__(self, learning_rate = 0.1 , epochs = 20): 
        self.lr = learning_rate
        self.weights = None
        self.bias = None #this is because we haven't yet seen the data , so we cannot define the bias or weights vector
        self.epochs = epochs
        self.errors_per_epoch = [] 
    
    #prediciting the output
    def predict(self, X): #now we know, we have dot multiplication of input vector and weight matrix followed by adding bias vector
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where (linear_output >= 0,1,0 ) #np.where (condition, value if true, value if false)

    #initialisation
    def fit(self, X, y ): 
        n_samples, n_features = X.shape
        self.bias = 0.0
        self.weights = np.zeros(n_features) #this is initialisation i.e. we are saying that all the weights are 0 for now , we did it like this because weights are present as matrix
        for _ in range(self.epochs): #number of times we repeating this updating process
            errors =0
            for xi,target in zip (X,y):
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = 1 if linear_output>=0 else 0
                update = self.lr* (target- y_pred)
                self.weights = self.weights+ update *xi
                self.bias = self.bias + update
                errors += int(update != 0) #if update is not equal to 0. i.e. error is not zero, i.e. we have some error in that case count the number of errors
                self.errors_per_epoch.append(errors)


#training the perceptron on OR data
p_or = Perceptron(learning_rate= 0.1, epochs =20)
p_or.fit(X_or, y_or) #giving our input as parameters - input values and true labels
print("Weights:" , p_or.weights)
print("Bias:", p_or.bias)
print("Predictions:", p_or.predict(X_or))

