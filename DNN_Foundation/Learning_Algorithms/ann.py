# Backpropagation implementation for the Fully Connected Neural Network with 2 hidden layers

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

from util import getData, softmax, cost2, y2indicator, error_rate
from sklearn.utils import shuffle


class ANN(object):
    def __init__(self, M1, M2):
        self.M1 = M1
        self.M2 = M2

    def fit(self, X, Y, Xvalid, Yvalid, learning_rate=1e-6, reg=1e-6, epochs=500, show_fig=False):

        N, D = X.shape
        K = len(set(Y))
        T = y2indicator(Y)

        self.W1 = np.random.randn(D, self.M1) / np.sqrt(D)
        self.b1 = np.zeros(self.M1)

        self.W2 = np.random.randn(self.M1, self.M2) / np.sqrt(self.M1)
        self.b2 = np.zeros(self.M2)

        self.W3 = np.random.randn(self.M2, K) / np.sqrt(self.M2)
        self.b3 = np.zeros(K)

        costs = []
        best_validation_error = 1
        for i in range(epochs):
            # forward propagation and cost calculation
            pY, Z1, Z2 = self.forward(X)

            # gradient descent step
            delta_3 = pY - T
            self.W3 -= learning_rate * (Z2.T.dot(delta_3) + reg * self.W3)
            self.b3 -= learning_rate * (np.sum(delta_3, axis=0) + reg * self.b3)

            delta_2 = delta_3.dot(self.W3.T)*(1 - Z2*Z2)
            self.W2 -= learning_rate * (Z1.T.dot(delta_2) + reg * self.W2)
            self.b2 -= learning_rate * (np.sum(delta_2, axis=0) + reg * self.b2)

            delta_1 = delta_2.dot(self.W2.T)*(1 - Z1*Z1)
            self.W1 -= learning_rate * (X.T.dot(delta_1) + reg * self.W1)
            self.b1 -= learning_rate * (np.sum(delta_1, axis=0) + reg * self.b1)

            if i % 10 == 0:
                pYvalid, _, _ = self.forward(Xvalid)
                c = cost2(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
                print("i:", i, "cost:", c, "error:", e)
                if e < best_validation_error:
                    best_validation_error = e
        print("best_validation_error:", best_validation_error)

        if show_fig:
            plt.plot(costs)
            plt.show()


    def forward(self, X):
        Z1 = np.tanh(X.dot(self.W1) + self.b1)
        Z2 = np.tanh(Z1.dot(self.W2) + self.b2)
        return softmax(Z2.dot(self.W3) + self.b3), Z1, Z2

    def predict(self, X):
        pY, _, _ = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


def main():
    Xtrain, Ytrain, Xvalid, Yvalid = getData()
    
    model = ANN(200, 100)
    model.fit(Xtrain, Ytrain, Xvalid, Yvalid, reg=0, show_fig=True)
    print(model.score(Xvalid, Yvalid))

if __name__ == '__main__':
    main()
