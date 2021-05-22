from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

from util import getData, softmax, cost2, y2indicator, error_rate, relu
from sklearn.utils import shuffle


class ANN(object):
    def __init__(self, M1, M2):
        self.M1 = M1
        self.M2 = M2

    def fit(self, X, Y, Xvalid, Yvalid, learning_rate=1e-6, reg=1e-6, epochs=1000, show_fig=False):

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
        cache_W3 = 1
        cache_b3 = 1
        cache_W2 = 1
        cache_b2 = 1
        cache_W1 = 1
        cache_b1 = 1
        decay_rate = 0.999
        eps = 1e-10
        for i in range(epochs):
            # forward propagation and cost calculation
            pY, Z1, Z2 = self.forward(X)

            delta_3 = pY - T
            gW3 = Z2.T.dot(delta_3) + reg * self.W3
            gb3 = np.sum(delta_3, axis=0) + reg * self.b3

            delta_2 = delta_3.dot(self.W3.T) * (1 - Z2 * Z2)
            gW2 = Z1.T.dot(delta_2) + reg * self.W2
            gb2 = np.sum(delta_2, axis=0) + reg * self.b2

            delta_1 = delta_2.dot(self.W2.T) * (1 - Z1 * Z1)
            gW1 = X.T.dot(delta_1) + reg * self.W1
            gb1 = np.sum(delta_1, axis=0) + reg * self.b1

            cache_W3 = decay_rate * cache_W3 + (1 - decay_rate) * gW3 * gW3
            cache_b3 = decay_rate * cache_b3 + (1 - decay_rate) * gb3 * gb3

            cache_W2 = decay_rate * cache_W2 + (1 - decay_rate) * gW2 * gW2
            cache_b2 = decay_rate * cache_b2 + (1 - decay_rate) * gb2 * gb2

            cache_W1 = decay_rate * cache_W1 + (1 - decay_rate) * gW1 * gW1
            cache_b1 = decay_rate * cache_b1 + (1 - decay_rate) * gb1 * gb1

            self.W3 -= learning_rate * gW3 / (np.sqrt(cache_W3) + eps)
            self.b3 -= learning_rate * gb3 / (np.sqrt(cache_b3) + eps)

            self.W2 -= learning_rate * gW2 / (np.sqrt(cache_W2) + eps)
            self.b2 -= learning_rate * gb2 / (np.sqrt(cache_b2) + eps)

            self.W1 -= learning_rate * gW1 / (np.sqrt(cache_W1) + eps)
            self.b1 -= learning_rate * gb1 / (np.sqrt(cache_b1) + eps)

            if i % 10 == 0:
                pYvalid, _, _ = self.forward(Xvalid)
                c = cost2(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
                print("epoch:", i, "cost:", c, "error:", e)
                if e < best_validation_error:
                    best_validation_error = e
        print("best_validation_error:", best_validation_error)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        # Z = relu(X.dot(self.W1) + self.b1)
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
