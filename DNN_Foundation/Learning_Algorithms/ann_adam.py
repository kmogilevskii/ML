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

        # 1st moment
        mW1 = 0
        mb1 = 0
        mW2 = 0
        mb2 = 0
        mW3 = 0
        mb3 = 0

        # 2nd moment
        vW1 = 0
        vb1 = 0
        vW2 = 0
        vb2 = 0
        vW3 = 0
        vb3 = 0

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = 1
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

            mW1 = beta1 * mW1 + (1 - beta1) * gW1
            mb1 = beta1 * mb1 + (1 - beta1) * gb1
            mW2 = beta1 * mW2 + (1 - beta1) * gW2
            mb2 = beta1 * mb2 + (1 - beta1) * gb2
            mW3 = beta1 * mW3 + (1 - beta1) * gW3
            mb3 = beta1 * mb3 + (1 - beta1) * gb3

            vW1 = beta2 * vW1 + (1 - beta2) * gW1 * gW1
            vb1 = beta2 * vb1 + (1 - beta2) * gb1 * gb1
            vW2 = beta2 * vW2 + (1 - beta2) * gW2 * gW2
            vb2 = beta2 * vb2 + (1 - beta2) * gb2 * gb2
            vW3 = beta2 * vW3 + (1 - beta2) * gW3 * gW3
            vb3 = beta2 * vb3 + (1 - beta2) * gb3 * gb3

            # bias correction
            correction1 = 1 - beta1 ** t
            hat_mW1 = mW1 / correction1
            hat_mb1 = mb1 / correction1
            hat_mW2 = mW2 / correction1
            hat_mb2 = mb2 / correction1
            hat_mW3 = mW3 / correction1
            hat_mb3 = mb3 / correction1


            correction2 = 1 - beta2 ** t
            hat_vW1 = vW1 / correction2
            hat_vb1 = vb1 / correction2
            hat_vW2 = vW2 / correction2
            hat_vb2 = vb2 / correction2
            hat_vW3 = vW3 / correction2
            hat_vb3 = vb3 / correction2

            t += 1

            self.W3 -= learning_rate * hat_mW3 / (np.sqrt(hat_vW3) + eps)
            self.b3 -= learning_rate * hat_mb3 / (np.sqrt(hat_vb3) + eps)

            self.W2 -= learning_rate * hat_mW2 / (np.sqrt(hat_vW2) + eps)
            self.b2 -= learning_rate * hat_mb2 / (np.sqrt(hat_vb2) + eps)

            self.W1 -= learning_rate * hat_mW1 / (np.sqrt(hat_vW1) + eps)
            self.b1 -= learning_rate * hat_mb1 / (np.sqrt(hat_vb1) + eps)

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
