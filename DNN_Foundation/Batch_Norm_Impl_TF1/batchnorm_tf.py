import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def init_weight(M1, M2):
    return np.random.randn(M1, M2) * np.sqrt(2. / M1)


class HiddenLayerBatchNorm(object):

    def __init__(self, M1, M2, f):
        self.f = f
        self.W = tf.Variable(init_weight(M1, M2).astype(np.float32))
        self.gamma = tf.Variable(np.ones(M2).astype(np.float32))
        self.beta = tf.Variable(np.zeros(M2).astype(np.float32))
        self.running_mean = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)
        self.running_var = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)

    def forward(self, X, is_training, decay=.9):
        activations = tf.matmul(X, self.W)
        if is_training:
            batch_mean, batch_var = tf.nn.moments(activations, [0])
            update_running_mean = tf.assign(self.running_mean, decay*self.running_mean + (1-decay)*self.running_mean)
            update_running_var = tf.assign(self.running_var, decay*self.running_var + (1-decay)*self.running_var)
            with tf.control_dependencies([update_running_mean, update_running_var]):
                out = tf.nn.batch_normalization(activations, batch_mean, batch_var, self.gamma, self.beta, 1e-4)
        else:
            out = tf.nn.batch_normalization(activations, self.running_mean, self.running_var, self.gamma, self.beta, 1e-4)
        return self.f(out)


class LinearTransformation(object):

    def __init__(self, M1, M2):
        self.W = tf.Variable(init_weight(M1, M2).astype(np.float32))
        self.b = tf.Variable(np.zeros(M2).astype(np.float32))

    def forward(self, X):
        return tf.matmul(X, self.W) + self.b


class ANN(object):

    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = []

    def set_session(self, session):
        self.session = session

    def fit(self, X, Y, Xtest, Ytest, activation=tf.nn.relu, learning_rate=1e-2, print_period=100, batch_sz=100, show_fig=True, epochs=15):
        N, D = np.shape(X)
        M1 = D
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayerBatchNorm(M1, M2, activation)
            self.layers.append(h)
            M1 = M2
        K = len(set(Y))
        h = LinearTransformation(M1, K)
        self.layers.append(h)
        tfX = tf.placeholder(tf.float32, shape=(None, D), name='X')
        tfY = tf.placeholder(tf.int32, shape=(None,), name='Y')
        self.tfX = tfX
        logits = self.forward(tfX, is_training=True)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tfY))
        training_op = tf.train.MomentumOptimizer(learning_rate, momentum=.9, use_nesterov=True).minimize(cost)
        test_logits = self.forward(tfX, is_training=False)
        self.predict_op = tf.argmax(test_logits, axis=1)
        self.session.run(tf.global_variables_initializer())
        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                start_idx = j * batch_sz
                end_idx = start_idx + batch_sz
                Xbatch = X[start_idx:end_idx]
                Ybatch = Y[start_idx:end_idx]
                c, _, lgts = self.session.run([cost, training_op, logits], feed_dict={tfX: Xbatch, tfY: Ybatch})
                costs.append(c)
                if (j + 1) % print_period == 0:
                    acc = np.mean(Ybatch == np.argmax(lgts, axis=1))
                    print(f"epoch: {i}/{epochs}, batch: {j}/{n_batches}, cost: {c}, accuracy: {acc}")
            print(f"Train accuracy: {self.score(X, Y)}, test accuracy: {self.score(Xtest, Ytest)}")
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X, is_training):
        out = X
        for layer in self.layers[:-1]:
            out = layer.forward(out, is_training)
        return self.layers[-1].forward(out)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.tfX: X})


def main():
    df = pd.read_csv('train.csv')
    data = df.values.astype(np.float32)
    X = data[:, 1:]
    Y = data[:, 0]
    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std
    ann = ANN([500, 300])
    session = tf.InteractiveSession()
    ann.set_session(session)
    ann.fit(Xtrain, Ytrain, Xtest, Ytest)


if __name__ == '__main__':
    main()