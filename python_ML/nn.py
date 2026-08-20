import math
import time

import numpy as np
from tqdm import tqdm

import pandas as pd
import numpy as np


# 此文件采用相同的算法，在明文下进行实现，从而可以方便进行比较
class Net:
    def __init__(self,  input_size, layers, learning_rate=0.035, batch_size=128):
        last_size = input_size
        self.layers = []
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        for layer in layers:
            if layer['activation'] == 'linear':
                self.layers.append(Layer(last_size, layer['size'], learning_rate))
            if layer['activation'] == 'relu':
                self.layers.append(ReluLayer(last_size, layer['size'], learning_rate))
            if layer['activation'] == 'sigmoid':
                self.layers.append(SigmoidLayer(last_size, layer['size'], learning_rate))
            last_size = layer['size']

    def forward(self, x_batch):
        output = x_batch
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, error):
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def fit(self, x, y, test_x, test_y, epochs):
        iterations = 0
        l = x.shape[0]
        batch_number = math.ceil(l / self.batch_size)
        start = time.time()
        for epoch in range(epochs):
            total_error = 0
            right = 0
            print("epoch: ", epoch)
            for i in tqdm(range(batch_number)):
                if(iterations % 100 == 0):
                    print("第",iterations,"迭代")
                    self.test(test_x, test_y)
                iterations = iterations + 1
                x_batch = x[i * self.batch_size: min((i + 1) * self.batch_size, l)]
                y_batch = y[i * self.batch_size: min((i + 1) * self.batch_size, l)]
                y_pred = self.forward(x_batch)
                pred = y_pred.argmax(axis=1)
                label = y_batch.argmax(axis=1)
                error = y_pred - y_batch
                self.backward(error)
                # 计算误差
                total_error += np.sum(error * error)
                right += (pred == label).sum().item()
            # print("training error: ", total_error/l)
            # print("accuracy ", right / l)
            self.test(test_x, test_y)

        end = time.time()
        print("time:", end - start)

    def predict(self, x):
        return self.forward(x)

    def test(self, x, y):
        l = x.shape[0]
        right = 0
        y_pred = self.predict(x)
        pred = y_pred.argmax(axis=1)
        label = y.argmax(axis=1)
        right += (pred == label).sum().item()
        # batch_number = math.ceil(l / self.batch_size)
        # for i in range(batch_number):
        #     x_batch = x[i * self.batch_size: min((i + 1) * self.batch_size, l)]
        #     y_batch = y[i * self.batch_size: min((i + 1) * self.batch_size, l)]
        #     y_pred = self.predict(x_batch)
        #     # 计算准确率
        #     pred = y_pred.argmax(axis=1)
        #     label = y_batch.argmax(axis=1)
        #     right += (pred == label).sum().item()
        print("test accuracy ", right / l)


class Layer:
    def __init__(self, input_size, size, learning_rate, has_bias=False):
        self.has_bias = has_bias
        # self.weights = np.load(str(input_size) + "-" + str(size)+".npy")
        self.weights = np.random.normal(0, 0.05, size=(input_size, size))
        # self.weights = np.zeros((input_size, size))
        if has_bias:
            self.bias = np.random.random((size,))
        self.learning_rate = learning_rate

    def forward(self, input):
        self.input = input
        output = self.input.dot(self.weights)
        if self.has_bias:
            output = output + self.bias
        return output

    def backward(self, error):
        next_error = error.dot(self.weights.T)
        w_gradient = self.input.T.dot(error) * self.learning_rate / error.shape[0]
        self.weights = self.weights - w_gradient
        if self.has_bias:
            b_gradient = error.sum() * self.learning_rate / error.shape[0]
            self.bias = self.bias - b_gradient
        return next_error


class ReluLayer(Layer):
    def forward(self, input):
        input = super(ReluLayer, self).forward(input)
        self.sign = input > 0
        output = self.sign * input
        return output

    def backward(self, error):
        error = error * self.sign
        return super(ReluLayer, self).backward(error)


class SigmoidLayer(Layer):
    def forward(self, input):
        input = super(SigmoidLayer, self).forward(input)
        output = 1 / (1 + np.exp(-input))
        self.output = output
        return output

    def backward(self, error):
        error = error * self.output * (1 - self.output)
        return super(SigmoidLayer, self).backward(error)


def load_test_data():
    path = "../data/mnist/mnist_test.csv"
    test_data = pd.read_csv(path, header=None)
    header = ['label']
    for i in range(784):
        header.append(str(i))
    test_data.columns = header
    # change to array and normalized
    test_images = test_data.drop(['label'], axis=1).to_numpy().reshape(-1,  784) / 255.0
    test_labels = test_data['label'].to_numpy()
    # change to one_hot
    test_labels = np.eye(10)[test_labels]
    return test_images, test_labels


def load_train_data():
    # 读入数据
    path = "../data/mnist/mnist_train.csv"
    train_data = pd.read_csv(path, header=None)
    header = ['label']
    for i in range(784):
        header.append(str(i))
    train_data.columns = header
    # change to array and normalized
    train_images = train_data.drop(['label'], axis=1).to_numpy().reshape(-1,  784) / 255.0
    train_labels = train_data['label'].to_numpy()
    # change to one_hot
    train_labels = np.eye(10)[train_labels]
    return train_images, train_labels

def nn_test():
    my_x, my_y = load_train_data()
    features, labels = load_test_data()
    layers = [
        {'size': 128, 'activation': 'linear'},
        {'size': 128, 'activation': 'relu'},
        {'size': 128, 'activation': 'linear'},
        {'size': 128, 'activation': 'relu'},
        {'size': 10, 'activation': 'linear'}
    ]
    net = Net(input_size=784, layers=layers)
    net.fit(my_x, my_y, features, labels, 5)

nn_test()